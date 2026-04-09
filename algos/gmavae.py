from typing import List, Dict, Tuple
from models import generate_action_encoder, generate_image_encoder
from utils import maps_tree, compute_lamda
import torch.nn as nn
import torch
from algos.algos import Algo

# Our proposed method

class GMAVAE(Algo):
    """
    Our proposed method: (Group Masked) Action-based VAE
    """
    def __init__(self,
                 nfo:Dict, 
                 z_dim: int,
                 image_specs: Dict,
                 action_specs: Dict,
                 group_masking: bool, # whether to use group masking though pi
                                      # False: A-VAE; True: GMA-VAE
                 lamda_r: float = 0,
                 lamda_a: float = 0,
                 lamda_d: float = 0,
                 device: str = "cpu",
                 target_d: Dict = None, #target value for the disentanglement loss
                 focal_loss: float = None,
                 groups: List[List[int]] = None #groups of actions
                 ):
        
        super().__init__()


        self.device = device

        x_dims: List[int] = nfo["x_dims"]
        self.n_action: int = nfo["n_action"]
        groups = nfo["group"] if groups is None else groups

        image_specs["x_dims"] = x_dims
        image_specs["z_dim"] = z_dim

        action_specs["z_dim"] = z_dim
        assert "dense" in action_specs["type"], "GMAVAE only works with dense action encoder"
        if "continuous" in action_specs["type"] :
            # if the action encoder is continuous
            assert focal_loss is None, "Focal loss not implemented for continuous action encoder"
            self.continuous_action_encoder = True
            action_specs["input_dim"] = nfo["action_dim"]
        else : 
            self.continuous_action_encoder = False
            action_specs["n_action"] = self.n_action

            # initialize group masks
            self.groups = [-1] * self.n_action
            c=0
            for g in groups:
                for x in g:
                    self.groups[x] = c
                c+=1
            self.groups = torch.tensor(self.groups).int().to(self.device)

        self.z_dim = z_dim
        self.image_encoder = generate_image_encoder(image_specs)
        self.action_encoder = generate_action_encoder(action_specs)
        self.lamda_r = lamda_r
        self.lamda_a = lamda_a
        self.lamda_d = lamda_d
        self.target_d = target_d
        self.focal_loss = focal_loss
        self.group_masking = group_masking

        #initialize dist to get pi thourgh softmax
        self.n_groups = len(groups)
        self.dist = nn.Parameter(torch.normal(0,0,size = (self.n_groups, z_dim-self.n_groups), device=self.device),
                                    requires_grad=True)

    @property
    def pi(self):
        pi = torch.softmax(self.dist, dim=0)

        # force first n_group dims to be attributed to each group
        I = torch.eye(self.n_groups).to(self.device)
        pi = torch.cat([I, pi], dim=1)
        
        return pi # shape [n_group, z_dim]
    
    def encode_action(self, A):
        Az = self.action_encoder(A)

        #pification
        if self.group_masking :
            batch_size = Az.shape[:-2]

            # get pi
            if self.continuous_action_encoder :
                # assume that the first dimension of action corresponds to the group index
                pi = self.pi[A[...,0].long()]  # shape [..., z_dim]
            else :
                pi = self.pi[self.groups[A]]  # shape [..., z_dim]
            
            # compute mask
            Pi = torch.einsum('...i,...j->...ij', pi, pi) # shape [..., z_dim, z_dim]

            I = torch.eye(self.z_dim).to(self.device)
            I = I.reshape(tuple(1 for _ in range(len(batch_size))) + (self.z_dim, self.z_dim))
            I = I.repeat(*(batch_size + (1,1)))

            # apply mask to Az
            Az = Pi * (Az - I) + I

        return Az
    
    def apply_action(self,
                     Z, 
                     A):
        Az = self.encode_action(A)
        Z = torch.einsum('bij,bj->bi',Az,Z)
        return Z
    
    def decode_image(self,
                     Z,
                     sample = False) :
        return self.image_encoder.decode(Z, sample)

    def compute_loss(self,
             X:torch.Tensor, #[B,m+1,...]
             A:torch.Tensor, #[B,m,...]
             eval:bool = False) -> Tuple[Dict, Dict]:
        B,m = A.shape[:2] # sequence length
        
        # IMAGE ENCODER LOSS
        image_encoder_loss, image_encoder_coeff, Z = self.image_encoder.loss(X.flatten(0,1),
                                                                             return_sampled=False,
                                                                             iter=self.counter_iter)
        Z = Z.reshape(-1, m+1, self.z_dim)
        
        # ACTION LOSS
        Az = self.encode_action(A) # shape [B,m,z_dim,z_dim]
        error = torch.zeros(B,m).float().to(self.device) # shape [B,m]
        for i in range(m):
            Z_hat = torch.einsum('bij,bj->bi',Az[:,i],Z[:,i])
            error[:,i] = ((Z[:,i+1] - Z_hat)**2).mean(dim=1)
        
        weights = torch.ones_like(error)
        if self.focal_loss :
            error_group = torch.zeros_like(error).to(self.device)
            for a in range(self.n_action) :
                if a in A:
                    error_group[A==a] = error[A == a].mean()
            w = error_group.detach()**self.focal_loss
            if w.sum() > 1e-10 :
                weights = error_group.detach()**self.focal_loss
                
        action_loss = (error * weights).sum()/weights.sum()

        # DISENTANGLEMENT LOSS
        dist_loss = -(torch.exp(self.dist) * self.dist).sum(axis=0) \
                    /torch.exp(self.dist).sum(axis=0) \
                    +torch.logsumexp(self.dist, dim = 0)

        if self.target_d is not None :
            target_d = compute_lamda(self.target_d, self.counter_iter)
            dist_loss = torch.abs(target_d - dist_loss)
        dist_loss = dist_loss.mean()
            

        loss = {
            "image": image_encoder_loss,
            "action": action_loss,
            "dist": dist_loss,
        }

        coeff = {
            "image": maps_tree(image_encoder_coeff, lambda x : self.lamda_r * x),
            "action": compute_lamda(self.lamda_a,self.counter_iter),
            "dist": compute_lamda(self.lamda_d, self.counter_iter),
        }
        if not eval :
            self.counter_iter += 1

        return loss, coeff