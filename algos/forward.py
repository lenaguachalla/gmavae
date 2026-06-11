from typing import List, Tuple, Dict
from models import generate_action_encoder, generate_image_encoder
from utils import maps_tree, compute_lamda
import torch
from algos.algos import Algo

class Forward(Algo):
    """
    from Symmetry-Based Disentangled Representation Learning requires Interaction with Environments
    Caselles-Dupré et al 2019
    arXiv:1904.00243
    """
    def __init__(self,
                 nfo:dict, 
                 z_dim: int,
                 image_specs: dict,
                 action_specs: dict,
                 lamda_r: float = 0,
                 lamda_a: float = 0,
                 device: str = "cpu",
                 ):
        
        super().__init__()

        x_dims: List[int] = nfo["x_dims"]
        self.n_action: int = nfo["n_action"]
        groups = nfo["group"]

        image_specs["x_dims"] = x_dims
        image_specs["z_dim"] = z_dim
        assert action_specs["type"] == "forward", "Only forward action encoder is supported"
        action_specs["z_dim"] = z_dim
        action_specs["n_action"] = self.n_action
        action_specs["groups"] = groups

        self.z_dim = z_dim
        self.image_encoder = generate_image_encoder(image_specs)
        self.action_encoder = generate_action_encoder(action_specs)
        self.lamda_r = lamda_r
        self.lamda_a = lamda_a
        self.device = device

    
    def encode_action(self, A):
        Az = self.action_encoder(A)
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
             A:torch.Tensor, #[B,m]
             eval:bool = False) -> Tuple[Dict, Dict]:
        m = A.shape[1] # sequence length
        
        # IMAGE ENCODER LOSS
        image_encoder_loss, image_encoder_coeff, Z = self.image_encoder.loss(X.flatten(0,1),
                                                                             iter=self.counter_iter)
        Z = Z.reshape(-1, m+1, self.z_dim)
        
        # ACTION LOSS
        Az = self.encode_action(A) # shape [B,m,z_dim,z_dim]
        error = torch.zeros_like(A).float() # shape [B,m]
        for i in range(m):
            Z_hat = torch.einsum('bij,bj->bi',Az[:,i],Z[:,i])
            error[:,i] = ((Z[:,i+1] - Z_hat)**2).mean(dim=1)
        action_loss = error.mean()

        loss = {
            "image": image_encoder_loss,
            "action": action_loss,
        }

        coeff = {
            "image": maps_tree(image_encoder_coeff, lambda x : self.lamda_r * x),
            "action": compute_lamda(self.lamda_a,self.counter_iter),
        }
        if not eval :
            self.counter_iter += 1

        return loss, coeff