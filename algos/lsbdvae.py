from typing import List, Dict, Tuple
from models import generate_action_encoder, generate_image_encoder
from models.image.diffvae import DiffVAE
from utils import maps_tree, compute_lamda
import torch.nn as nn
import torch
from algos.algos import Algo


class LSBDVAE(Algo):
    """
    from Quantifying and Learning Linear Symmetry-Based Disentanglement
    Tonnaer et al 2022
    arXiv:2011.06070
    """
    def __init__(self,
                 nfo:Dict, 
                 image_specs: Dict,
                 action_specs: Dict,
                 lamda_r: float = 0,
                 lamda_a: float = 0,
                 device:str  = "cpu",
                 action_loss_on_sampled = True,
                 ):
        
        super().__init__()

        x_dims: List[int] = nfo["x_dims"]
        n_action: int = nfo["n_action"]
        image_specs["x_dims"] = x_dims

        assert image_specs["type"] == "diffvae", "LSBDVAE only works with DiffVAE"
        self.image_encoder: DiffVAE = generate_image_encoder(image_specs)

        action_specs["z_dim"] = self.image_encoder.z_dim
        action_specs["n_action"] = n_action

        self.action_encoder = generate_action_encoder(action_specs)
        assert self.image_encoder.z_dim == self.action_encoder.z_dim, "Image and action encoders must have the same z_dim"
        
        self.z_dim = self.image_encoder.z_dim

        self.lamda_r = lamda_r
        self.lamda_a = lamda_a
        self.action_loss_on_sampled = action_loss_on_sampled

        if device is None :
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else : 
            self.device = device

    def encode_action(self, A):
        Ashape = A.shape
        A = A.reshape(-1)
        Az = self.action_encoder(A)
        Az = Az.reshape(Ashape + (self.z_dim,self.z_dim))
        return Az

    def apply_action(self,
                     Z, 
                     A):
        Az = self.encode_action(A)
        Z = torch.einsum('bij,bj->bi',Az,Z)
        Z = self.image_encoder.project(Z)
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
                                                                             return_sampled=True,
                                                                             iter=self.counter_iter)
        Z = Z.reshape(-1, m+1, self.z_dim)
        
        # ACTION LOSS
        Az = self.encode_action(A)
        error = torch.zeros_like(A).float()
        for i in range(m):
            Z_hat = torch.einsum('bij,bj->bi',Az[:,i],Z[:,i])

            Zs = torch.stack([Z[:,i+1],Z_hat],dim=1) # shape [B,2,z_dim]
            p = self.image_encoder.average(Zs) # projection of mean onto latent space

            error[:,i] = .5 * nn.MSELoss()(Z_hat, p) +\
                         .5 * nn.MSELoss()(Z[:,i+1], p)
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