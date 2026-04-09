from models import generate_image_encoder, generate_action_encoder
import torch.nn as nn
import torch
from algos.algos import Algo
from typing import List, Tuple, Dict
from utils import compute_lamda


class HAE(Algo):
    """
    from Homomorphism Autoencoder — Learning Group Structured Representations  from Observed Transitions
    Keurti et al 2022
    arXiv:2207.12067
    """
    def __init__(self,
                 nfo: dict,
                 z_dim: int,
                 lamda_r: float,
                 lamda_p: float,
                 image_specs: dict,
                 action_specs: dict,
                 device = None,
                 ):
        super().__init__()
        if device is None :
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else : 
            self.device = device
        x_dims: List[int] = nfo["x_dims"]

        image_specs["x_dims"] = x_dims
        image_specs["z_dim"] = z_dim
        self.z_dim = z_dim
        action_specs["input_dim"] = nfo["action_dim"]
        self.image_encoder = generate_image_encoder(image_specs)
        self.action_encoder = generate_action_encoder(action_specs)

        self.lamda_p = lamda_p
        self.lamda_r = lamda_r

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
        B,m = A.shape[:2]

        Z = self.image_encoder.encode(X.flatten(0,1))
        Z = Z.reshape(-1, m+1, self.z_dim)
        
        
        Az = self.encode_action(A)
        latent_error = torch.zeros(B,m).float().to(self.device)
        obs_error = torch.zeros(B,m+1).float().to(self.device)

        Z_hat = Z[:,0]
        X_hat = self.image_encoder.decode(Z_hat)
        obs_error[:,0] = nn.MSELoss()(X_hat, X[:,0])

        for i in range(m):
            Z_hat = torch.einsum('bij,bj->bi',Az[:,i],Z_hat)

            # PREDICTION LOSS
            latent_error[:,i] = nn.MSELoss()(Z_hat,Z[:,i+1])

            # RECONSTRUCTION LOSS
            X_hat = self.image_encoder.decode(Z_hat)
            obs_error[:,i+1] = nn.MSELoss()(X_hat, X[:,i+1])

        prediction_loss = latent_error.mean()
        reconstruction_loss = obs_error.mean()

        loss = {
            "prediction": prediction_loss,
            "reconstruction": reconstruction_loss,
        }

        coeff = {
            "prediction": self.lamda_p,
            "reconstruction": compute_lamda(self.lamda_r, self.counter_iter),
        }

        if not eval :
            self.counter_iter += 1

        return loss, coeff