from models import generate_image_encoder
import torch.nn as nn
import torch
from algos.algos import Algo
from models.action.rotation import RotationActionEncoder
from models.action.rotation_continuous import RotationContinuousActionEncoder
from typing import List, Tuple, Dict
from utils import compute_lamda


class SOBDRL(Algo):
    """
    from Learning Disentangled Representations and Group Structure of Dynamical Environments
    Quessard et al 2020
    arXiv:2002.06991
    """
    def __init__(self,
                 nfo: dict,
                 z_dim: int,
                 lamda_p: float,
                 image_specs: dict,
                 action_specs: dict,
                 device = "cpu",
                 lamda_d: float = 0,
                 action_lr_rate: float = 1.,
                 ):
        super().__init__()
        
        self.device = device
        x_dims: List[int] = nfo["x_dims"]
        n_action: int = nfo["n_action"]

        image_specs["x_dims"] = x_dims
        image_specs["z_dim"] = z_dim
        self.z_dim = z_dim
        self.image_encoder = generate_image_encoder(image_specs)
        assert "rotation" in action_specs["type"], "SOBDRL only works with rotation action encoder"
        if "continuous" in action_specs["type"] :
            action_input_dim = nfo["action_dim"]
            self.action_encoder = RotationContinuousActionEncoder(action_input_dim, action_specs["hidden_dims"], z_dim)
        else :
            self.action_encoder = RotationActionEncoder(n_action, z_dim)
        self.lamda_p = lamda_p
        self.lamda_d = lamda_d
        
        self.action_lr_rate = action_lr_rate

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
        B,m = A.shape[:2] # sequence length
        
        # PREDICTION LOSS
        Az = self.encode_action(A)
        error = torch.zeros(B,m).float().to(self.device) # shape [B,m]
        Z_hat = self.image_encoder.encode(X[:,0], sample = True)
        for i in range(m):
            Z_hat = torch.einsum('bij,bj->bi',Az[:,i],Z_hat)
            Xp_hat = self.image_encoder.decode(Z_hat, sample = True)
            if self.image_encoder.loss_fn == "mse":
                error[:,i] = nn.MSELoss()(Xp_hat, X[:,i+1])
            elif self.image_encoder.loss_fn == "bce":
                error[:,i] = nn.BCELoss()(Xp_hat, X[:,i+1])
        prediction_loss = error.mean()

        # DISENTANGLEMENT LOSS
        theta_squared = (self.action_encoder.theta**2).sort(dim=-1)[0]
        dist_loss = theta_squared[...,:-1].mean()

        loss = {
            "prediction": prediction_loss,
            "dist": dist_loss,
        }

        coeff = {
            "prediction": self.lamda_p,
            "dist": compute_lamda(self.lamda_d, self.counter_iter),
        }

        if not eval :
            self.counter_iter += 1

        return loss, coeff
    
    @property
    def params_lr_coeff(self):
        return [{"params": self.image_encoder.parameters(), "lr": 1.0},
                {"params": self.action_encoder.parameters(), "lr": self.action_lr_rate}]