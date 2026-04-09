from models.image.image import ImageEncoder
from models.image.image import generate_encoder, generate_decoder
from utils import compute_lamda
from torch import nn
import torch
from typing import List, Tuple, Dict

"""
A classical AutoEncoder (AE) for images.
"""

class AE(ImageEncoder):
    def __init__(self,
                 encoder_specs:dict,
                 decoder_specs:dict,
                 x_dims:List[int],
                 z_dim:int,
                 normalize: bool = False,
                 loss: str = "mse",
                 z_noise: float = 0):
        super().__init__()
        self.x_dims = x_dims
        self.encoder = generate_encoder(encoder_specs,
                                        x_dims,
                                        z_dim)
        self.decoder = generate_decoder(decoder_specs,
                                        z_dim,
                                        x_dims)
        self.normalize = normalize
        self.z_noise = z_noise
        self.loss_fn = loss

    def encode(self,
               X:torch.Tensor,
               sample: bool = False,
               iter: int = None
               ) -> torch.Tensor :
        Z = self.encoder(X)
        if self.normalize == True:
            Z = nn.functional.normalize(Z,dim=1,eps=1e-5)
        if iter is not None :
            z_noise = compute_lamda(self.z_noise, iter)
            if z_noise != 0 and sample:
                Z = Z + torch.randn_like(Z) * z_noise
        return Z
    
    def decode(self,
               Z:torch.Tensor,
               sample: bool = False,
               **kwargs
               ) -> torch.Tensor :
        X_hat = self.decoder(Z, **kwargs)
        return X_hat
    
    def forward(self,
                X:torch.Tensor,
                sample: bool = False
                ) -> torch.Tensor:
        return self.decode(self.encode(X,sample=sample),sample=sample)


    def loss(self,
             X:torch.Tensor,
             return_sampled:bool = True,
             iter:int = None
             ) -> Tuple[Dict[str,torch.Tensor], Dict[str,float], torch.Tensor]:
        Z = self.encode(X, sample=False)
        if iter is not None and compute_lamda(self.z_noise, iter) != 0:
            z_noise = compute_lamda(self.z_noise, iter)
            Zl = Z + torch.randn_like(Z) * z_noise
        else:
            Zl = Z
        X_hat = self.decode(Zl)
        if self.loss_fn == "mse" :
            loss_rec = nn.MSELoss()(X_hat,X)
        elif self.loss_fn == "bce" :
            loss_rec = nn.BCELoss()(X_hat,X)

        loss = {
            "rec": loss_rec
        }
        coeff = {
            "rec": 1.
        }

        if return_sampled :
            return loss, coeff, Zl
        else :
            return loss, coeff, Z
