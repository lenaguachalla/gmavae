from models.image.image import ImageEncoder
from models.image.image import generate_encoder, generate_decoder
from utils import compute_lamda
import torch
import numpy as np
from typing import List, Tuple, Dict

class BetaVAE(ImageEncoder):
    """
    from beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
    Higgins et al 2017
    openreview.net/forum?id=Sy2fzU9gl
    """
    def __init__(self,
                 encoder_specs:dict,
                 decoder_specs:dict,
                 x_dims:List[int],
                 z_dim:int,
                 beta:float = 1,
                 loss:str = "mse",
                 remove_enc_mu_norm: bool = False,
                 fixed_sigma: float = None):
        super().__init__()
        if fixed_sigma :
            z_dim_encoder = z_dim
        else :
            z_dim_encoder = 2*z_dim
        self.encoder = generate_encoder(encoder_specs,
                                        x_dims,
                                        z_dim_encoder)
        self.decoder = generate_decoder(decoder_specs,
                                        z_dim,
                                        x_dims)

        self.z_dim = z_dim
        self.x_dims = x_dims
        self.beta = beta
        self.fixed_sigma = fixed_sigma
        self.remove_enc_mu_norm = remove_enc_mu_norm
        self.loss_fn = loss

    def split(self,
              coding: torch.Tensor) :
        if self.fixed_sigma:
            mu = coding
            logvar = torch.ones_like(mu) * 2 * np.log(self.fixed_sigma)
        else :
            dim = coding.shape[-1]//2
            mu, logvar = coding[...,:dim], coding[...,dim:]

        return mu, logvar

    
    def encode(self,
               X:torch.Tensor,
               sample:bool = False
               ) -> torch.Tensor:
        encoding = self.encoder(X)
        mu, logvar = self.split(encoding)

        if sample: 
            sigma = torch.exp(0.5*logvar)
            epsilon = torch.normal(torch.zeros_like(mu), 1.0)
            Z = mu + sigma * epsilon

            return Z
        else:
            return mu
        
    def decode(self,
               Z:torch.Tensor,
               sample:bool = False
               ) -> torch.Tensor:
        mu = self.decoder(Z)

        return mu

    def forward(self,
                X:torch.Tensor,
                sample: bool = False
                ) -> torch.Tensor:
        return self.decode(self.encode(X,sample=sample),sample=sample)

    def loss(self,
             X:torch.Tensor,
             return_sampled:bool = False,
             iter: int = None,
             ) -> Tuple[Dict[str,torch.Tensor], Dict[str,float], torch.Tensor]:

        # KL LOSS
        encoding = self.encoder(X)
        enc_mu, enc_logvar = self.split(encoding)
        if self.remove_enc_mu_norm:
            loss_kl = (- enc_logvar\
                    + torch.exp(enc_logvar)).mean() / 2
        else :
            loss_kl = (- enc_logvar\
                    + torch.exp(enc_logvar) +\
                    + enc_mu**2).mean() / 2

        # RECONSTRUCTION LOSS
        # sampling
        enc_sigma = torch.exp(0.5*enc_logvar)
        epsilon = torch.normal(torch.zeros_like(enc_mu), 1.0).to(enc_sigma.device)
        Z = enc_mu + enc_sigma * epsilon

        dec_mu = self.decoder(Z)

        if self.loss_fn == "mse":
            loss_rec = torch.nn.MSELoss()(dec_mu,X)
        elif self.loss_fn == "bce":
            loss_rec = torch.nn.BCELoss()(dec_mu,X)

        loss = {
            "kl": loss_kl,
            "rec": loss_rec
        }

        coeff = {
            "kl": compute_lamda(self.beta, iter),
            "rec": 1.
        }

        if return_sampled:
            return loss, coeff, Z
        else:
            return loss, coeff, enc_mu