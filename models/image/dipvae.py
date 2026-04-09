from models.image.image import ImageEncoder
from models.image.image import generate_encoder, generate_decoder
from utils import compute_lamda
import torch
from typing import List, Tuple, Dict

class DIPVAE(ImageEncoder):
    """
    from Variational Inference of Disentangled Latent Concepts from Unlabeled Observations
    Kumer et al 2017
    arXiv:1711.00848
    """
    def __init__(self,
                 encoder_specs:dict,
                 decoder_specs:dict,
                 x_dims:List[int],
                 z_dim:int,
                 beta: float = 1,
                 lamda_od:float = 1,
                 lamda_d:float = 1,
                 mode: int = 1,
                 loss:str = "mse",):
        super().__init__()
        assert mode in [1,2], "Mode must be either 1 or 2"
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
        self.lamda_od = lamda_od
        self.lamda_d = lamda_d
        self.loss_fn = loss
        self.mode = mode

    def split(self,
              coding: torch.Tensor) :
        
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

        # DIP Loss
        if self.mode == 1:
            cov = torch.cov(Z.T)
        elif self.mode == 2:
            cov = torch.cov(Z.T) + torch.diag_embed(torch.exp(enc_logvar))

        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        off_diag = cov - torch.diag_embed(diag)

        diag_loss = torch.nn.MSELoss()(diag, torch.ones_like(diag))
        offdiag_loss = torch.nn.MSELoss()(off_diag, torch.zeros_like(off_diag))

        loss = {
            "kl": loss_kl,
            "rec": loss_rec,
            "diag": diag_loss,
            "offdiag": offdiag_loss
        }

        coeff = {
            "kl": compute_lamda(self.beta, iter),
            "rec": 1.,
            "diag": compute_lamda(self.lamda_d, iter),
            "offdiag": compute_lamda(self.lamda_od, iter)
        }

        if return_sampled:
            return loss, coeff, Z
        else:
            return loss, coeff, enc_mu