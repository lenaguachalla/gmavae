from models.image.image import ImageEncoder
from models.image.image import generate_encoder, generate_decoder
from utils import compute_lamda
from torch import nn
import torch
import numpy as np
from typing import List, Tuple, Dict

class LatentSpace:
    def __init__(self,
                 z_dim:int,
                 steps:int = 10):
        self.z_dim = z_dim
        self.steps = steps
        self.t_dim = None
    def project(self,
                z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self,
               mu:torch.Tensor,
               logt:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def kl_loss(self, mu:torch.Tensor ,logt:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def logt_clipper(self, logt:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def average(self, z:torch.Tensor) -> torch.Tensor:
        #z is of shape (batch, m, z_dim)
        raise NotImplementedError

class EuclidianSpace(LatentSpace):
    def __init__(self,
                 z_dim:int,
                 steps:int = 10):
        self.z_dim = z_dim
        self.steps = steps
        self.t_dim = z_dim

    def sample(self,
               mu:torch.Tensor,
               logt:torch.Tensor) -> torch.Tensor:
        epsilon = torch.normal(torch.zeros_like(mu), 1.0)
        z_sample = mu + torch.exp(logt) * epsilon
        return z_sample
    def project(self, z):
        return z
    
    def kl_loss(self, mu, logt):
        kl_loss = - 0.5 * torch.sum(1 + 2*logt - torch.square(mu) - torch.exp(2*logt), axis=-1)
        return kl_loss.mean()
    
    def logt_clipper(self, logt:torch.Tensor) -> torch.Tensor:
        return logt
    
    def average(self, z:torch.Tensor) -> torch.Tensor:
        #z is of shape (batch, m, z_dim)
        return z.mean(dim=-2)
    
class HyperSphericalSpace(LatentSpace):
    def __init__(self,
                 z_dim:int,
                 steps:int = 10,
                 min_logt:float = -10,
                 max_logt:float = -5,
                 fix_logt: bool = False):
        self.z_dim = z_dim
        self.t_dim = 1
        self.steps = steps
        self.min_logt = min_logt
        self.max_logt = max_logt
        self.fix_logt = fix_logt

    def sample(self,
               mu:torch.Tensor,
               logt:torch.Tensor) -> torch.Tensor:
        z = mu
        for _ in range(self.steps):
            eps = torch.normal(torch.zeros_like(mu), 1.0)
            step = torch.exp(0.5 * logt) * eps / np.sqrt(self.steps)
            z = self.project(z + step)
        return z
    def project(self, z):
        return torch.nn.functional.normalize(z, p=2, dim=-1)
    
    def kl_loss(self, mu, logt):
        scalar_curv = self.z_dim * (self.z_dim - 1)
        logt = torch.squeeze(logt, dim=-1)
        kl_loss = - self.z_dim * logt / 2.0 \
                  + scalar_curv * torch.exp(logt) / 4.0
        return kl_loss.mean()
    
    def logt_clipper(self, logt):
        if self.fix_logt:
            return self.max_logt * torch.ones_like(logt)
        mean_logt = (self.min_logt + self.max_logt) / 2.0
        window = (self.max_logt - self.min_logt) / 2.0
        logt = nn.Tanh()(logt) * window + mean_logt
        return logt
    
    def average(self, z:torch.Tensor) -> torch.Tensor:
        #z is of shape (batch, m, z_dim)
        z = torch.sum(z, dim=-2)
        return self.project(z)

def get_latent_space(params:Dict) -> LatentSpace:
    params = params.copy()
    type = params.pop("type")
    if type == "euclidian":
        return EuclidianSpace(**params)
    elif type == "hyperspherical":
        return HyperSphericalSpace(**params)
    else:
        raise ValueError(f"Unknown latent space type: {type}")

class DiffVAE(ImageEncoder):
    """
    from Diffusion Variational Autoencoders
    Rey et al 2020
    10.24963/ijcai.2020/375
    """
    def __init__(self,
                 encoder_specs:dict,
                 decoder_specs:dict,
                 x_dims:List[int],
                 latent_spaces:List[dict],
                 loss: str = "mse",
                 beta:float = 1.0,):
        super().__init__()

        self.latent_spaces = [get_latent_space(params) for params in latent_spaces]
        self.z_dim = sum([ls.z_dim for ls in self.latent_spaces]) 
        self.x_dims = x_dims
        self.loss_fn = loss
        self.beta = beta

        #encoder
        z_dim_encoder = self.z_dim + sum([ls.t_dim for ls in self.latent_spaces]) 
        self.encoder = generate_encoder(encoder_specs,
                                        x_dims,
                                        z_dim_encoder)
        
        #decoder
        self.decoder = generate_decoder(decoder_specs,
                                        self.z_dim,
                                        x_dims)
    
    def split(self,
              encoding: torch.Tensor) :
        mus = []
        logts = []
        kz = 0
        kt = 0
        for ls in self.latent_spaces:
            mus.append(encoding[...,kz:kz+ls.z_dim])
            logts.append(encoding[...,self.z_dim+kt:self.z_dim+kt+ls.t_dim])
            kz+=ls.z_dim
            kt+=ls.t_dim
        mus = [ls.project(mu) for mu,ls in zip(mus,self.latent_spaces)]
        logts = [ls.logt_clipper(logt) for logt, ls in zip(logts, self.latent_spaces)]

        return mus, logts
    
    def forward(self,
                X:torch.Tensor,
                sample: bool = False
                ) -> torch.Tensor:
        return self.decode(self.encode(X,sample=sample),sample=sample)
    
    def split_z(self, z:torch.Tensor) -> List[torch.Tensor]:
        zs = []
        k = 0
        for ls in self.latent_spaces:
            zs.append(z[...,k:k+ls.z_dim])
            k+=ls.z_dim
        return zs
    
    def average(self, z:torch.Tensor) -> torch.Tensor:
        zs = self.split_z(z)
        return torch.cat([ls.average(z) for z, ls in zip(zs, self.latent_spaces)], dim=-1)
    
    def project(self, z:torch.Tensor) -> torch.Tensor:
        zs = self.split_z(z)
        return torch.cat([ls.project(z) for z, ls in zip(zs, self.latent_spaces)], dim=-1)
    
    def encode(self,
               X:torch.Tensor,
               sample:bool = False
               ) -> torch.Tensor:
        encoding = self.encoder(X)
        mus, logts = self.split(encoding)

        if sample: 
            zs = [ls.sample(mu, logt) for mu, logt, ls in zip(mus, logts, self.latent_spaces)]
        else:
            zs = mus

        return torch.cat(zs, dim=-1)
        
    def decode(self,
               Z:torch.Tensor,
               sample:bool = False
               ) -> torch.Tensor:
        return self.decoder(Z)
    

    def loss(self,
             X:torch.Tensor,
             return_sampled:bool = False,
             iter: int = None
             ) -> Tuple[Dict[str,torch.Tensor], Dict[str,float], torch.Tensor]:

        # KL LOSS
        encoding = self.encoder(X)
        enc_mus, enc_logts = self.split(encoding)
        losses_kl = [ls.kl_loss(mu, logt) for mu, logt, ls in zip(enc_mus, enc_logts, self.latent_spaces)]
        loss_kl = torch.sum(torch.stack(losses_kl))

        # RECONSTRUCTION LOSS
        Zs = [ls.sample(mu, logt) for mu, logt, ls in zip(enc_mus, enc_logts, self.latent_spaces)]
        Z = torch.cat(Zs, dim=-1)

        dec_mu = self.decoder(Z)
        if self.loss_fn == "mse":
            loss_rec = nn.MSELoss()(dec_mu, X)
        elif self.loss_fn == "bce":
            loss_rec = nn.BCELoss()(dec_mu, X)
        else:
            raise NotImplementedError

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
            return loss, coeff, torch.cat(enc_mus, dim=-1)