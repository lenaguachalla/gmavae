from models.image.image import ImageEncoder
from models.image.image import generate_encoder, generate_decoder
from utils import compute_lamda
from models.architectures.architectures import generate_nn
import torch
from typing import List, Tuple, Dict

class FactorVAE(ImageEncoder):
    """
    from Disentangling by Factorising
    Kim and Mnih 2015
    arXiv:1802.05983
    """
    def __init__(self,
                 encoder_specs:dict,
                 decoder_specs:dict,
                 discriminator_specs:dict,
                 x_dims:List[int],
                 z_dim:int,
                 beta: float = 1,
                 gamma: float = 1,
                 lamda_d:float = 1,
                 loss:str = "mse",):
        super().__init__()
        z_dim_encoder = 2*z_dim
        self.encoder = generate_encoder(encoder_specs,
                                        x_dims,
                                        z_dim_encoder)
        self.decoder = generate_decoder(decoder_specs,
                                        z_dim,
                                        x_dims)
        
        discriminator_specs["input_dim"] = z_dim
        discriminator_specs["output_dims"] = [1]
        
        self.discriminator = generate_nn(discriminator_specs)

        self.z_dim = z_dim
        self.x_dims = x_dims
        self.beta = beta
        self.gamma = gamma
        self.lamda_d = lamda_d
        self.loss_fn = loss

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

    def discriminate(self,
                     Z:torch.Tensor,
                     freeze:bool = False) -> torch.Tensor:
        if freeze:
            for param in self.discriminator.parameters():
                param.requires_grad = False

        p = self.discriminator(Z).squeeze(-1)

        if freeze:
            for param in self.discriminator.parameters():
                param.requires_grad = True

        return p
    
    @staticmethod
    def permut_dims(z:torch.Tensor) -> torch.Tensor:
        """
        Permute the dimensions of z to ensure that the latent space is not biased towards any particular dimension.
        """
        B, z_dim = z.shape
        z_permuted = []
        for d in range(z_dim) :
            z_permuted.append(z[:,d][torch.randperm(B)])
        return torch.stack(z_permuted, dim=1)

    def loss(self,
             X:torch.Tensor,
             return_sampled:bool = False,
             iter: int = None,
             ) -> Tuple[Dict[str,torch.Tensor], Dict[str,float], torch.Tensor]:

        N = X.shape[0]//2
        X1, X2 = X[:N], X[N:]
        # KL LOSS
        encoding = self.encoder(X1)
        enc_mu, enc_logvar = self.split(encoding)
        loss_kl = (- enc_logvar\
                + torch.exp(enc_logvar) +\
                + enc_mu**2).mean() / 2

        # RECONSTRUCTION LOSS
        # sampling
        enc_sigma = torch.exp(0.5*enc_logvar)
        epsilon = torch.normal(torch.zeros_like(enc_mu), 1.0).to(enc_sigma.device)
        Z1 = enc_mu + enc_sigma * epsilon

        dec_mu = self.decoder(Z1)

        if self.loss_fn == "mse":
            loss_rec = torch.nn.MSELoss()(dec_mu,X1)
        elif self.loss_fn == "bce":
            loss_rec = torch.nn.BCELoss()(dec_mu,X1)

        # TC loss
        p = self.discriminate(Z1, freeze=True)
        epsilon = 1e-5
        loss_tc = torch.mean(torch.log(p/(1-p) + epsilon))

        # Discriminator loss
        Z1 = Z1.detach()
        Y1 = torch.ones(Z1.shape[0], device=Z1.device)

        with torch.no_grad():
            encoding = self.encoder(X2)
            enc_mu, enc_logvar = self.split(encoding)
            enc_sigma = torch.exp(0.5*enc_logvar)
            epsilon = torch.normal(torch.zeros_like(enc_mu), 1.0).to(enc_sigma.device)
            Z2 = enc_mu + enc_sigma * epsilon
            Z2 = self.permut_dims(Z2)
        Y2 = torch.zeros(Z2.shape[0], device=Z2.device)

        Z = torch.cat([Z1, Z2], dim=0)
        Y = torch.cat([Y1, Y2], dim=0)

        p = self.discriminate(Z, freeze=False)
        epsilon = 1e-5
        p = torch.clamp(p, epsilon, 1-epsilon)

        loss_d = torch.nn.BCELoss()(p, Y)


        loss = {
            "kl": loss_kl,
            "rec": loss_rec,
            "tc": loss_tc,
            "d": loss_d,
        }

        coeff = {
            "kl": compute_lamda(self.beta, iter),
            "rec": 1.,
            "tc": compute_lamda(self.gamma, iter),
            "d": compute_lamda(self.lamda_d, iter),
        }

        if return_sampled:
            return loss, coeff, Z
        else:
            return loss, coeff, enc_mu

    @property
    def params_lr_coeff(self):
        return [{"params": self.encoder.parameters(), "lr": 1.0},
                {"params": self.decoder.parameters(), "lr": 1.0},
                {"params": self.discriminator.parameters(), "lr": 1.0}]