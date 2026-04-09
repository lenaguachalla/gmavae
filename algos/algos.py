import torch.nn as nn
import torch
from copy import deepcopy
from typing import Tuple, Dict

class Algo(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter_iter = 0

    def compute_loss(self,
             X:torch.Tensor, #[B,m+1,...]
             A:torch.Tensor, #[B,m,...]
             eval:bool = False) -> Tuple[Dict, Dict]:
        """
        X [B,M+1,...]: observations (float)
        A [B,M]: actions (int)
        """
        raise NotImplementedError
    
    def encode_image(self,
                     X:torch.Tensor,
                     A:torch.Tensor = None,
                     sample: bool = False
                     ) -> torch.Tensor :
        """
        X [B,...]: observations (float)
        A [B,M,...]: actions (int)
        """
        Z = self.image_encoder.encode(X, sample)

        if A is not None :
            for i in range(A.shape[1]):
                Z = self.apply_action(Z, A[:,i])
        
        return Z
    
    def apply_action(self,
                     Z:torch.Tensor,
                     A:torch.Tensor) -> torch.Tensor:
        """
        Apply the action A on the latent representation Z.
        Z [B,z_dim]: latent representation
        A [B,...]: action
        """
        raise NotImplementedError

    def decode_image(self,
                     Z:torch.Tensor) -> torch.Tensor :
        raise NotImplementedError
    
    def encode_action(self, A:torch.Tensor) -> torch.Tensor :
        raise NotImplementedError
    
    def forward(self,
                X:torch.Tensor,
                A:torch.Tensor = None,
                sample: bool = False) -> torch.Tensor :
        """
        X [B,...]: observations (float)
        A [B,M,...]: actions (int)
        """
        Z = self.encode_image(X, A, sample)
        return self.decode_image(Z)
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    @property
    def params_lr_coeff(self):
        return [{"params": self.parameters(), "lr": 1.0}]

    def save(self, path: str):
        torch.save(self, path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device(self.device), weights_only = False).state_dict())

    def reset_counter_iter(self):
        self.counter_iter = 0

def generate_algo(specs: dict,
                  nfo: dict) -> Algo:
    type = specs["type"]
    tspecs = deepcopy(specs)
    del tspecs["type"]

    match type:
        case "gmavae" :
            from algos.gmavae import GMAVAE as Algo
        case "sobdrl" :
            from algos.sobdrl import SOBDRL as Algo
        case "lsbdvae" :
            from algos.lsbdvae import LSBDVAE as Algo
        case "forward" :
            from algos.forward import Forward as Algo
        case "ae" :
            from algos.ae import AE as Algo
        case "hae" :
            from algos.hae import HAE as Algo
        case _ :
            raise ValueError(f"Unknown algorithm type: {type}")
    
    return Algo(**tspecs, nfo = nfo)