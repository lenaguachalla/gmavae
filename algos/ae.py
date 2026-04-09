from typing import List, Dict, Tuple
from models import generate_image_encoder
import torch
from algos.algos import Algo

class AE(Algo):
    """
    Classical AutoEncoder
    """
    def __init__(self,
                 nfo:Dict, 
                 z_dim: int,
                 image_specs: Dict,
                 device: str = "cpu",
                 ):
        
        super().__init__()

        self.device = device

        x_dims: List[int] = nfo["x_dims"]

        image_specs["x_dims"] = x_dims
        image_specs["z_dim"] = z_dim

        self.z_dim = z_dim
        self.image_encoder = generate_image_encoder(image_specs)

        self.counter_iter = 0
        
    def encode_action(self, A):
        raise ValueError("No Action Encoder in AE")
    
    def encode_image(self,
                     X,
                     A = None,
                     sample = False) :
        assert A is None, "No Action Encoder in AE"
        Z = self.image_encoder.encode(X, sample)
        return Z
    
    def decode_image(self,
                     Z,
                     sample = False) :
        return self.image_encoder.decode(Z, sample)

    def compute_loss(self,
             X:torch.Tensor, #[B,m+1,...]
             A:torch.Tensor, #[B,m]
             eval:bool = False) -> Tuple[Dict, Dict]:
        
        loss, coeffs, _ = self.image_encoder.loss(X.flatten(0,1), iter=self.counter_iter)

        loss = {
            "image": loss,
        }

        coeff = {
            "image": coeffs,
        }

        if not eval :
            self.counter_iter += 1

        return loss, coeff