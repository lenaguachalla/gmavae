from models.action.action import ActionEncoder
from torch import nn
import torch
from typing import List


class ConcatActionEncoder(ActionEncoder):
    """
    Concatenate multiple action encoders into a single encoder.
    The resulting matrices will be block diagonal, with each block corresponding to an encoder.
    """
    def __init__(self,
                 action_encoders: List[ActionEncoder]
                 ):
        super().__init__()
        self.encoders = nn.ModuleList(action_encoders)
        self.z_dim = sum([encoder.z_dim for encoder in action_encoders])
    
    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        Az = None 
        kz = 0
        for encoder in self.encoders:
            Az_ = encoder(a)
            z_dim = encoder.z_dim
            if Az is None :
                Az = torch.zeros(Az_.shape[:-2] +(self.z_dim, self.z_dim)).to(a.device)
            Az[..., kz:kz+z_dim, kz:kz+z_dim] = Az_
            kz += z_dim

        return Az