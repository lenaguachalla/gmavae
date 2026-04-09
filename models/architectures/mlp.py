from torch import nn
import torch
import numpy as np
from typing import List
from models.architectures.architectures import get_activation_fn

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: List[int],
                 output_dims: List[int],
                 final_activation_fn: str = None,
                 activation_fn: str = "relu"):
        super().__init__()
        if type(output_dims) is int:
          output_dims = [output_dims]
        self.output_dims = output_dims
        if len(hidden_dim)>0:
          self.fct = nn.Sequential(nn.Linear(input_dim,hidden_dim[0]))
          for k in range(len(hidden_dim)-1) :
            self.fct.append(get_activation_fn(activation_fn))
            self.fct.append(nn.Linear(hidden_dim[k],hidden_dim[k+1]))

          self.fct.append(get_activation_fn(activation_fn))
          self.fct.append(nn.Linear(hidden_dim[-1],np.prod(output_dims)))
        else :
          self.fct = nn.Linear(input_dim,np.prod(output_dims))

        self.final_activation_fn = get_activation_fn(final_activation_fn)

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z = z.flatten(start_dim=1)
        z = self.fct(z)
        z = z.reshape([B] + self.output_dims)
        z = self.final_activation_fn(z)
        return z
