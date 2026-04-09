from models.action.action import ActionEncoder
from models.architectures.mlp import MLP
from torch import nn
import torch

class DenseContinuousActionEncoder(ActionEncoder):
    """
    Action encoder parametrized by an MLP that outputs the full matrix.
    """
    def __init__(self,
                 input_dim:int,
                 hidden_dims: list,
                 z_dim:int,
                 activation_fn: str = None,
                 normalize: str = None,
                 ):
        super().__init__()
        self.encoder = MLP(input_dim=input_dim,
                           hidden_dim=hidden_dims,
                           output_dims=z_dim**2)
        self.z_dim = z_dim
        self.normalize = normalize
        self.activation_fn = activation_fn

    
    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        batch_sizes = a.shape[:-1]
        a = a.reshape((-1, a.shape[-1]))
        B = a.shape[0]

        if self.activation_fn is None:
            activation_fn = lambda x: x
        elif self.activation_fn == "tanh" :
            activation_fn = lambda x: torch.tanh(x)
        elif self.activation_fn == "matrix_exp" :
            activation_fn = lambda x: torch.linalg.matrix_exp(x)
        else :
            raise ValueError(f"Activation function {activation_fn} not supported")


        Az = activation_fn(self.encoder(a).reshape((B,self.z_dim, self.z_dim)))
        if self.normalize == True :
            Az= nn.functional.normalize(Az,dim=(-1,-2), eps=1e-5)
        elif type(self.normalize) in [float, int] :
            Az= nn.functional.normalize(Az,dim=(-1,-2), eps=1e-5) * self.normalize
        elif self.normalize == "line" :
            Az= nn.functional.normalize(Az,dim=-2, eps=1e-5)
        elif self.normalize == "column" :
            Az= nn.functional.normalize(Az,dim=-1, eps=1e-5)

        Az = Az.reshape(batch_sizes + (self.z_dim, self.z_dim))
        return Az