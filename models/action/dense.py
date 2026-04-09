from models.action.action import ActionEncoder
from torch import nn
import torch

class DenseActionEncoder(ActionEncoder):
    """
    Action encoder with a learnable matrix for each action.
    """
    def __init__(self,
                 n_action:int,
                 z_dim:int,
                 sigma:float = 1,
                 activation_fn: str = None,
                 normalize: str = None,
                 ):
        super().__init__()
        self.encoder = nn.Parameter(torch.normal(0,sigma,size = (n_action, z_dim**2)),
                                    requires_grad=True)
        self.z_dim = z_dim
        self.normalize = normalize
        self.activation_fn = activation_fn

    
    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        if self.activation_fn is None:
            activation_fn = lambda x: x
        elif self.activation_fn == "tanh" :
            activation_fn = lambda x: torch.tanh(x)
        else :
            raise ValueError(f"Activation function {activation_fn} not supported")


        Az = activation_fn(self.encoder)[a].reshape(a.shape + (self.z_dim, self.z_dim))
        if self.normalize == True :
            Az= nn.functional.normalize(Az,dim=(-1,-2), eps=1e-5)
        elif type(self.normalize) in [float, int] :
            Az= nn.functional.normalize(Az,dim=(-1,-2), eps=1e-5) * self.normalize
        elif self.normalize == "line" :
            Az= nn.functional.normalize(Az,dim=-2, eps=1e-5)
        elif self.normalize == "column" :
            Az= nn.functional.normalize(Az,dim=-1, eps=1e-5)
        return Az