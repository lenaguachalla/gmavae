from models.action.action import ActionEncoder
from torch import nn
import numpy as np
import torch
from typing import List

# from Symmetry-Based Disentangled Representation Learning requires Interaction with Environments
# Caselles-Dupré et al 2019
# arXiv:1904.00243

class ForwardActionEncoder(ActionEncoder):
    def __init__(self,
                 n_action:int,
                 z_dim:int,
                 groups: List[List[int]], #list of groups of actions
                 dims: List[int], #list of dimensions given for each group
                 sigma:float = 1,
                 activation_fn: str = None,
                 ):
        super().__init__()
        assert len(groups) == len(dims), "groups and dims must have the same length"
        assert sum([len(g) for g in groups]) == n_action, "groups must cover all actions"
        assert sum(dims) == z_dim, "dims must sum to z_dim"
        self.groups = [np.array(g) for g in groups]
        n_parameters = sum([len(g)*d**2 for g,d in zip(groups,dims)])
        self.n_action = n_action
        self.encoder = nn.Parameter(torch.normal(0,sigma,size = (n_parameters,)),
                                    requires_grad=True)
        self.z_dim = z_dim
        self.dims = dims
        self.activation_fn = activation_fn

    def encode(self) -> torch.Tensor:
        if self.activation_fn is None:
            activation_fn = lambda x: x
        elif self.activation_fn.startswith("tanh") :
            if "_" in self.activation_fn :
                c = float(self.activation_fn.split("_")[1])
            else :
                c = 1
            
            activation_fn = lambda x: c*torch.tanh(x)
        else :
            raise ValueError(f"Activation function {activation_fn} not supported")
        Az = torch.eye(self.z_dim).reshape((1, self.z_dim, self.z_dim)).repeat(self.n_action, 1, 1).to(self.encoder.device)
        id = 0
        ie = 0
        for g,d in zip(self.groups, self.dims):
            for a in g:
                Az[a, id:id+d, id:id+d] = activation_fn(self.encoder)[ie:ie+d**2].reshape((d,d))
                ie += d**2
            id += d
        return Az
    
    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        return self.encode()[a]
