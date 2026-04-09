from models.action.action import ActionEncoder
from torch import nn
import torch

class RotationActionEncoder(ActionEncoder):
    """
    Action encoder parametrized by a SO(z_dim) rotation for each action.
    """
    def __init__(self,
                 n_action:int,
                 z_dim:int):
        super().__init__()
        self.z_dim = z_dim
        self.n_action = n_action
        self.encoder = nn.Parameter(torch.normal(0,1,size = (n_action, z_dim*(z_dim-1)//2)),
                                    requires_grad=True)

    @property
    def theta(self) -> torch.Tensor :
        A = torch.arange(self.n_action).to(self.encoder.data.device)
        return self.encoder[A]
    
    @property
    def matrices(self) -> torch.Tensor :
        theta = self.theta
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        R = torch.zeros((self.n_action,self.z_dim,self.z_dim)).to(self.encoder.data.device)
        R[...,:,:] = torch.eye(self.z_dim)

        k = 0
        for i in range(self.z_dim - 1) :
            for j in range(i+1, self.z_dim) :
                Ri = torch.zeros((self.n_action,self.z_dim,self.z_dim)).to(self.encoder.data.device)
                Ri[...,:,:] = torch.eye(self.z_dim)
                Ri[...,i,i] = cos[...,k]
                Ri[...,i,j] = sin[...,k]
                Ri[...,j,j] = cos[...,k]
                Ri[...,j,i] = -sin[...,k]
                
                R = torch.einsum('bij,bjk->bik', R, Ri)

                k+=1

        return R
    
    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        return self.matrices[a]
