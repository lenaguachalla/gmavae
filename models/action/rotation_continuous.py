from models.action.action import ActionEncoder
from models.architectures.mlp import MLP
import torch

class RotationContinuousActionEncoder(ActionEncoder):
    """
    Action encoder parametrized by an MLP that outputs the angles for the rotation matrix.
    """
    def __init__(self,
                 input_dim:int,
                 hidden_dims: list,
                 z_dim:int):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = MLP(input_dim=input_dim,
                           hidden_dim=hidden_dims,
                           output_dims=z_dim*(z_dim-1)//2)
        self.theta = None

    def compute_matrices(self) -> torch.Tensor :
        theta = self.theta
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        R = torch.zeros(theta.shape[:-1] + (self.z_dim,self.z_dim)).to(self.theta.device)
        R[...,:,:] = torch.eye(self.z_dim)

        k = 0
        for i in range(self.z_dim - 1) :
            for j in range(i+1, self.z_dim) :
                Ri = torch.zeros(theta.shape[:-1] + (self.z_dim,self.z_dim)).to(self.theta.device)
                Ri[...,:,:] = torch.eye(self.z_dim)
                Ri[...,i,i] = cos[...,k]
                Ri[...,i,j] = sin[...,k]
                Ri[...,j,j] = cos[...,k]
                Ri[...,j,i] = -sin[...,k]
                
                R = torch.einsum('...ij,...jk->...ik', R, Ri)

                k+=1

        return R
    
    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        batch_sizes = a.shape[:-1]
        a = a.reshape((-1, a.shape[-1]))
        self.theta = self.encoder(a)
        self.theta = self.theta.reshape(batch_sizes + (self.theta.shape[-1],))
        return self.compute_matrices()