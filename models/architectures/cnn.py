from torch import nn
import torch
from models.architectures.mlp import MLP
from typing import List, Tuple, Dict
from models.architectures.architectures import get_activation_fn

class Conv2D(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channels: List[int],
                 kernel_size: List[Tuple[int]],
                 stride: List[Tuple[int]],
                 padding: List[Tuple[int]],
                 output_dim: int,
                 latent_dim: int,
                 hidden_dim: List[int],
                 pooling: Dict = None,
                 final_activation_fn: str = None,
                 batch_norm: bool = False,
                 activation_fn: str = "relu"):
        super().__init__()
        self.cnn = nn.Sequential()
        for k in range(len(hidden_channels)) :
            if k == 0 :
                self.cnn.append(nn.Conv2d(input_channel,
                                        hidden_channels[0],
                                        kernel_size=kernel_size[0],
                                        stride=stride[0],
                                        padding=padding[0]))
            else :
                self.cnn.append(nn.Conv2d(hidden_channels[k-1],
                                        hidden_channels[k],
                                        kernel_size=kernel_size[k],
                                        stride=stride[k],
                                        padding=padding[k]))
            if batch_norm :
                self.cnn.append(nn.BatchNorm2d(hidden_channels[k]))
            self.cnn.append(get_activation_fn(activation_fn))
            if pooling is not None :
                if pooling["type"] == "max" :
                    self.cnn.append(nn.MaxPool2d(kernel_size=pooling["kernel_size"][k],
                                                 stride=pooling["stride"][k]))
                elif pooling["type"] == "avg" :
                    self.cnn.append(nn.AvgPool2d(kernel_size=pooling["kernel_size"][k],
                                                 stride=pooling["stride"][k]))
                else :
                    raise ValueError(pooling["type"])
        self.mlp = MLP(input_dim=latent_dim,
                            hidden_dim=hidden_dim,
                            output_dims=[output_dim],
                            activation_fn=activation_fn,
                            final_activation_fn=final_activation_fn)


    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z=z.permute(0,3,1,2)
        z=self.cnn(z)
        z=torch.flatten(z,start_dim=1)
        z=self.mlp(z)
        return z
    
class Deconv2D(nn.Module):
    def __init__(self,
                 hidden_channels: List[int],
                 output_dims: List[int],
                 latent_dims: List[int],
                 kernel_size: List[Tuple[int]],
                 stride: List[Tuple[int]],
                 padding: List[Tuple[int]],
                 input_dim: int,
                 hidden_dim: List[int],
                 pooling: Dict = None,
                 final_activation_fn: str = None,
                 activation_fn: str = "relu"):
        super().__init__()
        if pooling is not None :
            raise NotImplementedError("Pooling not implemented for Deconv2D")

        self.mlp = MLP(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       output_dims=latent_dims,
                       activation_fn=activation_fn,
                       final_activation_fn=activation_fn)

        self.cnn = nn.Sequential()
        
        for k in range(len(hidden_channels)-1) :
            self.cnn.append(nn.ConvTranspose2d(hidden_channels[k],
                                            hidden_channels[k+1],
                                            kernel_size=kernel_size[k],
                                            stride=stride[k],
                                            padding=padding[k]))
            self.cnn.append(get_activation_fn(activation_fn))
            if pooling is not None :
                pass

        self.cnn.append(nn.ConvTranspose2d(hidden_channels[-1],
                                            output_dims[-1],
                                            kernel_size=kernel_size[-1],
                                            stride=stride[-1],
                                            padding=padding[-1]))
        
        self.final_activation_fn = get_activation_fn(final_activation_fn)

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z=self.mlp(z)
        z=z.permute(0,3,1,2)
        z=self.cnn(z)
        z=z.permute(0,2,3,1)
        z=self.final_activation_fn(z)
        return z
