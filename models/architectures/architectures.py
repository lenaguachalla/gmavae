import torch
from copy import deepcopy

def generate_nn(specs: dict) -> torch.nn.Module:
    type = specs["type"]
    tspecs = deepcopy(specs)
    del tspecs["type"]
    if type == "mlp" :
        from models.architectures.mlp import MLP as Model
    elif type == "conv2d" :
        from models.architectures.cnn import Conv2D as Model
    elif type == "deconv2d" :
        from models.architectures.cnn import Deconv2D as Model
    else :
        raise ValueError(type)
    
    return Model(**tspecs)

def get_activation_fn(name: str) -> torch.nn.Module:
    if name is None :
        return torch.nn.Identity()
    elif name == "relu" :
        return torch.nn.ReLU()
    elif name == "leaky_relu" :
        return torch.nn.LeakyReLU()
    elif name == "sigmoid" :
        return torch.nn.Sigmoid()
    elif name == "tanh" :
        return torch.nn.Tanh()
    else :
        raise ValueError(name)
    

class ConcatenatedModule(torch.nn.Module):
    def __init__(self, modules):
        super(ConcatenatedModule, self).__init__()
        self.ms = torch.nn.ModuleList(modules)

    def forward(self, x):
        for module in self.ms:
            x = module(x)
        return x