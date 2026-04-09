import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from typing import List, Tuple, Dict
from models.architectures.architectures import generate_nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self,
               X:torch.Tensor,
               sample:bool = False
               ) -> torch.Tensor:
        raise NotImplementedError
    
    def decode(self,
               Z:torch.Tensor,
               sample:bool = False
               ) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, X) -> Tuple[Dict[str,torch.Tensor], Dict[str,float], torch.Tensor]:
        raise NotImplementedError
    
def generate_image_encoder(specs: Dict) -> ImageEncoder:
    type = specs["type"]
    tspecs = deepcopy(specs)
    del tspecs["type"]

    match type :
        case "ae" :
            from models.image.ae import AE as Model
        case "betavae" :
            from models.image.betavae import BetaVAE as Model
        case "diffvae" :
            from models.image.diffvae import DiffVAE as Model
        case "dipvae" :
            from models.image.dipvae import DIPVAE as Model
        case "factorvae" :
            from models.image.factorvae import FactorVAE as Model
        case _ :
            raise ValueError(f"Unknown image model type: {type}")
    
    return Model(**tspecs)

def to_tuple_list(x, n=None) :
    """
    Convert any type of input into the shape
      [(x1, y1), (x2, y2), ...]
    """
    def to_tuple(x) :
        if type(x) == int :
            return (x,x)
        elif type(x) == str :
            a,b = x.split("-")
            return (int(a),int(b))
        else:
            raise ValueError(x)
    if type(x) in [int, str] :
        return [to_tuple(x)]*n
    elif type(x) == list :
        return [to_tuple(a) for a in x]
    else:
        raise ValueError(x)
def generate_encoder(encoder_specs:dict,
                        x_dims:List[int],
                        z_dim:int) -> nn.Module:
                        
    if encoder_specs["type"] == "mlp" :
        encoder_specs["input_dim"] = np.prod(x_dims)
        encoder_specs["output_dims"] = [z_dim]
    elif encoder_specs["type"] == "conv2d" :
        assert len(x_dims) == 3

        x,y,input_channel = x_dims

        #compute output dimension according to the specs
        n=len(encoder_specs["hidden_channels"])
        padding = to_tuple_list(encoder_specs["padding"], n)
        stride = to_tuple_list(encoder_specs["stride"], n)
        kernel_size = to_tuple_list(encoder_specs["kernel_size"], n)
        pooling = encoder_specs.get("pooling", None)
        if pooling is not None :
            pooling_kernel_size = to_tuple_list(pooling["kernel_size"], n)
            pooling_stride = to_tuple_list(pooling["stride"], n)
        dilation = 1
        for k,((px, py), (sx,sy), (kx, ky)) in enumerate(zip(padding,stride,kernel_size)) :
            x = int((x+2*px - dilation*(kx-1) - 1)/sx) + 1
            y = int((y+2*py - dilation*(ky-1) - 1)/sy) + 1
            if pooling is not None :
                x = int((x-pooling_kernel_size[k][0])/pooling_stride[k][0]) + 1
                y = int((y-pooling_kernel_size[k][1])/pooling_stride[k][1]) + 1

            assert x>0, f"Invalid dimension {x}"
            assert y>0, f"Invalid dimension {y}"

        encoder_specs["padding"] = padding
        encoder_specs["stride"] = stride
        encoder_specs["kernel_size"] = kernel_size
        if pooling is not None :
            encoder_specs["pooling"]["kernel_size"] = pooling_kernel_size
            encoder_specs["pooling"]["stride"] = pooling_stride
        encoder_specs["latent_dim"] = x*y*encoder_specs["hidden_channels"][-1]
        encoder_specs["input_channel"] = input_channel
        encoder_specs["output_dim"] = z_dim

    return generate_nn(encoder_specs)

def generate_decoder(decoder_specs:dict,
                        z_dim:int,
                        x_dims:List[int]) :
    if decoder_specs["type"] == "mlp" :
        decoder_specs["input_dim"] = z_dim
        decoder_specs["output_dims"] = x_dims
    elif decoder_specs["type"] == "deconv2d" :
        assert len(x_dims) == 3
        if decoder_specs.get("pooling", None) is not None :
            raise NotImplementedError("Pooling not implemented for deconv2d")
        decoder_specs["input_dim"] = z_dim
        decoder_specs["output_dims"] = x_dims

        x,y,_ = x_dims

        #compute input dimension according to the specs
        n=len(decoder_specs["hidden_channels"])
        padding = to_tuple_list(decoder_specs["padding"],n)
        stride = to_tuple_list(decoder_specs["stride"],n)
        kernel_size = to_tuple_list(decoder_specs["kernel_size"],n)
        dilation = 1
        for (px, py), (sx,sy), (kx, ky) in zip(padding[::-1],stride[::-1],kernel_size[::-1]) :
            x = (x+2*px - dilation*(kx-1) - 1)/sx + 1
            y = (y+2*py - dilation*(ky-1) - 1)/sy + 1
            assert int(x) == x and x>0, f"Invalid dimension {x}"
            assert int(y) == y and y>0, f"Invalid dimension {y}"
            x = int(x)
            y = int(y)

        decoder_specs["padding"] = padding
        decoder_specs["stride"] = stride
        decoder_specs["kernel_size"] = kernel_size
        decoder_specs["latent_dims"] = [x,y,decoder_specs["hidden_channels"][0]]

    return generate_nn(decoder_specs)