import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = None

def generate_action_encoder(specs:dict) -> ActionEncoder:
    specs = specs.copy()
    type = specs.pop("type")
    if type == "concat" :
        from models.action.concat import ConcatActionEncoder
        specs_list = specs["list"]
        if "n_action" in specs:
            for spec in specs_list:
                spec["n_action"] = specs["n_action"]
        if "input_dim" in specs:
            for spec in specs_list:
                spec["input_dim"] = specs["input_dim"]
        return ConcatActionEncoder([generate_action_encoder(spec) for spec in specs_list])
    elif type == "dense" :
        from models.action.dense import DenseActionEncoder as Model
    elif type == "rotation" :
        from models.action.rotation import RotationActionEncoder as Model
    elif type == "hardcoded" :
        from models.action.hardcoded import HardcodedActionEncoder as Model
    elif type == "forward" :
        from models.action.forward import ForwardActionEncoder as Model
    elif type == "dense_continuous" :
        from models.action.dense_continuous import DenseContinuousActionEncoder as Model
    elif type == "rotation_continuous" :
        from models.action.rotation_continuous import RotationContinuousActionEncoder as Model
    else:
        raise ValueError(type)
    
    return Model(**specs)
