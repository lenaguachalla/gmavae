from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
import torch 
from collections import OrderedDict
import numpy as np
import random

def maps_tree(tree, f:callable) :
    return tree_map(f, tree)

def merge_trees(*trees,
                f:callable = lambda x : sum(x)/len(x)):
    """
    Take several trees of same structure and apply f on the leaves
    such that the output have the same structure
    """
    # Check that all strctures are the same 
    flattened_trees = [tree_flatten(tree) for tree in trees]
    all_flattened_values = [flattened[0] for flattened in flattened_trees]
    all_specifications = [flattened[1] for flattened in flattened_trees]

    if not all(spec == all_specifications[0] for spec in all_specifications):
        raise ValueError("Les arbres n'ont pas la même structure.")

    # Apply f on the leaves
    aggregated_values = [f(values) for values in zip(*all_flattened_values)]
    # Reconstruct the tree
    aggregated_tree = tree_unflatten(aggregated_values, all_specifications[0])
    return aggregated_tree

def shrink_tree(tree,
                f:callable = lambda x : sum(x)):
    values = tree_flatten(tree)[0]
    return f(values)

def unfold_dicts(dico: dict,
                 prefix: str= '') -> dict:
    if type(dico) != dict :
        return {prefix : dico}
    else :
        s={}
        for k,v in dico.items() :
            s.update(unfold_dicts(v,prefix+'/'+k))
    return s

def fold_dicts(dico: dict, delimiter: str = '/') -> dict:
    d = {}
    for k,v in dico.items():
        keys = k.split(delimiter)
        d_ = d
        for key in keys[1:-1]:
            if key not in d_:
                d_[key] = {}
            d_ = d_[key]
        d_[keys[-1]] = v
    return d

def print_params(model: torch.nn.Module):
    def dfs (dico:dict, indent = 0):
        for k,v in dico.items():
            if type(v) == dict :
                print(' '*indent + k + ":")
                dfs(v,indent+1)
            else :
                print(' '*indent + k + ":",list(v.shape))
    dico = model.state_dict()
    dico = OrderedDict(("." + k,v) for k,v in dico.items())
    dico = fold_dicts(dico, delimiter='.')
    dfs(dico)
    print("Total number of parameters:", sum(v.numel() for v in model.parameters()))

def compute_lamda(lamda, iter: int) :
    if type(lamda) in [float, int]:
        return lamda
    elif lamda["type"] == "affine" :
        timesteps = lamda["timesteps"]
        values = lamda["values"]
        if iter < timesteps[0] :
            return values[0]
        elif iter >= timesteps[-1] :
            return values[-1]
        else :
            for i in range(len(timesteps)-1) :
                if iter >= timesteps[i] and iter < timesteps[i+1] :
                    return (values[i+1]-values[i])/(timesteps[i+1]-timesteps[i])*(iter-timesteps[i]) + values[i]
    elif lamda["type"] == "stair" :
        timesteps = lamda["timesteps"]
        values = lamda["values"]
        if iter < timesteps[0] :
            return 0
        for i in range(len(timesteps)-1) :
            if iter >= timesteps[i] and iter < timesteps[i+1] :
                return values[i]
        return values[-1]
    elif lamda["type"] == "geometric" :
        c,a,b = lamda["c"], lamda["a"], lamda["b"]
        return c * a**(b*iter)
    else :
        raise ValueError(f"Unknown lamda type: {lamda['type']}")
    
def compute_grad_norm(algo):
    total_norm = 0
    for p in algo.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def pretty_print(v) -> str:
    if type(v) in [list,np.ndarray] :
        mean = np.mean(v)
        std = np.std(v)
        n = len(v)
        if mean < 10**-2 :
            return '{:.1e}'.format(mean) + ' ± ' + '{:.1e}'.format(std/np.sqrt(n))
        else :
            return f"{mean:.2f}" + ' ± ' + f"{std/np.sqrt(n):.2f}"
    else :
        if v < 10**-2 :
            return '{:.1e}'.format(v)
        else :
            return f"{v:.2f}"

def set_seeds(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


