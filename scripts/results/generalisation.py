"""
Compute the prediction error of seen and unseen transitions
"""

import os
import yaml
import torch
from data_generator import load_data, get_generator, Generator
from utils import pretty_print
from typing import List, Dict
import pickle
import numpy as np
import sys
import json

data_folder:str = "./data/"
image_folder:str = "./images/"

def generate_transitions(generator: Generator,
                         mask_actions: np.ndarray) :
    """
    Generate all the one-step transitions for the given mask of actions
    """
    n_states, n_actions = mask_actions.shape
    idxs0 = []
    actions = []
    for i in range(n_states):
        n_available_actions = np.sum(mask_actions[i])
        idxs0.append(np.repeat(i, n_available_actions))
        actions.append(np.where(mask_actions[i])[0])
    idxs0 = np.concatenate(idxs0).flatten()
    actions = np.concatenate(actions).flatten()  # [B]

    idxs1 = generator.group.transition(idxs0, actions)  # [B]

    X0 = generator.generate(idxs0)
    X1 = generator.generate(idxs1)
    X = np.stack([X0, X1], axis=1)  # [B, 2, ...]

    X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
    actions = torch.tensor(actions, dtype=torch.int64)  # Convert to tensor

    return X, actions

def test_algo_aux(algo,
                  X,
                  actions,
                  ) :
    algo.to('cpu')
    
    with torch.no_grad():
        X1_hat = algo.forward(X[:,0], actions[:,None])

    return torch.mean((X[:,1] - X1_hat) ** 2)

def test_algo(expe_folder:str,
              white_seeds: List[int] = None) :
    
    if os.path.exists(f"{expe_folder}/generalization_prediction_error.pkl") :
        # If the metrics were already computed, we load them
        with open(f"{expe_folder}/generalization_prediction_error.pkl", "rb") as f:
            values = pickle.load(f)
    else :
        # Load the generator to generate the sequences
        with open(f"{expe_folder}config.yaml", 'r') as file: 
            config = yaml.safe_load(file)

        _, nfo = load_data(config["dataset"]["name"])
        del _

        generator = get_generator(nfo["environment"],nfo["specs"])

        if "n_actions_per_state" in config["dataset"] :
            # IID setting
            n_available_actions = config["dataset"]["n_actions_per_state"]
            
            # Generate seen actions
            n_actions = generator.group.n_actions
            n_states = generator.group.n_states
            np.random.seed(0)
            available_actions = np.array([np.random.choice(n_actions, size=n_available_actions, replace=False) for _ in range(n_states)])  # [n_states, n_actions_per_state]

            mask_actions = np.zeros(shape=(n_states, n_actions), dtype=np.bool_)  # [n_states, n_actions]
            for i in range(n_states) :
                mask_actions[i, available_actions[i]] = True

            seen_transitions, seen_actions = generate_transitions(generator, mask_actions)  # [n_states, 2, ...]
            unseen_transitions, unseen_actions = generate_transitions(generator, np.logical_not(mask_actions))  # [n_states, 2, ...]
        elif config["dataset"].get("ood", False) :
            ## OOD setting
            mask_actions = generator.ood_actions(np.arange(generator.group.n_states))
            seen_transitions, seen_actions = generate_transitions(generator, mask_actions)
            unseen_transitions, unseen_actions = generate_transitions(generator, np.logical_not(mask_actions))
        else :
            raise ValueError("The dataset does not have a valid configuration for OOD prediction.")
        
        
        seeds = os.listdir(expe_folder)
        seeds = [seed for seed in seeds if seed.isdigit()]
        seeds.sort()
        values = {"seen": [], "unseen": []}
        for seed in seeds :
            algo = torch.load(f"{expe_folder}/{seed}/last_model", map_location=torch.device('cpu'), weights_only=False)
            values["seen"].append(test_algo_aux(algo, seen_transitions, seen_actions))
            values["unseen"].append(test_algo_aux(algo, unseen_transitions, unseen_actions))
    
        with open(f"{expe_folder}/generalization_prediction_error.pkl", "wb") as f:
            pickle.dump(values, f)

    if white_seeds is not None :
        white_seeds = np.array(white_seeds)
        values["seen"] = np.array(values["seen"])[white_seeds]
        values["unseen"] = np.array(values["unseen"])[white_seeds]

    return values

def main(algos:Dict[str, Dict[str, str]],
         verbose: bool = True,
         ) :
    for name, values in algos.items():
        print(name)
        if "entangled_seeds" in values:
            seeds = values["entangled_seeds"]
            results = test_algo(expe_folder=values["expe_folder"], white_seeds=seeds)
            if verbose:
                print(f"Entangled seen transitions: {pretty_print(results['seen'])}")
                print(f"Entangled unseen transitions: {pretty_print(results['unseen'])}")

            seeds = values["disentangled_seeds"]
            results = test_algo(expe_folder=values["expe_folder"], white_seeds=seeds)
            if verbose:
                print(f"Disentangled seen transitions: {pretty_print(results['seen'])}")
                print(f"Disentangled unseen transitions: {pretty_print(results['unseen'])}")
        
        else :
            results = test_algo(expe_folder=values["expe_folder"])
            if verbose:
                print(f"Seen transitions: {pretty_print(results['seen'])}")
                print(f"Unseen transitions: {pretty_print(results['unseen'])}")

if __name__ == "__main__":
    with open("./scripts/results/folders.json", 'r') as f :
        folders = json.loads(f.read())
    expe = sys.argv[1] if len(sys.argv) > 1 else "coil3_ood"
    algos = folders[expe]
    main(algos = algos,)