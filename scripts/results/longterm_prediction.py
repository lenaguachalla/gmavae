"""
Compute the prediction error for different sequence lengths
"""

import os
import sys
import yaml
import torch
from data_generator import load_data, get_generator, Group, Generator
from scripts.results.plot import set_rcParams, plot_std
from typing import List, Dict
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json

data_folder:str = "./data/"
image_folder:str = "./images/"
max_length = 10000
batch_size = 8

def generate_sequences(generator: Generator, length: int) :
    np.random.seed(0)
    group: Group = generator.group

    # Generate a batch of initial observations
    initial_idx = np.random.choice(range(group.n_states), size=batch_size)

    # Generate a batch of actions and transition through the group
    actions = np.random.choice(range(group.n_actions), size=(batch_size, length))
    
    # Convert to tensors
    actions = torch.tensor(actions, dtype=torch.int64)

    return initial_idx, actions, #[B,L+1,...],[B,L]

def test_algo_aux(algo,
                  initial_idx, # [B]
                  actions, # [B,L]
                  generator: Generator, # to generate the observations
                  ) :
    algo.to('cpu')
    group: Group = generator.group

    errors = []
    
    with torch.no_grad():
        idx = initial_idx
        observations = torch.tensor(generator.generate(idx), dtype=torch.float32)  # [B, ...]
        Z = algo.encode_image(observations[:])  # [B, z_dim]
        for i in tqdm.tqdm(range(actions.shape[1])):

            next_idx = group.transition(idx, actions[:, i])
            next_observations = torch.tensor(generator.generate(next_idx), dtype=torch.float32)  # [B, ...]

            Z = algo.apply_action(Z, actions[:, i])
            prediction = algo.decode_image(Z)
            errors.append(torch.mean((next_observations - prediction) ** 2, dim=list(range(1, prediction.ndim))).unsqueeze(1))  # [B, 1]

            idx = next_idx
            observations = next_observations

    return torch.cat(errors, dim=1).numpy()  # [B, L]

def test_algo(expe_folder:str,
              white_seeds: List[int] = None,
         ) :
    
    if os.path.exists(f"{expe_folder}/longterm_prediction_error.pkl") :
        # If the metrics were already computed, we load them
        with open(f"{expe_folder}/longterm_prediction_error.pkl", "rb") as f:
            values = pickle.load(f)
    else : 
        values = None



    # Compute values
    if values is None :

        # Load the generator to generate the sequences
        with open(f"{expe_folder}config.yaml", 'r') as file: 
            config = yaml.safe_load(file)

        _, nfo = load_data(config["dataset"]["name"])
        del _

        generator = get_generator(nfo["environment"],nfo["specs"])

        # Generate the sequences
        initial_idx, actions = generate_sequences(generator, max_length)

        # Retrieve the seeds
        seeds = os.listdir(expe_folder)
        seeds = [seed for seed in seeds if seed.isdigit()]
        seeds.sort()
        values = np.zeros(shape=(0, actions.shape[1]), dtype=np.float32)  # Initialize an empty array for the errors
        for seed in seeds :
            print(f"Testing seed {seed}...")
            algo = torch.load(f"{expe_folder}{seed}/last_model", map_location=torch.device('cpu'), weights_only=False)
            values = np.concatenate((values, test_algo_aux(algo, initial_idx, actions, generator)), axis=0)

        with open(f"{expe_folder}/longterm_prediction_error.pkl", "wb") as f:
            pickle.dump(values, f)

    if white_seeds is not None:
        white_seeds = np.array(white_seeds)
        values = np.concatenate([values[s*batch_size:(s+1)*batch_size] for s in white_seeds], axis=0)

    return values

def main(algos:Dict[str, Dict[str, str]],
         ) :
    for name, values in algos.items():
        print(name)
        if "entangled_seeds" in values:
            seeds = values["entangled_seeds"]
            if len(seeds) > 0 :
                results = test_algo(expe_folder=values["expe_folder"], white_seeds=seeds)
                plot_std(results, label=name + "(entangled)")

            seeds = values["disentangled_seeds"]
            if len(seeds) > 0 :
                results = test_algo(expe_folder=values["expe_folder"], white_seeds=seeds)
                plot_std(results, label=name + "(disentangled)")

        else :
            results = test_algo(expe_folder=values["expe_folder"])
            if results.size > 0:
                plot_std(results, label=name)

    # set logscale
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1,results.shape[1])
    plt.xlabel("Sequence length", weight='bold')
    plt.ylabel("Prediction error", weight='bold')
    #plt.legend()

if __name__ == "__main__":
    set_rcParams(**{"font.weight": "bold"})

    fig = plt.figure(figsize=(4, 2.4))
    ax = fig.add_subplot(111)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    with open("./scripts/results/folders.json", 'r') as f :
        folders = json.loads(f.read())
    expe = sys.argv[1] if len(sys.argv) > 1 else "coil3"
    algos = folders[expe]
    white_list = ["A-VAE", "LSBD-VAE", "SOBDRL", "GMA-VAE"]
    for name in list(algos.keys()):
        if name not in white_list :
            del algos[name]
    main(algos = algos,)

    os.makedirs(f"{image_folder}{expe}/", exist_ok=True)
    plt.savefig(f"{image_folder}{expe}/longterm_prediction.pdf", bbox_inches='tight')