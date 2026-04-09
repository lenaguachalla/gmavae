"""
Apply the dislib library on learned representations
"""

import os
import torch
import numpy as np
import yaml
import sys
import json
from typing import Tuple
from scripts.results.plot import set_rcParams, boxplot
from data_generator.groups import GroupProduct
from data_generator import load_data, get_generator
from utils import pretty_print
import matplotlib.pyplot as plt
import pickle
from dislib import BetaVAEMetric, MigMetric, DciMetric, Modularity, SapMetric, IndependanceMetric

data_folder:str = "./data/"
image_folder:str = "./images/"

# create ground truth data
class Dataset:
    def __init__(self, folder: str, random_seed: int = 0):

        with open(f"{folder}config.yaml", 'r') as file: 
            config = yaml.safe_load(file)
            dataname = config["dataset"]["name"]

        _,nfo = load_data(dataname, data_folder=data_folder)
        nfo["dataname"] = dataname
        del _

        self.gen = get_generator(nfo["environment"],
                                 specs=nfo["specs"])
            
        self.group: GroupProduct = self.gen.group
        self.latents_sizes = np.array([g.n_states for g in self.group.groups])

        states = np.arange(self.group.n_states)
        np.random.seed(random_seed)
        states = np.random.permutation(states)
        self.train_set, self.test_set = states[:int(len(states)*0.8)], states[int(len(states)*0.8):]

    def sample_diff_latents(self, idxs: np.ndarray) -> Tuple[np.ndarray]:
        """Sample a pair of latents derived from two action of different supgroups"""
        subgroup_idxs = np.random.choice(np.arange(len(self.latents_sizes)), size=2, replace=False)
        idxs_1 = np.copy(idxs)
        idxs_1[subgroup_idxs[0]] = np.random.choice([i for i in range(self.latents_sizes[subgroup_idxs[0]]) if i != idxs[subgroup_idxs[0]]])
        idxs_2 = np.copy(idxs)
        idxs_2[subgroup_idxs[1]] = np.random.choice([i for i in range(self.latents_sizes[subgroup_idxs[1]]) if i != idxs[subgroup_idxs[1]]])
        return idxs_1, idxs_2


    def sample_latent(self, mode:str = None) :
        if mode is None :
            idx = np.random.randint(0, self.group.n_states)
        elif mode == "train" :
            idx = np.random.choice(self.train_set)
        elif mode == "test" :
            idx = np.random.choice(self.test_set)
        else :
            raise ValueError(f"Unknown mode {mode}")
        
        return np.copy(self.group.idx_to_idxs[idx])
    
    def get_img_by_latent(self, latent) :
        idxs = latent[None]
        idx = self.group.idxs_to_idx[tuple(idxs.T)]
        images = self.gen.generate(idx)[0]

        return torch.Tensor(images).float()

    def sample(self, num_points_iter: int = 1, random_state: int = None) :
        idxs = np.random.randint(0, self.latents_sizes, size=(num_points_iter, len(self.latents_sizes)))
        idx = self.group.idxs_to_idx[tuple(idxs.T)]
        images = self.gen.generate(idx)

        return idxs, torch.Tensor(images).float()
    

metrics = [BetaVAEMetric, IndependanceMetric, Modularity, DciMetric, SapMetric, MigMetric]
keys = ['dmetric/val_hig_acc', 'independance_score', "dmetric/explicitness_score_test", "dmetric/disentanglement", "dmetric/SAP_score", 'dmetric/discrete_mig']
metric_names = ["Beta-VAE", "Inde", "Mod", "DCI", "SAP", "MIG"]

def compute_metrics(folder, ds) :

    # create representation function
    algo = torch.load(f"{folder}last_model", map_location=torch.device("cpu"), weights_only=False)
    algo.to("cpu")

    values = np.array([Metric(ds)(algo)[k] for k,Metric in zip(keys, metrics)])

    return values


def test_algo(expe_folder:str,
              verbose = False,
              best_seed = None):
    if os.path.exists(f"{expe_folder}dislib.pkl") :
        with open(f"{expe_folder}dislib.pkl", "rb") as f:
            result = pickle.load(f)
    else :
        seeds = os.listdir(expe_folder)
        seeds = [seed for seed in seeds if seed.isdigit()]
        seeds.sort()
        ds = Dataset(expe_folder)
        result = np.stack([compute_metrics(expe_folder + seed + "/", ds) for seed in seeds],-1)
    if verbose:
        for k in range(len(metrics)) :
            print(f"{metric_names[k]}: {pretty_print(result[k])}")
        print()
        if best_seed is not None:
            print("best: ")
            for k in range(len(metrics)) :
                print(f"best {metric_names[k]}: {pretty_print(result[k, best_seed])}")

        print()
        print()

    # save results
    with open(f"{expe_folder}dislib.pkl", "wb") as f:
        pickle.dump(result, f)


    return result


def main(algos,
         verbose = True,
         image_folder:str = "./images/",
         plot: bool = False
         ) :
    results = []
    
    for name, values in algos.items():
        print(name)

        results.append(test_algo(expe_folder=values["expe_folder"],
                                 best_seed=values.get("best_seed", None),
                                 verbose=verbose))
    if plot :
        boxplot(results,
                x_names=[name for name in algos.keys()],
                hue_names=metric_names)
    return results

if __name__ == "__main__" :
    plt.figure(figsize=(5,4))
    set_rcParams(xtick_labelsize = 14,
                 ytick_labelsize = 14,)

    with open("./scripts/results/folders.json", 'r') as f :
        folders = json.loads(f.read())
    expe = sys.argv[1] if len(sys.argv) > 1 else "shapes"
    algos = folders[expe]
    main(algos = algos,
         plot = True)

    
    os.makedirs(f"{image_folder}{expe}/", exist_ok=True)
    plt.savefig(f"{image_folder}{expe}/dislib.pdf", bbox_inches='tight')
    plt.clf()