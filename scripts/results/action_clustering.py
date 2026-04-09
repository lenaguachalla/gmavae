"""
Compute the accuracy of the action clustering of step 2
Show the mean group distance matrix across seeds
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from data_generator import load_data
from metrics import get_metric
import json
import sys
from scripts.results.plot import set_rcParams

image_folder:str = "./images/"
data_folder:str = "./data/"

threshold = 0.1

def main(expe_folder:str):
    # choose ticks
    ticks_dict = {
        "flatland/cyclic": ["$x^+$", "$x^-$", "$y^+$", "$y^-$", "$c^+$", "$c^-$"],
        "flatland/permutation": ["$x^+$", "$x^-$", "$y^+$", "$y^-$",
                                 r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$", r"$\sigma_5$"],
        "coil/2": ["$r_1^+$", "$r_1^-$", "$r_2^+$", "$r_2^-$", r"$\sigma$"],
        "coil/3": ["$r_1^+$", "$r_1^-$", "$r_2^+$", "$r_2^-$", "$r_3^+$", "$r_3^-$",
                   r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$", r"$\sigma_5$"],
        "shapes/ss2": ["$g_1^+$", "$g_1^-$", "$g_2^+$", "$g_2^-$", "$g_3^+$", "$g_3^-$",
                       "$g_4^+$", "$g_4^-$", "$g_5$", "$g_6^+$", "$g_6^-$"],
    }

    # load available seeds
    seeds = os.listdir(expe_folder)
    seeds = [seed for seed in seeds if seed.isdigit()]

    # load dataset
    with open(f"{expe_folder}config.yaml", 'r') as file: 
        config = yaml.safe_load(file)
        dataname = config["dataset"]["name"]

        _,nfo = load_data(dataname, data_folder=data_folder)
        nfo["dataname"] = dataname
        del _

    # compute metrics and matrices
    values = []
    matrices = []

    for seed in seeds:
        algo = torch.load(f"{expe_folder}{seed}/last_model", map_location=torch.device('cpu'), weights_only=False)
        algo.to("cpu")
        group_metric = get_metric("groups", algo, nfo, {})

        metrics, dG = group_metric.compute_metrics(returns_dG=True)
        value = metrics[str(threshold)]
        values.append(value)
        matrices.append(dG)
        
    print("mean accuracy:", np.mean(values))
    print("success rate:", np.mean(np.array(values) == 1))

    mean_matrix = np.mean(np.stack(matrices), axis=0)


    metrics = group_metric.compute_metrics(dG = mean_matrix)
    value = metrics[str(threshold)]
    print("accuracy for mean matrix:", value)

    mean_matrix[np.arange(mean_matrix.shape[0]), np.arange(mean_matrix.shape[0])] = 0
    plt.imshow(mean_matrix,cmap='gray')
    plt.colorbar()
    if dataname in ticks_dict:
        ticks = ticks_dict[dataname]
        plt.xticks(ticks=range(len(ticks)), labels=ticks)
        plt.yticks(ticks=range(len(ticks)), labels=ticks)


if __name__ == "__main__":
    set_rcParams()
    plt.figure(figsize=(3.5, 2.5))

    with open("./scripts/results/folders.json", 'r') as f :
        folders = json.loads(f.read())
    expe = sys.argv[1] if len(sys.argv) > 1 else "coil2"

    if "A-VAE" in folders[expe] :
        expe_folder = folders[expe]["A-VAE"]["expe_folder"]
        main(expe_folder = expe_folder,)

        os.makedirs(image_folder + expe, exist_ok=True)
        plt.savefig(image_folder + expe + "/clustering_matrix.pdf", bbox_inches='tight')
        plt.clf()
    elif expe == "coil_randomaction" :
        for algo in folders[expe].keys() :
            expe_folder = folders[expe][algo]["expe_folder"]
            main(expe_folder = expe_folder,)

            os.makedirs(image_folder + expe + "/" + algo, exist_ok=True)
            plt.savefig(image_folder + expe + "/" + algo + "/clustering_matrix.pdf", bbox_inches='tight')
            plt.clf()