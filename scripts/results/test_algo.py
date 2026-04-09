"""
Test an algorithm on a dataset with a set of metrics
Return a dictionary with the metrics
"""

import os
import yaml
import json
import sys
import torch
from metrics import get_metric
from data_generator import get_loader
from utils import pretty_print
from typing import List
import pickle
from typing import Dict

data_folder:str = "./data/"

def test_algo_aux(algo, metrics) :
    algo.to('cpu')
    values = {}
    for metric in metrics :
        metric.set_algo(algo)
        for k,v in metric.compute_metrics().items() :
            values[f"{metric}/{k}"] = v
    return values

def test_algo(expe_folder:str,
            metric_names:List[str],
            verbose = False,
            best_seed = None,
         ) :
    
    if os.path.exists(f"{expe_folder}/test_results.pkl") :
        # If the metrics were already computed, we load them
        with open(f"{expe_folder}/test_results.pkl", "rb") as f:
            values = pickle.load(f)
    else :
        values = {}
    
    values_keys = set(k.split('/')[0] for k in values.keys())

    # Load the configuration file
    with open(f"{expe_folder}config.yaml", 'r') as file: 
        config = yaml.safe_load(file)
    loader_specs = config["dataset"]
    loader_specs.pop("action_noise_std", None) # remove action noise

    loader = get_loader(loader_specs=loader_specs,
                        batch_size=2048,
                        device="cpu",)
    nfo = loader.nfo
    nfo["dataname"] = config["dataset"]["name"]

    # Create the metrics
    metrics = [get_metric(metric_name, None, nfo, loader)\
               for metric_name in metric_names if metric_name not in values_keys]

    seeds = os.listdir(expe_folder)
    seeds = [seed for seed in seeds if seed.isdigit()]
    seeds.sort()
    list_values = []
    for seed in seeds :
        algo = torch.load(f"{expe_folder}/{seed}/last_model", map_location=torch.device('cpu'), weights_only=False)
        list_values.append(test_algo_aux(algo, metrics))
    values.update({k: [v[k] for v in list_values] for k in list_values[0].keys()})
    with open(f"{expe_folder}/test_results.pkl", "wb") as f:
        pickle.dump(values, f)

    if verbose :
        for k, v in values.items() :
            print(f"{k}: {pretty_print(v)}")
            if best_seed is not None:
                print(f"best: {pretty_print(v[best_seed])}")
            print()

    return values

def main(algos:Dict[str, Dict[str, str]],
         verbose = True,
         metric_names:List[str] =["prediction"]
         ) :
    results = []
    for name, values in algos.items():
        print(name)
        results.append(test_algo(expe_folder=values["expe_folder"],
                                 best_seed=values.get("best_seed", None),
                                 metric_names=metric_names,
                                 verbose=verbose))
    return results

if __name__ == "__main__":
    metric_names = ["prediction"]
    
    with open("./scripts/results/folders.json", 'r') as f :
        folders = json.loads(f.read())
    expe = sys.argv[1] if len(sys.argv) > 1 else "coil2"
    algos = folders[expe]
    main(algos = algos,
         metric_names=metric_names,
         verbose=True,)