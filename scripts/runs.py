import os
import yaml
from train import train
import shutil
import sys
from datetime import datetime
config_folder = "./configs/"

"""
Run all the configs in the config folder
"""

def compute_list(input) :
    if type(input) == list:
        return input
    elif type(input) == str and input.startswith("range"):
        input = input[6:-1]
        if "," in input:
            input = input.split(",")
            start, end = int(input[0]), int(input[1])
            return list(range(start, end))
        else:
            end = int(input)
            return list(range(end))


def run_seeds(config, config_file):
    seeds = compute_list(config["seed"])
    name_expe = config.get("name_expe", None)
    if name_expe is None :
        name_expe = str(datetime.now())
    os.makedirs(config_folder + name_expe, exist_ok=True)
    os.makedirs( f"./expe/"+name_expe, exist_ok=True)
    shutil.copyfile(config_folder + config_file, f"./expe/"+name_expe + '/' +"config.yaml")

    for seed in seeds:
        print(f"  with seed {seed}")
        config_seed = config.copy()
        config_seed["seed"] = seed
        config_seed["name_expe"] = name_expe + f"/{seed}"
        if config.get("load", None):
            config_seed["load"] = f"{config['load']}/{seed}"
        with open(config_folder + name_expe + f"/{seed}.yaml", 'w') as f :
            yaml.dump(config_seed, f)
        train(config_folder + name_expe + f"/{seed}.yaml")

def run_dims(config, config_file):
    dims = compute_list(config["algo_specs"]["z_dim"])
    name_expe = config.get("name_expe", None)
    if name_expe is None :
        name_expe = str(datetime.now())
    os.makedirs(config_folder + name_expe, exist_ok=True)
    os.makedirs( f"./expe/"+name_expe, exist_ok=True)
    shutil.copyfile(config_folder + config_file, f"./expe/"+name_expe + '/' +"config.yaml")
    for dim in dims:
        print(f" with dim {dim}")
        config_dim = config.copy()
        config_dim["algo_specs"]["z_dim"] = dim
        config_dim["name_expe"] = name_expe + f"/d{dim}"
        if config.get("load", None):
            config_dim["load"] = f"{config['load']}/d{dim}"
        with open(config_folder + name_expe + f"/d{dim}.yaml", 'w') as f :
            yaml.dump(config_dim, f)
        run_seeds(config_dim, f"{name_expe}/d{dim}.yaml")
        
def main(prefix=""):
    config_files = os.listdir(config_folder)
    config_files = [config_file for config_file in config_files if os.path.isfile(config_folder + config_file)]
    config_files = [config_file for config_file in config_files if config_file.startswith(prefix)]
    config_files.sort()
    print("List of configs:")
    for config_file in config_files:
        print(" -", config_file)
    
    for config_file in config_files:
        print(f"Running {config_file}")
        with open(config_folder + config_file, 'r') as f :
            config = yaml.safe_load(f)

        # If several z dimensions and several seeds
        if "z_dim" in config["algo_specs"] and type(config["algo_specs"]["z_dim"]) in [str, list]:
            assert type(config["seed"]) in [str, list], ""
            run_dims(config, config_file)

        # Elif several seeds
        elif type(config["seed"]) in [str, list]:
            run_seeds(config, config_file)

        # Else just one config
        else :
            train(config_folder + config_file)

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv)>1 else ""
    main(prefix)
