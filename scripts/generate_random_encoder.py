from algos import generate_algo
from data_generator import load_data
import os
import yaml
from utils import set_seeds

def main(dataname:str,
         model_folder:str,
         n_seeds:int = 5,) :
    
    nfo = load_data(dataname)[-1]

    
    config = {
        "dataset": {
            "name": dataname,
            "m": 5
        },
        "algo_specs": algo_specs,
    }

    os.makedirs(model_folder, exist_ok=True)
    with open(model_folder + "config.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    

    for seed in range(n_seeds) :
        set_seeds(seed)
        algo = generate_algo(algo_specs, nfo)

        os.makedirs(model_folder + str(seed), exist_ok=True)
        algo.save(model_folder+f"{seed}/last_model")


        #dump config as yaml
        with open(model_folder + f"{seed}/config.yaml", 'w') as file:
            yaml.dump(config, file, default_flow_style=False)


algo_specs = {
    "type": "ae",
    "z_dim": 6,
    "image_specs": {
        "type": "ae",
        "encoder_specs": {
            "type": "conv2d",
            "hidden_channels": [32, 64],
            "kernel_size": 8,
            "pooling": None,
            "stride": 4,
            "padding": 2,
            "hidden_dim": [256]
        },
        "decoder_specs": {
            "type": "mlp",
            "hidden_dim": [],
            "final_activation_fn": "sigmoid"
        }
    }
}

if __name__ == "__main__":

    #Generates random encoders for selected experiments

    '''inputs = [{"dataname": "coil/2",
               "model_folder": "./expe/coil2/random/"},
              {"dataname": "coil/3",
               "model_folder": "./expe/coil3/random/"},
              {"dataname": "flatland/cyclic",
               "model_folder": "./expe/flc/random/"},
              {"dataname": "flatland/permutation",
               "model_folder": "./expe/flp/random/"},
              {"dataname": "shapes/ss2",
               "model_folder": "./expe/shapes2/random/"},
              {"dataname": "mpi3d/lie",
               "model_folder": "./expe/mpi3d/random/"}]'''
    
    inputs = [{"dataname": "flatland/cyclic",
               "model_folder": "./expe/flc_noisy_obs2_v3/random/"}]

    for input in inputs :
        main(**input, n_seeds=5)