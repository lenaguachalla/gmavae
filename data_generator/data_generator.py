import numpy as np
import os
import torch
import json
from typing import Tuple
from data_generator.groups import Group
from time import time
import h5py 

class Generator () :
    def __init__ (self):
        self.group: Group = None
        self.e: bool = None

    def full_interaction_generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate all the possible one-action transitions
        """
        data_image = []
        data_action = []

        A = np.arange(self.group.n_actions)

        for idx0 in range(self.group.n_states):
            I0 = idx0 * np.ones(A.size).astype(int)
            I1 = self.group.transition(I0, A).astype(int)
            data_image.append(np.stack([self.generate(I0), self.generate(I1)], axis=1))
            data_action.append(A)
        return np.vstack(data_image), np.hstack(data_action)[:,None]
    
    def generate(self, idx: np.ndarray) -> np.ndarray :
        """
        Generate the images corresponding to the states idx
        """
        raise NotImplementedError

    def full_image_generate(self, batch_size: int = 500) -> np.ndarray:
        """
        Generate all the possible images
        """

        if batch_size > self.group.n_states:
            return self.generate(np.arange(self.group.n_states))
        
        else :
            data = []
            for i in range(0, self.group.n_states, batch_size):
                idx = np.arange(i, min(i + batch_size, self.group.n_states))
                data.append(self.generate(idx))
            return np.vstack(data)
    
    def sample(self) -> np.ndarray:
        raise NotImplementedError
    
    def get_nfo(self) -> dict :
        return
    
    def process_action(self, a: np.ndarray) -> np.ndarray :
        return a

    def add_action_noise(self,
                         idx: np.ndarray | torch.Tensor,
                         std: float) -> np.ndarray | torch.Tensor :
        if std > 0.0 :
            raise NotImplementedError
        return idx

def get_generator(environnement: str,
                  specs: dict,) -> Generator:
    match environnement :
        case "flatland" :
            from data_generator.environments.flatland import FlatlandGenerator as DataGenerator
        case "coil" :
            from data_generator.environments.coil import CoilGenerator as DataGenerator
        case "shapes" :
            from data_generator.environments.shapes import ShapesGenerator as DataGenerator
        case "mpi3d" :
            from data_generator.environments.mpi3d import MPI3DGenerator as DataGenerator
        case _ :
            raise ValueError(f"Unknown environment: {environnement}")

    return DataGenerator(**specs)


def generate_data(environment: str,
                  specs: dict,
                  name: str,
                  verbose: bool = False):
    """
    Given an environment and its specs, generate the data and save it
    """
    t0 = time()
    
    generator = get_generator(environment, specs)

    if verbose :
        print(f"Generator: {environment}/{name}")

    folder = f"./data/{environment}/{name}/"
    os.makedirs(folder, exist_ok=True)

    # generate data
    images = generator.full_image_generate()
    if images.size < 1.e8:
        np.save(folder + 'images', images)
    else :
        with h5py.File(folder + 'images.h5', 'w') as f:
            f.create_dataset('dataset', data=images, compression='gzip', compression_opts=4)


    nfo = generator.get_nfo()
    with open(folder + "nfo", "w") as f:
        f.write(json.dumps(nfo, indent=2))

    if verbose :
        print(f"{len(images)} generated images")
        print(f"Time taken: {time()-t0:.2f} s")


def load_data(dataname: str,
              data_folder = "data/",
              torched:bool = True
              ): 
    """
    Load observations from data folder
    """
    folder = data_folder + dataname + "/"
    if os.path.exists(folder + 'images.npy'):
        images = np.load(folder + 'images.npy')
    elif os.path.exists(folder + 'images.h5'):
        with h5py.File(folder + 'images.h5', 'r') as f:
            images = np.array(f['dataset'])
    else :
        raise FileNotFoundError(f"Data not found in {folder}. Please generate the data first.")
    with open(folder + 'nfo', 'r') as f:
        nfo = json.load(f)

    if torched:
        images = torch.from_numpy(images).float()

    return images,\
           nfo
