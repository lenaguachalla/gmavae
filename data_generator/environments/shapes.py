import numpy as np
from data_generator.data_generator import Generator
from data_generator.groups import GroupProduct, CyclicGroup
import h5py 
import os

# data to be downloaded at https://github.com/google-deepmind/3d-shapes

class ShapesGenerator(Generator) :
    def __init__(self,
                 data_folder = "./data_raw/3dshapes/",
                 subsampling: int = 2,
                 ):
        if not os.path.exists(data_folder + "3dshapes.h5"):
            raise ValueError(f"Data folder {data_folder} does not exists, please download it at https://github.com/google-deepmind/3d-shapes and set the correct path.")
        
        self.subsampling = subsampling

        n_values = [10,10,10,8,4,15]  # [floor, wall, object, scale, shape, orientation]


        self.group = GroupProduct([CyclicGroup(n=n//subsampling, m=1) for n in n_values])
        dataset = h5py.File(data_folder + "3dshapes.h5", 'r')
        self.images = np.array(dataset['images']) # [N, 64, 64, 3]

        labels = np.array(dataset['labels']).copy() # [N, 6]

        # Convert labels in int
        for k in range(3):
            labels[:,k] = labels[:,k] * n_values[k]
        labels[:,3] = (labels[:,3] - 0.75) * 2 * (n_values[3] - 1)  # scale
        labels[:,-1] = (labels[:,-1] + 30)/60* (n_values[-1]-1)
        labels = np.rint(labels).astype(int)

        # subsample the labels
        self.labels = np.zeros([n//subsampling for n in n_values], dtype=int) # [...//subsampling]
        for k, l in enumerate(labels):
            if np.all(l%subsampling == 0) and np.all(l//subsampling < np.array(n_values)//subsampling):
                self.labels[tuple(l//subsampling)] = k

    def __repr__(self):
        return "shapes"
    
    def generate(self, idx: np.ndarray) -> np.ndarray :
        state = self.group.get_state(idx)  # [B, len_state]
        labels = self.labels[tuple(state.T)]  # [B]
        return self.images[labels]/255.0
    
    @property
    def specs(self) -> dict:
        return {
            "subsampling": self.subsampling,
        }
    
    def get_nfo(self) -> dict:        
        return {
            "x_dims": [64,64,3],
            "n_action": self.group.n_actions,
            "group": self.group.groups_list,
            "specs": self.specs,
            "environment": "shapes",
        }