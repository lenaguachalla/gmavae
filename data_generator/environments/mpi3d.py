import numpy as np
from data_generator.data_generator import Generator
from data_generator.groups import GroupProduct, CyclicGroup
import torch
import os

# data "mpi3d_real_complex" to be downloaded at https://github.com/rr-learning/disentanglement_dataset

class MPI3DGenerator(Generator) :
    def __init__(self,
                 data_folder = "./data_raw/mpi3d/",
                 ):

        self.group = GroupProduct([CyclicGroup(n=40, m=None) for _ in range(2)])
        #load npy file
        if not os.path.exists(data_folder + "mpi3d.npy"):
            preprocess(data_folder, "real3d_complicated_shapes_ordered.npz")
        
        self.images = np.load(data_folder + "mpi3d.npy", allow_pickle=True)
            
        assert self.images.shape == (40,40,64,64,3)

    def __repr__(self):
        return "shapes"
    
    def generate(self, idx: np.ndarray) -> np.ndarray :
        state = self.group.get_state(idx)  # [B, 2]
        return self.images[state[:,0],state[:,1]]/255.0
    
    def process_action(self,
                       a: np.ndarray | torch.Tensor
                       ) -> np.ndarray | torch.Tensor:
        """
        Return the action in the form [n,r]
          with n the group index
               r the angle of rotation in radian
        """
        if type(a) is torch.Tensor :
            n = (a>39).float()
            r = (a+1-39*n).float()/40*2*np.pi
            return torch.stack([n, r], dim=-1)
        elif type(a) is np.ndarray :
            n = (a>39).astype(np.float32)
            r = (a+1-39*n).astype(np.float32)/40*2*np.pi
            return np.stack([n, r], axis=-1)
        else :
            raise NotImplementedError
    
    @property
    def specs(self) -> dict:
        return {
        }
    
    def get_nfo(self) -> dict:        
        return {
            "x_dims": [64,64,3],
            "n_action": self.group.n_actions,
            "action_dim": 2,
            "group": self.group.groups_list,
            "specs": self.specs,
            "environment": "mpi3d",
            
        }
    def add_action_noise(self, a, std):
        if std == 0.0 :
            return a
        
        if type(a) is torch.Tensor :
            n = (a>39).int()
            r = (a-39*n).float()
            r += torch.randn_like(r)*std/(2*np.pi)*40
            r = torch.round(r) % 39
            return (n*39 + r).long()
        elif type(a) is np.ndarray :
            n = (a>39).astype(np.int64)
            r = (a-39*n).astype(np.float32)
            r += np.random.randn(*r.shape)*std/(2*np.pi)*40
            r = np.round(r) % 39
            return (n*39 + r).astype(np.int64)
    
def preprocess(folder_name:str,
               file_name:str) :
    if not os.path.exists(folder_name + file_name):
        raise ValueError(f"File {folder_name + file_name} does not exist. Please download it at https://github.com/rr-learning/disentanglement_dataset and the setup the correct path.")

    # load npz file
    data = np.load(folder_name + file_name)["images"]
    data = data.reshape([4,4,2,3,3,40,40,64,64,3])
    data = data[0,0,0,1,0]
    # save as np file without compression
    np.save(folder_name + "mpi3d.npy", data)

if __name__ == "__main__" :
    folder_name = "./data_raw/mpi3d/"
    file_name = "real3d_complicated_shapes_ordered.npz"
    preprocess(folder_name, file_name)