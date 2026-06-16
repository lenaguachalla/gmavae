import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generator.data_generator import Generator
from typing import List
import einops
import os
from data_generator.groups import GroupProduct, PermutationGroup, CyclicGroup

# data to be downloaded at https://www.kaggle.com/datasets/jessicali9530/coil100

class CoilGenerator(Generator) :
    def __init__(self,
                 objects:List[int],
                 rotations:List[int],
                 available_rotations:List[List[int] | None | int] = None, #rotation shift available for each object
                    # None means all rotations are available, int means only one rotation is available, list means the available actions
                 available_permut:List[int] = None, #permutation available for each object
                    # None means all permutations are available, list means the available actions
                 coil_path: str = "./data_raw/coil-100/", # path to the coil dataset, to be downloaded with the above link
                 permutation: bool = True, 
                 entangled_actions: int = None, #if not None, number of subgroup for each action
                 e: bool = False, #if True, identity action is added
                 ):
        assert len(objects) == len(rotations), "number of objects and rotations must be the same"
        if not os.path.exists(coil_path):
            raise ValueError(f"coil_path {coil_path} does not exist. Please download the COIL-100 dataset from https://www.kaggle.com/datasets/jessicali9530/coil100 and set the correct path.")
        
        self.n_objects = len(objects)
        self.objects = objects
        self.rotations = rotations
        self.available_permut = available_permut

        if available_rotations is None :
            if entangled_actions : # if we entangle the actions, we want the subactions to be any element of the subgroup
                self.available_rotations = [None for _ in range(self.n_objects)]
            else :
                self.available_rotations = [1 for _ in range(self.n_objects)]
        else :
            self.available_rotations = available_rotations

        if permutation :
            self.permutation_group = PermutationGroup(self.n_objects, available_actions=available_permut)
            self.group = GroupProduct([CyclicGroup(n, m=m) for n,m in zip(rotations, self.available_rotations)] + [self.permutation_group],
                                      entangled_actions=entangled_actions,
                                      e=e)
        else :
            self.permutation_group = None
            self.group = GroupProduct([CyclicGroup(n, m=m) for n,m in zip(rotations, self.available_rotations)],
                                      entangled_actions=entangled_actions,
                                      e=e)
        

        # Load COIL Images
        self.images = []
        for o, r in zip(objects, rotations):
            images = []
            for k in range(r) :
                theta = 5 * round(360 * k / r / 5)
                # load a png file as np array
                img = plt.imread(coil_path + f"obj{o}__{theta}.png")
                img = img[::2,::2,:]
                images.append(img)
            self.images.append(np.stack(images)) #[r,64,64,3]

    def __repr__(self):
        return "coil"
    
    def generate(self, idx: np.ndarray) -> np.ndarray :
        X, Y = 64, 64
        B = idx.shape[0]
        #idx [B]

        state = self.group.get_state(idx) #[B, len_state]
        rotation_states = state[...,:self.n_objects].astype(int) #[B, n_objects]
        permutation_state = state[...,self.n_objects:].astype(int) #[B, n_objects]
            # is empty if no permutation group

        images = []

        for k in range(self.n_objects):
            # rotation_states[:,k] [B]
            # permutation_state [B]
            images.append(self.images[k][rotation_states[:,k],...])

        images = np.stack(images, axis = 1) #[B, n_objects, X, Y, 3]
        if self.permutation_group :
            images = images[np.arange(B)[:,None],permutation_state] #[B, n_objects, X, Y, 3]
        images = einops.rearrange(images, 'b n x y c -> b x (n y) c') #[B, n_objects*X, Y, 3]

        return images
    
    def ood_actions(self, idx: np.ndarray) -> np.ndarray:
        """
        Returns the mask of actions available for the given state idx.
        Only the right most item can rotate
        """
        assert self.permutation_group is not None
        mask = np.zeros((idx.shape[0], self.group.n_actions), dtype=bool)
        # always allow permutation
        mask[:, self.group.n_actions - self.permutation_group.n_actions:] = True

        # allow rotation of the right most object
        state = self.group.get_state(idx) #[B, len_state]
        permutation_state = state[...,self.n_objects:].astype(int) #[B, n_objects]
        right_most = permutation_state[:, -1] # [B] id of the right most object in the permutation

        for i in range(self.n_objects):
            for action in self.group.groups_list[i] :
                mask[right_most==i, action] = True
        return mask

    @property
    def specs(self) -> dict :
        return {
            "objects": self.objects,
            "rotations": self.rotations,
            "permutation": self.permutation_group is not None,
            "entangled_actions": self.group.entangled_actions,
            "available_rotations": self.available_rotations,
            "available_permut": self.available_permut,
            "e": self.group.e,
        }

    def get_nfo(self) -> dict:
        
        return {
            "x_dims": [64,64*self.n_objects,3],
            "n_action": self.group.n_actions,
            "group": self.group.groups_list,
            "specs": self.specs,
            "environment": "coil",
        }
