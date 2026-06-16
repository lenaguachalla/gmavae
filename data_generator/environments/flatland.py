import numpy as np
import torch
from data_generator.data_generator import Generator
from typing import List
from data_generator.groups import generate_group, GroupProduct, PermutationGroup, CyclicGroup

class FlatlandGenerator(Generator) :
    def __init__(self,
                 n_pos:int,
                 color_type:str, # "cyclic" or "permutation"
                 l:int = 17, #size of spherical objects
                 img_shape: List[int] = [64,64], #for img mode, size of the image
                 ):
        self.n_pos = n_pos
        self.color_type = color_type
        self.x_group = CyclicGroup(n=n_pos, m=1) # x axis cyclic group
        self.y_group = CyclicGroup(n=n_pos, m=1) # y axis cyclic group
        
        if color_type == "cyclic":
            color_specs = {"type": "cyclic", "n": 3, "m": 1}
        elif color_type == "permutation":
            color_specs = {"type": "permutation", "n": 3}
        self.color_group = generate_group(color_specs)
        self.group = GroupProduct([self.x_group,self.y_group, self.color_group])

        if isinstance(self.color_group, CyclicGroup) :
            assert self.color_group.n <= len(COLORS), f"number of color must be inferior to {COLORS}"

        self.img_shape = img_shape
        self.l = l

        # set the color dimension ie the number of channels
        if self.color_group is None :
            self.color_dim = 1
        elif isinstance(self.color_group, CyclicGroup):
            self.color_dim = 3
        elif isinstance(self.color_group, PermutationGroup):
            self.color_dim  = self.color_group.len_state

        # set the pattern of the spherical objects
        assert self.l % 2 == 1, "l must be odd sinon c'est moche"
        self.pattern_x_indices = []
        self.pattern_y_indices = []
        for x in range(self.l):
            for y in range(self.l):
                if (x - self.l // 2) ** 2 + (y - self.l // 2) ** 2 <= (self.l // 2) ** 2:
                    self.pattern_x_indices.append(x)
                    self.pattern_y_indices.append(y)
        self.pattern_x_indices = np.array(self.pattern_x_indices)
        self.pattern_y_indices = np.array(self.pattern_y_indices)

    def __repr__(self):
        return "flatland"
    
    def generate(self, idx: np.ndarray) -> np.ndarray :
        B = idx.shape[0]
        #idx [B]

        state = self.group.get_state(idx) #[B, len_state]
        position_state = state[...,:2].astype(int)
        color_state = state[...,2:]
        if self.color_group is None :
            color = 1
            background = 0
        elif isinstance(self.color_group, CyclicGroup):
            background = np.array([0.,0.,0.])
            color = COLORS[color_state[:,0]]
        elif isinstance(self.color_group, PermutationGroup) :
            background = 0
            color = (color_state+1) / self.color_group.n

        # compute the position of the spherical objects
        X = (self.img_shape[0]-self.l)*position_state[:,0]//(self.n_pos-1)
        Y = (self.img_shape[1]-self.l)*position_state[:,1]//(self.n_pos-1)

        # create the blank images
        image = background * np.ones([B] + self.img_shape + [self.color_dim])
        n = len(self.pattern_x_indices)
        B_indices = np.arange(B)[:,None].repeat(n,axis=1).flatten()

        # paste the spherical objects in the image
        X_indices = self.pattern_x_indices[None,:].repeat(B,axis=0)
        X_indices += X[:,None].repeat(n,axis=1)
        X_indices = X_indices.flatten()

        Y_indices = self.pattern_y_indices[None,:].repeat(B,axis=0)
        Y_indices += Y[:,None].repeat(n,axis=1)
        Y_indices = Y_indices.flatten()

        color = color[:,None,:].repeat(n,axis=1).reshape(-1,self.color_dim)

        image[B_indices,X_indices,Y_indices] = color
        
        return image

    @property
    def specs(self) -> dict:
        return {
            "n_pos": self.n_pos,
            "color_type": self.color_type,
            "l": self.l,
            "img_shape": self.img_shape,
        }
    
    def get_nfo(self) -> dict:        
        return {
            "x_dims": self.img_shape + [self.color_dim],
            "n_action": self.group.n_actions,
            "group": self.group.groups_list,
            "specs": self.specs,
            "environment": "flatland",
        }

COLORS=np.array([[1,0,0],
                [0,1,0],
                [0,0,1],
                [1,1,0],
                [0,1,1],
                [1,0,1],
                [1,1,1]])