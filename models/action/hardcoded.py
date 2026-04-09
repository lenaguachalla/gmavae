from models.action.action import ActionEncoder
from torch import nn
from math import  factorial
import numpy as np
import torch
import itertools
from typing import List, Dict


class HardcodedActionEncoder(ActionEncoder):
    """
    Hardcoded action encoder for known group structures.
    """
    def __init__(self,
                 n_action:int,
                 z_dim:int,
                 group_list: List[Dict]):
        super().__init__()
        self.z_dim = z_dim
        self.n_action = n_action

        self.matrices = nn.Parameter(torch.zeros((n_action, z_dim, z_dim)), requires_grad=False)
        for i in range(n_action):
            self.matrices[i] = torch.eye(z_dim)

        k_dim = 0
        k_action = 0
        for action_specs in group_list:
            if action_specs["type"] == "cyclic": 
                theta = 2 * np.pi / action_specs["n_state"]
                self.matrices[k_action, k_dim:k_dim+2, k_dim:k_dim+2] = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                                                                      [np.sin(theta), np.cos(theta)]])
                k_action += 1

                if action_specs["n_state"] > 2:
                    theta = - 2 * np.pi / action_specs["n_state"]
                    self.matrices[k_action, k_dim:k_dim+2, k_dim:k_dim+2] = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                                                                        [np.sin(theta), np.cos(theta)]])
                    k_action += 1

                k_dim += 2

            elif action_specs["type"] == "permutation":
                n = action_specs["n"]
                permutations = list(itertools.permutations(range(n)))
                for k in range(1, factorial(n)) :
                    self.matrices[k_action, k_dim:k_dim+n, k_dim:k_dim+n] = 0
                    self.matrices[k_action, k_dim:k_dim+n, k_dim:k_dim+n][np.arange(n), permutations[k]] = 1
                    k_action += 1

                k_dim += n

        assert k_action == n_action, f"Expected {n_action} actions, got {k_action}"
        assert k_dim == z_dim, f"Expected {z_dim} dimensions, got {k_dim}"



    def forward(self, a:torch.Tensor) -> torch.Tensor : 
        return self.matrices[a]