import numpy as np
from data_generator import load_data, get_generator
from torch import nn
from itertools import combinations
import torch
from metrics.metrics import Metric

""""
Compute the independance metric representing the distanglement
from Linear Disentangled Representations and Unsupervised Action Estimation, Painter et Al, 2020
"""

def compute_cosine(x, y):
    """
    Compute the cosine similarity between two tensors
    """
    x = nn.functional.normalize(x, dim=-1)
    y = nn.functional.normalize(y, dim=-1)
    return torch.sum(x * y, dim=-1)

class IndependanceMetric(Metric) :
    def __init__(self, algo, nfo, loaders):
        super().__init__(algo, nfo, loaders)
        self.generator = get_generator(nfo["environment"], specs=nfo["specs"])
        self.data_images = load_data(nfo["dataname"])[0].to(self.device)

    def __repr__(self) :
        return "inde"
    
    def compute_metrics(self):
        algo = self.algo
        batch_size = 16

        latent_sizes = np.array([g.n_states for g in self.generator.group.groups])
        self.n_subgroup = len(latent_sizes)
        
        dependences = []

        for _ in range(batch_size):
            # sample idxs
            for i1, i2 in combinations(range(self.n_subgroup), 2) :
                #i1 and i2 latent
                idxs = np.array([np.random.randint(0, latent_sizes[j], size=batch_size) for j in range(len(latent_sizes))]).T
                idx = self.generator.group.idxs_to_idx[tuple(idxs.T)]

                # sample two different latents
                idxs_1 = np.copy(idxs)
                idxs_1[:, i1] = [np.random.choice([j for j in range(latent_sizes[i1]) if j != idxs[k, i1]]) for k in range(batch_size)]
                idx_1 = self.generator.group.idxs_to_idx[tuple(idxs_1.T)]
                idxs_2 = np.copy(idxs)
                idxs_2[:, i2] = [np.random.choice([j for j in range(latent_sizes[i2]) if j != idxs[k, i2]]) for k in range(batch_size)]
                idx_2 = self.generator.group.idxs_to_idx[tuple(idxs_2.T)]

                # get images
                images = self.data_images[idx].to(self.device)
                images_1 = self.data_images[idx_1].to(self.device)
                images_2 = self.data_images[idx_2].to(self.device)

                # compute Z
                with torch.no_grad():
                    Z = algo.encode_image(images)
                    Z_1 = algo.encode_image(images_1)
                    Z_2 = algo.encode_image(images_2)
                
                # compute cosine similarity
                cosines = compute_cosine(Z-Z_1, Z-Z_2)

                dependences.append(cosines)
        inde = 1 - torch.abs(torch.cat(dependences, dim=0)).mean().item()
        return {"R": inde}