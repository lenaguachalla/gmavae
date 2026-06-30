"""Implementation of Independance Metric.

Based on "Linear Disentangled Representations and Unsupervised Action Estimation" .
Implementation based on https://github.com/MattPainter01/UnsupervisedActionEstimation
itself based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import torch
from dislib import utils

def compute_cosine(x:np.ndarray, y:np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    if np.linalg.norm(x) * np.linalg.norm(y) == 0:
        #raise KeyError
        return 1.
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

class IndependanceMetric:
    def __init__(self, ds, num_points=utils.NUM_POINTS):
        super().__init__()
        self.ds = ds
        self.num_points = num_points

    def __repr__(self):
        return "Independance"

    def _compute_cosine(self, rep_fn):
        with torch.no_grad():
            cosines = []

            for _ in range(self.num_points):
                latent = self.ds.sample_latent()
                latent_1, latent_2 = self.ds.sample_diff_latents(latent)
                
                img = self.ds.get_img_by_latent(latent)
                img1 = self.ds.get_img_by_latent(latent_1)
                img2 = self.ds.get_img_by_latent(latent_2)
                
                z = rep_fn(img).cpu().numpy()
                z1 = rep_fn(img1).cpu().numpy()
                z2 = rep_fn(img2).cpu().numpy()

                cosines.append(np.abs(compute_cosine(z-z1, z-z2)))
        return np.mean(cosines)


    def __call__(self, pymodel):
        rep_fn = lambda x: pymodel.encode_image(x.unsqueeze(0))[0]
        
        return {'independance_score': 1-self._compute_cosine(rep_fn)}
