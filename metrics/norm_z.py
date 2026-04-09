import torch
from metrics.metrics import Metric

class NormZMetric(Metric) :
    """
    Compute the mean norm of the latent representation
    """
    def __repr__(self) :
        return "norm_z"
    
    def compute_metrics(self) :
        algo = self.algo
        L_norm = []
        for X, A in self.loader :
            X = X.flatten(0,1).to(algo.device)
            Xp_hat = algo.encode_image(X)
            norm = torch.linalg.norm(Xp_hat, dim = -1).detach().mean().item()

            L_norm.append(norm)

        return {"norm": sum(L_norm)/len(L_norm)}