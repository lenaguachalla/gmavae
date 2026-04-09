import torch
from metrics.metrics import Metric

class EntropyMetric(Metric) :
    """
    Compute the entropy of the action distribution for GMAVAE
    """
    def __repr__(self) :
        return "entropy"
    
    def compute_metrics(self) :
        algo = self.algo
        if getattr(algo, "pi", None) is None :
            return {}
        dist = algo.dist
        entropy = -(torch.exp(dist) * dist).sum(axis=0) \
                    /torch.exp(dist).sum(axis=0) \
                    +torch.log(torch.exp(dist).sum(axis=0))
        
        return {"entropy":entropy.mean().item()}