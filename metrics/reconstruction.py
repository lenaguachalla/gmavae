import torch
from metrics.metrics import Metric

class ReconstructionMetric(Metric) :
    """
    Compute the reconstruction MSE
    """

    def __repr__(self) :
        return "reconstruction"
    
    def compute_metrics(self) :
        algo = self.algo
        L_rec = []
        for X, _ in self.loader :
            X = X.flatten(0,1)
            X_hat = algo.forward(X)
            rec = torch.nn.MSELoss()(X_hat, X).detach().item()

            L_rec.append(rec)

        return {"error": sum(L_rec)/len(L_rec)}