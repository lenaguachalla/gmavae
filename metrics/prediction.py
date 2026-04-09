import torch
from metrics.metrics import Metric
from algos.ae import AE

class PredictionMetric(Metric) :
    """
    Compute the prediction MSE
    """
    def __repr__(self) :
        return "prediction"
    
    def compute_metrics(self) :
        if isinstance(self.algo, AE):
            return {}
        algo = self.algo
        L_pred = []
        for X, A in self.loader :
            X_hat = algo.forward(X[:,0], A[:,:1])
            pred = torch.nn.MSELoss()(X_hat, X[:,1]).detach().item()

            L_pred.append(pred)

        return {"error": sum(L_pred)/len(L_pred)}