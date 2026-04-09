from metrics.metrics import Metric

class DimensionMetric(Metric) :
    """
    Compute:
    for each group, compute the number attributed by action distanglement
        with the hardcoding argmax on pi
    Additionally compute the highest pi not given to the group
    """
    def __repr__(self) :
        return "dim"
    
    def compute_metrics(self) :
        algo = self.algo
        
        if  not hasattr(algo, "pi") :
            return {}
        
        pi = algo.pi.cpu().detach().numpy()
        
        n_groups, z_dim = pi.shape
        n = {k:0 for k in range(n_groups)}

        for i in range(z_dim) :
            k=pi[:,i].argmax()
            n[k] += 1
            pi[k,i] = 0

        p = {k:pi[k,:].max() for k in range(n_groups)}        

        metrics = {f"n/{k}":v for k,v in n.items()}
        metrics.update({f"p/{k}":v for k,v in p.items()})

        return metrics