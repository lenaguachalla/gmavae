from algos import Algo
from typing import Dict
from data_generator import EnvLoader

class Metric():
    def __init__(self, algo: Algo, nfo: Dict, loader: EnvLoader):
        self.algo = algo
        self.nfo = nfo
        self.loader = loader
        self.device = algo.device if algo is not None else "cpu"

    def compute_metrics(self) -> dict:
        raise NotImplementedError
    
    def set_algo(self, algo):
        self.algo = algo
    
def get_metric(type:str, algo, nfo ,loader, kwargs = {}) -> Metric :
    match type :
        case "groups":
            from metrics.groups import GroupMetric as Metric
        case "dim":
            from metrics.dimension import DimensionMetric as Metric
        case "prediction":
            from metrics.prediction import PredictionMetric as Metric
        case "reconstruction":
            from metrics.reconstruction import ReconstructionMetric as Metric
        case "inde":
            from metrics.independance import IndependanceMetric as Metric
        case "norm_z":
            from metrics.norm_z import NormZMetric as Metric
        case "entropy":
            from metrics.entropy import EntropyMetric as Metric
        case "sigma":
            from metrics.sigma import SigmaMetric as Metric
        case _:
            raise ValueError(f"Metric type {type} not recognized")
    
    return Metric(algo, nfo, loader, **kwargs) 