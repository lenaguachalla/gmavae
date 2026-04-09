import torch
from metrics.metrics import Metric
from models.image.diffvae import DiffVAE, EuclidianSpace, HyperSphericalSpace
from models.image.betavae import BetaVAE
from models.image.dipvae import DIPVAE
from models.image.factorvae import FactorVAE

class SigmaMetric(Metric) :
    """
    Compute the mean sigma output of a VAE
    """
    def __repr__(self) :
        return "sigma"
    
    def compute_metrics(self) :
        image_encoder = self.algo.image_encoder
        if not hasattr(image_encoder, "beta") :
            return {}
        
        metrics = {}

        for X, _ in self.loader :
            X = X.flatten(0,1).to(self.device)
            encoding = image_encoder.encoder(X)
            
            if type(image_encoder) == DiffVAE:
                enc_mu, enc_logts = image_encoder.split(encoding)
                if any (isinstance(image_encoder.latent_spaces[i], EuclidianSpace) for i in range(len(enc_logts))) :
                    enc_sigma = torch.stack([torch.exp(enc_logts[i]) for i in range(len(enc_logts))\
                                            if isinstance(image_encoder.latent_spaces[i], EuclidianSpace)])
                    if "enc_sigma" not in metrics :
                        metrics["enc_sigma"] = [enc_sigma.mean().detach().item()]
                    else :
                        metrics["enc_sigma"].append(enc_sigma.mean().detach().item())
                if any (isinstance(image_encoder.latent_spaces[i], HyperSphericalSpace) for i in range(len(enc_logts))) :
                    enc_sigma = torch.stack([enc_logts[i] for i in range(len(enc_logts))\
                                            if isinstance(image_encoder.latent_spaces[i], HyperSphericalSpace)])
                    if "logt" not in metrics :
                        metrics["logt"] = [enc_sigma.mean().detach().item()]
                    else :
                        metrics["logt"].append(enc_sigma.mean().detach().item())

            elif type(image_encoder) in [BetaVAE, DIPVAE, FactorVAE] :
                enc_mu, enc_logvar = image_encoder.split(encoding)
                enc_sigma = torch.exp(0.5*enc_logvar)
                if "enc_sigma" not in metrics :
                    metrics["enc_sigma"] = [enc_sigma.mean().detach().item()]
                else :
                    metrics["enc_sigma"].append(enc_sigma.mean().detach().item())

        return metrics