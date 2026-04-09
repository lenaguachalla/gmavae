import numpy as np
import torch 
from algos import Algo
from metrics.metrics import Metric
from sklearn.metrics import adjusted_rand_score
from data_generator import load_data

def compute_norm_h(A:np.ndarray,
                   Z:np.ndarray) -> np.ndarray:
    # A[...,d,d]
    # Z[b,d]
    # Returns h norm of A wrt Z
    Az = np.einsum('...ij,bj->...bi', A, Z)
    return np.sqrt(np.square(Az).sum(axis=-1)).mean(axis=-1)

class GroupMetric(Metric) :
    """
    Compute The accuracy of the group prediction
    M represents the max number of the same action to get from one action to another
    corresponds to Step 2 of our proposed method
    """
    def __init__(self,
                 algo:Algo,
                 nfo:dict,
                 loaders,
                 M:int=3,
                 s:float= 0.1,):
        super().__init__(algo, nfo, loaders)
        self.data_images = load_data(nfo["dataname"], torched=True)[0].to(self.device)
        self.M = M
        self.s = s

    def __repr__(self) :
        return "groups"    
        
    def compute_metrics(self,
                        dG=None,
                        returns_dG=False):
        if not getattr(self.algo, "encode_action", None) :
            return {}
        algo = self.algo
        n_action = self.nfo["n_action"]
        z_dim = algo.z_dim
        Z = algo.encode_image(self.data_images).cpu().detach().numpy()

        if dG is None :
            A = torch.arange(0,n_action).int().to(self.device)
            Az = algo.encode_action(A).cpu().detach().numpy()
            # [n,z,z]

            # Compute matrices product
            # Ml[i,j,m,:,:] = A_j^m @ A_i
            # Mr[i,j,m,:,:] = A_i @ A_j^m
            Ml = np.broadcast_to(Az[:,None,None,:,:], (n_action, n_action, 1, z_dim, z_dim))
            Mr = np.broadcast_to(Az[:,None,None,:,:], (n_action, n_action, 1, z_dim, z_dim))

            for _ in range(self.M) :
                M = np.einsum("cij,bcjk->bcik", Az, Ml[:,:,-1])
                Ml = np.concatenate([Ml,M[:,:,None]], axis=2)
                M = np.einsum("bcij,cjk->bcik", Mr[:,:,-1], Az)
                Mr = np.concatenate([Mr,M[:,:,None]], axis=2)

            # Compute group distance
            dl = compute_norm_h(Ml[:,:,:,None]-Az[None,None,None,:], Z)
            # dl[i,j,k,l] = ||A_j^k @ A_i - A_l||
            dl = dl[:,:,1:,:] #k=0 is useless

            dl = dl.min(axis=(1,2))
            # dl[i,l] = min_{j,k}||A_j^k @ A_i - A_l||
            
            dr = compute_norm_h(Mr[:,:,:,None]-Az[None,None,None,:], Z)
            # d[i,j,k,l] = ||A_i @ A_j^k - A_l||
            dr = dr[:,:,1:,:] #k=0 is useless

            dr = dr.min(axis=(1,2))
            # d[i,l] = min_{j,k}||A_i @ A_j^k - A_l||

            dG = np.min(np.stack([dl,dl.T,dr,dr.T]), axis=0)
            dG[np.arange(n_action), np.arange(n_action)] = np.inf

        group_gt = self.nfo["group"]
        group_gt = list(set(g) for g in group_gt)

        # check if identity action is in group_gt
        e = sum(len(group) for group in group_gt) == n_action - 1
        
        # initialize groups
        group = [np.array([k]) for k in range(n_action)]
        group_dist = dG.copy()

        i,j = np.unravel_index(group_dist.argmin(), group_dist.shape)
        d = group_dist[i,j]

        while len(group) > 1 and d < self.s:
            #concatenate groups
            group[i] = np.concatenate([group[i],group[j]])
            group.pop(j)

            group_dist[i] = np.maximum(group_dist[i], group_dist[j])
            group_dist[:,i] = np.maximum(group_dist[:,i], group_dist[:,j])
            group_dist = np.delete(group_dist, j, axis=0)
            group_dist = np.delete(group_dist, j, axis=1)

            i,j = np.unravel_index(group_dist.argmin(), group_dist.shape)
            d = group_dist[i,j]

        group = [set(g) for g in group]

        # as identity action can be attributed to any subgroup
        # delete identity action if it exists
        if e:
            for g in group :
                if n_action - 1 in g:
                    g.remove(n_action - 1)
                    if len(g) == 0:
                        group.remove(g)
                    break
            
        def compute_ARI(group):
            index_group_gt = []
            k = 0
            for g in group_gt :
                for _ in g :
                    index_group_gt.append(k)
                k += 1
            index_group = []
            k = 0
            for g in group :
                for _ in g :
                    index_group.append(k)
                k += 1
            return adjusted_rand_score(index_group_gt, index_group)
        
        ari = compute_ARI(group)

        metric = {f"{self.s}" : ari}

        if returns_dG:
            return metric, dG
        else :
            return metric