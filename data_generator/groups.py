import numpy as np
from itertools import permutations, product
from math import factorial
from typing import List, Dict

"""
Define some useful groups and their product
"""

class Group:
    def __init__(self) :
        self.state_table: np.ndarray = None
        self.transition_table: np.ndarray = None
    
    @property
    def n_states(self) -> int:
        # return the number of states possible
        raise NotImplementedError
    
    @property
    def n_actions(self) -> int:
        # return the number of actions possible
        raise NotImplementedError

    def transition(self, idx:np.ndarray, a:np.ndarray) -> np.ndarray:
        # map an array of states ids and action idx to their next state
        # -1 is corresponding to the identity action
        return self.transition_table[idx,a]
    
    def get_state(self, idx:np.ndarray) -> np.ndarray:
        # get a state from its id
        return self.state_table[idx]
    
    @property
    def groups_list(self) -> np.ndarray:
        # return the partition of subgroups
        # useful for the product of groups
        return [list(range(self.n_actions))]
    
    @property
    def len_state(self) -> int:
        # return the length of the state
        return len(self.get_state(0))
    
class CyclicGroup(Group):
    def __init__(self, n, m=None) :
        # Group corresponding to Z/nZ
        # If m is None, all the actions are available (except the identity)
        # If m is int, the only actions are -m and +m
        # If m is list of int, the actions are the elements of m

        self.n = n
        self.m = m
        if m is None :
            self.shifts = list(range(1,n))
        elif isinstance(m, int):
            assert m%n !=0 
            if m%n == -m%n :
                self.shifts = [m%n]
            else :
                self.shifts = [-m%n, m%n]
        elif isinstance(m, list):
            seen = set()
            for mi in m:
                assert mi%n != 0
                assert mi%n not in seen
                seen.add(mi)
            self.shifts = [mi%n for mi in m]
        else :
            raise ValueError("m should be None, an int or a list of int")

        # Transition table
        self.transition_table = np.zeros((n,self.n_actions)).astype(int)
        for idx in range(n):
            for k,s in enumerate(self.shifts):
                self.transition_table[idx,k] = (idx + s) % n
                    
        # State table
        self.state_table = np.arange(n).reshape(-1,1)
    def __repr__(self):
        if self.m is None:
            return f"z{self.n}"
        elif isinstance(self.m, int):
            return f"x{self.n}d{self.m}"
        elif isinstance(self.m, list):
            return f"x{self.n}d{'_'.join(map(str,self.m))}"
        
    @property
    def n_states(self) -> int:
        return self.n
    
    @property
    def n_actions(self) -> int:
        return len(self.shifts)

class PermutationGroup(Group):
    def __init__(self, n, available_actions: List[int] = None):
        # If available_actions is None, all the actions are available
        # If available_actions is a list, only those actions are available
        if available_actions is None :
            self.available_actions = list(range(1, factorial(n)))
        else :
            seen = set()
            for a in available_actions:
                assert a > 0 and a < factorial(n)
                assert a not in seen
                seen.add(a)
            self.available_actions = available_actions
        self.available_actions = np.array(self.available_actions).astype(int)
        # Group corresponding to the permutation group of n elements

        self.n = n
        self.permuts = np.array(list(permutations(range(n)))).astype(int)

        state_to_idx = {}
        for idx,p in enumerate(self.permuts):
            state_to_idx[tuple(p)] = idx
        
        # Transition table
        self.transition_table = np.zeros((factorial(n),self.n_actions))
        for idx in range(factorial(n)):
            for k,a in enumerate(self.available_actions):
                self.transition_table[idx,k] = state_to_idx[tuple(self.permuts[idx][self.permuts[a]])]


        # State table
        self.state_table = self.permuts
    def __repr__(self):
        repr = f"s{self.n}"
        if len(self.available_actions) < factorial(self.n) - 1:
            repr += "_" + "_".join(map(str, self.available_actions))
        return repr
    
    @property
    def n_states(self) -> int:
        return factorial(self.n)
    
    @property
    def n_actions(self) -> int:
        return len(self.available_actions)
    
    def get_state(self, idx):
        return self.permuts[idx]
    
class GroupProduct(Group):
    def __init__(self,
                 groups:List[Group],
                 e: bool = False,
                 entangled_actions: int = None) :
        assert not (e and entangled_actions is not None), "Cannot have entangled actions with identity action"

        self.entangled_actions = entangled_actions
        self.e = e 

        # Product of groups
        self.groups = groups

        # map groupproduct state to subgroups states
        self.idxs_to_idx = np.zeros([g.n_states for g in self.groups], dtype=int)
        self.idx_to_idxs = np.zeros([self.n_states, len(self.groups)], dtype=int)
        for k,idxs in enumerate(product(*[range(g.n_states) for g in self.groups])):
            self.idx_to_idxs[k] = np.array(idxs)
            self.idxs_to_idx[tuple(idxs)] = k

        # map groupproduct action to subgroups actions
        if entangled_actions is not None :
            assert entangled_actions <= len(self.groups)
            self.A_to_as = []
            self.as_to_A = {}
            np.random.seed(0)
            for _ in range(self.n_actions) :
                subgroup = np.random.choice(range(len(self.groups)), size=entangled_actions, replace=False)
                actions = tuple(np.random.randint(0,self.groups[g].n_actions) if g in subgroup else -1 for g in range(len(self.groups)))
                self.as_to_A[actions] = len(self.A_to_as)
                self.A_to_as.append(actions)
            self.A_to_as = np.array(self.A_to_as) # [n_a,n_g]
        else :
            self.A_to_a = []
            for k,g in enumerate(self.groups):
                self.A_to_a += [(k,a) for a in range(g.n_actions)]
            if e :
                self.A_to_a += [(0,-1)] # identity action
            self.A_to_a = np.array(self.A_to_a)
        super().__init__()

    def __repr__(self):
        repr = ""
        for g in self.groups:
            repr += f"_{g}"
        return repr[1:]

    @property
    def n_states(self) -> int:
        return int(np.prod([g.n_states for g in self.groups]))
    
    @property
    def n_actions(self) -> int:
        if self.entangled_actions is not None :
            return 5
        else:
            return int(np.sum([g.n_actions for g in self.groups])) + self.e
    
    def transition(self, idx, A) :
        #idx [B]
        #A [B]
        if self.entangled_actions:
            B = len(idx)
            actions = self.A_to_as[A] #[B,G]
            actions[A==-1] = -1
            idxs = self.idx_to_idxs[idx] #[B,G]
            for l in range(len(self.groups)):
                idxs[:,l] = self.groups[l].transition(idxs[:,l],actions[:,l])
            
            idx = self.idxs_to_idx[tuple(idxs.T)] #[B]
        else :
            B = len(idx)
            ka = self.A_to_a[A]
            k,a = ka[...,0],ka[...,1] #[B]
            idxs = self.idx_to_idxs[idx] #[B,G]

            pass
            for l in range(len(self.groups)):
                idxs[k==l,l] = self.groups[l].transition(idxs[k==l,l],a[k==l])
            
            idx = self.idxs_to_idx[tuple(idxs.T)] #[B]
        return idx

    def get_state(self, idx) :
        #idx [B]
        idxs = self.idx_to_idxs[idx] #[B,G]
        state = np.zeros([len(idx),0]).astype(int)
        for l in range(len(self.groups)):
            state = np.hstack([state,self.groups[l].get_state(idxs[:,l])])
        return state
    
    @property
    def groups_list(self) -> np.ndarray:
        offset = 0
        groups = []
        for G in self.groups:
            for g in G.groups_list:
                groups.append([a+offset for a in g])
            offset += G.n_actions
        return groups
    
def generate_group(specs:Dict) -> Group:
    specs = specs.copy()
    type = specs.pop("type")
    if type == "cyclic":
        Group = CyclicGroup
    elif type == "permutation":
        Group =  PermutationGroup
    else:
        raise ValueError(type)

    return Group(**specs)