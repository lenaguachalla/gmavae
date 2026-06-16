import numpy as np
import torch
from data_generator.data_generator import load_data, get_generator
from typing import Dict

class EnvLoader() :
    def __init__(self,
                 name:str,
                 batch_size:int,
                 m: int = 1,                        #sequence length
                 n_actions_per_state: int = None,   #number of actions available per state
                 action_noise_std: float = 0.0,     #additive noise on the action
                 obs_noise_std: float = 0.0,        #additive noise on the observation
                 ood: bool = False,                 #out of distribution setup
                 device = "cpu",
                 ):
        """
        Dataloader for the environment
        The data is generated on the fly as in data_generator
        """

        assert not (ood and n_actions_per_state is not None), "ood and n_actions_per_state cannot be used together"

        environment = name.split("/")[0]
        self.dataset, self.nfo = load_data(name)

        self.generator = get_generator(environment, specs=self.nfo["specs"])
        self.batch_size = batch_size
        self.m = m
        self.action_noise_std = action_noise_std
        self.obs_noise_std = obs_noise_std
        self.send = False
        self.device = device
        self.dataset = self.dataset.to(self.device)
        self.ood = ood
        if n_actions_per_state :
            n_action = self.generator.group.n_actions
            A = torch.arange(n_action)
            b = self.generator.group.n_states
            np.random.seed(0)
            self.available_actions = torch.Tensor([np.random.choice(A, size=n_actions_per_state, replace=False) for _ in range(b)]).int()
        else :
            self.available_actions = None

    def __iter__(self):
        self.send = False
        return self
    
    def __next__(self):
        if self.send :
            raise StopIteration
        else :
            i = torch.randint(self.generator.group.n_states, size=(self.batch_size,))
            images = [self.dataset[i]]
            actions = []
            for _ in range(self.m) :
                if self.ood :
                    available_actions = self.generator.ood_actions(i)
                    A = np.array([np.random.choice(np.where(available_actions[k])[0]) for k in range(self.batch_size)])
                    A = torch.from_numpy(A)
                elif self.available_actions is not None :
                    indices = torch.randint(0, self.available_actions.shape[1], size=(self.batch_size,))
                    A = self.available_actions[i, indices]
                else :
                    A = torch.randint(0, self.generator.group.n_actions, size=(self.batch_size,))
                
                i = self.generator.group.transition(i, self.generator.add_action_noise(A, self.action_noise_std))
                
                "Add gaussian noise to the observation"
                img = self.dataset[i]
                if self.obs_noise_std > 0.0:
                    img = img + torch.randn_like(img) * self.obs_noise_std
                    img = torch.clamp(img, 0.0, 1.0)

                images.append(img)
                actions.append(A)

            images = torch.stack(images, axis=1)
            actions = torch.stack(actions, axis=1).to(self.device)
            actions = self.generator.process_action(actions)

            self.send = True
            return images, actions

    def __len__(self):
        return 1

    def get_nfo(self, k:str = None) :
        if k is None :
            return self.nfo
        else :
            return self.nfo[k]
        

def get_loader(loader_specs: Dict,
               batch_size: int,
               device: str) -> EnvLoader:
    """
    Get a loader from the specifications
    """
    loader_specs = loader_specs.copy()

    return EnvLoader(batch_size=batch_size,
                     device=device,
                     **loader_specs)