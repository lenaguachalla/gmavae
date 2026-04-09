# LSBD: GMA-VAE
This repository is the code for the paper 

> **Disentangled representation learning through unsupervised symmetry group discovery**
> Barthélémy Dang-Nhu, Louis Annabi, Sylvain Argentieri
> at *ICLR 2026*

📄 arXiv: https://arxiv.org/abs/2603.11790  
🔗 OpenReview: https://openreview.net/forum?id=I6xjMoLY3j 

## Overview of the method
Our method aims to learn a disentangled representation following the Linear Symmetry-Based Disentanglement (LSBD) framework introduced by [Higgins et al., 2018](https://arxiv.org/abs/1812.02230).  

The full pipeline consists of three main steps:

1. __Learn an entangled representation__ using A-VAE:  
this can be done with the implementation in `algos/gmavae` by setting `group_masking=False`.

2. __Recover the group decomposition__:  
   this is achieved by applying the metric in `metrics/groups` to the learned A-VAE model, which clusters actions into subgroups and returns the ARI metric.

3. __Learn a disentangled representation__ using GMA-VAE:  
   this can be done with the implementation in `algos/gmavae` by setting `group_masking=True`. The ground-truth group decomposition is provided via the `groups` argument.


## Run Experiments
- Install the requirements:  
  ```apt install -r requirements.txt```
- Generate data:  
  ```python scripts/generate_data.py```
- Run the experiments:  
  ```python scripts/runs.py```  
  It will run all the experiments at once. If only some experiments have to be runned, add the wanted prefix of config files of the ```config``` folder (e.g. ```python scripts/runs.py flc```). All the trainings have to be runned on GPU for reproductibility.
- Generate the random encoders:  
  ```python scripts/generate_random_encoder```
- Generate all the results from the trained models:  
  ```sh scripts/results/generate.sh```  
  All the numerical results will be printed in the terminal and the plots will be generated in the ```images``` folder
  
## Methods implemented in ```algos``` folder
### Supervised methods
- [LSBD-VAE](https://arxiv.org/abs/2011.06070)

### Self-supervised methods
- GMA-VAE (Ours)
- A-VAE (Ours)
- [Forward-VAE](https://arxiv.org/abs/1904.00243)
- [SOBDRL](https://arxiv.org/abs/2002.06991)
- LSBD-VAE* adapted from [LSBD-VAE](https://arxiv.org/abs/2011.06070)
- [HAE](https://arxiv.org/abs/2207.12067)

### Purely unsupervised methods
- Auto-Encoder
- [$\beta$-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
- [DIP-VAE I/II](https://arxiv.org/abs/1711.00848)
- [Factor-VAE](https://arxiv.org/abs/1802.05983)


## 📂 Repository Structure
```bash
.
├── algos/                # LSBD methods
├── configs/              # YAML experiment configurations
├── data_generator/       # generate and load datasets
  └── environments/       # environments used in the experiments
├── dislib/               # library for disentangled metrics
├── models/               # neural network architectures
├── scripts/              # scripts to generate the figures of the paper
├── train.py              # training loop     
├── utils.py 
├── requirements.txt
└── README.md
```
