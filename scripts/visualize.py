"""
Visualize noisy observations and their reconstructions from a trained model

    Single model (3 rows): clean, noisy, recon
    python scripts/visualize.py flc 0.2 

    Compare two models (4 rows) clean, noisy, recon1, recon2
    python scripts/visualize.py flc 0.2 flc_noisy_obs2

Arguments:
    1: model name (like flc, flc_noisy_obs, coil2)
    2: obs_noise_std (float)
    3: (optional) second model name for comparison

Obs: models are set to be gmavae on seed 0. Output path and number of samples can be changed 
    
"""

import torch
import matplotlib.pyplot as plt
import yaml
import os
import sys
from data_generator import load_data

def add_obs_noise(images, std):
    if std == 0.0:
        return images
    images = images + torch.randn_like(images) * std
    return torch.clamp(images, 0.0, 1.0)

def to_numpy_img(tensor):
    img = tensor.cpu().numpy()
    return img

def plot_rows(rows, title, output, n_samples):
    fig, axes = plt.subplots(len(rows), n_samples, figsize=(n_samples * 2, len(rows) * 2))
    fig.suptitle(title)
    for row_idx, row_data in enumerate(rows):
        for col_idx in range(n_samples):
            ax = axes[row_idx][col_idx]
            ax.imshow(to_numpy_img(row_data[col_idx]))
            ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    print(f"Saved to {output}")
    plt.show()

if __name__ == "__main__":

    model_path = ('expe/' + sys.argv[1] + '/gmavae/0/last_model') if len(sys.argv) > 1 else KeyError("Provide the model as the first argument")
    obs_noise_std = float(sys.argv[2]) if len(sys.argv) > 2 else KeyError("Provide the observation noise as the second argument")
    model_path2 = ('expe/' + sys.argv[3] + 'gmavae/0/last_model') if len(sys.argv) > 3 else None
    
    with open('expe/' + sys.argv[1] + '/gmavae/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]["name"]

    output = "images/visu/reconstruction.png" # Change output path as to not overwrite every time
    n_samples = 8 # Default number of samples to visualize

    torch.manual_seed(42)

    images, nfo = load_data(dataset)
    indices = torch.randperm(len(images))[:n_samples]
    clean = images[indices].float()
    noisy = add_obs_noise(clean, obs_noise_std)

    algo = torch.load(model_path, map_location='cpu', weights_only=False)
    algo.to('cpu')
    algo.eval()
    with torch.no_grad():
        recon = algo.forward(noisy)

    if model_path2 is not None:

        algo2 = torch.load(model_path2, map_location='cpu', weights_only=False)
        algo2.to('cpu')
        algo2.eval()
        with torch.no_grad():
            recon2 = algo2.forward(noisy)

        rows = [clean, noisy, recon, recon2]
        title = f"Model 1: {sys.argv[1]}  |  Model 2: {sys.argv[3]} \n Dataset: {dataset}  |  Noise std = {obs_noise_std}"

    else:

        rows = [clean, noisy, recon]
        title = f"Model: {sys.argv[1]} \n Dataset: {dataset}  |  Noise std = {obs_noise_std}"

    plot_rows(rows, title, output, n_samples)