"""
Plotting utilities for results visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import product
from typing import List
import pandas as pd
import seaborn as sns

colors = [cm.get_cmap('tab20')(i) for i in np.linspace(0, 1, 20)]
COLORS = {
    "LSBD-VAE": colors[0],
    "LSBD-VAE*": colors[10],
    "SOBDRL": colors[2],
    "GMA-VAE": colors[14],
    "A-VAE": colors[4],
    "Beta-VAE": colors[5],
    "SOBDRL(entangled)": colors[6],
    "SOBDRL(disentangled)": colors[8],
}
def boxplot(results:List[np.ndarray],
            x_names: List[str],
            hue_names: List[str]) :
    rows = []

    for r,h in product(range(len(results)), range(len(hue_names))):
        for x in range(results[r].shape[1]):
            rows.append({"Algorithm": x_names[r],
                        "Metric": hue_names[h],
                        "Value": results[r][h,x]})
    df = pd.DataFrame(rows)
    g = sns.catplot(data=df,
                 x="Algorithm",
                 y="Value",
                 hue="Metric",
                 kind="bar",
                 height=3,
                 aspect=1.5)
    g.set(xlabel=None)  # Remove x-axis label
    g.set(ylabel=None)  # Remove x-axis label
    g.set_xticklabels(rotation=-45)
    g._legend.remove()


def plot_std(ys: np.ndarray,
             x: np.ndarray = None,
             label: str = None,
             mode: str = "median",
             **kwargs) :
    ys = np.array(ys)

    if ys.ndim == 1:
        if x is None :
            x = np.arange(len(ys))

        plt.plot(x, ys, label=label, **kwargs)
    elif ys.ndim == 2:
        b,l = ys.shape
        if x is None :
            x = np.arange(l)

        if mode == "median":
            y = np.median(ys, axis=0)
            upper = np.percentile(ys, 75, axis=0)
            lower = np.percentile(ys, 25, axis=0)
        elif mode == "mean":
            y = ys.mean(axis=0)
            upper = ys.mean(axis=0) + ys.std(axis=0)/np.sqrt(b)
            lower = ys.mean(axis=0) - ys.std(axis=0)/np.sqrt(b)

        plt.plot(x, y, label=label, **kwargs, color=COLORS.get(label, None))
        plt.fill_between(x, upper, lower, alpha=0.3, color=COLORS.get(label, None))

def set_rcParams(**kwargs) -> None:
    plt.rcParams.update({
        "text.usetex" : kwargs.get("usetex", False),
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.family": kwargs.get("font_family", "serif"),
        "font.size": kwargs.get("font_size", 11),
        "font.weight": kwargs.get("font_weight", "medium"),
        "axes.labelweight": kwargs.get("axes_labelweight", "medium"),
        "axes.titlesize": kwargs.get("axes_titlesize", 10),
        "axes.labelsize": kwargs.get("axes_labelsize", 11),
        "legend.fontsize": kwargs.get("legend_fontsize", 9),
        "xtick.labelsize": kwargs.get("xtick_labelsize", 10),
        "ytick.labelsize": kwargs.get("ytick_labelsize", 10),
    })