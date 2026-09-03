"""Visualization utilities.

Plot histograms and evolution charts of data distributions.
All chart text is in English to avoid missing CJK font issues.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt


def setup_matplotlib_chinese():
    """Configure matplotlib with an English-safe default font.

    Kept for backward compatibility: charts now use English labels,
    so no CJK font is required.
    """
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def plot_single_histogram(
    data: np.ndarray,
    title: str,
    save_path: Path,
    bins: int = 50,
    xlabel: str = 'Value',
    ylabel: str = 'Frequency'
):
    """Plot a histogram of a single data distribution.

    Args:
        data: 1-D NumPy array
        title: chart title
        save_path: path to save the figure
        bins: number of histogram bins
        xlabel: x-axis label
        ylabel: y-axis label
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    stats_text = f'mean: {np.mean(data):.4f}\nstd: {np.std(data):.4f}\n'
    stats_text += f'min: {np.min(data):.4f}\nmax: {np.max(data):.4f}'
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_distribution_evolution(
    data_dict: Dict[int, np.ndarray],
    layer_name: str,
    network_name: str,
    data_type: str,
    save_path: Path
):
    """Plot how a data distribution evolves over training steps.

    Args:
        data_dict: mapping from training step to data
        layer_name: layer name
        network_name: network name
        data_type: data type ('weight', 'gradient', 'activation')
        save_path: path to save the figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    steps = sorted(data_dict.keys())

    means = [np.mean(data_dict[s]) for s in steps]
    stds = [np.std(data_dict[s]) for s in steps]
    mins = [np.min(data_dict[s]) for s in steps]
    maxs = [np.max(data_dict[s]) for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top: mean and std
    ax1.plot(steps, means, 'b-', label='mean', linewidth=2)
    ax1.fill_between(steps,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, label='±1 std')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'{network_name} - {layer_name} - {data_type} Evolution (Mean)',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: min and max
    ax2.plot(steps, mins, 'g-', label='min', linewidth=2)
    ax2.plot(steps, maxs, 'r-', label='max', linewidth=2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'{network_name} - {layer_name} - {data_type} Evolution (Range)',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_network_comparison(
    policy_data: np.ndarray,
    q1_data: np.ndarray,
    q2_data: np.ndarray,
    layer_name: str,
    data_type: str,
    save_path: Path,
    bins: int = 50
):
    """Plot a comparison across the three networks.

    Args:
        policy_data: policy network data
        q1_data: Q1 network data
        q2_data: Q2 network data
        layer_name: layer name
        data_type: data type ('weight', 'gradient', 'activation')
        save_path: path to save the figure
        bins: number of histogram bins
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    data_list = [policy_data, q1_data, q2_data]
    titles = ['Policy Network', 'Q1 Network', 'Q2 Network']
    colors = ['blue', 'green', 'red']

    for ax, data, title, color in zip(axes, data_list, titles, colors):
        ax.hist(data, bins=bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{title}\n{layer_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        stats_text = f'mean={np.mean(data):.3f}\nstd={np.std(data):.3f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    fig.suptitle(f'Network Comparison - {layer_name} - {data_type}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_multi_layer_comparison(
    data_dict: Dict[str, np.ndarray],
    network_name: str,
    data_type: str,
    save_path: Path,
    bins: int = 50
):
    """Plot a comparison of multiple layers within one network.

    Args:
        data_dict: mapping from layer name to data
        network_name: network name
        data_type: data type
        save_path: path to save the figure
        bins: number of histogram bins
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_layers = len(data_dict)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (layer_name, data) in enumerate(sorted(data_dict.items())):
        ax = axes[idx]
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        stats_text = f'mean={np.mean(data):.3f}\nstd={np.std(data):.3f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{network_name} - {data_type} Distribution by Layer',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
