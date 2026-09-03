"""可视化工具

绘制数据分布的直方图和演化图。
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def setup_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    # 尝试多个中文字体
    chinese_fonts = ['SimHei', 'STHeiti', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            continue
    
    # 如果都不可用，使用默认字体
    if 'font.sans-serif' not in plt.rcParams:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False


def plot_single_histogram(
    data: np.ndarray,
    title: str,
    save_path: Path,
    bins: int = 50,
    xlabel: str = '数值',
    ylabel: str = '频数'
):
    """绘制单个数据分布的直方图
    
    Args:
        data: 一维NumPy数组
        title: 图表标题
        save_path: 保存路径
        bins: 直方图的bin数量
        xlabel: x轴标签
        ylabel: y轴标签
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f'均值: {np.mean(data):.4f}\n标准差: {np.std(data):.4f}\n'
    stats_text += f'最小值: {np.min(data):.4f}\n最大值: {np.max(data):.4f}'
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
    """绘制数据分布随训练步数的演化
    
    Args:
        data_dict: 训练步数到数据的映射 {step: data}
        layer_name: 层名称
        network_name: 网络名称
        data_type: 数据类型（'weight', 'gradient', 'activation'）
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 排序步数
    steps = sorted(data_dict.keys())
    
    # 计算每个步数的统计量
    means = [np.mean(data_dict[s]) for s in steps]
    stds = [np.std(data_dict[s]) for s in steps]
    mins = [np.min(data_dict[s]) for s in steps]
    maxs = [np.max(data_dict[s]) for s in steps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上图：均值和标准差
    ax1.plot(steps, means, 'b-', label='均值', linewidth=2)
    ax1.fill_between(steps,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, label='±1标准差')
    ax1.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('数值', fontsize=12)
    ax1.set_title(f'{network_name} - {layer_name} - {data_type} 演化（均值）', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：最小值和最大值
    ax2.plot(steps, mins, 'g-', label='最小值', linewidth=2)
    ax2.plot(steps, maxs, 'r-', label='最大值', linewidth=2)
    ax2.set_xlabel('训练步数', fontsize=12)
    ax2.set_ylabel('数值', fontsize=12)
    ax2.set_title(f'{network_name} - {layer_name} - {data_type} 演化（范围）', 
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
    """绘制三个网络的数据对比
    
    Args:
        policy_data: Policy网络数据
        q1_data: Q1网络数据
        q2_data: Q2网络数据
        layer_name: 层名称
        data_type: 数据类型（'weight', 'gradient', 'activation'）
        save_path: 保存路径
        bins: 直方图的bin数量
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 数据和标题
    data_list = [policy_data, q1_data, q2_data]
    titles = ['Policy网络', 'Q1网络', 'Q2网络']
    colors = ['blue', 'green', 'red']
    
    for ax, data, title, color in zip(axes, data_list, titles, colors):
        ax.hist(data, bins=bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel('数值', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(f'{title}\n{layer_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    fig.suptitle(f'网络对比 - {layer_name} - {data_type}', 
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
    """绘制同一网络多个层的对比
    
    Args:
        data_dict: 层名称到数据的映射
        network_name: 网络名称
        data_type: 数据类型
        save_path: 保存路径
        bins: 直方图的bin数量
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
        ax.set_xlabel('数值', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{network_name} - 各层{data_type}分布对比', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
