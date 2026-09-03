"""统计分析工具

计算数据分布的基本统计量和直方图。
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import json


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """计算数据的基本统计量
    
    Args:
        data: 一维NumPy数组
        
    Returns:
        统计量字典，包含均值、标准差、最小值、最大值、中位数、分位数
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'count': int(len(data))
    }


def compute_histogram(data: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """计算直方图数据
    
    Args:
        data: 一维NumPy数组
        bins: 直方图的bin数量
        
    Returns:
        (counts, bin_edges) 元组
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    return counts, bin_edges


def aggregate_multi_step_data(data_list: List[np.ndarray]) -> Dict[str, Any]:
    """聚合多个训练步的数据
    
    Args:
        data_list: 数据数组列表，每个元素是一个训练步的数据
        
    Returns:
        聚合后的统计信息
    """
    # 合并所有数据
    all_data = np.concatenate(data_list)
    
    # 计算整体统计量
    overall_stats = compute_statistics(all_data)
    
    # 计算每个步骤的统计量
    step_stats = [compute_statistics(data) for data in data_list]
    
    return {
        'overall': overall_stats,
        'per_step': step_stats,
        'num_steps': len(data_list)
    }


def save_statistics_json(stats: Dict, filepath: Path):
    """保存统计结果为JSON文件
    
    Args:
        stats: 统计量字典
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_collected_data(data_dir: Path) -> Dict[str, List[Dict]]:
    """加载收集的数据文件
    
    Args:
        data_dir: 数据目录
        
    Returns:
        数据字典 {'weights': [...], 'gradients': [...], 'activations': [...]}
    """
    data_dir = Path(data_dir)
    
    result = {
        'weights': [],
        'gradients': [],
        'activations': [],
    }
    
    # 统一加载三类文件：weights_*.npz / gradients_*.npz / activations_*.npz
    file_specs = [
        ('weights', 'weights_*.npz'),
        ('gradients', 'gradients_*.npz'),
        ('activations', 'activations_*.npz'),
    ]
    
    for key, pattern in file_specs:
        for data_file in sorted(data_dir.glob(pattern)):
            data = np.load(data_file, allow_pickle=True)
            
            # 解析文件中的所有item
            i = 0
            while f'item_{i}_step' in data:
                item = {
                    'step': int(data[f'item_{i}_step']),
                    'network': str(data[f'item_{i}_network']),
                    'layer': str(data[f'item_{i}_layer']),
                    'data': data[f'item_{i}_data']
                }
                result[key].append(item)
                i += 1
    
    return result


def group_data_by_network_layer(data_list: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """按网络和层分组数据
    
    Args:
        data_list: 数据项列表
        
    Returns:
        嵌套字典 {network: {layer: [items]}}
    """
    grouped = {}
    
    for item in data_list:
        network = item['network']
        layer = item['layer']
        
        if network not in grouped:
            grouped[network] = {}
        
        if layer not in grouped[network]:
            grouped[network][layer] = []
        
        grouped[network][layer].append(item)
    
    return grouped
