#!/usr/bin/env python3
"""分析训练数据分布的主脚本

用法示例:
    python analyze_distribution.py --data_dir logs/HalfCheetah-v4_sdac_xxx/data_distribution
"""

import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, List
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from zczhou.data_distribution.utils.statistics import (
    compute_statistics,
    compute_histogram,
    save_statistics_json,
    load_collected_data,
    group_data_by_network_layer
)
from zczhou.data_distribution.utils.visualization import (
    plot_single_histogram,
    plot_distribution_evolution,
    plot_network_comparison,
    plot_multi_layer_comparison,
    setup_matplotlib_chinese
)


def analyze_weights(data_list: List[Dict], output_dir: Path):
    """分析权重数据
    
    Args:
        data_list: 权重数据列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n分析权重数据，共 {len(data_list)} 个样本")
    
    # 按网络和层分组
    grouped = group_data_by_network_layer(data_list)
    
    # 对每个网络的每个层进行分析
    for network_name, layers in grouped.items():
        print(f"\n  网络: {network_name}")
        network_dir = output_dir / network_name
        network_dir.mkdir(parents=True, exist_ok=True)
        
        layer_data_for_comparison = {}
        
        for layer_name, items in layers.items():
            print(f"    层: {layer_name}, 样本数: {len(items)}")
            
            # 计算该层所有数据的统计量
            all_data = np.concatenate([item['data'] for item in items])
            stats = compute_statistics(all_data)
            
            # 保存统计结果
            stats_file = network_dir / f"{layer_name.replace('/', '_')}_stats.json"
            save_statistics_json(stats, stats_file)
            
            # 绘制直方图
            hist_file = network_dir / f"{layer_name.replace('/', '_')}_hist.png"
            plot_single_histogram(
                all_data,
                title=f'{network_name} - {layer_name} 权重分布',
                save_path=hist_file,
                xlabel='权重值',
                ylabel='频数'
            )
            
            # 如果有多个训练步，绘制演化图
            if len(items) > 1:
                step_data = {item['step']: item['data'] for item in items}
                evolution_file = network_dir / f"{layer_name.replace('/', '_')}_evolution.png"
                plot_distribution_evolution(
                    step_data,
                    layer_name=layer_name,
                    network_name=network_name,
                    data_type='权重',
                    save_path=evolution_file
                )
            
            # 收集用于跨网络对比
            layer_data_for_comparison[layer_name] = all_data
        
        # 绘制该网络内多层对比
        if len(layer_data_for_comparison) > 1:
            comparison_file = network_dir / f"{network_name}_layers_comparison.png"
            plot_multi_layer_comparison(
                layer_data_for_comparison,
                network_name=network_name,
                data_type='权重',
                save_path=comparison_file
            )
    
    # 跨网络对比（对相同的层）
    networks = list(grouped.keys())
    if len(networks) >= 3 and 'policy' in networks and 'q1' in networks and 'q2' in networks:
        # 找到所有网络共有的层
        common_layers = set(grouped['policy'].keys())
        for net in ['q1', 'q2']:
            common_layers &= set(grouped[net].keys())
        
        for layer_name in common_layers:
            # 获取三个网络该层的数据
            policy_data = np.concatenate([item['data'] for item in grouped['policy'][layer_name]])
            q1_data = np.concatenate([item['data'] for item in grouped['q1'][layer_name]])
            q2_data = np.concatenate([item['data'] for item in grouped['q2'][layer_name]])
            
            comparison_file = output_dir / f"network_comparison_{layer_name.replace('/', '_')}.png"
            plot_network_comparison(
                policy_data, q1_data, q2_data,
                layer_name=layer_name,
                data_type='权重',
                save_path=comparison_file
            )


def analyze_gradients(data_list: List[Dict], output_dir: Path):
    """分析梯度数据
    
    Args:
        data_list: 梯度数据列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n分析梯度数据，共 {len(data_list)} 个样本")
    
    # 按网络和层分组
    grouped = group_data_by_network_layer(data_list)
    
    # 对每个网络的每个层进行分析
    for network_name, layers in grouped.items():
        print(f"\n  网络: {network_name}")
        network_dir = output_dir / network_name
        network_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_name, items in layers.items():
            print(f"    层: {layer_name}, 样本数: {len(items)}")
            
            # 计算该层所有数据的统计量
            all_data = np.concatenate([item['data'] for item in items])
            stats = compute_statistics(all_data)
            
            # 保存统计结果
            stats_file = network_dir / f"{layer_name.replace('/', '_')}_stats.json"
            save_statistics_json(stats, stats_file)
            
            # 绘制直方图
            hist_file = network_dir / f"{layer_name.replace('/', '_')}_hist.png"
            plot_single_histogram(
                all_data,
                title=f'{network_name} - {layer_name} 梯度分布',
                save_path=hist_file,
                xlabel='梯度值',
                ylabel='频数'
            )


def generate_markdown_report(output_dir: Path):
    """生成Markdown格式的汇总报告
    
    Args:
        output_dir: 分析结果目录
    """
    report_path = output_dir / "summary_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 训练数据分布分析报告\n\n")
        f.write("## 概述\n\n")
        f.write("本报告汇总了训练过程中收集的权重和梯度数据的分布统计。\n\n")
        
        # 权重分析
        weight_dir = output_dir / "weight"
        if weight_dir.exists():
            f.write("## 权重分布分析\n\n")
            
            for network_dir in sorted(weight_dir.iterdir()):
                if network_dir.is_dir():
                    network_name = network_dir.name
                    f.write(f"### {network_name} 网络\n\n")
                    
                    # 读取统计文件
                    for stats_file in sorted(network_dir.glob("*_stats.json")):
                        layer_name = stats_file.stem.replace('_stats', '')
                        
                        with open(stats_file, 'r') as sf:
                            stats = json.load(sf)
                        
                        f.write(f"#### {layer_name}\n\n")
                        f.write(f"- 均值: {stats['mean']:.6f}\n")
                        f.write(f"- 标准差: {stats['std']:.6f}\n")
                        f.write(f"- 最小值: {stats['min']:.6f}\n")
                        f.write(f"- 最大值: {stats['max']:.6f}\n")
                        f.write(f"- 中位数: {stats['median']:.6f}\n\n")
                        
                        # 添加直方图
                        hist_file = network_dir / f"{layer_name}_hist.png"
                        if hist_file.exists():
                            rel_path = hist_file.relative_to(output_dir)
                            f.write(f"![{network_name} {layer_name} 权重分布]({rel_path})\n\n")
        
        # 梯度分析
        gradient_dir = output_dir / "gradient"
        if gradient_dir.exists() and any(gradient_dir.iterdir()):
            f.write("## 梯度分布分析\n\n")
            f.write("（梯度分析结果）\n\n")
        
        f.write("## 结论\n\n")
        f.write("1. 权重分布特征\n")
        f.write("2. 训练稳定性评估\n")
        f.write("3. 网络间差异分析\n")
    
    print(f"\n报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='分析训练数据分布')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='训练时保存数据的目录')
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='分析结果输出目录（默认为data_dir的父目录）')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误：数据目录不存在: {data_dir}")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 默认输出到zczhou/data_distribution
        script_dir = Path(__file__).parent
        output_dir = script_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 设置中文支持
    setup_matplotlib_chinese()
    
    # 加载数据
    print("\n加载数据...")
    data = load_collected_data(data_dir)
    
    # 分析权重
    if data['weights']:
        weight_output = output_dir / 'weight'
        analyze_weights(data['weights'], weight_output)
    else:
        print("\n警告：未找到权重数据")
    
    # 分析梯度（如果有）
    if data['gradients']:
        gradient_output = output_dir / 'gradient'
        analyze_gradients(data['gradients'], gradient_output)
    else:
        print("\n提示：未找到梯度数据")
    
    # 生成报告
    print("\n生成汇总报告...")
    generate_markdown_report(output_dir)
    
    print(f"\n✓ 分析完成，结果保存在：{output_dir}")


if __name__ == '__main__':
    main()
