#!/usr/bin/env python3
"""Main script for analyzing training data distributions.

Usage:
    python analyze_distribution.py --data_dir logs/HalfCheetah-v4/sdac_xxx/data_distribution
"""

import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, List
import sys

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

# 网络对应的Haiku模块名前缀，用于剥离后得到统一的层名
NETWORK_MODULE_PREFIX = {
    'policy': 'dacer_policy_net/',
    'q1': 'q_net/',
    'q2': 'q_net/',
}


def normalize_layer_name(layer_name: str, network_name: str) -> str:
    """剥离层名开头的网络模块前缀，得到统一的层名。

    例如 'dacer_policy_net/linear_1/w' -> 'linear_1/w'，
    'q_net/linear_1/w' -> 'linear_1/w'，便于跨网络对比。

    Args:
        layer_name: 收集到的原始层路径
        network_name: 网络名称

    Returns:
        去掉模块前缀后的层名
    """
    prefix = NETWORK_MODULE_PREFIX.get(network_name)
    if prefix and layer_name.startswith(prefix):
        return layer_name[len(prefix):]
    return layer_name


def analyze_weights(data_list: List[Dict], output_dir: Path):
    """Analyze weight data.

    Args:
        data_list: list of weight data items
        output_dir: output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing weight data: {len(data_list)} samples")

    # 统一层名（剥离网络模块前缀），便于跨网络对比
    for item in data_list:
        item['layer'] = normalize_layer_name(item['layer'], item['network'])

    grouped = group_data_by_network_layer(data_list)

    for network_name, layers in grouped.items():
        print(f"\n  Network: {network_name}")
        network_dir = output_dir / network_name
        network_dir.mkdir(parents=True, exist_ok=True)

        layer_data_for_comparison = {}

        for layer_name, items in layers.items():
            print(f"    Layer: {layer_name}, samples: {len(items)}")

            all_data = np.concatenate([item['data'] for item in items])
            stats = compute_statistics(all_data)

            stats_file = network_dir / f"{layer_name.replace('/', '_')}_stats.json"
            save_statistics_json(stats, stats_file)

            hist_file = network_dir / f"{layer_name.replace('/', '_')}_hist.png"
            plot_single_histogram(
                all_data,
                title=f'{network_name} - {layer_name} Weight Distribution',
                save_path=hist_file,
                xlabel='Weight Value',
                ylabel='Frequency'
            )

            if len(items) > 1:
                step_data = {item['step']: item['data'] for item in items}
                evolution_file = network_dir / f"{layer_name.replace('/', '_')}_evolution.png"
                plot_distribution_evolution(
                    step_data,
                    layer_name=layer_name,
                    network_name=network_name,
                    data_type='Weight',
                    save_path=evolution_file
                )

            layer_data_for_comparison[layer_name] = all_data

        if len(layer_data_for_comparison) > 1:
            comparison_file = network_dir / f"{network_name}_layers_comparison.png"
            plot_multi_layer_comparison(
                layer_data_for_comparison,
                network_name=network_name,
                data_type='Weight',
                save_path=comparison_file
            )

    # 跨网络对比（对相同的层）
    networks = list(grouped.keys())
    if all(net in grouped for net in ('policy', 'q1', 'q2')):
        common_layers = set(grouped['policy'].keys())
        for net in ['q1', 'q2']:
            common_layers &= set(grouped[net].keys())

        for layer_name in common_layers:
            policy_data = np.concatenate([item['data'] for item in grouped['policy'][layer_name]])
            q1_data = np.concatenate([item['data'] for item in grouped['q1'][layer_name]])
            q2_data = np.concatenate([item['data'] for item in grouped['q2'][layer_name]])

            comparison_file = output_dir / f"network_comparison_{layer_name.replace('/', '_')}.png"
            plot_network_comparison(
                policy_data, q1_data, q2_data,
                layer_name=layer_name,
                data_type='Weight',
                save_path=comparison_file
            )


def analyze_gradients(data_list: List[Dict], output_dir: Path):
    """Analyze gradient data.

    Args:
        data_list: list of gradient data items
        output_dir: output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing gradient data: {len(data_list)} samples")

    for item in data_list:
        item['layer'] = normalize_layer_name(item['layer'], item['network'])

    grouped = group_data_by_network_layer(data_list)

    for network_name, layers in grouped.items():
        print(f"\n  Network: {network_name}")
        network_dir = output_dir / network_name
        network_dir.mkdir(parents=True, exist_ok=True)

        for layer_name, items in layers.items():
            print(f"    Layer: {layer_name}, samples: {len(items)}")

            all_data = np.concatenate([item['data'] for item in items])
            stats = compute_statistics(all_data)

            stats_file = network_dir / f"{layer_name.replace('/', '_')}_stats.json"
            save_statistics_json(stats, stats_file)

            hist_file = network_dir / f"{layer_name.replace('/', '_')}_hist.png"
            plot_single_histogram(
                all_data,
                title=f'{network_name} - {layer_name} Gradient Distribution',
                save_path=hist_file,
                xlabel='Gradient Value',
                ylabel='Frequency'
            )


def generate_markdown_report(output_dir: Path):
    """Generate a Markdown summary report.

    Args:
        output_dir: analysis output directory
    """
    report_path = output_dir / "summary_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Training Data Distribution Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report summarizes distribution statistics of weights (and gradients, if collected) during training.\n\n")

        weight_dir = output_dir / "weight"
        if weight_dir.exists():
            f.write("## Weight Distribution Analysis\n\n")

            for network_dir in sorted(weight_dir.iterdir()):
                if network_dir.is_dir():
                    network_name = network_dir.name
                    f.write(f"### {network_name} Network\n\n")

                    for stats_file in sorted(network_dir.glob("*_stats.json")):
                        layer_name = stats_file.stem.replace('_stats', '')

                        with open(stats_file, 'r') as sf:
                            stats = json.load(sf)

                        f.write(f"#### {layer_name}\n\n")
                        f.write(f"- mean: {stats['mean']:.6f}\n")
                        f.write(f"- std: {stats['std']:.6f}\n")
                        f.write(f"- min: {stats['min']:.6f}\n")
                        f.write(f"- max: {stats['max']:.6f}\n")
                        f.write(f"- median: {stats['median']:.6f}\n\n")

                        hist_file = network_dir / f"{layer_name}_hist.png"
                        if hist_file.exists():
                            rel_path = hist_file.relative_to(output_dir)
                            f.write(f"![{network_name} {layer_name} weight distribution]({rel_path})\n\n")

        gradient_dir = output_dir / "gradient"
        if gradient_dir.exists() and any(gradient_dir.iterdir()):
            f.write("## Gradient Distribution Analysis\n\n")
            f.write("(Gradient analysis results)\n\n")

        f.write("## Conclusions\n\n")
        f.write("1. Weight distribution characteristics\n")
        f.write("2. Training stability assessment\n")
        f.write("3. Cross-network differences\n")

    print(f"\nReport generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training data distributions')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing data saved during training')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Analysis output directory (default: zczhou/data_distribution)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory does not exist: {data_dir}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    setup_matplotlib_chinese()

    print("\nLoading data...")
    data = load_collected_data(data_dir)

    if data['weights']:
        weight_output = output_dir / 'weight'
        analyze_weights(data['weights'], weight_output)
    else:
        print("\nWarning: no weight data found")

    if data['gradients']:
        gradient_output = output_dir / 'gradient'
        analyze_gradients(data['gradients'], gradient_output)
    else:
        print("\nNote: no gradient data found")

    print("\nGenerating summary report...")
    generate_markdown_report(output_dir)

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
