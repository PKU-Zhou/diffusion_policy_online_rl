"""数据分布统计工具包"""

from .data_collector import DataCollector
from .statistics import (
    compute_statistics,
    compute_histogram,
    aggregate_multi_step_data,
    save_statistics_json
)
from .visualization import (
    plot_single_histogram,
    plot_distribution_evolution,
    plot_network_comparison,
    setup_matplotlib_chinese
)

__all__ = [
    'DataCollector',
    'compute_statistics',
    'compute_histogram',
    'aggregate_multi_step_data',
    'save_statistics_json',
    'plot_single_histogram',
    'plot_distribution_evolution',
    'plot_network_comparison',
    'setup_matplotlib_chinese',
]
