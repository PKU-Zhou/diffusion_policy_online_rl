"""训练过程中的数据收集器

用于收集网络权重、激活和梯度的数据分布。
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk


class DataCollector:
    """训练过程中的数据收集器
    
    收集指定层的权重和梯度数据，用于后续的分布分析。
    """
    
    def __init__(
        self,
        output_dir: Path,
        sample_interval: int = 1000,
        layer_patterns: Optional[List[str]] = None,
        save_batch_size: int = 10,
        activation_max_samples: int = 2048,
        collect_types: Optional[List[str]] = None,
    ):
        """初始化数据收集器
        
        Args:
            output_dir: 数据输出目录
            sample_interval: 采样间隔（训练步数）
            layer_patterns: 要采样的层名称模式，如 ['linear_0', 'linear_1']
                          如果为None，则收集所有层
            save_batch_size: 累积多少次采样后保存一次到磁盘
            activation_max_samples: 每层激活值最多保留的元素数（展平后截断）
            collect_types: 要收集的数据类型列表，如 ['weight', 'gradient', 'activation']
                          如果为None，则收集全部类型
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_interval = sample_interval
        self.layer_patterns = layer_patterns or []
        self.save_batch_size = save_batch_size
        self.activation_max_samples = activation_max_samples
        self.collect_types = set(collect_types) if collect_types else {'weight', 'gradient', 'activation'}
        
        # 数据缓存
        self.weight_cache = []  # [(step, network_name, layer_name, data), ...]
        self.gradient_cache = []  # [(step, network_name, layer_name, data), ...]
        self.activation_cache = []  # [(step, network_name, layer_name, data), ...]
        
        # 计数器
        self.samples_since_save = 0
        
    def should_collect_type(self, data_type: str) -> bool:
        """判断是否需要收集指定类型的数据
        
        Args:
            data_type: 数据类型（'weight', 'gradient', 'activation'）
            
        Returns:
            是否需要收集
        """
        return data_type in self.collect_types
        
    def should_collect(self, step: int) -> bool:
        """判断当前步数是否需要收集数据
        
        Args:
            step: 当前训练步数
            
        Returns:
            是否需要收集数据
        """
        return step % self.sample_interval == 0
    
    def _match_layer(self, layer_path: str) -> bool:
        """判断层名称是否匹配过滤模式
        
        Args:
            layer_path: 层路径（不含 w/b 后缀），如 'q_net/linear_1'
            
        Returns:
            是否匹配
        """
        if not self.layer_patterns:
            return True
        
        # 取路径最后一段作为层名
        base = layer_path.split('/')[-1]
        
        for pattern in self.layer_patterns:
            pattern = pattern.strip()
            if not pattern:
                continue
            if pattern == base:
                return True
            # Haiku 首层不带下标：linear_0 匹配首层 linear
            if pattern.endswith('_0') and pattern[:-2] == base:
                return True
        return False
    
    def _extract_layers(self, params: hk.Params) -> Dict[str, np.ndarray]:
        """从参数树中提取匹配的层
        
        Args:
            params: Haiku参数树
            
        Returns:
            层名称到数据的映射 {layer_name: data}
        """
        result = {}
        
        def extract_recursive(d, prefix=''):
            if isinstance(d, dict):
                for key, value in d.items():
                    new_prefix = f"{prefix}/{key}" if prefix else key
                    extract_recursive(value, new_prefix)
            elif isinstance(d, (jnp.ndarray, np.ndarray)):
                # prefix 形如 'q_net/linear_1/w'，去掉最后的 w/b 得到层路径
                layer_path = '/'.join(prefix.split('/')[:-1])
                if self._match_layer(layer_path):
                    result[prefix] = np.array(d)  # 转换为numpy数组
        
        extract_recursive(params)
        return result
    
    def collect_weights(self, params: hk.Params, network_name: str, step: int = 0):
        """收集网络权重
        
        Args:
            params: 网络参数
            network_name: 网络名称（如 'policy', 'q1', 'q2'）
            step: 当前训练步数
        """
        layers = self._extract_layers(params)
        
        for layer_name, data in layers.items():
            self.weight_cache.append({
                'step': step,
                'network': network_name,
                'layer': layer_name,
                'data': data.flatten()  # 展平为一维数组
            })
    
    def collect_gradients(self, grads: hk.Params, network_name: str, step: int = 0):
        """收集网络梯度
        
        Args:
            grads: 网络梯度
            network_name: 网络名称（如 'policy', 'q1', 'q2'）
            step: 当前训练步数
        """
        layers = self._extract_layers(grads)
        
        for layer_name, data in layers.items():
            self.gradient_cache.append({
                'step': step,
                'network': network_name,
                'layer': layer_name,
                'data': data.flatten()  # 展平为一维数组
            })
    
    def collect_activations(self, activations: Dict[str, Any], network_name: str, step: int = 0):
        """收集网络激活值
        
        Args:
            activations: 层名称到激活值数组的映射，如 {'linear_1': array(batch, 256)}
                        层名称可以是带模块前缀的路径（如 'dacer_policy_net/linear_1'）
            network_name: 网络名称（如 'policy', 'q1', 'q2'）
            step: 当前训练步数
        """
        for layer_path, data in activations.items():
            # 统一层名处理：支持 'linear_1' 或 'dacer_policy_net/linear_1/w' 形式
            # 拦截器给出的是模块名（如 'linear_1'），直接匹配
            if not self._match_layer(layer_path):
                continue
            
            arr = np.asarray(data).flatten()
            # 截断以控制体积
            if len(arr) > self.activation_max_samples:
                arr = arr[:self.activation_max_samples]
            
            self.activation_cache.append({
                'step': step,
                'network': network_name,
                'layer': layer_path,
                'data': arr
            })
    
    def save_batch(self, step: int):
        """将累积的数据保存到磁盘
        
        Args:
            step: 当前训练步数
        """
        self.samples_since_save += 1
        
        if self.samples_since_save >= self.save_batch_size:
            self._flush_to_disk()
            self.samples_since_save = 0
    
    def _flush_to_disk(self):
        """将缓存数据写入磁盘"""
        import time
        timestamp = int(time.time())
        
        # 保存权重数据
        if self.weight_cache:
            weight_file = self.output_dir / f"weights_{timestamp}.npz"
            data_dict = {}
            for i, item in enumerate(self.weight_cache):
                prefix = f"item_{i}"
                data_dict[f"{prefix}_step"] = item['step']
                data_dict[f"{prefix}_network"] = item['network']
                data_dict[f"{prefix}_layer"] = item['layer']
                data_dict[f"{prefix}_data"] = item['data']
            
            np.savez_compressed(weight_file, **data_dict)
            print(f"已保存 {len(self.weight_cache)} 个权重样本到 {weight_file}")
            self.weight_cache.clear()
        
        # 保存梯度数据
        if self.gradient_cache:
            gradient_file = self.output_dir / f"gradients_{timestamp}.npz"
            data_dict = {}
            for i, item in enumerate(self.gradient_cache):
                prefix = f"item_{i}"
                data_dict[f"{prefix}_step"] = item['step']
                data_dict[f"{prefix}_network"] = item['network']
                data_dict[f"{prefix}_layer"] = item['layer']
                data_dict[f"{prefix}_data"] = item['data']
            
            np.savez_compressed(gradient_file, **data_dict)
            print(f"已保存 {len(self.gradient_cache)} 个梯度样本到 {gradient_file}")
            self.gradient_cache.clear()
        
        # 保存激活数据
        if self.activation_cache:
            activation_file = self.output_dir / f"activations_{timestamp}.npz"
            data_dict = {}
            for i, item in enumerate(self.activation_cache):
                prefix = f"item_{i}"
                data_dict[f"{prefix}_step"] = item['step']
                data_dict[f"{prefix}_network"] = item['network']
                data_dict[f"{prefix}_layer"] = item['layer']
                data_dict[f"{prefix}_data"] = item['data']
            
            np.savez_compressed(activation_file, **data_dict)
            print(f"已保存 {len(self.activation_cache)} 个激活样本到 {activation_file}")
            self.activation_cache.clear()
    
    def finalize(self):
        """训练结束时保存剩余数据"""
        if self.weight_cache or self.gradient_cache or self.activation_cache:
            print("保存剩余数据...")
            self._flush_to_disk()
            print("数据收集完成")
