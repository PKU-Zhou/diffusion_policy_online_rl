"""
性能计时工具模块

提供Timer上下文管理器和TimerRegistry单例，用于测量训练过程各阶段的耗时。
"""

import time
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager


class TimerRegistry:
    """
    计时器注册表，用于统一管理所有计时记录
    
    使用单例模式，支持嵌套计时和层级关系。
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()
    
    def reset(self):
        """重置所有计时数据"""
        self._records: List[Dict[str, Any]] = []
        self._active_timers: List[str] = []
        self._cumulative: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}
    
    def start(self, name: str):
        """开始计时"""
        self._start_times[name] = time.perf_counter()
        self._active_timers.append(name)
    
    def stop(self, name: str) -> float:
        """停止计时并返回耗时"""
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        end_time = time.perf_counter()
        duration = end_time - self._start_times[name]
        
        # 记录到累计统计
        self._cumulative[name] += duration
        self._counts[name] += 1
        
        # 记录单次计时
        parent = self._active_timers[-2] if len(self._active_timers) > 1 else None
        self._records.append({
            'name': name,
            'duration': duration,
            'parent': parent,
            'timestamp': end_time
        })
        
        # 从活动计时器列表中移除
        if self._active_timers and self._active_timers[-1] == name:
            self._active_timers.pop()
        
        del self._start_times[name]
        return duration
    
    def record(self, name: str, duration: float, parent: Optional[str] = None):
        """直接记录一个计时结果（用于非上下文管理器方式）"""
        self._cumulative[name] += duration
        self._counts[name] += 1
        self._records.append({
            'name': name,
            'duration': duration,
            'parent': parent,
            'timestamp': time.perf_counter()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            包含总耗时、平均耗时、调用次数等统计信息的字典
        """
        stats = {}
        for name in self._cumulative:
            stats[name] = {
                'total_duration': self._cumulative[name],
                'count': self._counts[name],
                'average_duration': self._cumulative[name] / self._counts[name] if self._counts[name] > 0 else 0
            }
        return stats
    
    def get_hierarchy(self) -> Dict[str, Any]:
        """
        获取层级结构的计时信息
        
        Returns:
            嵌套字典，表示计时器的父子关系
        """
        hierarchy = {}
        
        # 构建层级关系
        for record in self._records:
            name = record['name']
            parent = record['parent']
            
            if parent is None:
                if name not in hierarchy:
                    hierarchy[name] = {
                        'total_duration': 0,
                        'count': 0,
                        'children': {}
                    }
                hierarchy[name]['total_duration'] += record['duration']
                hierarchy[name]['count'] += 1
            else:
                # 确保父节点存在
                if parent not in hierarchy:
                    hierarchy[parent] = {
                        'total_duration': 0,
                        'count': 0,
                        'children': {}
                    }
                
                # 添加到父节点的children中
                if name not in hierarchy[parent]['children']:
                    hierarchy[parent]['children'][name] = {
                        'total_duration': 0,
                        'count': 0
                    }
                hierarchy[parent]['children'][name]['total_duration'] += record['duration']
                hierarchy[parent]['children'][name]['count'] += 1
        
        return hierarchy
    
    def export_json(self, filepath: str):
        """
        导出计时结果为JSON文件
        
        Args:
            filepath: 输出文件路径
        """
        stats = self.get_stats()
        hierarchy = self.get_hierarchy()
        
        output = {
            'summary': stats,
            'hierarchy': hierarchy,
            'records': self._records
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def get_cumulative_time(self, name: str) -> float:
        """获取某个计时器的累计时间"""
        return self._cumulative.get(name, 0.0)
    
    def get_count(self, name: str) -> int:
        """获取某个计时器的调用次数"""
        return self._counts.get(name, 0)


class Timer:
    """
    计时器上下文管理器
    
    用法:
        with Timer('operation_name'):
            # 要计时的代码
            pass
        
    或者指定registry:
        registry = TimerRegistry()
        with Timer('operation_name', registry=registry):
            # 要计时的代码
            pass
    """
    
    def __init__(self, name: str, registry: Optional[TimerRegistry] = None):
        """
        初始化计时器
        
        Args:
            name: 计时器名称
            registry: TimerRegistry实例，如果为None则使用全局单例
        """
        self.name = name
        self.registry = registry if registry is not None else TimerRegistry()
        self.duration = None
    
    def __enter__(self):
        self.registry.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = self.registry.stop(self.name)
        return False


@contextmanager
def timer(name: str, registry: Optional[TimerRegistry] = None):
    """
    计时器装饰器版本
    
    用法:
        with timer('operation_name'):
            # 要计时的代码
            pass
    """
    t = Timer(name, registry)
    with t:
        yield t
