"""Performance monitoring and bottleneck detection utilities."""

import time
import gc
import logging
import sys
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import threading
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.function_timings = defaultdict(list)
        self.memory_snapshots = []
        self.start_time = time.time()
        self.monitoring_active = False
        self._lock = threading.Lock()

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        self.start_time = time.time()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")

    def record_metric(self, name: str, value: float, unit: str = ""):
        """Record a performance metric."""
        if not self.monitoring_active:
            return

        with self._lock:
            self.metrics[name].append({
                'timestamp': time.time(),
                'value': value,
                'unit': unit
            })

    def record_function_timing(self, func_name: str, duration: float):
        """Record function execution time."""
        if not self.monitoring_active:
            return

        with self._lock:
            self.function_timings[func_name].append({
                'timestamp': time.time(),
                'duration': duration
            })

    def take_memory_snapshot(self):
        """Take a memory usage snapshot."""
        if not self.monitoring_active:
            return

        snapshot = {
            'timestamp': time.time(),
            'rss_mb': 0,  # Will be filled if psutil is available
            'vms_mb': 0,
            'cpu_percent': 0
        }

        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                snapshot.update({
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'cpu_percent': process.cpu_percent()
                })
            except Exception as e:
                logger.warning(f"Error getting memory snapshot: {e}")

        with self._lock:
            self.memory_snapshots.append(snapshot)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.monitoring_active:
            return {}

        uptime = time.time() - self.start_time

        # Calculate function timing statistics
        function_stats = {}
        for func_name, timings in self.function_timings.items():
            if timings:
                durations = [t['duration'] for t in timings]
                function_stats[func_name] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': np.mean(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'calls_per_second': len(durations) / uptime if uptime > 0 else 0
                }

        # Calculate memory statistics
        memory_stats = {}
        if self.memory_snapshots:
            memory_values = [s['rss_mb'] for s in self.memory_snapshots]
            memory_stats = {
                'current_mb': memory_values[-1] if memory_values else 0,
                'peak_mb': max(memory_values) if memory_values else 0,
                'avg_mb': np.mean(memory_values) if memory_values else 0,
                'min_mb': min(memory_values) if memory_values else 0
            }

        # Get system metrics
        system_stats = self._get_system_metrics()

        return {
            'uptime_seconds': uptime,
            'function_timings': function_stats,
            'memory_stats': memory_stats,
            'system_stats': system_stats,
            'metrics_count': {name: len(values) for name, values in self.metrics.items()}
        }

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not HAS_PSUTIL:
            return {}

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': disk.percent
            }
        except Exception as e:
            logger.warning(f"Error getting system metrics: {e}")
            return {}

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze performance data to identify bottlenecks."""
        if not self.function_timings:
            return {'bottlenecks': [], 'recommendations': []}

        # Find slowest functions
        slow_functions = []
        for func_name, timings in self.function_timings.items():
            if timings:
                avg_time = np.mean([t['duration'] for t in timings])
                total_time = sum([t['duration'] for t in timings])
                call_count = len(timings)

                slow_functions.append({
                    'function': func_name,
                    'avg_time': avg_time,
                    'total_time': total_time,
                    'call_count': call_count,
                    'time_percentage': (total_time / (time.time() - self.start_time)) * 100 if time.time() > self.start_time else 0
                })

        # Sort by total time
        slow_functions.sort(key=lambda x: x['total_time'], reverse=True)

        # Identify potential bottlenecks
        bottlenecks = []
        recommendations = []

        for func in slow_functions[:10]:  # Top 10 slowest functions
            if func['time_percentage'] > 10:  # Functions taking >10% of total time
                bottlenecks.append(func)

                # Generate recommendations
                if func['call_count'] > 100:
                    recommendations.append({
                        'function': func['function'],
                        'issue': 'High call frequency',
                        'recommendation': 'Consider caching results or batching operations'
                    })
                elif func['avg_time'] > 1.0:  # Functions taking >1 second
                    recommendations.append({
                        'function': func['function'],
                        'issue': 'Slow execution time',
                        'recommendation': 'Consider optimization or parallelization'
                    })

        return {
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'total_functions_tracked': len(self.function_timings)
        }

    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        try:
            report = self.get_performance_report()
            bottleneck_analysis = self.get_bottleneck_analysis()

            export_data = {
                'performance_report': report,
                'bottleneck_analysis': bottleneck_analysis,
                'raw_metrics': dict(self.metrics),
                'function_timings': dict(self.function_timings)
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")


# Global performance monitor
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _global_monitor


def time_function(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            _global_monitor.record_function_timing(func.__name__, duration)
    return wrapper


@contextmanager
def monitor_performance(operation_name: str):
    """Context manager for monitoring performance of a code block."""
    start_time = time.time()
    start_memory = None

    try:
        # Get initial memory
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB

        yield

    finally:
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB

        duration = end_time - start_time
        memory_delta = end_memory - (start_memory or 0)

        _global_monitor.record_metric(f"{operation_name}_duration", duration, "seconds")
        _global_monitor.record_metric(f"{operation_name}_memory_delta", memory_delta, "MB")


def profile_memory_usage():
    """Get current memory usage profile."""
    if not HAS_PSUTIL:
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'memory_percent': 0
        }

    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'memory_percent': process.memory_percent()
        }
    except Exception as e:
        logger.warning(f"Error getting memory profile: {e}")
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'memory_percent': 0
        }


def force_garbage_collection():
    """Force garbage collection and return memory stats."""
    gc.collect()
    return profile_memory_usage()


def get_dataframe_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed memory usage for a DataFrame."""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()

    return {
        'total_mb': total_memory / (1024 * 1024),
        'per_column_mb': (memory_usage / (1024 * 1024)).to_dict(),
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }


def detect_memory_leaks(threshold_mb: float = 100) -> List[str]:
    """Detect potential memory leaks."""
    warnings = []

    # Check if memory usage is growing significantly
    if len(_global_monitor.memory_snapshots) >= 10:
        recent_memory = [s['rss_mb'] for s in _global_monitor.memory_snapshots[-10:]]
        memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]

        if memory_trend > 1.0:  # Memory growing by more than 1MB per snapshot
            warnings.append(f"Potential memory leak detected: {memory_trend:.2f} MB/snapshot growth")

    # Check current memory usage
    current_memory = profile_memory_usage()['rss_mb']
    if current_memory > threshold_mb:
        warnings.append(f"High memory usage: {current_memory:.2f} MB")

    return warnings