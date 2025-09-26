"""Real-time performance monitoring dashboard for the trading system."""

import psutil
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from typing import Dict, List, Any
import threading
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


class PerformanceMonitor:
    """Monitor system performance in real-time."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes of data at 1s intervals
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics storage
        self.metrics = {
            'timestamp': deque(maxlen=window_size),
            'cpu_percent': deque(maxlen=window_size),
            'memory_percent': deque(maxlen=window_size),
            'memory_mb': deque(maxlen=window_size),
            'disk_io_read': deque(maxlen=window_size),
            'disk_io_write': deque(maxlen=window_size),
            'network_sent': deque(maxlen=window_size),
            'network_recv': deque(maxlen=window_size),
            'thread_count': deque(maxlen=window_size),
            'process_count': deque(maxlen=window_size)
        }
        
        # Component-specific metrics
        self.component_metrics = {
            'data_fetch_time': deque(maxlen=100),
            'feature_eng_time': deque(maxlen=100),
            'model_train_time': deque(maxlen=100),
            'backtest_time': deque(maxlen=100),
            'prediction_time': deque(maxlen=100)
        }
        
        # Alerts and thresholds
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'response_time': 1.0  # seconds
        }
        
        self.alerts = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize baseline metrics
        self.baseline_metrics = None
        
    def start_monitoring(self):
        """Start the monitoring thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        # Get initial disk and network stats
        disk_io_start = psutil.disk_io_counters()
        net_io_start = psutil.net_io_counters()
        
        while self.monitoring:
            try:
                # Get current metrics
                timestamp = datetime.now()
                
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                memory_mb = memory_info.used / (1024 * 1024)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read = (disk_io.read_bytes - disk_io_start.read_bytes) / (1024 * 1024)  # MB
                disk_write = (disk_io.write_bytes - disk_io_start.write_bytes) / (1024 * 1024)  # MB
                disk_io_start = disk_io
                
                # Network I/O
                net_io = psutil.net_io_counters()
                net_sent = (net_io.bytes_sent - net_io_start.bytes_sent) / (1024 * 1024)  # MB
                net_recv = (net_io.bytes_recv - net_io_start.bytes_recv) / (1024 * 1024)  # MB
                net_io_start = net_io
                
                # Process info
                process = psutil.Process()
                thread_count = process.num_threads()
                process_count = len(psutil.pids())
                
                # Store metrics
                self.metrics['timestamp'].append(timestamp)
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_percent'].append(memory_percent)
                self.metrics['memory_mb'].append(memory_mb)
                self.metrics['disk_io_read'].append(disk_read)
                self.metrics['disk_io_write'].append(disk_write)
                self.metrics['network_sent'].append(net_sent)
                self.metrics['network_recv'].append(net_recv)
                self.metrics['thread_count'].append(thread_count)
                self.metrics['process_count'].append(process_count)
                
                # Check thresholds
                self._check_thresholds(cpu_percent, memory_percent)
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _check_thresholds(self, cpu_percent: float, memory_percent: float):
        """Check if metrics exceed thresholds and generate alerts."""
        if cpu_percent > self.thresholds['cpu_percent']:
            alert = {
                'timestamp': datetime.now(),
                'type': 'CPU',
                'message': f'High CPU usage: {cpu_percent:.1f}%',
                'severity': 'warning' if cpu_percent < 90 else 'critical'
            }
            self.alerts.append(alert)
            self.logger.warning(alert['message'])
        
        if memory_percent > self.thresholds['memory_percent']:
            alert = {
                'timestamp': datetime.now(),
                'type': 'Memory',
                'message': f'High memory usage: {memory_percent:.1f}%',
                'severity': 'warning' if memory_percent < 95 else 'critical'
            }
            self.alerts.append(alert)
            self.logger.warning(alert['message'])
    
    def record_component_metric(self, component: str, execution_time: float):
        """Record execution time for a specific component."""
        if component in self.component_metrics:
            self.component_metrics[component].append({
                'timestamp': datetime.now(),
                'time': execution_time
            })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.metrics['timestamp']:
            return {}
        
        latest_idx = -1
        return {
            'timestamp': self.metrics['timestamp'][latest_idx],
            'cpu_percent': self.metrics['cpu_percent'][latest_idx],
            'memory_percent': self.metrics['memory_percent'][latest_idx],
            'memory_mb': self.metrics['memory_mb'][latest_idx],
            'disk_io_read': self.metrics['disk_io_read'][latest_idx],
            'disk_io_write': self.metrics['disk_io_write'][latest_idx],
            'network_sent': self.metrics['network_sent'][latest_idx],
            'network_recv': self.metrics['network_recv'][latest_idx],
            'thread_count': self.metrics['thread_count'][latest_idx],
            'process_count': self.metrics['process_count'][latest_idx]
        }
    
    def get_average_metrics(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get average metrics over the specified window."""
        if not self.metrics['timestamp']:
            return {}
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        # Find indices within window
        indices = [i for i, ts in enumerate(self.metrics['timestamp']) if ts > cutoff_time]
        
        if not indices:
            return {}
        
        return {
            'avg_cpu_percent': np.mean([self.metrics['cpu_percent'][i] for i in indices]),
            'avg_memory_percent': np.mean([self.metrics['memory_percent'][i] for i in indices]),
            'avg_memory_mb': np.mean([self.metrics['memory_mb'][i] for i in indices]),
            'avg_disk_read_mb': np.mean([self.metrics['disk_io_read'][i] for i in indices]),
            'avg_disk_write_mb': np.mean([self.metrics['disk_io_write'][i] for i in indices]),
            'avg_network_sent_mb': np.mean([self.metrics['network_sent'][i] for i in indices]),
            'avg_network_recv_mb': np.mean([self.metrics['network_recv'][i] for i in indices])
        }
    
    def get_component_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for each component."""
        performance = {}
        
        for component, metrics in self.component_metrics.items():
            if metrics:
                times = [m['time'] for m in metrics]
                performance[component] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'std_time': np.std(times),
                    'count': len(times)
                }
        
        return performance
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current = self.get_current_metrics()
        avg_1min = self.get_average_metrics(60)
        avg_5min = self.get_average_metrics(300)
        component_perf = self.get_component_performance()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current,
            'avg_metrics_1min': avg_1min,
            'avg_metrics_5min': avg_5min,
            'component_performance': component_perf,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': str(psutil.Process().exe())
            }
        }
        
        return report
    
    def plot_realtime_metrics(self, save_path: str = None):
        """Create real-time performance plots."""
        if len(self.metrics['timestamp']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Metrics', fontsize=16)
        
        # Convert timestamps to seconds from start
        timestamps = list(self.metrics['timestamp'])
        start_time = timestamps[0]
        time_seconds = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # CPU Usage
        ax1 = axes[0, 0]
        ax1.plot(time_seconds, list(self.metrics['cpu_percent']), 'b-', linewidth=2)
        ax1.axhline(y=self.thresholds['cpu_percent'], color='r', linestyle='--', alpha=0.7)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('CPU Utilization')
        ax1.grid(True, alpha=0.3)
        
        # Memory Usage
        ax2 = axes[0, 1]
        ax2.plot(time_seconds, list(self.metrics['memory_percent']), 'g-', linewidth=2)
        ax2.axhline(y=self.thresholds['memory_percent'], color='r', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Memory Utilization')
        ax2.grid(True, alpha=0.3)
        
        # Disk I/O
        ax3 = axes[1, 0]
        ax3.plot(time_seconds, list(self.metrics['disk_io_read']), 'r-', label='Read', linewidth=2)
        ax3.plot(time_seconds, list(self.metrics['disk_io_write']), 'orange', label='Write', linewidth=2)
        ax3.set_ylabel('MB/s')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Disk I/O')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Network I/O
        ax4 = axes[1, 1]
        ax4.plot(time_seconds, list(self.metrics['network_sent']), 'purple', label='Sent', linewidth=2)
        ax4.plot(time_seconds, list(self.metrics['network_recv']), 'brown', label='Received', linewidth=2)
        ax4.set_ylabel('MB/s')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_title('Network I/O')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Performance plot saved to {save_path}")
        else:
            plt.show()
    
    def save_metrics_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        report = self.generate_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to {filepath}")


def create_performance_comparison_plot(before_metrics: Dict, after_metrics: Dict, save_path: str):
    """Create comparison plot showing performance improvements."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    components = ['Data Fetching', 'Feature Engineering', 'ML Training', 'Backtesting']
    before_times = [
        before_metrics.get('data_fetching', {}).get('original_time', 0),
        before_metrics.get('feature_engineering', {}).get('original_time', 0),
        before_metrics.get('ml_training', {}).get('original_time', 0),
        before_metrics.get('backtesting', {}).get('original_time', 0)
    ]
    after_times = [
        after_metrics.get('data_fetching', {}).get('optimized_time', 0),
        after_metrics.get('feature_engineering', {}).get('optimized_time', 0),
        after_metrics.get('ml_training', {}).get('optimized_time', 0),
        after_metrics.get('backtesting', {}).get('optimized_time', 0)
    ]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before_times, width, label='Before Optimization', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, after_times, width, label='After Optimization', color='green', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom')
    
    # Add improvement percentages
    for i, (before, after) in enumerate(zip(before_times, after_times)):
        if before > 0:
            improvement = ((before - after) / before) * 100
            ax.text(i, max(before, after) * 1.1, f'{improvement:.0f}%↑', 
                   ha='center', va='bottom', color='blue', fontweight='bold')
    
    ax.set_xlabel('Components', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Performance Optimization Results', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    print("🖥️  Performance Monitor Started")
    print("=" * 50)
    print("Monitoring system performance...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(5)
            metrics = monitor.get_current_metrics()
            if metrics:
                print(f"\n📊 Current Metrics:")
                print(f"   CPU: {metrics['cpu_percent']:.1f}%")
                print(f"   Memory: {metrics['memory_percent']:.1f}%")
                print(f"   Threads: {metrics['thread_count']}")
    
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        monitor.stop_monitoring()
        
        # Generate final report
        report = monitor.generate_performance_report()
        
        # Save metrics
        monitor.save_metrics_to_file('performance_metrics.json')
        
        # Create plot
        monitor.plot_realtime_metrics('performance_plot.png')
        
        print("\n✅ Performance monitoring complete!")
        print(f"📄 Metrics saved to: performance_metrics.json")
        print(f"📊 Plot saved to: performance_plot.png")