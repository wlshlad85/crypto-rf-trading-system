"""
ULTRATHINK Performance Dashboard
Week 5 - DAY 29-30 Implementation

Real-time performance metrics visualization and analysis dashboard.
Provides insights into system performance, bottlenecks, and trends.
"""

import time
import json
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

from performance_profiler import PerformanceProfiler, PerformanceMetric


class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, profiler: PerformanceProfiler, 
                 update_interval: float = 1.0):
        self.profiler = profiler
        self.update_interval = update_interval
        self.is_running = False
        
        # Data buffers for real-time plotting
        self.time_buffer = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.latency_buffer = deque(maxlen=300)
        self.cpu_buffer = deque(maxlen=300)
        self.memory_buffer = deque(maxlen=300)
        self.throughput_buffer = deque(maxlen=300)
        
        # Operation-specific buffers
        self.operation_latencies = defaultdict(lambda: deque(maxlen=100))
        
        # Performance regression detection
        self.baseline_latencies = {}
        self.regression_threshold = 1.5  # 50% increase triggers alert
        
        # Initialize plots
        self.setup_plots()
    
    def setup_plots(self):
        """Initialize matplotlib figures"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('ULTRATHINK Performance Dashboard', fontsize=16)
        
        # Create subplots
        self.ax_latency = plt.subplot(2, 3, 1)
        self.ax_cpu = plt.subplot(2, 3, 2)
        self.ax_memory = plt.subplot(2, 3, 3)
        self.ax_throughput = plt.subplot(2, 3, 4)
        self.ax_bottlenecks = plt.subplot(2, 3, 5)
        self.ax_heatmap = plt.subplot(2, 3, 6)
        
        # Configure axes
        self.ax_latency.set_title('Trading Decision Latency')
        self.ax_latency.set_ylabel('Latency (ms)')
        self.ax_latency.set_ylim(0, 20)
        
        self.ax_cpu.set_title('CPU Usage')
        self.ax_cpu.set_ylabel('CPU %')
        self.ax_cpu.set_ylim(0, 100)
        
        self.ax_memory.set_title('Memory Usage')
        self.ax_memory.set_ylabel('Memory (MB)')
        self.ax_memory.set_ylim(0, 8192)
        
        self.ax_throughput.set_title('System Throughput')
        self.ax_throughput.set_ylabel('Operations/sec')
        
        self.ax_bottlenecks.set_title('Top Bottlenecks')
        self.ax_heatmap.set_title('Operation Latency Heatmap')
        
        plt.tight_layout()
    
    def update_plots(self, frame):
        """Update all plots with latest data"""
        current_time = time.time()
        
        # Get latest metrics
        summary = self.profiler.get_performance_summary()
        recent_metrics = list(self.profiler.recent_metrics)
        
        # Update time buffer
        self.time_buffer.append(current_time)
        
        # Update latency
        if recent_metrics:
            avg_latency = np.mean([m.duration_ms for m in recent_metrics[-10:]])
            self.latency_buffer.append(avg_latency)
        else:
            self.latency_buffer.append(0)
        
        # Update system metrics
        self.cpu_buffer.append(summary['system']['cpu_percent'])
        self.memory_buffer.append(summary['system']['memory_mb'])
        
        # Calculate throughput
        if len(self.time_buffer) > 1:
            time_diff = self.time_buffer[-1] - self.time_buffer[-2]
            ops_count = len([m for m in recent_metrics 
                           if m.start_time >= self.time_buffer[-2]])
            throughput = ops_count / time_diff if time_diff > 0 else 0
            self.throughput_buffer.append(throughput)
        else:
            self.throughput_buffer.append(0)
        
        # Clear axes
        for ax in [self.ax_latency, self.ax_cpu, self.ax_memory, 
                  self.ax_throughput]:
            ax.clear()
        
        # Convert time to relative seconds
        if self.time_buffer:
            time_array = np.array(list(self.time_buffer))
            time_relative = time_array - time_array[0]
        else:
            time_relative = []
        
        # Plot latency with threshold line
        self.ax_latency.plot(time_relative, list(self.latency_buffer), 
                            'g-', linewidth=2, label='Avg Latency')
        self.ax_latency.axhline(y=10, color='r', linestyle='--', 
                               label='10ms Threshold')
        self.ax_latency.axhline(y=5, color='y', linestyle='--', 
                               label='5ms Target')
        self.ax_latency.set_title('Trading Decision Latency')
        self.ax_latency.set_ylabel('Latency (ms)')
        self.ax_latency.set_ylim(0, 20)
        self.ax_latency.legend(loc='upper right')
        self.ax_latency.grid(True, alpha=0.3)
        
        # Plot CPU usage
        self.ax_cpu.plot(time_relative, list(self.cpu_buffer), 
                        'b-', linewidth=2)
        self.ax_cpu.axhline(y=80, color='r', linestyle='--', 
                           label='80% Threshold')
        self.ax_cpu.set_title('CPU Usage')
        self.ax_cpu.set_ylabel('CPU %')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.legend(loc='upper right')
        self.ax_cpu.grid(True, alpha=0.3)
        
        # Plot memory usage
        self.ax_memory.plot(time_relative, list(self.memory_buffer), 
                           'c-', linewidth=2)
        self.ax_memory.axhline(y=4096, color='r', linestyle='--', 
                              label='4GB Threshold')
        self.ax_memory.axhline(y=3072, color='y', linestyle='--', 
                              label='3GB Target')
        self.ax_memory.set_title('Memory Usage')
        self.ax_memory.set_ylabel('Memory (MB)')
        self.ax_memory.set_ylim(0, 8192)
        self.ax_memory.legend(loc='upper right')
        self.ax_memory.grid(True, alpha=0.3)
        
        # Plot throughput
        self.ax_throughput.plot(time_relative, list(self.throughput_buffer), 
                               'm-', linewidth=2)
        self.ax_throughput.set_title('System Throughput')
        self.ax_throughput.set_ylabel('Operations/sec')
        self.ax_throughput.grid(True, alpha=0.3)
        
        # Update bottlenecks bar chart
        self._update_bottlenecks_chart()
        
        # Update operation heatmap
        self._update_operation_heatmap()
        
        # Check for performance regressions
        self._check_regressions()
        
        return self.fig.axes
    
    def _update_bottlenecks_chart(self):
        """Update bottlenecks bar chart"""
        self.ax_bottlenecks.clear()
        
        bottlenecks = self.profiler.get_bottlenecks(7)
        if bottlenecks:
            operations = [b.operation[-30:] for b in bottlenecks]  # Truncate names
            latencies = [b.avg_duration_ms for b in bottlenecks]
            
            colors = ['red' if l > 10 else 'yellow' if l > 5 else 'green' 
                     for l in latencies]
            
            bars = self.ax_bottlenecks.barh(operations, latencies, color=colors)
            self.ax_bottlenecks.set_xlabel('Average Latency (ms)')
            self.ax_bottlenecks.set_title('Top Bottlenecks')
            
            # Add value labels
            for i, (bar, latency) in enumerate(zip(bars, latencies)):
                self.ax_bottlenecks.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                                        f'{latency:.1f}ms', va='center')
    
    def _update_operation_heatmap(self):
        """Update operation latency heatmap"""
        self.ax_heatmap.clear()
        
        # Collect recent metrics by operation
        operation_data = defaultdict(list)
        for metric in list(self.profiler.recent_metrics)[-100:]:
            operation_data[metric.operation].append(metric.duration_ms)
        
        if operation_data:
            # Create heatmap data
            operations = list(operation_data.keys())[:10]  # Top 10 operations
            time_bins = 10
            heatmap_data = []
            
            for op in operations:
                latencies = operation_data[op]
                if len(latencies) >= time_bins:
                    # Bin the latencies
                    bins = np.array_split(latencies, time_bins)
                    bin_avgs = [np.mean(b) for b in bins]
                else:
                    # Pad with zeros
                    bin_avgs = latencies + [0] * (time_bins - len(latencies))
                heatmap_data.append(bin_avgs)
            
            # Plot heatmap
            if heatmap_data:
                sns.heatmap(heatmap_data, 
                           xticklabels=[f't-{i}' for i in range(time_bins-1, -1, -1)],
                           yticklabels=[op[-20:] for op in operations],
                           cmap='YlOrRd', 
                           ax=self.ax_heatmap,
                           cbar_kws={'label': 'Latency (ms)'})
                self.ax_heatmap.set_title('Operation Latency Heatmap')
                self.ax_heatmap.set_xlabel('Time Bins')
    
    def _check_regressions(self):
        """Check for performance regressions"""
        current_profiles = self.profiler.profiles
        
        for operation, profile in current_profiles.items():
            if profile.count < 10:  # Need sufficient samples
                continue
            
            # Initialize baseline if not exists
            if operation not in self.baseline_latencies:
                self.baseline_latencies[operation] = profile.avg_duration_ms
                continue
            
            # Check for regression
            baseline = self.baseline_latencies[operation]
            current = profile.avg_duration_ms
            
            if current > baseline * self.regression_threshold:
                self._alert_regression(operation, baseline, current)
    
    def _alert_regression(self, operation: str, baseline: float, current: float):
        """Alert on performance regression"""
        increase_pct = ((current - baseline) / baseline) * 100
        print(f"\n[PERFORMANCE REGRESSION] {datetime.now()}")
        print(f"  Operation: {operation}")
        print(f"  Baseline: {baseline:.2f}ms")
        print(f"  Current: {current:.2f}ms")
        print(f"  Increase: {increase_pct:.1f}%")
    
    def start(self):
        """Start the dashboard"""
        self.is_running = True
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_plots,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            blit=False
        )
        
        plt.show()
    
    def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
    
    def export_snapshot(self, output_path: str = "optimization/performance_snapshot.png"):
        """Export current dashboard snapshot"""
        Path(output_path).parent.mkdir(exist_ok=True)
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard snapshot saved to {output_path}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance analysis report"""
        summary = self.profiler.get_performance_summary()
        
        # Calculate trends
        latency_trend = self._calculate_trend(self.latency_buffer)
        cpu_trend = self._calculate_trend(self.cpu_buffer)
        memory_trend = self._calculate_trend(self.memory_buffer)
        
        # Identify critical issues
        issues = []
        if self.latency_buffer and max(self.latency_buffer) > 10:
            issues.append("Latency exceeded 10ms threshold")
        if self.cpu_buffer and max(self.cpu_buffer) > 80:
            issues.append("CPU usage exceeded 80% threshold")
        if self.memory_buffer and max(self.memory_buffer) > 4096:
            issues.append("Memory usage exceeded 4GB threshold")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "trends": {
                "latency": {
                    "current": self.latency_buffer[-1] if self.latency_buffer else 0,
                    "average": np.mean(list(self.latency_buffer)) if self.latency_buffer else 0,
                    "max": max(self.latency_buffer) if self.latency_buffer else 0,
                    "trend": latency_trend
                },
                "cpu": {
                    "current": self.cpu_buffer[-1] if self.cpu_buffer else 0,
                    "average": np.mean(list(self.cpu_buffer)) if self.cpu_buffer else 0,
                    "max": max(self.cpu_buffer) if self.cpu_buffer else 0,
                    "trend": cpu_trend
                },
                "memory": {
                    "current": self.memory_buffer[-1] if self.memory_buffer else 0,
                    "average": np.mean(list(self.memory_buffer)) if self.memory_buffer else 0,
                    "max": max(self.memory_buffer) if self.memory_buffer else 0,
                    "trend": memory_trend
                }
            },
            "issues": issues,
            "regressions": [
                {
                    "operation": op,
                    "baseline_ms": baseline,
                    "current_ms": self.profiler.profiles[op].avg_duration_ms
                }
                for op, baseline in self.baseline_latencies.items()
                if op in self.profiler.profiles and 
                self.profiler.profiles[op].avg_duration_ms > baseline * self.regression_threshold
            ]
        }
        
        return report
    
    def _calculate_trend(self, buffer: deque) -> str:
        """Calculate trend direction"""
        if len(buffer) < 10:
            return "insufficient_data"
        
        values = list(buffer)
        recent = np.mean(values[-10:])
        older = np.mean(values[-20:-10])
        
        if recent > older * 1.1:
            return "increasing"
        elif recent < older * 0.9:
            return "decreasing"
        else:
            return "stable"


class PerformanceMonitor:
    """Automated performance monitoring system"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.dashboard = PerformanceDashboard(profiler)
        self.monitoring_task = None
    
    async def start_monitoring(self, report_interval: float = 60.0):
        """Start automated monitoring"""
        self.profiler.start_continuous_profiling()
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(report_interval)
        )
    
    async def _monitoring_loop(self, interval: float):
        """Monitoring loop"""
        while True:
            await asyncio.sleep(interval)
            
            # Generate and log report
            report = self.dashboard.generate_report()
            
            # Log critical issues
            if report['issues']:
                print(f"\n[PERFORMANCE MONITOR] {datetime.now()}")
                print("Critical Issues Detected:")
                for issue in report['issues']:
                    print(f"  - {issue}")
            
            # Save report
            report_path = f"optimization/monitor_report_{int(time.time())}.json"
            Path(report_path).parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Export dashboard snapshot
            self.dashboard.export_snapshot(
                f"optimization/dashboard_{int(time.time())}.png"
            )
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        self.profiler.stop_continuous_profiling()


if __name__ == "__main__":
    # Demo the dashboard
    from performance_profiler import profiler
    
    # Create dashboard
    dashboard = PerformanceDashboard(profiler)
    
    # Simulate some operations for testing
    @profiler.profile("test_trading_decision")
    def simulate_trading_decision():
        """Simulate trading decision"""
        time.sleep(np.random.uniform(0.003, 0.015))  # 3-15ms
    
    @profiler.profile("test_data_processing")
    def simulate_data_processing():
        """Simulate data processing"""
        time.sleep(np.random.uniform(0.001, 0.005))  # 1-5ms
    
    def simulation_thread():
        """Run simulations"""
        while True:
            simulate_trading_decision()
            simulate_data_processing()
            time.sleep(0.1)
    
    # Start simulation in background
    import threading
    sim_thread = threading.Thread(target=simulation_thread, daemon=True)
    sim_thread.start()
    
    # Start profiler
    profiler.start_continuous_profiling()
    
    # Launch dashboard
    print("Starting Performance Dashboard...")
    print("Simulating trading operations...")
    
    try:
        dashboard.start()
    except KeyboardInterrupt:
        dashboard.stop()
        profiler.stop_continuous_profiling()
        
        # Generate final report
        report = dashboard.generate_report()
        print("\nFinal Performance Report:")
        print(json.dumps(report, indent=2))