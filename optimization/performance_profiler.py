"""
ULTRATHINK Advanced Performance Profiler
Week 5 - DAY 29-30 Implementation

Comprehensive performance profiling system for production optimization.
Tracks CPU, memory, async operations, and identifies bottlenecks.
"""

import time
import asyncio
import psutil
import cProfile
import pstats
import io
import functools
import tracemalloc
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import sqlite3
from pathlib import Path
import threading
import signal
import sys


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    cpu_percent: float
    memory_mb: float
    async_task: bool = False
    db_queries: int = 0
    network_calls: int = 0
    exception: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration * 1000


@dataclass
class PerformanceProfile:
    """Aggregated performance profile for an operation"""
    operation: str
    count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    p50_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    durations: List[float] = field(default_factory=list)
    
    def update(self, metric: PerformanceMetric):
        """Update profile with new metric"""
        self.count += 1
        self.total_duration += metric.duration
        self.min_duration = min(self.min_duration, metric.duration)
        self.max_duration = max(self.max_duration, metric.duration)
        self.durations.append(metric.duration)
        
        # Update running averages
        self.avg_cpu_percent = ((self.avg_cpu_percent * (self.count - 1) + 
                                metric.cpu_percent) / self.count)
        self.avg_memory_mb = ((self.avg_memory_mb * (self.count - 1) + 
                              metric.memory_mb) / self.count)
    
    def calculate_percentiles(self):
        """Calculate duration percentiles"""
        if self.durations:
            sorted_durations = sorted(self.durations)
            self.p50_duration = np.percentile(sorted_durations, 50)
            self.p95_duration = np.percentile(sorted_durations, 95)
            self.p99_duration = np.percentile(sorted_durations, 99)
    
    @property
    def avg_duration(self) -> float:
        """Average duration"""
        return self.total_duration / self.count if self.count > 0 else 0
    
    @property
    def avg_duration_ms(self) -> float:
        """Average duration in milliseconds"""
        return self.avg_duration * 1000


class PerformanceProfiler:
    """Advanced performance profiling system"""
    
    def __init__(self, db_path: str = "optimization/performance_metrics.db"):
        self.db_path = db_path
        self.process = psutil.Process()
        self.metrics: List[PerformanceMetric] = []
        self.profiles: Dict[str, PerformanceProfile] = defaultdict(
            lambda: PerformanceProfile("")
        )
        self.is_profiling = False
        self.profile_thread: Optional[threading.Thread] = None
        self.db_lock = threading.Lock()
        
        # Performance thresholds
        self.latency_threshold_ms = 10.0  # Trading decision threshold
        self.memory_threshold_mb = 4096    # 4GB threshold
        self.cpu_threshold_percent = 80.0
        
        # Circular buffer for recent metrics
        self.recent_metrics = deque(maxlen=1000)
        
        # Initialize database
        self._init_database()
        
        # Memory tracking
        tracemalloc.start()
    
    def _init_database(self):
        """Initialize performance metrics database"""
        Path(self.db_path).parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    operation TEXT,
                    duration REAL,
                    cpu_percent REAL,
                    memory_mb REAL,
                    async_task BOOLEAN,
                    db_queries INTEGER,
                    network_calls INTEGER,
                    exception TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_operation 
                ON performance_metrics(operation)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_metrics(timestamp)
            """)
    
    def profile(self, operation: str = None):
        """Decorator for profiling functions"""
        def decorator(func):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync(op_name, func, args, kwargs)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async(op_name, func, args, kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def _profile_sync(self, operation: str, func: Callable, args, kwargs):
        """Profile synchronous function"""
        start_time = time.time()
        cpu_start = self.process.cpu_percent()
        memory_start = self.process.memory_info().rss / 1024 / 1024  # MB
        
        exception = None
        result = None
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            cpu_percent = self.process.cpu_percent() - cpu_start
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            metric = PerformanceMetric(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb - memory_start,
                async_task=False,
                exception=exception
            )
            
            self._record_metric(metric)
        
        return result
    
    async def _profile_async(self, operation: str, func: Callable, args, kwargs):
        """Profile asynchronous function"""
        start_time = time.time()
        cpu_start = self.process.cpu_percent()
        memory_start = self.process.memory_info().rss / 1024 / 1024  # MB
        
        exception = None
        result = None
        
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            exception = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            cpu_percent = self.process.cpu_percent() - cpu_start
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            metric = PerformanceMetric(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb - memory_start,
                async_task=True,
                exception=exception
            )
            
            self._record_metric(metric)
        
        return result
    
    def _record_metric(self, metric: PerformanceMetric):
        """Record performance metric"""
        # Add to memory buffers
        self.metrics.append(metric)
        self.recent_metrics.append(metric)
        
        # Update profile
        profile = self.profiles[metric.operation]
        profile.operation = metric.operation
        profile.update(metric)
        
        # Check thresholds
        self._check_thresholds(metric)
        
        # Store in database
        self._store_metric(metric)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds"""
        alerts = []
        
        if metric.duration_ms > self.latency_threshold_ms:
            alerts.append(f"LATENCY: {metric.operation} took {metric.duration_ms:.2f}ms")
        
        if metric.memory_mb > self.memory_threshold_mb:
            alerts.append(f"MEMORY: {metric.operation} used {metric.memory_mb:.2f}MB")
        
        if metric.cpu_percent > self.cpu_threshold_percent:
            alerts.append(f"CPU: {metric.operation} used {metric.cpu_percent:.2f}%")
        
        if alerts:
            self._log_performance_alert(metric, alerts)
    
    def _log_performance_alert(self, metric: PerformanceMetric, alerts: List[str]):
        """Log performance alerts"""
        timestamp = datetime.fromtimestamp(metric.start_time)
        print(f"\n[PERFORMANCE ALERT] {timestamp}")
        for alert in alerts:
            print(f"  - {alert}")
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in database"""
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO performance_metrics 
                        (timestamp, operation, duration, cpu_percent, memory_mb,
                         async_task, db_queries, network_calls, exception)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.start_time,
                        metric.operation,
                        metric.duration,
                        metric.cpu_percent,
                        metric.memory_mb,
                        metric.async_task,
                        metric.db_queries,
                        metric.network_calls,
                        metric.exception
                    ))
            except Exception as e:
                print(f"Error storing metric: {e}")
    
    def start_continuous_profiling(self, interval: float = 1.0):
        """Start continuous system profiling"""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.profile_thread = threading.Thread(
            target=self._continuous_profiling_loop,
            args=(interval,),
            daemon=True
        )
        self.profile_thread.start()
    
    def stop_continuous_profiling(self):
        """Stop continuous profiling"""
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join(timeout=5)
    
    def _continuous_profiling_loop(self, interval: float):
        """Continuous profiling loop"""
        while self.is_profiling:
            try:
                # System-wide metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                metric = PerformanceMetric(
                    operation="system",
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0,
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    network_calls=int(net_io.packets_sent + net_io.packets_recv),
                    db_queries=int(disk_io.read_count + disk_io.write_count)
                )
                
                self._record_metric(metric)
                
            except Exception as e:
                print(f"Profiling error: {e}")
            
            time.sleep(interval)
    
    def get_bottlenecks(self, top_n: int = 10) -> List[PerformanceProfile]:
        """Identify performance bottlenecks"""
        # Calculate percentiles for all profiles
        for profile in self.profiles.values():
            profile.calculate_percentiles()
        
        # Sort by average duration
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.avg_duration,
            reverse=True
        )
        
        return sorted_profiles[:top_n]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        if not self.metrics:
            return {}
        
        # Recent metrics analysis
        recent_durations = [m.duration_ms for m in self.recent_metrics]
        
        # Memory snapshot
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        
        summary = {
            "total_operations": len(self.metrics),
            "unique_operations": len(self.profiles),
            "recent_metrics": {
                "count": len(self.recent_metrics),
                "avg_duration_ms": np.mean(recent_durations) if recent_durations else 0,
                "p95_duration_ms": np.percentile(recent_durations, 95) if recent_durations else 0,
                "p99_duration_ms": np.percentile(recent_durations, 99) if recent_durations else 0,
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                "traced_memory_mb": current_memory / 1024 / 1024,
                "peak_memory_mb": peak_memory / 1024 / 1024,
            },
            "bottlenecks": [
                {
                    "operation": p.operation,
                    "count": p.count,
                    "avg_duration_ms": p.avg_duration_ms,
                    "p95_duration_ms": p.p95_duration * 1000,
                    "p99_duration_ms": p.p99_duration * 1000,
                }
                for p in self.get_bottlenecks(5)
            ]
        }
        
        return summary
    
    def export_report(self, output_path: str = "optimization/performance_report.json"):
        """Export comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "profiles": {
                name: {
                    "operation": profile.operation,
                    "count": profile.count,
                    "total_duration_s": profile.total_duration,
                    "avg_duration_ms": profile.avg_duration_ms,
                    "min_duration_ms": profile.min_duration * 1000,
                    "max_duration_ms": profile.max_duration * 1000,
                    "p50_duration_ms": profile.p50_duration * 1000,
                    "p95_duration_ms": profile.p95_duration * 1000,
                    "p99_duration_ms": profile.p99_duration * 1000,
                    "avg_cpu_percent": profile.avg_cpu_percent,
                    "avg_memory_mb": profile.avg_memory_mb,
                }
                for name, profile in self.profiles.items()
                if profile.count > 0
            }
        }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def profile_cprofile(self, func: Callable, *args, **kwargs):
        """Profile function with cProfile for detailed analysis"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        
        # Save to file
        output_path = f"optimization/cprofile_{func.__name__}_{int(time.time())}.txt"
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(profile_output)
        
        return result, profile_output
    
    def __enter__(self):
        """Context manager entry"""
        self.start_continuous_profiling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_continuous_profiling()
        self.export_report()


# Global profiler instance
profiler = PerformanceProfiler()


def profile(operation: str = None):
    """Convenience decorator for profiling"""
    return profiler.profile(operation)


if __name__ == "__main__":
    # Example usage and testing
    import random
    
    # Test synchronous function
    @profile("test_calculation")
    def test_calculation(n: int) -> float:
        """Test CPU-intensive calculation"""
        result = 0
        for i in range(n):
            result += random.random() ** 2
        return result
    
    # Test async function
    @profile("test_async_operation")
    async def test_async_operation(delay: float) -> str:
        """Test async operation"""
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"
    
    async def run_tests():
        """Run performance tests"""
        print("Starting Performance Profiler Tests...")
        
        with profiler:
            # Test synchronous operations
            for i in range(10):
                test_calculation(100000)
            
            # Test async operations
            tasks = []
            for i in range(5):
                tasks.append(test_async_operation(random.uniform(0.1, 0.5)))
            
            await asyncio.gather(*tasks)
            
            # Test with cProfile
            result, profile_output = profiler.profile_cprofile(
                test_calculation, 1000000
            )
            
            # Get performance summary
            summary = profiler.get_performance_summary()
            print("\nPerformance Summary:")
            print(json.dumps(summary, indent=2))
            
            # Get bottlenecks
            bottlenecks = profiler.get_bottlenecks()
            print("\nTop Bottlenecks:")
            for b in bottlenecks[:5]:
                print(f"  - {b.operation}: {b.avg_duration_ms:.2f}ms (count: {b.count})")
    
    # Run tests
    asyncio.run(run_tests())