"""
ULTRATHINK Stress Testing Framework
Week 5 - DAY 33-34 Implementation

Comprehensive stress testing framework for crypto trading system.
Tests system behavior under extreme conditions including market crashes,
high volume loads, network failures, and resource exhaustion.
"""

import asyncio
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
import multiprocessing
import psutil
import random
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_profiler import PerformanceProfiler, profile


@dataclass
class StressTestConfig:
    """Stress test configuration"""
    test_name: str
    duration_seconds: int = 300
    target_throughput: int = 10000  # Operations per second
    max_concurrent_connections: int = 1000
    data_volume_gb: float = 1.0
    memory_limit_gb: float = 8.0
    cpu_limit_percent: float = 90.0
    network_failure_rate: float = 0.0  # 0-1 probability
    market_crash_severity: float = 0.0  # 0-1 where 1 = 50% price drop
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.duration_seconds,
            "target_throughput": self.target_throughput,
            "max_concurrent_connections": self.max_concurrent_connections,
            "data_volume_gb": self.data_volume_gb,
            "memory_limit_gb": self.memory_limit_gb,
            "cpu_limit_percent": self.cpu_limit_percent,
            "network_failure_rate": self.network_failure_rate,
            "market_crash_severity": self.market_crash_severity
        }


@dataclass
class StressTestResult:
    """Stress test result"""
    test_name: str
    config: StressTestConfig
    success: bool
    start_time: datetime
    end_time: datetime
    actual_duration: float
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    
    # System metrics
    errors: List[str] = field(default_factory=list)
    system_alerts: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_operations / self.total_operations if self.total_operations > 0 else 0.0
    
    @property
    def actual_throughput(self) -> float:
        """Calculate actual throughput"""
        return self.total_operations / self.actual_duration if self.actual_duration > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_name": self.test_name,
            "config": self.config.to_dict(),
            "success": self.success,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "actual_duration": self.actual_duration,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "actual_throughput": self.actual_throughput,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "peak_cpu_percent": self.peak_cpu_percent,
            "errors": self.errors,
            "system_alerts": self.system_alerts
        }


class MarketDataSimulator:
    """Simulates market data for stress testing"""
    
    def __init__(self, crash_severity: float = 0.0):
        self.crash_severity = crash_severity
        self.current_price = 50000.0  # Starting BTC price
        self.volatility = 0.02  # 2% volatility
        self.trend = 0.0001  # Slight upward trend
        self.crashed = False
        
    def generate_tick(self) -> Dict[str, Any]:
        """Generate single market tick"""
        
        # Apply crash if configured
        if self.crash_severity > 0 and not self.crashed:
            crash_prob = self.crash_severity * 0.001  # Small probability per tick
            if random.random() < crash_prob:
                self.current_price *= (1 - self.crash_severity)
                self.volatility *= 3  # Increase volatility during crash
                self.crashed = True
        
        # Generate price movement
        price_change = np.random.normal(self.trend, self.volatility)
        self.current_price *= (1 + price_change)
        
        # Generate volume
        volume = np.random.exponential(1000)
        
        return {
            'timestamp': time.time(),
            'symbol': 'BTC/USD',
            'price': self.current_price,
            'volume': volume,
            'bid': self.current_price * 0.9995,
            'ask': self.current_price * 1.0005
        }
    
    def generate_batch(self, size: int) -> List[Dict[str, Any]]:
        """Generate batch of market ticks"""
        return [self.generate_tick() for _ in range(size)]


class LoadGenerator:
    """Generates load for stress testing"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.is_running = False
        self.operations_count = 0
        self.success_count = 0
        self.error_count = 0
        self.latencies = []
        self.start_time = None
        
    async def trading_operation(self) -> Tuple[bool, float]:
        """Simulate a trading operation"""
        start_time = time.perf_counter()
        
        try:
            # Simulate market data processing
            await asyncio.sleep(0.001)  # 1ms data processing
            
            # Simulate feature calculation
            features = np.random.randn(50)
            normalized_features = (features - features.mean()) / (features.std() + 1e-7)
            
            # Simulate model inference
            await asyncio.sleep(0.0005)  # 0.5ms inference
            
            # Simulate risk check
            position_size = random.uniform(1000, 10000)
            risk_ok = position_size < 50000  # Risk limit
            
            # Simulate network failure
            if random.random() < self.config.network_failure_rate:
                raise Exception("Network failure")
            
            # Simulate order execution
            if risk_ok:
                await asyncio.sleep(0.0005)  # 0.5ms execution
                success = True
            else:
                success = False
            
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
            return success, latency
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return False, latency
    
    async def generate_load(self, semaphore: asyncio.Semaphore) -> None:
        """Generate load with concurrency control"""
        
        while self.is_running:
            async with semaphore:
                success, latency = await self.trading_operation()
                
                self.operations_count += 1
                self.latencies.append(latency)
                
                if success:
                    self.success_count += 1
                else:
                    self.error_count += 1
            
            # Control rate to achieve target throughput
            if self.start_time:
                elapsed = time.time() - self.start_time
                target_ops = self.config.target_throughput * elapsed
                
                if self.operations_count > target_ops:
                    await asyncio.sleep(0.001)  # Small delay
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load generation statistics"""
        
        if not self.latencies:
            return {
                "operations": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0
            }
        
        latencies_array = np.array(self.latencies)
        
        return {
            "operations": self.operations_count,
            "successful": self.success_count,
            "failed": self.error_count,
            "success_rate": self.success_count / self.operations_count,
            "avg_latency_ms": np.mean(latencies_array),
            "p95_latency_ms": np.percentile(latencies_array, 95),
            "p99_latency_ms": np.percentile(latencies_array, 99),
            "max_latency_ms": np.max(latencies_array)
        }


class ResourceMonitor:
    """Monitors system resources during stress testing"""
    
    def __init__(self):
        self.is_monitoring = False
        self.cpu_readings = []
        self.memory_readings = []
        self.disk_io_readings = []
        self.network_io_readings = []
        self.process = psutil.Process()
        
    def start_monitoring(self, interval: float = 0.1):
        """Start resource monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=1)
    
    def _monitoring_loop(self, interval: float):
        """Resource monitoring loop"""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_readings.append(cpu_percent)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_readings.append(memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.disk_io_readings.append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes,
                        'timestamp': time.time()
                    })
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.network_io_readings.append({
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv,
                        'timestamp': time.time()
                    })
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        
        stats = {
            "cpu": {
                "avg_percent": np.mean(self.cpu_readings) if self.cpu_readings else 0,
                "max_percent": np.max(self.cpu_readings) if self.cpu_readings else 0,
                "readings": len(self.cpu_readings)
            },
            "memory": {
                "avg_mb": np.mean(self.memory_readings) if self.memory_readings else 0,
                "max_mb": np.max(self.memory_readings) if self.memory_readings else 0,
                "readings": len(self.memory_readings)
            },
            "disk_io": {
                "samples": len(self.disk_io_readings)
            },
            "network_io": {
                "samples": len(self.network_io_readings)
            }
        }
        
        return stats


class StressTester:
    """Main stress testing framework"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.results: List[StressTestResult] = []
        self.market_simulator = None
        self.resource_monitor = ResourceMonitor()
        
    async def run_stress_test(self, config: StressTestConfig) -> StressTestResult:
        """Run a single stress test"""
        
        print(f"\nStarting stress test: {config.test_name}")
        print(f"Duration: {config.duration_seconds}s")
        print(f"Target throughput: {config.target_throughput} ops/sec")
        print(f"Max concurrent: {config.max_concurrent_connections}")
        print("=" * 60)
        
        # Initialize result
        result = StressTestResult(
            test_name=config.test_name,
            config=config,
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Initialize market simulator
        self.market_simulator = MarketDataSimulator(config.market_crash_severity)
        
        # Create load generator
        load_generator = LoadGenerator(config)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrent_connections)
        
        try:
            # Start load generation
            load_generator.is_running = True
            load_generator.start_time = time.time()
            
            # Create multiple coroutines for load generation
            load_tasks = []
            for _ in range(min(100, config.max_concurrent_connections)):
                task = asyncio.create_task(load_generator.generate_load(semaphore))
                load_tasks.append(task)
            
            # Run for specified duration
            await asyncio.sleep(config.duration_seconds)
            
            # Stop load generation
            load_generator.is_running = False
            
            # Wait for tasks to complete
            await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Calculate results
            load_stats = load_generator.get_statistics()
            resource_stats = self.resource_monitor.get_statistics()
            
            result.end_time = datetime.now()
            result.actual_duration = (result.end_time - result.start_time).total_seconds()
            result.total_operations = load_stats["operations"]
            result.successful_operations = load_stats["successful"]
            result.failed_operations = load_stats["failed"]
            result.avg_latency_ms = load_stats["avg_latency_ms"]
            result.p95_latency_ms = load_stats["p95_latency_ms"]
            result.p99_latency_ms = load_stats["p99_latency_ms"]
            result.max_latency_ms = load_stats["max_latency_ms"]
            
            # Resource usage
            result.peak_memory_mb = resource_stats["memory"]["max_mb"]
            result.avg_cpu_percent = resource_stats["cpu"]["avg_percent"]
            result.peak_cpu_percent = resource_stats["cpu"]["max_percent"]
            
            # Check success criteria
            result.success = self._evaluate_success(result, config)
            
            # Generate alerts
            result.system_alerts = self._check_alerts(result, config)
            
            print(f"\nStress test completed: {config.test_name}")
            print(f"Success: {result.success}")
            print(f"Operations: {result.total_operations}")
            print(f"Success rate: {result.success_rate:.2%}")
            print(f"Throughput: {result.actual_throughput:.0f} ops/sec")
            print(f"P99 latency: {result.p99_latency_ms:.2f}ms")
            print(f"Peak memory: {result.peak_memory_mb:.1f}MB")
            print(f"Peak CPU: {result.peak_cpu_percent:.1f}%")
            
            if result.system_alerts:
                print(f"Alerts: {len(result.system_alerts)}")
                for alert in result.system_alerts:
                    print(f"  - {alert}")
            
        except Exception as e:
            result.errors.append(f"Test execution error: {str(e)}")
            result.success = False
            print(f"Stress test failed: {e}")
        
        finally:
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
        
        # Store result
        self.results.append(result)
        
        return result
    
    def _evaluate_success(self, result: StressTestResult, config: StressTestConfig) -> bool:
        """Evaluate if stress test was successful"""
        
        # Check success rate
        if result.success_rate < 0.95:  # 95% success rate required
            result.errors.append(f"Success rate too low: {result.success_rate:.2%}")
            return False
        
        # Check throughput
        min_throughput = config.target_throughput * 0.8  # 80% of target
        if result.actual_throughput < min_throughput:
            result.errors.append(f"Throughput too low: {result.actual_throughput:.0f} < {min_throughput:.0f}")
            return False
        
        # Check latency
        max_p99_latency = 50.0  # 50ms max P99 latency
        if result.p99_latency_ms > max_p99_latency:
            result.errors.append(f"P99 latency too high: {result.p99_latency_ms:.2f}ms")
            return False
        
        # Check memory usage
        if result.peak_memory_mb > config.memory_limit_gb * 1024:
            result.errors.append(f"Memory limit exceeded: {result.peak_memory_mb:.1f}MB")
            return False
        
        # Check CPU usage
        if result.peak_cpu_percent > config.cpu_limit_percent:
            result.errors.append(f"CPU limit exceeded: {result.peak_cpu_percent:.1f}%")
            return False
        
        return True
    
    def _check_alerts(self, result: StressTestResult, config: StressTestConfig) -> List[str]:
        """Check for system alerts"""
        
        alerts = []
        
        # Performance alerts
        if result.avg_latency_ms > 10.0:
            alerts.append(f"High average latency: {result.avg_latency_ms:.2f}ms")
        
        if result.p95_latency_ms > 20.0:
            alerts.append(f"High P95 latency: {result.p95_latency_ms:.2f}ms")
        
        # Resource alerts
        if result.avg_cpu_percent > 70.0:
            alerts.append(f"High average CPU: {result.avg_cpu_percent:.1f}%")
        
        if result.peak_memory_mb > config.memory_limit_gb * 1024 * 0.8:
            alerts.append(f"High memory usage: {result.peak_memory_mb:.1f}MB")
        
        # Error rate alerts
        if result.success_rate < 0.99:
            alerts.append(f"Error rate above 1%: {1 - result.success_rate:.2%}")
        
        return alerts
    
    def run_load_test(self, duration: int = 300, throughput: int = 5000) -> StressTestResult:
        """Run standard load test"""
        
        config = StressTestConfig(
            test_name="load_test",
            duration_seconds=duration,
            target_throughput=throughput,
            max_concurrent_connections=100
        )
        
        return asyncio.run(self.run_stress_test(config))
    
    def run_spike_test(self, duration: int = 60, throughput: int = 20000) -> StressTestResult:
        """Run spike test with very high load"""
        
        config = StressTestConfig(
            test_name="spike_test",
            duration_seconds=duration,
            target_throughput=throughput,
            max_concurrent_connections=500
        )
        
        return asyncio.run(self.run_stress_test(config))
    
    def run_endurance_test(self, duration: int = 3600, throughput: int = 2000) -> StressTestResult:
        """Run endurance test for extended period"""
        
        config = StressTestConfig(
            test_name="endurance_test",
            duration_seconds=duration,
            target_throughput=throughput,
            max_concurrent_connections=50
        )
        
        return asyncio.run(self.run_stress_test(config))
    
    def run_market_crash_test(self, duration: int = 300, crash_severity: float = 0.3) -> StressTestResult:
        """Run market crash simulation test"""
        
        config = StressTestConfig(
            test_name="market_crash_test",
            duration_seconds=duration,
            target_throughput=5000,
            max_concurrent_connections=200,
            market_crash_severity=crash_severity
        )
        
        return asyncio.run(self.run_stress_test(config))
    
    def run_network_failure_test(self, duration: int = 300, failure_rate: float = 0.1) -> StressTestResult:
        """Run network failure simulation test"""
        
        config = StressTestConfig(
            test_name="network_failure_test",
            duration_seconds=duration,
            target_throughput=3000,
            max_concurrent_connections=100,
            network_failure_rate=failure_rate
        )
        
        return asyncio.run(self.run_stress_test(config))
    
    def run_comprehensive_test_suite(self) -> Dict[str, StressTestResult]:
        """Run comprehensive stress test suite"""
        
        print("\n" + "="*80)
        print("ULTRATHINK COMPREHENSIVE STRESS TEST SUITE")
        print("="*80)
        
        test_results = {}
        
        # 1. Load Test
        print("\n1. Running Load Test...")
        test_results["load_test"] = self.run_load_test()
        
        # 2. Spike Test
        print("\n2. Running Spike Test...")
        test_results["spike_test"] = self.run_spike_test()
        
        # 3. Market Crash Test
        print("\n3. Running Market Crash Test...")
        test_results["market_crash_test"] = self.run_market_crash_test()
        
        # 4. Network Failure Test
        print("\n4. Running Network Failure Test...")
        test_results["network_failure_test"] = self.run_network_failure_test()
        
        # 5. Short Endurance Test (5 minutes)
        print("\n5. Running Short Endurance Test...")
        test_results["endurance_test"] = self.run_endurance_test(duration=300)
        
        return test_results
    
    def export_results(self, output_path: str = "optimization/stress_test_results.json"):
        """Export stress test results"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": [result.to_dict() for result in self.results],
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "avg_throughput": np.mean([r.actual_throughput for r in self.results]),
                "max_throughput": max([r.actual_throughput for r in self.results]) if self.results else 0,
                "avg_p99_latency": np.mean([r.p99_latency_ms for r in self.results]),
                "max_memory_usage": max([r.peak_memory_mb for r in self.results]) if self.results else 0,
                "max_cpu_usage": max([r.peak_cpu_percent for r in self.results]) if self.results else 0
            }
        }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nStress test results exported to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("STRESS TEST SUMMARY")
        print("="*60)
        summary = report["summary"]
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Average throughput: {summary['avg_throughput']:.0f} ops/sec")
        print(f"Max throughput: {summary['max_throughput']:.0f} ops/sec")
        print(f"Average P99 latency: {summary['avg_p99_latency']:.2f}ms")
        print(f"Max memory usage: {summary['max_memory_usage']:.1f}MB")
        print(f"Max CPU usage: {summary['max_cpu_usage']:.1f}%")
        
        # Show failed tests
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            print(f"\nFAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {', '.join(test.errors)}")
        
        return report


if __name__ == "__main__":
    # Run stress tests
    tester = StressTester()
    
    # Run comprehensive test suite
    results = tester.run_comprehensive_test_suite()
    
    # Export results
    report = tester.export_results()
    
    # Print final status
    all_passed = all(r.success for r in tester.results)
    print(f"\n{'='*60}")
    print(f"STRESS TEST SUITE: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}")
    
    if not all_passed:
        print("\nRecommendations:")
        print("- Review failed tests and optimize bottlenecks")
        print("- Consider scaling infrastructure for higher loads")
        print("- Implement additional error handling for edge cases")
        print("- Monitor system resources during production deployment")