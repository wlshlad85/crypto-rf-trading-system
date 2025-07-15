"""
ULTRATHINK Benchmark Suite
Week 5 - DAY 29-30 Implementation

Comprehensive benchmark suite for measuring system performance across
trading decisions, data processing, model inference, and end-to-end operations.
"""

import time
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import concurrent.futures
import psutil
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_profiler import PerformanceProfiler, profile


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    category: str
    iterations: int
    total_time: float
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    std_time: float
    p95_time: float
    p99_time: float
    throughput: float  # Operations per second
    passed: bool
    target_latency: Optional[float] = None
    
    @property
    def mean_time_ms(self) -> float:
        """Mean time in milliseconds"""
        return self.mean_time * 1000
    
    @property
    def p99_time_ms(self) -> float:
        """P99 time in milliseconds"""
        return self.p99_time * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "category": self.category,
            "iterations": self.iterations,
            "mean_time_ms": self.mean_time_ms,
            "p95_time_ms": self.p95_time * 1000,
            "p99_time_ms": self.p99_time_ms,
            "throughput_ops_sec": self.throughput,
            "passed": self.passed,
            "target_latency_ms": self.target_latency * 1000 if self.target_latency else None
        }


class BenchmarkSuite:
    """Comprehensive system benchmark suite"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.results: List[BenchmarkResult] = []
        
        # Performance targets
        self.targets = {
            "trading_decision": 0.005,  # 5ms target
            "data_processing": 0.001,   # 1ms for tick processing
            "model_inference": 0.001,   # 1ms for predictions
            "risk_check": 0.0005,       # 0.5ms for risk validation
            "order_execution": 0.003,   # 3ms for order placement
            "feature_calculation": 0.002, # 2ms for features
        }
        
        # Test data
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Setup test data for benchmarks"""
        # Create synthetic market data
        self.test_market_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='100ms'),
            'open': np.random.randn(10000).cumsum() + 100,
            'high': np.random.randn(10000).cumsum() + 101,
            'low': np.random.randn(10000).cumsum() + 99,
            'close': np.random.randn(10000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 10000)
        })
        
        # Create test features
        self.test_features = np.random.randn(100, 50)  # 100 samples, 50 features
        
        # Create test portfolio state
        self.test_portfolio = {
            'balance': 100000,
            'positions': {'BTC': 0.5, 'ETH': 10},
            'pending_orders': [],
            'risk_metrics': {'var': 0.02, 'sharpe': 1.5}
        }
    
    def run_benchmark(self, name: str, category: str, func: Callable,
                     iterations: int = 1000, warmup: int = 100,
                     target_latency: Optional[float] = None) -> BenchmarkResult:
        """Run a single benchmark"""
        print(f"Running benchmark: {name}")
        
        # Warmup runs
        for _ in range(warmup):
            func()
        
        # Timed runs
        times = []
        start_total = time.perf_counter()
        
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        # Calculate statistics
        times_array = np.array(times)
        
        # Use target from category if not specified
        if target_latency is None:
            target_latency = self.targets.get(category, 0.010)  # 10ms default
        
        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=iterations,
            total_time=total_time,
            min_time=np.min(times_array),
            max_time=np.max(times_array),
            mean_time=np.mean(times_array),
            median_time=np.median(times_array),
            std_time=np.std(times_array),
            p95_time=np.percentile(times_array, 95),
            p99_time=np.percentile(times_array, 99),
            throughput=iterations / total_time,
            passed=np.percentile(times_array, 99) < target_latency,
            target_latency=target_latency
        )
        
        self.results.append(result)
        return result
    
    async def run_async_benchmark(self, name: str, category: str, 
                                 async_func: Callable, iterations: int = 1000,
                                 concurrency: int = 10) -> BenchmarkResult:
        """Run asynchronous benchmark with concurrency"""
        print(f"Running async benchmark: {name} (concurrency: {concurrency})")
        
        async def run_batch():
            """Run a batch of async operations"""
            tasks = [async_func() for _ in range(concurrency)]
            return await asyncio.gather(*tasks)
        
        # Warmup
        for _ in range(10):
            await run_batch()
        
        # Timed runs
        times = []
        start_total = time.perf_counter()
        
        batches = iterations // concurrency
        for _ in range(batches):
            start = time.perf_counter()
            await run_batch()
            end = time.perf_counter()
            # Time per operation
            times.extend([(end - start) / concurrency] * concurrency)
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        # Calculate statistics
        times_array = np.array(times[:iterations])  # Ensure exact count
        
        target_latency = self.targets.get(category, 0.010)
        
        result = BenchmarkResult(
            name=f"{name}_async",
            category=category,
            iterations=iterations,
            total_time=total_time,
            min_time=np.min(times_array),
            max_time=np.max(times_array),
            mean_time=np.mean(times_array),
            median_time=np.median(times_array),
            std_time=np.std(times_array),
            p95_time=np.percentile(times_array, 95),
            p99_time=np.percentile(times_array, 99),
            throughput=iterations / total_time,
            passed=np.percentile(times_array, 99) < target_latency,
            target_latency=target_latency
        )
        
        self.results.append(result)
        return result
    
    def benchmark_trading_decision(self):
        """Benchmark trading decision latency"""
        def make_decision():
            # Simulate decision logic
            features = self.test_features[np.random.randint(0, 100)]
            portfolio = self.test_portfolio.copy()
            
            # Feature normalization
            normalized = (features - features.mean()) / (features.std() + 1e-7)
            
            # Risk check
            position_size = portfolio['balance'] * 0.02  # 2% risk
            
            # Simple decision logic
            signal = np.sum(normalized[:10]) > 0
            action = 'buy' if signal else 'hold'
            
            return action, position_size
        
        return self.run_benchmark(
            name="trading_decision",
            category="trading_decision",
            func=make_decision,
            iterations=10000
        )
    
    def benchmark_data_processing(self):
        """Benchmark data processing speed"""
        def process_tick():
            # Get random slice of data
            idx = np.random.randint(100, len(self.test_market_data) - 100)
            window = self.test_market_data.iloc[idx-100:idx]
            
            # Calculate indicators
            sma_20 = window['close'].rolling(20).mean().iloc[-1]
            rsi = self._calculate_rsi(window['close'], 14)
            
            # Volume analysis
            vwap = (window['close'] * window['volume']).sum() / window['volume'].sum()
            
            return {'sma_20': sma_20, 'rsi': rsi, 'vwap': vwap}
        
        return self.run_benchmark(
            name="data_tick_processing",
            category="data_processing",
            func=process_tick,
            iterations=10000
        )
    
    def benchmark_model_inference(self):
        """Benchmark ML model inference"""
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.weights = np.random.randn(50, 32)
                self.bias = np.random.randn(32)
                self.weights2 = np.random.randn(32, 3)
                self.bias2 = np.random.randn(3)
            
            def predict(self, features):
                # Simple 2-layer network
                hidden = np.tanh(np.dot(features, self.weights) + self.bias)
                output = np.dot(hidden, self.weights2) + self.bias2
                return np.argmax(output)
        
        model = MockModel()
        
        def inference():
            features = self.test_features[np.random.randint(0, 100)]
            return model.predict(features)
        
        return self.run_benchmark(
            name="model_inference",
            category="model_inference",
            func=inference,
            iterations=10000
        )
    
    def benchmark_risk_validation(self):
        """Benchmark risk management checks"""
        def validate_risk():
            portfolio = self.test_portfolio.copy()
            proposed_position = np.random.uniform(1000, 10000)
            
            # Position sizing check
            max_position = portfolio['balance'] * 0.5
            if proposed_position > max_position:
                proposed_position = max_position
            
            # Portfolio VaR check
            current_var = portfolio['risk_metrics']['var']
            if current_var > 0.05:  # 5% VaR limit
                return False, "VaR limit exceeded"
            
            # Drawdown check
            drawdown = np.random.uniform(0, 0.2)
            if drawdown > 0.15:  # 15% max drawdown
                return False, "Drawdown limit exceeded"
            
            return True, proposed_position
        
        return self.run_benchmark(
            name="risk_validation",
            category="risk_check",
            func=validate_risk,
            iterations=10000
        )
    
    def benchmark_feature_calculation(self):
        """Benchmark feature engineering"""
        def calculate_features():
            # Get data window
            idx = np.random.randint(200, len(self.test_market_data) - 10)
            window = self.test_market_data.iloc[idx-200:idx]
            
            features = []
            
            # Price features
            returns = window['close'].pct_change()
            features.extend([
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurt()
            ])
            
            # Technical indicators
            sma_20 = window['close'].rolling(20).mean().iloc[-1]
            sma_50 = window['close'].rolling(50).mean().iloc[-1]
            features.append((sma_20 - sma_50) / sma_50)
            
            # Volume features
            volume_ratio = window['volume'].iloc[-20:].mean() / window['volume'].mean()
            features.append(volume_ratio)
            
            return np.array(features)
        
        return self.run_benchmark(
            name="feature_calculation",
            category="feature_calculation",
            func=calculate_features,
            iterations=5000
        )
    
    def benchmark_database_operations(self):
        """Benchmark database operations"""
        import sqlite3
        
        # Setup test database
        db_path = "optimization/benchmark_test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                symbol TEXT,
                side TEXT,
                price REAL,
                quantity REAL
            )
        """)
        conn.close()
        
        def db_operation():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert
            cursor.execute("""
                INSERT INTO trades (timestamp, symbol, side, price, quantity)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), 'BTC', 'buy', 50000.0, 0.1))
            
            # Query
            cursor.execute("""
                SELECT * FROM trades 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (time.time() - 3600,))
            
            results = cursor.fetchall()
            conn.commit()
            conn.close()
            
            return results
        
        result = self.run_benchmark(
            name="database_operations",
            category="data_processing",
            func=db_operation,
            iterations=1000
        )
        
        # Cleanup
        os.remove(db_path)
        
        return result
    
    def benchmark_end_to_end(self):
        """Benchmark complete trading cycle"""
        model = self._create_mock_model()
        
        def trading_cycle():
            # 1. Process market data
            idx = np.random.randint(200, len(self.test_market_data) - 10)
            window = self.test_market_data.iloc[idx-200:idx]
            
            # 2. Calculate features
            features = self._calculate_features_fast(window)
            
            # 3. Model inference
            prediction = model.predict(features)
            
            # 4. Risk validation
            portfolio = self.test_portfolio.copy()
            position_size = portfolio['balance'] * 0.02
            
            # 5. Execute decision
            if prediction == 1:  # Buy signal
                order = {
                    'symbol': 'BTC',
                    'side': 'buy',
                    'quantity': position_size / window['close'].iloc[-1],
                    'timestamp': time.time()
                }
                return order
            
            return None
        
        return self.run_benchmark(
            name="end_to_end_trading_cycle",
            category="trading_decision",
            func=trading_cycle,
            iterations=5000
        )
    
    def run_stress_test(self, duration_seconds: int = 60):
        """Run stress test with maximum load"""
        print(f"\nRunning stress test for {duration_seconds} seconds...")
        
        operations_count = 0
        start_time = time.time()
        latencies = []
        
        # Run until duration expires
        while time.time() - start_time < duration_seconds:
            op_start = time.perf_counter()
            
            # Simulate heavy load
            for _ in range(10):
                self._simulate_trading_operation()
            
            op_end = time.perf_counter()
            latencies.append((op_end - op_start) / 10)  # Per operation
            operations_count += 10
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate stress test metrics
        latencies_array = np.array(latencies)
        
        stress_result = {
            "duration_seconds": total_duration,
            "total_operations": operations_count,
            "throughput_ops_sec": operations_count / total_duration,
            "latency_ms": {
                "mean": np.mean(latencies_array) * 1000,
                "p95": np.percentile(latencies_array, 95) * 1000,
                "p99": np.percentile(latencies_array, 99) * 1000,
                "max": np.max(latencies_array) * 1000
            },
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_mb": psutil.virtual_memory().used / 1024 / 1024
            }
        }
        
        return stress_result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks"""
        print("Starting ULTRATHINK Benchmark Suite...")
        print("=" * 60)
        
        # Run individual benchmarks
        self.benchmark_trading_decision()
        self.benchmark_data_processing()
        self.benchmark_model_inference()
        self.benchmark_risk_validation()
        self.benchmark_feature_calculation()
        self.benchmark_database_operations()
        self.benchmark_end_to_end()
        
        # Run stress test
        stress_results = self.run_stress_test(duration_seconds=30)
        
        # Compile results
        results_summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [result.to_dict() for result in self.results],
            "stress_test": stress_results,
            "summary": self._generate_summary(),
            "passed": all(r.passed for r in self.results)
        }
        
        return results_summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        if not self.results:
            return {}
        
        # Group by category
        category_stats = {}
        for result in self.results:
            if result.category not in category_stats:
                category_stats[result.category] = {
                    "benchmarks": [],
                    "avg_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "passed": True
                }
            
            cat_stats = category_stats[result.category]
            cat_stats["benchmarks"].append(result.name)
            cat_stats["avg_latency_ms"] = np.mean([
                r.mean_time_ms for r in self.results 
                if r.category == result.category
            ])
            cat_stats["p99_latency_ms"] = np.max([
                r.p99_time_ms for r in self.results 
                if r.category == result.category
            ])
            cat_stats["passed"] = cat_stats["passed"] and result.passed
        
        # Overall statistics
        all_latencies = [r.mean_time_ms for r in self.results]
        
        return {
            "total_benchmarks": len(self.results),
            "passed_benchmarks": sum(1 for r in self.results if r.passed),
            "failed_benchmarks": sum(1 for r in self.results if not r.passed),
            "overall_latency_ms": {
                "mean": np.mean(all_latencies),
                "median": np.median(all_latencies),
                "p95": np.percentile(all_latencies, 95),
                "p99": np.percentile(all_latencies, 99)
            },
            "category_stats": category_stats,
            "critical_failures": [
                {
                    "name": r.name,
                    "latency_ms": r.p99_time_ms,
                    "target_ms": r.target_latency * 1000
                }
                for r in self.results if not r.passed
            ]
        }
    
    def export_results(self, output_path: str = "optimization/benchmark_results.json"):
        """Export benchmark results"""
        results = self.run_all_benchmarks()
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBenchmark results saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        summary = results['summary']
        print(f"Total Benchmarks: {summary['total_benchmarks']}")
        print(f"Passed: {summary['passed_benchmarks']}")
        print(f"Failed: {summary['failed_benchmarks']}")
        print(f"\nOverall Latency (ms):")
        print(f"  Mean: {summary['overall_latency_ms']['mean']:.2f}")
        print(f"  P95: {summary['overall_latency_ms']['p95']:.2f}")
        print(f"  P99: {summary['overall_latency_ms']['p99']:.2f}")
        
        if summary['critical_failures']:
            print("\nCRITICAL FAILURES:")
            for failure in summary['critical_failures']:
                print(f"  - {failure['name']}: {failure['latency_ms']:.2f}ms " +
                      f"(target: {failure['target_ms']:.2f}ms)")
        
        return results
    
    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    def _calculate_features_fast(self, window: pd.DataFrame) -> np.ndarray:
        """Fast feature calculation"""
        features = []
        
        # Basic price features
        returns = window['close'].pct_change().dropna()
        features.extend([
            returns.iloc[-1],  # Last return
            returns.mean(),
            returns.std()
        ])
        
        # Simple technical indicators
        sma_20 = window['close'].rolling(20).mean().iloc[-1]
        current_price = window['close'].iloc[-1]
        features.append((current_price - sma_20) / sma_20)
        
        # Volume feature
        features.append(window['volume'].iloc[-1] / window['volume'].mean())
        
        return np.array(features)
    
    def _create_mock_model(self):
        """Create mock ML model"""
        class MockModel:
            def predict(self, features):
                # Simple threshold-based decision
                signal_strength = np.sum(features[:3])
                if signal_strength > 0.1:
                    return 1  # Buy
                elif signal_strength < -0.1:
                    return 2  # Sell
                return 0  # Hold
        
        return MockModel()
    
    def _simulate_trading_operation(self):
        """Simulate complete trading operation"""
        # Market data processing
        idx = np.random.randint(100, 1000)
        data = self.test_market_data.iloc[idx-100:idx]
        
        # Feature calculation
        features = self._calculate_features_fast(data)
        
        # Risk check
        position_ok = np.random.random() > 0.1
        
        # Order simulation
        if position_ok:
            order = {'symbol': 'BTC', 'quantity': 0.1}
        
        return position_ok


if __name__ == "__main__":
    # Run comprehensive benchmarks
    suite = BenchmarkSuite()
    
    # Run all benchmarks and export results
    results = suite.export_results()
    
    # Run async benchmarks demo
    async def async_demo():
        async def mock_async_operation():
            await asyncio.sleep(0.001)  # 1ms async operation
            return np.random.random()
        
        result = await suite.run_async_benchmark(
            name="async_data_fetch",
            category="data_processing",
            async_func=mock_async_operation,
            iterations=1000,
            concurrency=50
        )
        
        print(f"\nAsync Benchmark Result:")
        print(f"  Mean latency: {result.mean_time_ms:.2f}ms")
        print(f"  Throughput: {result.throughput:.0f} ops/sec")
    
    # Run async demo
    asyncio.run(async_demo())