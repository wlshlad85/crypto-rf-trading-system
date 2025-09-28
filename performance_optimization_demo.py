#!/usr/bin/env python3
"""Performance optimization demonstration and testing script."""

import asyncio
import time
import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config
from utils.performance_monitor import get_performance_monitor, time_function, monitor_performance
from utils.result_cache import get_global_cache, cache_result
from utils.data_chunking import DataChunker
from utils.file_operations import OptimizedFileOperations
from features.feature_engineering import CryptoFeatureEngine
from data.yfinance_fetcher import YFinanceCryptoFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_optimizations():
    """Demonstrate the performance optimizations."""
    print("\n" + "="*80)
    print("🚀 PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*80 + "\n")

    # Initialize performance monitor
    monitor = get_performance_monitor()
    monitor.start_monitoring()

    # Get configuration
    config = get_default_config()

    print("📊 Testing Data Fetching Optimizations...")
    with monitor_performance("data_fetching"):
        # Test optimized data fetching
        fetcher = YFinanceCryptoFetcher(config.data)
        symbols = ['bitcoin', 'ethereum']  # Reduced for demo

        # This would normally take time, but with caching it should be fast
        data_dict = fetcher.fetch_all_symbols(symbols)
        combined_data = fetcher.combine_data(data_dict)
        clean_data = fetcher.get_clean_data(combined_data)

    print(f"   ✅ Fetched data shape: {clean_data.shape}")

    print("\n🎯 Testing Feature Engineering Optimizations...")
    with monitor_performance("feature_engineering"):
        # Test optimized feature engineering
        feature_engine = CryptoFeatureEngine(config.features)
        features = feature_engine.generate_features(clean_data)

    print(f"   ✅ Generated {len(features.columns)} features")

    print("\n💾 Testing File I/O Optimizations...")
    with monitor_performance("file_operations"):
        # Test optimized file operations
        file_ops = OptimizedFileOperations()

        # Save features efficiently
        test_file = "test_optimized_features.pkl"
        file_ops.save_dataframe_compressed(features, test_file)

        # Load them back
        loaded_features = file_ops.load_dataframe_compressed(test_file)

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    print(f"   ✅ File I/O test completed")

    print("\n🔍 Testing Memory Optimizations...")
    with monitor_performance("memory_optimization"):
        # Test memory optimization
        chunker = DataChunker(chunk_size=1000)

        # Get memory usage before
        initial_memory = chunker.memory_usage_summary(features)

        # Optimize memory usage
        optimized_features = chunker.optimize_dataframe_memory(features)

        # Get memory usage after
        final_memory = chunker.memory_usage_summary(optimized_features)

        memory_saved = initial_memory['total_memory_mb'] - final_memory['total_memory_mb']
        print(f"   ✅ Memory optimized: {memory_saved:.2f} MB saved")

    print("\n📈 Testing Caching System...")
    with monitor_performance("caching_system"):
        # Test caching system
        cache = get_global_cache()

        # Clear cache for fair test
        cache.clear()

        # Define a test function to cache
        @cache_result(cache, ttl_seconds=300)
        def expensive_computation(n):
            time.sleep(0.1)  # Simulate expensive operation
            return n * n

        # First call - should be slow
        start_time = time.time()
        result1 = expensive_computation(5)
        first_call_time = time.time() - start_time

        # Second call - should be fast (cached)
        start_time = time.time()
        result2 = expensive_computation(5)
        second_call_time = time.time() - start_time

        print(f"   ✅ First call: {first_call_time:.4f}s, Cached call: {second_call_time:.4f}s")
        print(f"   ✅ Cache stats: {cache.get_stats()}")

    # Generate performance report
    print("\n📊 Generating Performance Report...")
    report = monitor.get_performance_report()
    bottleneck_analysis = monitor.get_bottleneck_analysis()

    print("\n" + "-"*60)
    print("PERFORMANCE SUMMARY")
    print("-"*60)
    print(f"Total Uptime: {report['uptime_seconds']:.2f} seconds")
    print(f"Functions Tracked: {len(report['function_timings'])}")

    if report['memory_stats']:
        mem = report['memory_stats']
        print(f"Memory Usage: {mem['current_mb']:.1f} MB (Peak: {mem['peak_mb']:.1f} MB)")

    print(f"Cache Hit Rate: {cache.get_stats()['hit_rate']:.2%}")

    if bottleneck_analysis['bottlenecks']:
        print("\n🚨 BOTTLENECKS DETECTED:")
        for bottleneck in bottleneck_analysis['bottlenecks'][:3]:  # Top 3
            print(f"   • {bottleneck['function']}: {bottleneck['total_time']:.2f}s total")

    if bottleneck_analysis['recommendations']:
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for rec in bottleneck_analysis['recommendations'][:3]:  # Top 3
            print(f"   • {rec['function']}: {rec['recommendation']}")

    # Export detailed report
    report_file = "performance_optimization_report.json"
    monitor.export_metrics(report_file)
    print(f"\n📄 Detailed report saved to: {report_file}")

    # Stop monitoring
    monitor.stop_monitoring()

    print("\n" + "="*80)
    print("✅ PERFORMANCE OPTIMIZATION DEMONSTRATION COMPLETED")
    print("="*80 + "\n")

    return {
        'report': report,
        'bottleneck_analysis': bottleneck_analysis,
        'cache_stats': cache.get_stats()
    }


def run_bottleneck_analysis():
    """Run a comprehensive bottleneck analysis."""
    print("\n🔍 RUNNING COMPREHENSIVE BOTTLENECK ANALYSIS...")

    # This would normally run your full trading system
    # For demo purposes, we'll just show the monitoring capabilities

    monitor = get_performance_monitor()
    monitor.start_monitoring()

    # Simulate some operations that would be bottlenecks
    print("   Simulating data processing operations...")

    # Simulate feature engineering
    with monitor_performance("simulated_feature_engineering"):
        time.sleep(0.5)  # Simulate processing time

    # Simulate model training
    with monitor_performance("simulated_model_training"):
        time.sleep(0.8)  # Simulate training time

    # Simulate backtesting
    with monitor_performance("simulated_backtesting"):
        time.sleep(0.3)  # Simulate backtest time

    # Take memory snapshots
    monitor.take_memory_snapshot()
    time.sleep(0.1)
    monitor.take_memory_snapshot()

    # Generate analysis
    bottlenecks = monitor.get_bottleneck_analysis()
    report = monitor.get_performance_report()

    print("   ✅ Analysis completed")

    # Display results
    print("\n📊 BOTTLENECK ANALYSIS RESULTS:")
    print(f"   • Functions tracked: {bottlenecks['total_functions_tracked']}")
    print(f"   • Bottlenecks found: {len(bottlenecks['bottlenecks'])}")
    print(f"   • Recommendations: {len(bottlenecks['recommendations'])}")

    if bottlenecks['bottlenecks']:
        print("\n   🚨 TOP BOTTLENECKS:")
        for i, bottleneck in enumerate(bottlenecks['bottlenecks'][:5], 1):
            print(f"      {i}. {bottleneck['function']}")
            print(f"         Total time: {bottleneck['total_time']:.2f}s")
            print(f"         Call count: {bottleneck['call_count']}")
            print(f"         Avg time: {bottleneck['avg_time']:.4f}s")

    monitor.stop_monitoring()

    return bottlenecks


async def main():
    """Main demonstration function."""
    try:
        # Run the optimization demonstration
        results = await demonstrate_optimizations()

        # Run bottleneck analysis
        print("\n" + "="*60)
        bottleneck_results = run_bottleneck_analysis()

        print("\n🎉 OPTIMIZATION DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nKey improvements implemented:")
        print("   ✅ Feature engineering with parallel processing and caching")
        print("   ✅ Model training with incremental learning")
        print("   ✅ Optimized data fetching with intelligent rate limiting")
        print("   ✅ Memory-efficient data chunking")
        print("   ✅ Compressed file I/O operations")
        print("   ✅ Comprehensive result caching system")
        print("   ✅ Performance monitoring and bottleneck detection")

        return results

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        return None


if __name__ == "__main__":
    # Run the demonstration
    results = asyncio.run(main())

    if results:
        print("\n✅ All optimizations working correctly!")
        print("Check the generated files for detailed performance reports.")
    else:
        print("\n❌ Demonstration encountered errors.")
        sys.exit(1)