"""Main script to run all optimizations and fix bottlenecks."""

import asyncio
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import optimized components
from optimization.data_fetcher_optimizer import OptimizedYFinanceFetcher
from optimization.feature_engineering_optimizer import OptimizedFeatureEngine
from optimization.ml_training_optimizer import OptimizedRandomForestModel
from optimization.backtest_engine_optimizer import OptimizedBacktestEngine

# Import original components for comparison
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.ultra_feature_engineering import UltraFeatureEngine
from models.random_forest_model import CryptoRandomForestModel
from backtesting.backtest_engine import CryptoBacktestEngine

from utils.config import load_config, get_default_config


class PerformanceOptimizer:
    """Main class to orchestrate all performance optimizations."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path) if config_path else get_default_config()
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Track performance metrics
        self.performance_metrics = {}
        
    def setup_logging(self):
        """Setup enhanced logging."""
        log_dir = Path("optimization_logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def run_data_fetching_comparison(self):
        """Compare original vs optimized data fetching."""
        symbols = self.config.data.symbols[:5]  # Test with 5 symbols
        
        self.logger.info("=" * 80)
        self.logger.info("DATA FETCHING OPTIMIZATION COMPARISON")
        self.logger.info("=" * 80)
        
        # Original fetcher
        original_fetcher = YFinanceCryptoFetcher(self.config.data)
        
        self.logger.info("\n1. Testing ORIGINAL data fetcher...")
        start_time = time.time()
        original_data = original_fetcher.fetch_all_symbols(symbols)
        original_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {original_time:.2f} seconds")
        self.logger.info(f"   Data fetched: {len(original_data)} symbols")
        
        # Optimized fetcher
        optimized_fetcher = OptimizedYFinanceFetcher(self.config.data)
        
        self.logger.info("\n2. Testing OPTIMIZED data fetcher...")
        start_time = time.time()
        optimized_data = await optimized_fetcher.fetch_all_symbols_async(symbols)
        optimized_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {optimized_time:.2f} seconds")
        self.logger.info(f"   Data fetched: {len(optimized_data)} symbols")
        
        # Calculate improvement
        improvement = ((original_time - optimized_time) / original_time) * 100
        self.logger.info(f"\n   🚀 IMPROVEMENT: {improvement:.1f}% faster")
        self.logger.info(f"   ⏱️  Time saved: {original_time - optimized_time:.2f} seconds")
        
        self.performance_metrics['data_fetching'] = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'improvement_percent': improvement,
            'symbols_tested': len(symbols)
        }
        
        # Clean up
        optimized_fetcher.close()
        
        return optimized_data
    
    def run_feature_engineering_comparison(self, data: Dict[str, pd.DataFrame]):
        """Compare original vs optimized feature engineering."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FEATURE ENGINEERING OPTIMIZATION COMPARISON")
        self.logger.info("=" * 80)
        
        # Get first symbol's data for testing
        symbol = list(data.keys())[0]
        test_data = data[symbol]
        
        # Original feature engine
        original_engine = UltraFeatureEngine()
        
        self.logger.info("\n1. Testing ORIGINAL feature engineering...")
        start_time = time.time()
        original_features = original_engine.generate_features(test_data, symbol)
        original_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {original_time:.2f} seconds")
        self.logger.info(f"   Features generated: {original_features.shape[1]} features")
        self.logger.info(f"   Memory usage: {original_features.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Optimized feature engine
        optimized_engine = OptimizedFeatureEngine()
        
        self.logger.info("\n2. Testing OPTIMIZED feature engineering...")
        start_time = time.time()
        optimized_features = optimized_engine.generate_features_optimized(test_data, symbol)
        optimized_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {optimized_time:.2f} seconds")
        self.logger.info(f"   Features generated: {optimized_features.shape[1]} features")
        self.logger.info(f"   Memory usage: {optimized_features.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Calculate improvement
        improvement = ((original_time - optimized_time) / original_time) * 100
        memory_improvement = ((original_features.memory_usage().sum() - optimized_features.memory_usage().sum()) / 
                            original_features.memory_usage().sum()) * 100
        
        self.logger.info(f"\n   🚀 TIME IMPROVEMENT: {improvement:.1f}% faster")
        self.logger.info(f"   💾 MEMORY IMPROVEMENT: {memory_improvement:.1f}% less memory")
        
        self.performance_metrics['feature_engineering'] = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'time_improvement_percent': improvement,
            'memory_improvement_percent': memory_improvement,
            'features_count': optimized_features.shape[1]
        }
        
        # Clean up
        optimized_engine.close()
        
        return optimized_features
    
    def run_ml_training_comparison(self, features: pd.DataFrame):
        """Compare original vs optimized ML training."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ML TRAINING OPTIMIZATION COMPARISON")
        self.logger.info("=" * 80)
        
        # Create target variable
        target = features['close'].pct_change(24).shift(-24).fillna(0)
        
        # Prepare data
        feature_cols = [col for col in features.columns if col not in ['close', 'open', 'high', 'low', 'volume']]
        X = features[feature_cols].fillna(0)
        y = target
        
        # Remove invalid samples
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Limit data size for testing
        X = X.iloc[:5000]
        y = y.iloc[:5000]
        
        # Original model
        original_model = CryptoRandomForestModel(self.config.model)
        
        self.logger.info("\n1. Testing ORIGINAL ML training...")
        start_time = time.time()
        original_model.train(X, y)
        original_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {original_time:.2f} seconds")
        
        # Optimized model
        optimized_model = OptimizedRandomForestModel(self.config.model)
        
        self.logger.info("\n2. Testing OPTIMIZED ML training...")
        start_time = time.time()
        optimized_model.train_optimized(X, y)
        optimized_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {optimized_time:.2f} seconds")
        
        # Test hyperparameter tuning
        self.logger.info("\n3. Testing OPTIMIZED hyperparameter tuning...")
        start_time = time.time()
        tuning_results = optimized_model.hyperparameter_tuning_optimized(X, y, n_trials=10)
        tuning_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {tuning_time:.2f} seconds")
        self.logger.info(f"   Best score: {tuning_results['best_score']:.4f}")
        
        # Calculate improvement
        improvement = ((original_time - optimized_time) / original_time) * 100
        
        self.logger.info(f"\n   🚀 TRAINING IMPROVEMENT: {improvement:.1f}% faster")
        
        self.performance_metrics['ml_training'] = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'improvement_percent': improvement,
            'tuning_time': tuning_time,
            'samples_used': len(X)
        }
        
        return optimized_model
    
    def run_backtesting_comparison(self, data: pd.DataFrame, model):
        """Compare original vs optimized backtesting."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BACKTESTING OPTIMIZATION COMPARISON")
        self.logger.info("=" * 80)
        
        # Generate simple signals for testing
        fast_ma = data['close'].rolling(10).mean()
        slow_ma = data['close'].rolling(30).mean()
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        # Limit data for testing
        test_data = data.iloc[:10000]
        test_signals = signals.iloc[:10000]
        
        # Optimized backtest
        optimized_engine = OptimizedBacktestEngine(self.config.backtest)
        
        self.logger.info("\n1. Testing OPTIMIZED backtesting...")
        start_time = time.time()
        optimized_results = optimized_engine.run_backtest_optimized(test_data, test_signals)
        optimized_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {optimized_time:.2f} seconds")
        self.logger.info(f"   Sharpe Ratio: {optimized_results['metrics']['sharpe_ratio']:.3f}")
        self.logger.info(f"   Total Return: {optimized_results['metrics']['total_return']:.2%}")
        
        # Test parallel backtesting
        param_combinations = [
            {'fast_period': 5, 'slow_period': 20},
            {'fast_period': 10, 'slow_period': 30},
            {'fast_period': 20, 'slow_period': 50},
            {'fast_period': 10, 'slow_period': 40},
        ]
        
        self.logger.info("\n2. Testing PARALLEL multi-parameter backtesting...")
        start_time = time.time()
        parallel_results = optimized_engine.run_parallel_backtest_multiple_params(
            test_data, param_combinations
        )
        parallel_time = time.time() - start_time
        
        self.logger.info(f"   Time taken: {parallel_time:.2f} seconds for {len(param_combinations)} backtests")
        self.logger.info(f"   Time per backtest: {parallel_time/len(param_combinations):.2f} seconds")
        self.logger.info(f"   Best Sharpe Ratio: {parallel_results[0]['metrics']['sharpe_ratio']:.3f}")
        
        self.performance_metrics['backtesting'] = {
            'optimized_time': optimized_time,
            'parallel_time': parallel_time,
            'backtests_run': len(param_combinations),
            'time_per_backtest': parallel_time / len(param_combinations)
        }
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("OPTIMIZATION SUMMARY REPORT")
        self.logger.info("=" * 80)
        
        total_original_time = sum(
            metrics.get('original_time', 0) 
            for metrics in self.performance_metrics.values()
        )
        
        total_optimized_time = sum(
            metrics.get('optimized_time', 0)
            for metrics in self.performance_metrics.values()
        )
        
        overall_improvement = ((total_original_time - total_optimized_time) / total_original_time) * 100 if total_original_time > 0 else 0
        
        self.logger.info(f"\n📊 OVERALL PERFORMANCE IMPROVEMENT: {overall_improvement:.1f}%")
        self.logger.info(f"⏱️  Total time saved: {total_original_time - total_optimized_time:.2f} seconds")
        
        # Component-wise summary
        self.logger.info("\n📈 Component-wise Improvements:")
        for component, metrics in self.performance_metrics.items():
            improvement = metrics.get('improvement_percent', metrics.get('time_improvement_percent', 0))
            self.logger.info(f"   • {component.replace('_', ' ').title()}: {improvement:.1f}% faster")
        
        # Save report
        report_path = Path("optimization_logs") / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        self.logger.info(f"\n💾 Detailed report saved to: {report_path}")
        
        # Key recommendations
        self.logger.info("\n🎯 KEY RECOMMENDATIONS:")
        self.logger.info("   1. Replace sleep-based rate limiting with async semaphores")
        self.logger.info("   2. Use vectorized operations for all numerical computations")
        self.logger.info("   3. Implement parallel processing for independent operations")
        self.logger.info("   4. Optimize memory usage with appropriate data types")
        self.logger.info("   5. Use Numba JIT compilation for performance-critical loops")
        self.logger.info("   6. Leverage GPU acceleration where available (CatBoost/LightGBM)")
    
    async def run_all_optimizations(self):
        """Run all optimization comparisons."""
        try:
            # 1. Data fetching optimization
            optimized_data = await self.run_data_fetching_comparison()
            
            # 2. Feature engineering optimization
            features = self.run_feature_engineering_comparison(optimized_data)
            
            # 3. ML training optimization
            model = self.run_ml_training_comparison(features)
            
            # 4. Backtesting optimization
            first_symbol_data = list(optimized_data.values())[0]
            self.run_backtesting_comparison(first_symbol_data, model)
            
            # 5. Generate report
            self.generate_optimization_report()
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run performance optimizations')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    args = parser.parse_args()
    
    print("\n🚀 CRYPTO TRADING SYSTEM PERFORMANCE OPTIMIZER")
    print("=" * 80)
    print("This tool will identify and fix performance bottlenecks in your system")
    print("=" * 80)
    
    optimizer = PerformanceOptimizer(args.config)
    await optimizer.run_all_optimizations()
    
    print("\n✅ Optimization analysis complete!")


if __name__ == "__main__":
    asyncio.run(main())