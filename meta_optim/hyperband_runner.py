#!/usr/bin/env python3
"""
Hyperband Parameter Explorer for Random Forest Meta-Optimization

Implements sophisticated parameter exploration using:
- Hyperband algorithm (multi-armed bandit approach)
- Random search with early stopping
- Bayesian optimization (optional)
- Progressive halving for efficient resource allocation

Usage: python3 hyperband_runner.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_optim.objective_fn import MetaObjectiveFunction
from meta_optim.retrain_worker import RetrainWorker

class HyperbandRunner:
    """Hyperband algorithm implementation for Random Forest hyperparameter optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Hyperband runner with configuration."""
        self.config = config or self._get_default_config()
        
        # Hyperband parameters
        self.max_iter = self.config.get('max_iter', 81)  # Maximum iterations per configuration
        self.eta = self.config.get('eta', 3)  # Downsampling rate
        self.s_max = int(np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter
        
        # Components
        self.objective_fn = MetaObjectiveFunction()
        self.retrain_worker = RetrainWorker()
        
        # Results tracking
        self.all_results = []
        self.best_configs = []
        self.exploration_history = []
        
        # Logging
        self.log_dir = "meta_optim/meta_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"🎯 Hyperband initialized: s_max={self.s_max}, B={self.B}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Hyperband."""
        return {
            'max_iter': 81,
            'eta': 3,
            'max_parallel': 4,
            'min_improvement': 0.05,  # 5% minimum improvement
            'parameter_space': {
                'entry_model': {
                    'n_estimators': [50, 100, 150, 200, 300, 500],
                    'max_depth': [6, 8, 10, 12, 15, 20],
                    'min_samples_split': [5, 10, 15, 20, 30],
                    'min_samples_leaf': [2, 5, 8, 10, 15],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]
                },
                'position_model': {
                    'n_estimators': [50, 100, 150, 200, 250],
                    'max_depth': [4, 6, 8, 10, 12],
                    'min_samples_split': [5, 10, 15, 20],
                    'min_samples_leaf': [2, 5, 8, 10]
                },
                'exit_model': {
                    'n_estimators': [50, 100, 150, 200, 300],
                    'max_depth': [6, 8, 10, 12, 15],
                    'min_samples_split': [5, 10, 15, 25],
                    'min_samples_leaf': [2, 5, 8, 12]
                },
                'profit_model': {
                    'n_estimators': [100, 150, 200, 250, 300],
                    'max_depth': [8, 10, 12, 15, 18],
                    'min_samples_split': [8, 12, 16, 20],
                    'min_samples_leaf': [3, 6, 9, 12]
                },
                'trading_params': {
                    'momentum_threshold': [1.2, 1.5, 1.78, 2.0, 2.5],
                    'position_range_min': [0.3, 0.4, 0.464, 0.5],
                    'position_range_max': [0.7, 0.8, 0.85, 0.9],
                    'confidence_threshold': [0.5, 0.6, 0.65, 0.7],
                    'exit_threshold': [0.4, 0.5, 0.55, 0.6]
                }
            }
        }
    
    def run_hyperband_optimization(self, n_iterations: int = 3) -> Dict[str, Any]:
        """Run full Hyperband optimization."""
        print("🚀 Starting Hyperband Meta-Optimization")
        print("=" * 60)
        
        start_time = datetime.now()
        all_results = []
        
        for i in range(n_iterations):
            print(f"\n🔄 Hyperband Iteration {i + 1}/{n_iterations}")
            
            for s in reversed(range(self.s_max + 1)):
                print(f"  📊 Successive Halving: s={s}")
                
                # Calculate resource allocation
                n = int(np.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
                r = self.max_iter * self.eta ** (-s)
                
                print(f"    Configurations: {n}, Initial resource: {r:.1f}")
                
                # Random sampling of configurations
                configurations = self._sample_configurations(n)
                
                # Successive halving
                results = self._successive_halving(configurations, r, s)
                all_results.extend(results)
                
                # Update best configurations
                self._update_best_configs(results)
        
        # Final analysis
        optimization_results = self._analyze_results(all_results, start_time)
        
        # Save results
        self._save_results(optimization_results)
        
        print("=" * 60)
        print("✅ Hyperband optimization complete!")
        
        return optimization_results
    
    def _sample_configurations(self, n: int) -> List[Dict[str, Any]]:
        """Sample random configurations from parameter space."""
        configurations = []
        param_space = self.config['parameter_space']
        
        for _ in range(n):
            config = {}
            
            for model_name, model_params in param_space.items():
                config[model_name] = {}
                for param_name, param_values in model_params.items():
                    config[model_name][param_name] = np.random.choice(param_values)
            
            # Add unique ID
            config['config_id'] = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
            config['timestamp'] = datetime.now().isoformat()
            
            configurations.append(config)
        
        return configurations
    
    def _successive_halving(self, configurations: List[Dict], r: float, s: int) -> List[Dict]:
        """Implement successive halving algorithm."""
        results = []
        current_configs = configurations.copy()
        current_r = r
        
        for i in range(s + 1):
            print(f"      Round {i + 1}: {len(current_configs)} configs, resource {current_r:.1f}")
            
            # Evaluate configurations with current resource allocation
            round_results = self._evaluate_configurations(current_configs, current_r)
            results.extend(round_results)
            
            # Sort by performance
            round_results.sort(key=lambda x: x.get('composite_score', -999), reverse=True)
            
            # Keep top configurations for next round
            if i < s:
                n_keep = max(1, len(current_configs) // self.eta)
                current_configs = [r['configuration'] for r in round_results[:n_keep]]
                current_r *= self.eta
                
                print(f"        Keeping top {n_keep} configurations")
        
        return results
    
    def _evaluate_configurations(self, configurations: List[Dict], resource: float) -> List[Dict]:
        """Evaluate configurations using parallel processing."""
        results = []
        
        # Use thread pool for parallel evaluation
        max_workers = min(self.config.get('max_parallel', 4), len(configurations))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_config = {
                executor.submit(self._evaluate_single_config, config, resource): config
                for config in configurations
            }
            
            # Collect results
            for future in as_completed(future_to_config, timeout=3600):  # 1 hour timeout
                config = future_to_config[future]
                
                try:
                    result = future.result()
                    result['configuration'] = config
                    results.append(result)
                    
                    print(f"        ✅ {config['config_id']}: {result.get('composite_score', 0):.3f}")
                    
                except Exception as e:
                    print(f"        ❌ {config['config_id']}: {str(e)}")
                    
                    # Add failed result
                    results.append({
                        'configuration': config,
                        'composite_score': -999.0,
                        'viable': False,
                        'error': str(e),
                        'evaluation_time': 0
                    })
        
        return results
    
    def _evaluate_single_config(self, config: Dict[str, Any], resource: float) -> Dict[str, Any]:
        """Evaluate a single configuration."""
        start_time = time.time()
        
        try:
            # Use resource allocation to determine training subset size
            training_fraction = min(1.0, resource / self.max_iter)
            
            # Train models with this configuration
            backtest_results = self.retrain_worker.train_and_evaluate(
                config, 
                training_fraction=training_fraction
            )
            
            # Evaluate using objective function
            evaluation = self.objective_fn.evaluate_strategy(backtest_results)
            
            # Add timing information
            evaluation['evaluation_time'] = time.time() - start_time
            evaluation['resource_used'] = resource
            evaluation['training_fraction'] = training_fraction
            
            return evaluation
            
        except Exception as e:
            return {
                'composite_score': -999.0,
                'viable': False,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'resource_used': resource
            }
    
    def _update_best_configs(self, results: List[Dict]):
        """Update list of best configurations."""
        valid_results = [r for r in results if r.get('viable', False)]
        
        if not valid_results:
            return
        
        # Sort by composite score
        valid_results.sort(key=lambda x: x.get('composite_score', -999), reverse=True)
        
        # Update best configs (keep top 10)
        for result in valid_results[:10]:
            # Check if significantly better than existing configs
            is_new_best = True
            
            for existing in self.best_configs:
                if abs(result['composite_score'] - existing['composite_score']) < 0.01:
                    is_new_best = False
                    break
            
            if is_new_best:
                self.best_configs.append(result)
        
        # Keep only top 10 overall
        self.best_configs.sort(key=lambda x: x.get('composite_score', -999), reverse=True)
        self.best_configs = self.best_configs[:10]
    
    def _analyze_results(self, all_results: List[Dict], start_time: datetime) -> Dict[str, Any]:
        """Analyze optimization results."""
        
        valid_results = [r for r in all_results if r.get('viable', False)]
        
        if not valid_results:
            return {
                'success': False,
                'reason': 'No viable configurations found',
                'total_evaluations': len(all_results),
                'optimization_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Sort by performance
        valid_results.sort(key=lambda x: x.get('composite_score', -999), reverse=True)
        best_result = valid_results[0]
        
        # Calculate statistics
        scores = [r['composite_score'] for r in valid_results]
        
        analysis = {
            'success': True,
            'total_evaluations': len(all_results),
            'viable_configurations': len(valid_results),
            'optimization_time': (datetime.now() - start_time).total_seconds(),
            'best_configuration': best_result,
            'top_5_configurations': valid_results[:5],
            'performance_statistics': {
                'best_score': max(scores),
                'worst_score': min(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'median_score': np.median(scores)
            },
            'parameter_insights': self._analyze_parameter_importance(valid_results)
        }
        
        return analysis
    
    def _analyze_parameter_importance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze which parameters correlate with better performance."""
        
        if len(results) < 5:
            return {'insufficient_data': True}
        
        # Extract parameter values and scores
        param_analysis = {}
        
        for result in results:
            config = result.get('configuration', {})
            score = result.get('composite_score', 0)
            
            for model_name, model_params in config.items():
                if model_name in ['config_id', 'timestamp']:
                    continue
                
                if model_name not in param_analysis:
                    param_analysis[model_name] = {}
                
                for param_name, param_value in model_params.items():
                    if param_name not in param_analysis[model_name]:
                        param_analysis[model_name][param_name] = []
                    
                    param_analysis[model_name][param_name].append((param_value, score))
        
        # Calculate correlations and best values
        insights = {}
        
        for model_name, model_params in param_analysis.items():
            insights[model_name] = {}
            
            for param_name, value_score_pairs in model_params.items():
                if len(value_score_pairs) < 3:
                    continue
                
                # Group by parameter value and calculate mean scores
                value_groups = {}
                for value, score in value_score_pairs:
                    if value not in value_groups:
                        value_groups[value] = []
                    value_groups[value].append(score)
                
                # Calculate mean score for each value
                value_means = {
                    value: np.mean(scores) 
                    for value, scores in value_groups.items()
                }
                
                if value_means:
                    best_value = max(value_means.keys(), key=lambda k: value_means[k])
                    insights[model_name][param_name] = {
                        'best_value': best_value,
                        'best_score': value_means[best_value],
                        'value_performance': value_means
                    }
        
        return insights
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = os.path.join(self.log_dir, f'hyperband_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save top configurations
        if results.get('success', False):
            top_configs = {
                'timestamp': timestamp,
                'best_configuration': results['best_configuration'],
                'top_5_configurations': results['top_5_configurations'],
                'performance_statistics': results['performance_statistics']
            }
            
            top_configs_file = os.path.join(self.log_dir, 'top_configs.json')
            with open(top_configs_file, 'w') as f:
                json.dump(top_configs, f, indent=2, default=str)
        
        print(f"📁 Results saved to {results_file}")
    
    def load_best_configuration(self) -> Optional[Dict[str, Any]]:
        """Load the best configuration from previous runs."""
        top_configs_file = os.path.join(self.log_dir, 'top_configs.json')
        
        if not os.path.exists(top_configs_file):
            return None
        
        try:
            with open(top_configs_file, 'r') as f:
                data = json.load(f)
                return data.get('best_configuration')
        except Exception as e:
            print(f"❌ Error loading best configuration: {e}")
            return None
    
    def get_optimization_summary(self) -> str:
        """Get human-readable optimization summary."""
        
        if not self.best_configs:
            return "No optimization results available."
        
        best = self.best_configs[0]
        
        summary = f"""
🎯 HYPERBAND OPTIMIZATION SUMMARY
=================================
Best Configuration Score: {best.get('composite_score', 0):.3f}
Strategy Viable: {best.get('viable', False)}

📊 Performance Metrics:
Sharpe Ratio: {best.get('individual_metrics', {}).get('sharpe_ratio', 0):.2f}
Profit Factor: {best.get('individual_metrics', {}).get('profit_factor', 0):.2f}
Max Drawdown: {best.get('individual_metrics', {}).get('max_drawdown', 0):.2f}%
Alpha Persistence: {best.get('individual_metrics', {}).get('alpha_persistence', 0):.2f}

🔧 Best Parameters:
"""
        
        config = best.get('configuration', {})
        for model_name, params in config.items():
            if model_name not in ['config_id', 'timestamp']:
                summary += f"\n{model_name.upper()}:\n"
                for param, value in params.items():
                    summary += f"  {param}: {value}\n"
        
        summary += f"\n📈 Total Configurations Evaluated: {len(self.best_configs)}"
        
        return summary

def main():
    """Run Hyperband optimization."""
    
    print("🎯 Random Forest Meta-Optimization with Hyperband")
    print("=" * 60)
    
    try:
        # Initialize Hyperband runner
        runner = HyperbandRunner()
        
        # Run optimization
        results = runner.run_hyperband_optimization(n_iterations=2)  # Reduced for demo
        
        if results['success']:
            print(f"\n🎉 Optimization successful!")
            print(f"Best score: {results['best_configuration']['composite_score']:.3f}")
            print(f"Total evaluations: {results['total_evaluations']}")
            print(f"Optimization time: {results['optimization_time']:.1f} seconds")
            
            # Print summary
            print(runner.get_optimization_summary())
            
        else:
            print(f"\n❌ Optimization failed: {results.get('reason', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error in Hyperband optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()