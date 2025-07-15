"""Comprehensive optimization suite for Crypto Random Forest Trading System."""

import asyncio
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config, Config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel, EnsembleRandomForestModel

# Setup colorful logging
import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CryptoOptimizer:
    """Advanced optimization suite for cryptocurrency Random Forest models."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.data = None
        self.features = None
        self.X = None
        self.y = None
        self.optimization_results = {}
        self.best_models = []
        
    async def run_full_optimization(self):
        """Run comprehensive optimization suite."""
        print("\n" + "="*80)
        print("🚀 CRYPTOCURRENCY RANDOM FOREST - OPTIMIZATION SUITE")
        print("="*80 + "\n")
        
        # Step 1: Prepare data
        print("📊 Step 1: Preparing Data for Optimization...")
        await self._prepare_data()
        
        # Step 2: Hyperparameter optimization
        print("\n🔧 Step 2: Running Hyperparameter Optimization...")
        hp_results = await self._optimize_hyperparameters()
        
        # Step 3: Feature optimization
        print("\n🎯 Step 3: Optimizing Feature Selection...")
        feature_results = await self._optimize_features()
        
        # Step 4: Target optimization
        print("\n📈 Step 4: Optimizing Target Variables...")
        target_results = await self._optimize_targets()
        
        # Step 5: Ensemble optimization
        print("\n🤖 Step 5: Building Optimized Ensemble...")
        ensemble_results = await self._optimize_ensemble()
        
        # Step 6: Strategy optimization
        print("\n💼 Step 6: Optimizing Trading Strategy...")
        strategy_results = await self._optimize_strategy()
        
        # Step 7: Generate comprehensive report
        print("\n📊 Step 7: Generating Optimization Report...")
        self._generate_optimization_report()
        
        return self.optimization_results
    
    async def _prepare_data(self):
        """Prepare cryptocurrency data for optimization."""
        # Use multiple cryptocurrencies for robust optimization
        self.config.data.symbols = ['bitcoin', 'ethereum', 'solana', 'binancecoin', 'cardano']
        self.config.data.days = 180  # 6 months for comprehensive optimization
        
        print(f"   📊 Fetching {len(self.config.data.symbols)} cryptocurrencies...")
        fetcher = YFinanceCryptoFetcher(self.config.data)
        data_dict = fetcher.fetch_all_symbols(self.config.data.symbols)
        
        if not data_dict:
            raise ValueError("Failed to fetch data for optimization")
        
        # Current prices
        prices = fetcher.get_latest_prices(self.config.data.symbols)
        print("   💰 Current prices:")
        for symbol, price in prices.items():
            print(f"      {symbol.upper()}: ${price:,.2f}")
        
        # Combine and clean data
        self.data = fetcher.get_clean_data(fetcher.combine_data(data_dict))
        
        # Generate features
        print(f"   🔧 Generating features...")
        feature_engine = CryptoFeatureEngine(self.config.features)
        self.features = feature_engine.generate_features(self.data)
        
        print(f"   ✅ Data prepared: {self.data.shape[0]} samples, {self.features.shape[1]} features")
        
    async def _optimize_hyperparameters(self):
        """Optimize Random Forest hyperparameters using Optuna."""
        print("   🔍 Running Bayesian optimization with Optuna...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            
            # Create model with suggested parameters
            config = self.config
            config.model.__dict__.update(params)
            model = CryptoRandomForestModel(config.model)
            
            # Create target (Bitcoin 6-hour returns)
            target = self.data['bitcoin_close'].pct_change(6).shift(-6).dropna()
            
            # Prepare data
            common_index = self.features.index.intersection(target.index)
            X = self.features.loc[common_index].copy()
            y = target.loc[common_index]
            X['target'] = y
            
            X_clean, y_clean = model.prepare_data(X, 'target')
            
            # Use smaller subset for speed
            if len(X_clean) > 2000:
                sample_size = 2000
                X_clean = X_clean.iloc[-sample_size:]
                y_clean = y_clean.iloc[-sample_size:]
            
            # Train and validate
            try:
                results = model.train(X_clean, y_clean, validation_split=0.3)
                
                # Calculate trading performance
                val_size = int(len(X_clean) * 0.3)
                X_val = X_clean.iloc[-val_size:]
                y_val = y_clean.iloc[-val_size:]
                
                predictions = model.predict(X_val)
                
                # Convert to trading signals
                pred_series = pd.Series(predictions)
                signals = np.zeros(len(predictions))
                signals[pred_series > pred_series.quantile(0.7)] = 1
                signals[pred_series < pred_series.quantile(0.3)] = -1
                
                # Calculate Sharpe ratio
                strategy_returns = signals[:-1] * y_val.values[1:]
                if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(365*24)
                    return sharpe
                else:
                    return -10  # Penalty for failed strategies
                    
            except Exception as e:
                return -10  # Penalty for failed trials
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)  # Reduced for faster execution
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"   ✅ Best hyperparameters found:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")
        print(f"   📊 Best Sharpe ratio: {best_score:.3f}")
        
        self.optimization_results['hyperparameters'] = {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
        
        return best_params
    
    async def _optimize_features(self):
        """Optimize feature selection and engineering."""
        print("   🎯 Testing different feature selection strategies...")
        
        # Test different numbers of features
        feature_counts = [50, 100, 200, 300, 'all']
        results = {}
        
        for n_features in feature_counts:
            print(f"      Testing {n_features} features...")
            
            try:
                # Create target
                target = self.data['bitcoin_close'].pct_change(6).shift(-6).dropna()
                common_index = self.features.index.intersection(target.index)
                X = self.features.loc[common_index].copy()
                y = target.loc[common_index]
                
                # Feature selection
                if n_features != 'all':
                    feature_engine = CryptoFeatureEngine(self.config.features)
                    X['target'] = y
                    X_selected = feature_engine.select_features(X, y, method='importance')
                    if len(X_selected.columns) > n_features:
                        # Use Random Forest to get top features
                        from sklearn.ensemble import RandomForestRegressor
                        rf = RandomForestRegressor(n_estimators=50, random_state=42)
                        X_temp = X.select_dtypes(include=[np.number])
                        rf.fit(X_temp, y)
                        importance = pd.Series(rf.feature_importances_, index=X_temp.columns)
                        top_features = importance.nlargest(n_features).index
                        X_selected = X[top_features]
                    X_selected['target'] = y
                else:
                    X['target'] = y
                    X_selected = X
                
                # Train model
                model = CryptoRandomForestModel(self.config.model)
                X_clean, y_clean = model.prepare_data(X_selected, 'target')
                
                # Quick validation
                train_results = model.train(X_clean, y_clean, validation_split=0.3)
                val_r2 = train_results['validation']['r2']
                
                results[n_features] = val_r2
                print(f"         R²: {val_r2:.4f}")
                
            except Exception as e:
                print(f"         Failed: {e}")
                results[n_features] = -1
        
        best_feature_count = max(results, key=results.get)
        print(f"   ✅ Optimal feature count: {best_feature_count} (R²: {results[best_feature_count]:.4f})")
        
        self.optimization_results['features'] = {
            'results': results,
            'best_count': best_feature_count
        }
        
        return best_feature_count
    
    async def _optimize_targets(self):
        """Optimize target variable definition."""
        print("   📈 Testing different target variable configurations...")
        
        # Test different prediction horizons
        horizons = [3, 6, 12, 24]  # Hours
        target_types = ['returns', 'log_returns']
        
        results = {}
        
        for horizon in horizons:
            for target_type in target_types:
                config_key = f"{target_type}_{horizon}h"
                print(f"      Testing {config_key}...")
                
                try:
                    # Create target
                    btc_close = self.data['bitcoin_close']
                    if target_type == 'returns':
                        target = btc_close.pct_change(horizon).shift(-horizon)
                    else:  # log_returns
                        target = (np.log(btc_close) - np.log(btc_close.shift(horizon))).shift(-horizon)
                    
                    target = target.dropna()
                    
                    # Prepare data
                    common_index = self.features.index.intersection(target.index)
                    X = self.features.loc[common_index].copy()
                    y = target.loc[common_index]
                    X['target'] = y
                    
                    # Train model
                    model = CryptoRandomForestModel(self.config.model)
                    X_clean, y_clean = model.prepare_data(X, 'target')
                    
                    if len(X_clean) > 1000:
                        X_clean = X_clean.iloc[-1000:]
                        y_clean = y_clean.iloc[-1000:]
                    
                    train_results = model.train(X_clean, y_clean, validation_split=0.3)
                    val_r2 = train_results['validation']['r2']
                    
                    results[config_key] = val_r2
                    print(f"         R²: {val_r2:.4f}")
                    
                except Exception as e:
                    print(f"         Failed: {e}")
                    results[config_key] = -1
        
        best_target = max(results, key=results.get)
        print(f"   ✅ Optimal target: {best_target} (R²: {results[best_target]:.4f})")
        
        self.optimization_results['targets'] = {
            'results': results,
            'best_target': best_target
        }
        
        return best_target
    
    async def _optimize_ensemble(self):
        """Create and optimize ensemble models."""
        print("   🤖 Building optimized ensemble...")
        
        # Use best parameters from previous optimizations
        best_params = self.optimization_results.get('hyperparameters', {}).get('best_params', {})
        
        # Create ensemble with different random seeds
        ensemble_sizes = [3, 5, 7]
        results = {}
        
        for size in ensemble_sizes:
            print(f"      Testing ensemble size: {size}...")
            
            try:
                # Create target
                target = self.data['bitcoin_close'].pct_change(6).shift(-6).dropna()
                common_index = self.features.index.intersection(target.index)
                X = self.features.loc[common_index].copy()
                y = target.loc[common_index]
                X['target'] = y
                
                # Create ensemble
                config = self.config
                if best_params:
                    config.model.__dict__.update(best_params)
                
                ensemble = EnsembleRandomForestModel(config.model, n_models=size)
                X_clean, y_clean = ensemble.models[0].prepare_data(X, 'target')
                
                # Use smaller dataset for speed
                if len(X_clean) > 1500:
                    X_clean = X_clean.iloc[-1500:]
                    y_clean = y_clean.iloc[-1500:]
                
                # Train ensemble
                ensemble_results = ensemble.train(X_clean, y_clean)
                
                # Calculate ensemble performance
                val_size = int(len(X_clean) * 0.2)
                X_val = X_clean.iloc[-val_size:]
                y_val = y_clean.iloc[-val_size:]
                
                predictions = ensemble.predict(X_val)
                
                # Calculate metrics
                from sklearn.metrics import r2_score
                r2 = r2_score(y_val, predictions)
                
                results[size] = r2
                print(f"         Ensemble R²: {r2:.4f}")
                
            except Exception as e:
                print(f"         Failed: {e}")
                results[size] = -1
        
        best_ensemble_size = max(results, key=results.get) if results else 5
        print(f"   ✅ Optimal ensemble size: {best_ensemble_size} (R²: {results.get(best_ensemble_size, 0):.4f})")
        
        self.optimization_results['ensemble'] = {
            'results': results,
            'best_size': best_ensemble_size
        }
        
        return best_ensemble_size
    
    async def _optimize_strategy(self):
        """Optimize trading strategy parameters."""
        print("   💼 Optimizing trading strategy parameters...")
        
        # Test different signal thresholds
        thresholds = [(0.6, 0.4), (0.7, 0.3), (0.75, 0.25), (0.8, 0.2)]
        results = {}
        
        # Create a model for testing
        target = self.data['bitcoin_close'].pct_change(6).shift(-6).dropna()
        common_index = self.features.index.intersection(target.index)
        X = self.features.loc[common_index].copy()
        y = target.loc[common_index]
        X['target'] = y
        
        model = CryptoRandomForestModel(self.config.model)
        X_clean, y_clean = model.prepare_data(X, 'target')
        
        if len(X_clean) > 1000:
            X_clean = X_clean.iloc[-1000:]
            y_clean = y_clean.iloc[-1000:]
        
        model.train(X_clean, y_clean, validation_split=0.3)
        
        # Test on validation set
        val_size = int(len(X_clean) * 0.3)
        X_val = X_clean.iloc[-val_size:]
        y_val = y_clean.iloc[-val_size:]
        predictions = model.predict(X_val)
        
        for buy_thresh, sell_thresh in thresholds:
            try:
                # Generate signals
                pred_series = pd.Series(predictions)
                signals = np.zeros(len(predictions))
                signals[pred_series > pred_series.quantile(buy_thresh)] = 1
                signals[pred_series < pred_series.quantile(sell_thresh)] = -1
                
                # Calculate performance
                strategy_returns = signals[:-1] * y_val.values[1:]
                
                if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                    total_return = np.prod(1 + strategy_returns) - 1
                    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(365*24)
                    win_rate = np.mean(strategy_returns > 0)
                    
                    results[(buy_thresh, sell_thresh)] = {
                        'total_return': total_return,
                        'sharpe_ratio': sharpe,
                        'win_rate': win_rate
                    }
                    
                    print(f"      Thresholds ({buy_thresh}, {sell_thresh}): Return={total_return:.2%}, Sharpe={sharpe:.2f}")
                
            except Exception as e:
                print(f"      Failed for thresholds ({buy_thresh}, {sell_thresh}): {e}")
        
        # Find best thresholds based on Sharpe ratio
        best_thresholds = max(results, key=lambda x: results[x]['sharpe_ratio']) if results else (0.7, 0.3)
        
        print(f"   ✅ Optimal thresholds: {best_thresholds}")
        print(f"      Performance: {results.get(best_thresholds, {})}")
        
        self.optimization_results['strategy'] = {
            'results': results,
            'best_thresholds': best_thresholds
        }
        
        return best_thresholds
    
    def _generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("   📊 Creating optimization report...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cryptocurrency Random Forest - Optimization Results', fontsize=16)
        
        # Plot 1: Hyperparameter optimization
        ax1 = axes[0, 0]
        if 'hyperparameters' in self.optimization_results:
            study = self.optimization_results['hyperparameters']['study']
            trials = study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            ax1.plot(range(len(values)), values, 'b-', alpha=0.7)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.set_title('Hyperparameter Optimization Progress')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature count optimization
        ax2 = axes[0, 1]
        if 'features' in self.optimization_results:
            feature_results = self.optimization_results['features']['results']
            counts = list(feature_results.keys())
            scores = list(feature_results.values())
            ax2.bar(range(len(counts)), scores)
            ax2.set_xticks(range(len(counts)))
            ax2.set_xticklabels(counts, rotation=45)
            ax2.set_ylabel('Validation R²')
            ax2.set_title('Feature Count Optimization')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Target optimization
        ax3 = axes[1, 0]
        if 'targets' in self.optimization_results:
            target_results = self.optimization_results['targets']['results']
            targets = list(target_results.keys())
            scores = list(target_results.values())
            ax3.bar(range(len(targets)), scores)
            ax3.set_xticks(range(len(targets)))
            ax3.set_xticklabels(targets, rotation=45)
            ax3.set_ylabel('Validation R²')
            ax3.set_title('Target Variable Optimization')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Strategy optimization
        ax4 = axes[1, 1]
        if 'strategy' in self.optimization_results:
            strategy_results = self.optimization_results['strategy']['results']
            thresholds = [f"{k[0]:.1f},{k[1]:.1f}" for k in strategy_results.keys()]
            sharpe_ratios = [v['sharpe_ratio'] for v in strategy_results.values()]
            ax4.bar(range(len(thresholds)), sharpe_ratios)
            ax4.set_xticks(range(len(thresholds)))
            ax4.set_xticklabels(thresholds, rotation=45)
            ax4.set_ylabel('Sharpe Ratio')
            ax4.set_title('Trading Strategy Optimization')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_filename = f"optimization_results_{timestamp}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved to {viz_filename}")
        
        # Save optimization results
        results_filename = f"optimization_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, value in self.optimization_results.items():
            if key == 'hyperparameters' and 'study' in value:
                serializable_results[key] = {
                    'best_params': value['best_params'],
                    'best_score': value['best_score'],
                    'n_trials': len(value['study'].trials)
                }
            else:
                serializable_results[key] = value
        
        with open(results_filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {results_filename}")
        
        # Print summary
        print("\n" + "="*80)
        print("🏆 OPTIMIZATION COMPLETE - SUMMARY")
        print("="*80)
        
        if 'hyperparameters' in self.optimization_results:
            best_params = self.optimization_results['hyperparameters']['best_params']
            best_score = self.optimization_results['hyperparameters']['best_score']
            print(f"\n🔧 Best Hyperparameters (Sharpe: {best_score:.3f}):")
            for param, value in best_params.items():
                print(f"   • {param}: {value}")
        
        if 'features' in self.optimization_results:
            best_features = self.optimization_results['features']['best_count']
            print(f"\n🎯 Optimal Feature Count: {best_features}")
        
        if 'targets' in self.optimization_results:
            best_target = self.optimization_results['targets']['best_target']
            print(f"\n📈 Optimal Target: {best_target}")
        
        if 'ensemble' in self.optimization_results:
            best_ensemble = self.optimization_results['ensemble']['best_size']
            print(f"\n🤖 Optimal Ensemble Size: {best_ensemble}")
        
        if 'strategy' in self.optimization_results:
            best_thresholds = self.optimization_results['strategy']['best_thresholds']
            print(f"\n💼 Optimal Strategy Thresholds: {best_thresholds}")
        
        print(f"\n📊 Files Generated:")
        print(f"   • {viz_filename}")
        print(f"   • {results_filename}")


async def main():
    """Run the optimization suite."""
    optimizer = CryptoOptimizer()
    results = await optimizer.run_full_optimization()
    
    print("\n✨ Optimization suite completed successfully!")
    return results


if __name__ == "__main__":
    asyncio.run(main())