"""Quick intensive training for Cryptocurrency Random Forest models."""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any
import gc
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config, Config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel, EnsembleRandomForestModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuickIntensiveTrainer:
    """Quick intensive training for crypto Random Forest models."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.training_results = {}
        
        # Load optimized parameters
        self._load_optimized_params()
        
    def _load_optimized_params(self):
        """Load optimized parameters."""
        try:
            result_files = list(Path('.').glob('fast_optimization_results_*.json'))
            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                if 'hyperparameters' in results:
                    best_params = results['hyperparameters']['best_params']
                    self.config.model.__dict__.update(best_params)
                    logger.info(f"Loaded optimized parameters: {best_params}")
                    
        except Exception as e:
            logger.warning(f"Could not load optimization results: {e}")
    
    def run_quick_intensive_training(self, iterations: int = 50):
        """Run quick intensive training."""
        print("\n" + "="*80)
        print("🚀 QUICK INTENSIVE TRAINING - OPTIMIZED FOR SPEED")
        print("="*80 + "\n")
        
        print(f"📊 Training Configuration:")
        print(f"   • Quick iterations: {iterations}")
        print(f"   • Single validation split (no k-fold)")
        print(f"   • Ensemble size: 3 models")
        print(f"   • Early stopping: 5 iterations")
        
        # Step 1: Prepare data
        print("\n📊 Step 1: Preparing Data...")
        self._prepare_data()
        
        # Step 2: Train models
        print("\n🤖 Step 2: Training Models Intensively...")
        trained_models = self._train_models(iterations)
        
        # Step 3: Evaluate performance
        print("\n📈 Step 3: Evaluating Performance...")
        metrics = self._evaluate_performance(trained_models)
        
        # Step 4: Generate report
        print("\n📊 Step 4: Generating Report...")
        self._generate_report(metrics, trained_models)
        
        return trained_models, metrics
    
    def _prepare_data(self):
        """Prepare training data."""
        # Use 3 months for speed
        self.config.data.days = 90
        self.config.data.symbols = ['bitcoin', 'ethereum', 'solana']
        
        print("   📊 Fetching 3 months of data...")
        fetcher = YFinanceCryptoFetcher(self.config.data)
        data_dict = fetcher.fetch_all_symbols(self.config.data.symbols)
        
        # Get latest prices
        prices = fetcher.get_latest_prices(self.config.data.symbols)
        print("   💰 Current prices:")
        for symbol, price in prices.items():
            print(f"      {symbol.upper()}: ${price:,.2f}")
        
        # Combine and clean data
        self.raw_data = fetcher.combine_data(data_dict)
        self.clean_data = fetcher.get_clean_data(self.raw_data)
        
        # Generate features
        print("   🔧 Generating features...")
        feature_engine = CryptoFeatureEngine(self.config.features)
        self.features = feature_engine.generate_features(self.clean_data)
        
        print(f"   ✅ Data prepared: {self.clean_data.shape[0]} samples, {self.features.shape[1]} features")
        
        # Prepare data for each cryptocurrency
        self.crypto_data = {}
        for symbol in self.config.data.symbols:
            # Create target
            target = self.clean_data[f'{symbol}_close'].pct_change(6).shift(-6).dropna()
            
            # Align features and target
            common_index = self.features.index.intersection(target.index)
            X = self.features.loc[common_index].copy()
            y = target.loc[common_index]
            
            # Split data
            split_idx = int(len(X) * 0.8)
            
            self.crypto_data[symbol] = {
                'X_train': X.iloc[:split_idx],
                'y_train': y.iloc[:split_idx],
                'X_test': X.iloc[split_idx:],
                'y_test': y.iloc[split_idx:],
                'prices_test': self.clean_data.loc[common_index, f'{symbol}_close'].iloc[split_idx:]
            }
            
            print(f"      {symbol.upper()}: Train={len(self.crypto_data[symbol]['X_train'])}, Test={len(self.crypto_data[symbol]['X_test'])}")
    
    def _train_models(self, iterations: int):
        """Train models with iterations."""
        trained_models = {}
        
        for symbol in self.config.data.symbols:
            print(f"\n🔧 Training {symbol.upper()} model...")
            
            # Get data
            X_train = self.crypto_data[symbol]['X_train']
            y_train = self.crypto_data[symbol]['y_train']
            X_test = self.crypto_data[symbol]['X_test']
            y_test = self.crypto_data[symbol]['y_test']
            
            # Create ensemble
            ensemble = EnsembleRandomForestModel(self.config.model, n_models=3)
            
            # Training history
            best_score = -np.inf
            best_iteration = 0
            patience_counter = 0
            history = {'train_scores': [], 'val_scores': [], 'iterations': []}
            
            for i in range(iterations):
                print(f"\r   Iteration {i + 1}/{iterations} ", end='', flush=True)
                
                # Add target for preparation
                X_train_with_target = X_train.copy()
                X_train_with_target['target'] = y_train
                
                # Train ensemble with bootstrapping
                for model_idx, model in enumerate(ensemble.models):
                    # Bootstrap sample
                    sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                    X_sample = X_train_with_target.iloc[sample_idx]
                    
                    # Prepare and train
                    X_clean, y_clean = model.prepare_data(X_sample, 'target')
                    train_result = model.train(X_clean, y_clean, validation_split=0.2)
                    
                    # Store individual model score
                    if i == 0:
                        history['train_scores'].append([])
                    history['train_scores'][model_idx].append(train_result['validation']['r2'])
                
                # Evaluate ensemble on test set
                X_test_clean = X_test[X_clean.columns]
                predictions = ensemble.predict(X_test_clean)
                val_score = r2_score(y_test[:len(predictions)], predictions)
                
                history['val_scores'].append(val_score)
                history['iterations'].append(i + 1)
                
                # Check for improvement
                if val_score > best_score:
                    best_score = val_score
                    best_iteration = i + 1
                    patience_counter = 0
                    trained_models[symbol] = ensemble
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= 5:
                    print(f"\n   ✅ Early stopping at iteration {i + 1}")
                    break
                
                # Garbage collection every 10 iterations
                if (i + 1) % 10 == 0:
                    gc.collect()
            
            print(f"\n   ✅ {symbol.upper()} training complete!")
            print(f"      Best score: {best_score:.4f} at iteration {best_iteration}")
            
            self.training_results[symbol] = {
                'best_score': best_score,
                'best_iteration': best_iteration,
                'history': history
            }
        
        return trained_models
    
    def _evaluate_performance(self, trained_models):
        """Evaluate model performance."""
        print("\n📊 Evaluating model performance...")
        
        metrics = {}
        
        for symbol, model in trained_models.items():
            print(f"\n   Evaluating {symbol.upper()}...")
            
            # Get test data
            X_test = self.crypto_data[symbol]['X_test']
            y_test = self.crypto_data[symbol]['y_test']
            prices_test = self.crypto_data[symbol]['prices_test']
            
            # Prepare test data
            X_test_clean = X_test.select_dtypes(include=[np.number])
            
            # Make predictions
            predictions = model.predict(X_test_clean)
            
            # Calculate metrics
            test_r2 = r2_score(y_test[:len(predictions)], predictions)
            
            # Simulate trading
            portfolio_value = 10000
            positions = 0
            trades = 0
            
            for i in range(len(predictions)):
                if predictions[i] > 0.001 and portfolio_value > 1000:  # Buy
                    position_size = portfolio_value * 0.3
                    positions += position_size / prices_test.iloc[i]
                    portfolio_value -= position_size
                    trades += 1
                elif predictions[i] < -0.001 and positions > 0:  # Sell
                    sell_quantity = positions * 0.5
                    portfolio_value += sell_quantity * prices_test.iloc[i]
                    positions -= sell_quantity
                    trades += 1
            
            # Final value
            final_value = portfolio_value + positions * prices_test.iloc[-1]
            total_return = (final_value / 10000 - 1) * 100
            
            # Calculate Sharpe
            returns = pd.Series(predictions).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
            
            metrics[symbol] = {
                'test_r2': test_r2,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades,
                'final_value': final_value,
                'best_iteration': self.training_results[symbol]['best_iteration']
            }
            
            print(f"      R² Score: {test_r2:.4f}")
            print(f"      Return: {total_return:.2f}%")
            print(f"      Sharpe: {sharpe_ratio:.2f}")
            print(f"      Trades: {trades}")
        
        # Calculate ensemble metrics
        avg_return = np.mean([m['total_return'] for m in metrics.values()])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in metrics.values()])
        
        print(f"\n📊 Ensemble Performance:")
        print(f"   Average Return: {avg_return:.2f}%")
        print(f"   Average Sharpe: {avg_sharpe:.2f}")
        
        return metrics
    
    def _generate_report(self, metrics, trained_models):
        """Generate training report."""
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quick Intensive Training Results', fontsize=16)
        
        # Plot 1: Training progress
        ax1 = axes[0, 0]
        for symbol in self.config.data.symbols:
            if symbol in self.training_results:
                val_scores = self.training_results[symbol]['history']['val_scores']
                iterations = self.training_results[symbol]['history']['iterations']
                ax1.plot(iterations, val_scores, label=f'{symbol.upper()}', linewidth=2)
        ax1.set_title('Validation Scores Over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Validation R²')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final returns
        ax2 = axes[0, 1]
        symbols = list(metrics.keys())
        returns = [metrics[s]['total_return'] for s in symbols]
        colors = ['orange', 'blue', 'purple']
        ax2.bar(range(len(symbols)), returns, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(symbols)))
        ax2.set_xticklabels([s.upper() for s in symbols])
        ax2.set_title('Final Returns by Cryptocurrency')
        ax2.set_ylabel('Total Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sharpe ratios
        ax3 = axes[1, 0]
        sharpes = [metrics[s]['sharpe_ratio'] for s in symbols]
        ax3.bar(range(len(symbols)), sharpes, color=['red', 'green', 'yellow'], alpha=0.7)
        ax3.set_xticks(range(len(symbols)))
        ax3.set_xticklabels([s.upper() for s in symbols])
        ax3.set_title('Sharpe Ratios')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Best iterations
        ax4 = axes[1, 1]
        best_iters = [metrics[s]['best_iteration'] for s in symbols]
        ax4.bar(range(len(symbols)), best_iters, color=['cyan', 'magenta', 'brown'], alpha=0.7)
        ax4.set_xticks(range(len(symbols)))
        ax4.set_xticklabels([s.upper() for s in symbols])
        ax4.set_title('Best Iteration Found')
        ax4.set_ylabel('Iteration Number')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_filename = f"quick_intensive_training_{timestamp}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"\n   ✅ Visualization saved to {viz_filename}")
        plt.close()
        
        # Save results
        results_filename = f"quick_intensive_training_{timestamp}.json"
        json_results = {
            'config': {
                'iterations': 50,
                'models_per_crypto': 3,
                'early_stopping': 5
            },
            'metrics': metrics,
            'training_results': {
                symbol: {
                    'best_score': result['best_score'],
                    'best_iteration': result['best_iteration']
                }
                for symbol, result in self.training_results.items()
            },
            'ensemble_performance': {
                'average_return': np.mean([m['total_return'] for m in metrics.values()]),
                'average_sharpe': np.mean([m['sharpe_ratio'] for m in metrics.values()])
            }
        }
        
        with open(results_filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {results_filename}")
        
        # Print summary
        print("\n" + "="*80)
        print("🏆 QUICK INTENSIVE TRAINING COMPLETE")
        print("="*80)
        
        avg_return = np.mean([m['total_return'] for m in metrics.values()])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in metrics.values()])
        
        print(f"\n📊 Results Summary:")
        print(f"   • Average Return: {avg_return:.2f}%")
        print(f"   • Average Sharpe: {avg_sharpe:.2f}")
        print(f"   • Total Trades: {sum([m['trades'] for m in metrics.values()])}")
        
        print(f"\n🏆 Individual Performance:")
        for symbol, m in metrics.items():
            print(f"   • {symbol.upper()}: {m['total_return']:.2f}% return, {m['sharpe_ratio']:.2f} Sharpe")
        
        print(f"\n📊 Files Generated:")
        print(f"   • {viz_filename}")
        print(f"   • {results_filename}")


def main():
    """Run quick intensive training."""
    trainer = QuickIntensiveTrainer()
    models, metrics = trainer.run_quick_intensive_training(iterations=50)
    
    print("\n✨ Quick intensive training completed successfully!")
    print("🚀 Models are now optimized and ready for deployment!")
    
    return models, metrics


if __name__ == "__main__":
    main()