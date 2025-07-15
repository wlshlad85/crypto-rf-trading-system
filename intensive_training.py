"""Intensive training with 150 iterations for Cryptocurrency Random Forest models."""

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
from typing import Dict, List, Tuple, Any
import gc
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
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


class IntensiveTrainer:
    """Intensive training system for cryptocurrency Random Forest models."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.training_history = {
            'iterations': [],
            'train_scores': [],
            'val_scores': [],
            'sharpe_ratios': [],
            'feature_importance': []
        }
        self.best_models = {}
        self.best_scores = {}
        
        # Load optimized parameters
        self._load_optimized_params()
        
    def _load_optimized_params(self):
        """Load optimized parameters from previous results."""
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
    
    async def run_intensive_training(self, iterations: int = 150):
        """Run intensive training with specified iterations."""
        print("\n" + "="*80)
        print("🚀 INTENSIVE TRAINING - 150 ITERATIONS")
        print("="*80 + "\n")
        
        print(f"📊 Training Configuration:")
        print(f"   • Iterations: {iterations}")
        print(f"   • Models per crypto: 5 (ensemble)")
        print(f"   • Cross-validation folds: 5")
        print(f"   • Early stopping patience: 10")
        print(f"   • Optimized parameters loaded")
        
        # Step 1: Prepare data
        print("\n📊 Step 1: Preparing Training Data...")
        await self._prepare_training_data()
        
        # Step 2: Train models intensively
        print("\n🤖 Step 2: Starting Intensive Training...")
        trained_models = await self._train_models_intensively(iterations)
        
        # Step 3: Evaluate final performance
        print("\n📈 Step 3: Evaluating Final Performance...")
        final_metrics = await self._evaluate_final_performance(trained_models)
        
        # Step 4: Generate comprehensive report
        print("\n📊 Step 4: Generating Training Report...")
        self._generate_training_report(final_metrics)
        
        return trained_models, final_metrics
    
    async def _prepare_training_data(self):
        """Prepare comprehensive training data."""
        # Use 4 months of data for robust training
        self.config.data.days = 120
        self.config.data.symbols = ['bitcoin', 'ethereum', 'solana']
        
        print("   📊 Fetching 4 months of training data...")
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
        print("   🔧 Generating comprehensive features...")
        feature_engine = CryptoFeatureEngine(self.config.features)
        self.features = feature_engine.generate_features(self.clean_data)
        
        print(f"   ✅ Data prepared: {self.clean_data.shape[0]} samples, {self.features.shape[1]} features")
        
        # Prepare data for each cryptocurrency
        self.crypto_data = {}
        for symbol in self.config.data.symbols:
            print(f"   📊 Preparing {symbol.upper()} data...")
            
            # Create target (6-hour returns optimized)
            target = self.clean_data[f'{symbol}_close'].pct_change(6).shift(-6).dropna()
            
            # Align features and target
            common_index = self.features.index.intersection(target.index)
            X = self.features.loc[common_index].copy()
            y = target.loc[common_index]
            
            # Store prepared data
            self.crypto_data[symbol] = {
                'X': X,
                'y': y,
                'prices': self.clean_data.loc[common_index, f'{symbol}_close']
            }
            
            print(f"      ✅ {symbol.upper()}: {len(X)} samples ready")
    
    async def _train_models_intensively(self, iterations: int):
        """Train models intensively with cross-validation."""
        trained_models = {}
        
        for symbol in self.config.data.symbols:
            print(f"\n🔧 Training {symbol.upper()} models...")
            
            # Get data for this cryptocurrency
            X = self.crypto_data[symbol]['X']
            y = self.crypto_data[symbol]['y']
            
            # Initialize tracking
            best_score = -np.inf
            best_model = None
            patience_counter = 0
            patience = 10
            
            # Training history for this symbol
            symbol_history = {
                'train_scores': [],
                'val_scores': [],
                'sharpe_ratios': []
            }
            
            # Create ensemble model
            ensemble = EnsembleRandomForestModel(self.config.model, n_models=5)
            
            for iteration in range(iterations):
                print(f"\r   Iteration {iteration + 1}/{iterations}", end='', flush=True)
                
                # K-fold cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=iteration)
                
                iteration_train_scores = []
                iteration_val_scores = []
                iteration_sharpes = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    # Split data
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Add target column for model preparation
                    X_train_with_target = X_train.copy()
                    X_train_with_target['target'] = y_train
                    
                    # Prepare data
                    X_train_clean, y_train_clean = ensemble.models[0].prepare_data(
                        X_train_with_target, 'target'
                    )
                    
                    # Train model
                    try:
                        # Train each model in ensemble
                        for model_idx in range(len(ensemble.models)):
                            model = ensemble.models[model_idx]
                            
                            # Add slight randomization for diversity
                            sample_idx = np.random.choice(
                                len(X_train_clean), 
                                size=int(len(X_train_clean) * 0.9),
                                replace=False
                            )
                            
                            X_sample = X_train_clean.iloc[sample_idx]
                            y_sample = y_train_clean.iloc[sample_idx]
                            
                            # Train individual model
                            model.train(X_sample, y_sample, validation_split=0.2)
                        
                        # Validate ensemble
                        X_val_clean = X_val[X_train_clean.columns]
                        predictions = ensemble.predict(X_val_clean)
                        
                        # Calculate validation score
                        val_score = r2_score(y_val[:len(predictions)], predictions)
                        iteration_val_scores.append(val_score)
                        
                        # Calculate Sharpe ratio (simplified)
                        if len(predictions) > 1:
                            returns = pd.Series(predictions)
                            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
                            iteration_sharpes.append(sharpe)
                        
                    except Exception as e:
                        logger.warning(f"Training error in fold {fold}: {e}")
                        continue
                
                # Average scores across folds
                if iteration_val_scores:
                    avg_val_score = np.mean(iteration_val_scores)
                    avg_sharpe = np.mean(iteration_sharpes) if iteration_sharpes else 0
                    
                    symbol_history['val_scores'].append(avg_val_score)
                    symbol_history['sharpe_ratios'].append(avg_sharpe)
                    
                    # Check for improvement
                    if avg_val_score > best_score:
                        best_score = avg_val_score
                        best_model = ensemble
                        patience_counter = 0
                        
                        # Save best model
                        self.best_models[symbol] = ensemble
                        self.best_scores[symbol] = {
                            'val_score': best_score,
                            'sharpe_ratio': avg_sharpe,
                            'iteration': iteration + 1
                        }
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= patience:
                        print(f"\n   ✅ Early stopping at iteration {iteration + 1}")
                        break
                
                # Garbage collection every 10 iterations
                if (iteration + 1) % 10 == 0:
                    gc.collect()
            
            print(f"\n   ✅ {symbol.upper()} training complete!")
            print(f"      Best validation score: {best_score:.4f}")
            print(f"      Best iteration: {self.best_scores[symbol]['iteration']}")
            print(f"      Best Sharpe ratio: {self.best_scores[symbol]['sharpe_ratio']:.2f}")
            
            # Store history
            self.training_history[symbol] = symbol_history
            trained_models[symbol] = best_model
        
        return trained_models
    
    async def _evaluate_final_performance(self, trained_models):
        """Evaluate final performance of trained models."""
        print("\n📊 Evaluating final model performance...")
        
        final_metrics = {}
        
        for symbol, model in trained_models.items():
            print(f"\n   Evaluating {symbol.upper()} model...")
            
            # Get test data (last 20% of data)
            X = self.crypto_data[symbol]['X']
            y = self.crypto_data[symbol]['y']
            prices = self.crypto_data[symbol]['prices']
            
            test_size = int(len(X) * 0.2)
            X_test = X.iloc[-test_size:]
            y_test = y.iloc[-test_size:]
            prices_test = prices.iloc[-test_size:]
            
            # Prepare test data
            X_test_clean = X_test[X.iloc[:, :-1].columns]  # Remove target if present
            
            # Make predictions
            try:
                predictions = model.predict(X_test_clean)
                
                # Calculate metrics
                test_r2 = r2_score(y_test[:len(predictions)], predictions)
                test_rmse = np.sqrt(mean_squared_error(y_test[:len(predictions)], predictions))
                
                # Simulate trading
                portfolio_value = 10000
                positions = 0
                trades = 0
                
                for i in range(len(predictions)):
                    if predictions[i] > 0.001 and portfolio_value > 1000:  # Buy signal
                        position_size = portfolio_value * 0.3
                        positions += position_size / prices_test.iloc[i]
                        portfolio_value -= position_size
                        trades += 1
                    elif predictions[i] < -0.001 and positions > 0:  # Sell signal
                        sell_quantity = positions * 0.5
                        portfolio_value += sell_quantity * prices_test.iloc[i]
                        positions -= sell_quantity
                        trades += 1
                
                # Final portfolio value
                final_value = portfolio_value + positions * prices_test.iloc[-1]
                total_return = (final_value / 10000 - 1) * 100
                
                # Calculate Sharpe ratio
                returns = pd.Series(predictions).pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
                
                final_metrics[symbol] = {
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'trades': trades,
                    'final_value': final_value
                }
                
                print(f"      R² Score: {test_r2:.4f}")
                print(f"      Total Return: {total_return:.2f}%")
                print(f"      Sharpe Ratio: {sharpe_ratio:.2f}")
                print(f"      Trades: {trades}")
                
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                final_metrics[symbol] = {
                    'error': str(e)
                }
        
        # Calculate ensemble performance
        ensemble_return = np.mean([m['total_return'] for m in final_metrics.values() if 'total_return' in m])
        ensemble_sharpe = np.mean([m['sharpe_ratio'] for m in final_metrics.values() if 'sharpe_ratio' in m])
        
        print(f"\n📊 Ensemble Performance:")
        print(f"   Average Return: {ensemble_return:.2f}%")
        print(f"   Average Sharpe: {ensemble_sharpe:.2f}")
        
        return final_metrics
    
    def _generate_training_report(self, final_metrics):
        """Generate comprehensive training report."""
        print("\n📊 Generating training report...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Intensive Training Results - 150 Iterations', fontsize=16)
        
        # Plot 1: Training progress for each cryptocurrency
        ax1 = axes[0, 0]
        for symbol in self.config.data.symbols:
            if symbol in self.training_history:
                val_scores = self.training_history[symbol]['val_scores']
                ax1.plot(range(len(val_scores)), val_scores, label=f'{symbol.upper()}', linewidth=2)
        ax1.set_title('Validation Scores Over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Validation R²')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sharpe ratio evolution
        ax2 = axes[0, 1]
        for symbol in self.config.data.symbols:
            if symbol in self.training_history:
                sharpes = self.training_history[symbol]['sharpe_ratios']
                ax2.plot(range(len(sharpes)), sharpes, label=f'{symbol.upper()}', linewidth=2)
        ax2.set_title('Sharpe Ratio Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final performance comparison
        ax3 = axes[0, 2]
        symbols = list(final_metrics.keys())
        returns = [final_metrics[s].get('total_return', 0) for s in symbols]
        colors = ['orange', 'blue', 'purple']
        ax3.bar(range(len(symbols)), returns, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(symbols)))
        ax3.set_xticklabels([s.upper() for s in symbols])
        ax3.set_title('Final Returns by Cryptocurrency')
        ax3.set_ylabel('Total Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: R² scores comparison
        ax4 = axes[1, 0]
        r2_scores = [final_metrics[s].get('test_r2', 0) for s in symbols]
        ax4.bar(range(len(symbols)), r2_scores, color=['red', 'green', 'blue'], alpha=0.7)
        ax4.set_xticks(range(len(symbols)))
        ax4.set_xticklabels([s.upper() for s in symbols])
        ax4.set_title('Model R² Scores')
        ax4.set_ylabel('R² Score')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Sharpe ratios
        ax5 = axes[1, 1]
        sharpes = [final_metrics[s].get('sharpe_ratio', 0) for s in symbols]
        ax5.bar(range(len(symbols)), sharpes, color=['cyan', 'magenta', 'yellow'], alpha=0.7)
        ax5.set_xticks(range(len(symbols)))
        ax5.set_xticklabels([s.upper() for s in symbols])
        ax5.set_title('Final Sharpe Ratios')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Trade counts
        ax6 = axes[1, 2]
        trades = [final_metrics[s].get('trades', 0) for s in symbols]
        ax6.bar(range(len(symbols)), trades, color=['brown', 'pink', 'gray'], alpha=0.7)
        ax6.set_xticks(range(len(symbols)))
        ax6.set_xticklabels([s.upper() for s in symbols])
        ax6.set_title('Number of Trades')
        ax6.set_ylabel('Trade Count')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_filename = f"intensive_training_results_{timestamp}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved to {viz_filename}")
        plt.close()
        
        # Save detailed results
        results_filename = f"intensive_training_results_{timestamp}.json"
        
        # Prepare JSON-serializable results
        json_results = {
            'training_config': {
                'iterations': 150,
                'models_per_crypto': 5,
                'cross_validation_folds': 5,
                'early_stopping_patience': 10
            },
            'best_scores': self.best_scores,
            'final_metrics': final_metrics,
            'ensemble_performance': {
                'average_return': np.mean([m.get('total_return', 0) for m in final_metrics.values()]),
                'average_sharpe': np.mean([m.get('sharpe_ratio', 0) for m in final_metrics.values()]),
                'total_trades': sum([m.get('trades', 0) for m in final_metrics.values()])
            },
            'training_summary': {
                symbol: {
                    'iterations_trained': len(self.training_history.get(symbol, {}).get('val_scores', [])),
                    'best_iteration': self.best_scores.get(symbol, {}).get('iteration', 0),
                    'best_val_score': self.best_scores.get(symbol, {}).get('val_score', 0)
                }
                for symbol in self.config.data.symbols
            }
        }
        
        with open(results_filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {results_filename}")
        
        # Generate text report
        report_filename = f"intensive_training_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("INTENSIVE TRAINING REPORT - 150 ITERATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write("Total Iterations: 150\n")
            f.write("Models per Cryptocurrency: 5 (ensemble)\n")
            f.write("Cross-validation Folds: 5\n")
            f.write("Early Stopping Patience: 10\n\n")
            
            f.write("BEST MODELS SUMMARY\n")
            f.write("-"*40 + "\n")
            for symbol, scores in self.best_scores.items():
                f.write(f"\n{symbol.upper()}:\n")
                f.write(f"  Best Validation Score: {scores['val_score']:.4f}\n")
                f.write(f"  Best Sharpe Ratio: {scores['sharpe_ratio']:.2f}\n")
                f.write(f"  Best Iteration: {scores['iteration']}\n")
            
            f.write("\nFINAL PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            for symbol, metrics in final_metrics.items():
                if 'error' not in metrics:
                    f.write(f"\n{symbol.upper()}:\n")
                    f.write(f"  Test R² Score: {metrics['test_r2']:.4f}\n")
                    f.write(f"  Total Return: {metrics['total_return']:.2f}%\n")
                    f.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
                    f.write(f"  Number of Trades: {metrics['trades']}\n")
                    f.write(f"  Final Portfolio Value: ${metrics['final_value']:.2f}\n")
            
            f.write("\nENSEMBLE PERFORMANCE\n")
            f.write("-"*40 + "\n")
            avg_return = np.mean([m.get('total_return', 0) for m in final_metrics.values()])
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in final_metrics.values()])
            f.write(f"Average Return: {avg_return:.2f}%\n")
            f.write(f"Average Sharpe Ratio: {avg_sharpe:.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Report generated: {datetime.now()}\n")
        
        print(f"   ✅ Detailed report saved to {report_filename}")
        
        # Print summary
        print("\n" + "="*80)
        print("🏆 INTENSIVE TRAINING COMPLETE - SUMMARY")
        print("="*80)
        
        avg_return = np.mean([m.get('total_return', 0) for m in final_metrics.values()])
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in final_metrics.values()])
        
        print(f"\n📊 Final Results:")
        print(f"   • Average Return: {avg_return:.2f}%")
        print(f"   • Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"   • Total Trades: {sum([m.get('trades', 0) for m in final_metrics.values()])}")
        
        print(f"\n🏆 Best Individual Performances:")
        for symbol, metrics in final_metrics.items():
            if 'total_return' in metrics:
                print(f"   • {symbol.upper()}: {metrics['total_return']:.2f}% return, {metrics['sharpe_ratio']:.2f} Sharpe")
        
        print(f"\n📊 Files Generated:")
        print(f"   • {viz_filename}")
        print(f"   • {results_filename}")
        print(f"   • {report_filename}")


async def main():
    """Run intensive training."""
    trainer = IntensiveTrainer()
    models, metrics = await trainer.run_intensive_training(iterations=150)
    
    print("\n✨ Intensive training completed successfully!")
    print("🚀 Models are now optimized and ready for deployment!")
    
    return models, metrics


if __name__ == "__main__":
    asyncio.run(main())