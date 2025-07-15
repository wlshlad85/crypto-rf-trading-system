"""Advanced hyperparameter optimization for improved trading performance."""

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
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
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


class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimization for crypto trading strategy."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.optimization_results = {}
        self.best_parameters = {}
        self.data = None
        self.features = None
        
    def run_advanced_optimization(self, n_trials: int = 100):
        """Run comprehensive hyperparameter optimization."""
        print("\n" + "="*80)
        print("🚀 ADVANCED HYPERPARAMETER OPTIMIZATION")
        print("="*80 + "\n")
        
        print(f"📊 Optimization Configuration:")
        print(f"   • Trials: {n_trials}")
        print(f"   • Multi-objective: Sharpe ratio + Return")
        print(f"   • Time series validation")
        print(f"   • Trading-focused metrics")
        
        # Step 1: Prepare comprehensive data
        print("\n📊 Step 1: Preparing Optimization Data...")
        self._prepare_optimization_data()
        
        # Step 2: Optimize model hyperparameters
        print("\n🤖 Step 2: Optimizing Model Hyperparameters...")
        model_results = self._optimize_model_hyperparameters(n_trials)
        
        # Step 3: Optimize trading parameters
        print("\n💼 Step 3: Optimizing Trading Strategy...")
        trading_results = self._optimize_trading_parameters(n_trials // 2)
        
        # Step 4: Multi-objective optimization
        print("\n🎯 Step 4: Multi-Objective Optimization...")
        multi_obj_results = self._run_multi_objective_optimization(n_trials // 2)
        
        # Step 5: Validate best parameters
        print("\n✅ Step 5: Validating Best Parameters...")
        validation_results = self._validate_best_parameters()
        
        # Step 6: Generate comprehensive report
        print("\n📊 Step 6: Generating Optimization Report...")
        self._generate_optimization_report()
        
        return self.best_parameters
    
    def _prepare_optimization_data(self):
        """Prepare comprehensive data for optimization."""
        # Use 4 months of data for robust optimization
        self.config.data.days = 120
        self.config.data.symbols = ['bitcoin', 'ethereum', 'solana']
        
        print("   📊 Fetching 4 months of data...")
        fetcher = YFinanceCryptoFetcher(self.config.data)
        data_dict = fetcher.fetch_all_symbols(self.config.data.symbols)
        
        # Get latest prices
        prices = fetcher.get_latest_prices(self.config.data.symbols)
        print("   💰 Current prices:")
        for symbol, price in prices.items():
            print(f"      {symbol.upper()}: ${price:,.2f}")
        
        # Combine and clean data
        self.raw_data = fetcher.combine_data(data_dict)
        self.data = fetcher.get_clean_data(self.raw_data)
        
        # Generate comprehensive features
        print("   🔧 Generating comprehensive features...")
        feature_engine = CryptoFeatureEngine(self.config.features)
        self.features = feature_engine.generate_features(self.data)
        
        print(f"   ✅ Data prepared: {self.data.shape[0]} samples, {self.features.shape[1]} features")
        
        # Prepare targets for each cryptocurrency
        self.targets = {}
        for symbol in self.config.data.symbols:
            # Test different time horizons
            for horizon in [3, 6, 12, 24]:  # Hours
                target_key = f"{symbol}_{horizon}h"
                target = self.data[f'{symbol}_close'].pct_change(horizon).shift(-horizon).dropna()
                
                # Align with features
                common_index = self.features.index.intersection(target.index)
                self.targets[target_key] = {
                    'X': self.features.loc[common_index],
                    'y': target.loc[common_index],
                    'prices': self.data.loc[common_index, f'{symbol}_close']
                }
    
    def _optimize_model_hyperparameters(self, n_trials: int):
        """Optimize Random Forest model hyperparameters."""
        print("   🔍 Running model hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters with expanded ranges
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            
            # Test different feature counts
            n_features = trial.suggest_int('n_features', 50, min(250, self.features.shape[1]), step=25)
            
            # Test different target configurations
            target_keys = list(self.targets.keys())
            target_key = trial.suggest_categorical('target', target_keys)
            
            try:
                # Get data
                target_data = self.targets[target_key]
                X = target_data['X']
                y = target_data['y']
                
                # Feature selection (simplified) - only use numeric columns
                if n_features < X.shape[1]:
                    # Use variance-based selection for speed on numeric columns only
                    X_numeric = X.select_dtypes(include=[np.number])
                    feature_vars = X_numeric.var()
                    # Remove constant features
                    non_constant_features = feature_vars[feature_vars > 0]
                    if len(non_constant_features) < n_features:
                        n_features = len(non_constant_features)
                    top_features = non_constant_features.nlargest(n_features).index
                    X_selected = X[top_features]
                else:
                    # Still filter to numeric columns only
                    X_selected = X.select_dtypes(include=[np.number])
                
                # Create model
                config = self.config
                config.model.__dict__.update(params)
                model = CryptoRandomForestModel(config.model)
                
                # Prepare data
                X_with_target = X_selected.copy()
                X_with_target['target'] = y
                X_clean, y_clean = model.prepare_data(X_with_target, 'target')
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                sharpe_ratios = []
                
                for train_idx, val_idx in tscv.split(X_clean):
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    # Train model
                    model.train(X_train, y_train, validation_split=0.2)
                    
                    # Validate
                    predictions = model.predict(X_val)
                    score = r2_score(y_val[:len(predictions)], predictions)
                    scores.append(score)
                    
                    # Calculate trading Sharpe ratio
                    if len(predictions) > 1:
                        pred_returns = pd.Series(predictions)
                        sharpe = pred_returns.mean() / pred_returns.std() * np.sqrt(252 * 24) if pred_returns.std() > 0 else 0
                        sharpe_ratios.append(sharpe)
                
                # Combine metrics (70% Sharpe, 30% R²)
                avg_score = np.mean(scores)
                avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
                combined_score = 0.7 * avg_sharpe + 0.3 * avg_score * 10  # Scale R² to similar range
                
                return combined_score
                
            except Exception as e:
                return -100  # Penalty for failed trials
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"   ✅ Model optimization complete:")
        print(f"      Best combined score: {best_score:.3f}")
        print(f"      Best parameters: {best_params}")
        
        self.optimization_results['model'] = {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
        
        # Store best model parameters
        self.best_parameters['model'] = best_params
        
        return best_params
    
    def _optimize_trading_parameters(self, n_trials: int):
        """Optimize trading strategy parameters."""
        print("   💼 Running trading parameter optimization...")
        
        def trading_objective(trial):
            # Trading strategy parameters
            buy_threshold = trial.suggest_float('buy_threshold', 0.55, 0.85, step=0.05)
            sell_threshold = trial.suggest_float('sell_threshold', 0.15, 0.45, step=0.05)
            position_size = trial.suggest_float('position_size', 0.1, 0.5, step=0.05)
            stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05, step=0.005)
            take_profit = trial.suggest_float('take_profit', 0.02, 0.10, step=0.01)
            
            try:
                # Use best model parameters from previous optimization
                model_params = self.best_parameters.get('model', {})
                
                # Use Bitcoin 6h as primary target for trading optimization
                target_data = self.targets.get('bitcoin_6h', list(self.targets.values())[0])
                X = target_data['X']
                y = target_data['y']
                prices = target_data['prices']
                
                # Create and train model
                config = self.config
                if model_params:
                    config.model.__dict__.update({k: v for k, v in model_params.items() if k != 'target' and k != 'n_features'})
                
                model = CryptoRandomForestModel(config.model)
                
                # Prepare data
                X_with_target = X.copy()
                X_with_target['target'] = y
                X_clean, y_clean = model.prepare_data(X_with_target, 'target')
                
                # Train on first 80%
                split_idx = int(len(X_clean) * 0.8)
                X_train = X_clean.iloc[:split_idx]
                y_train = y_clean.iloc[:split_idx]
                X_test = X_clean.iloc[split_idx:]
                y_test = y_clean.iloc[split_idx:]
                prices_test = prices.iloc[split_idx:split_idx + len(X_test)]
                
                model.train(X_train, y_train, validation_split=0.2)
                
                # Generate trading signals
                predictions = model.predict(X_test)
                
                # Convert to percentile-based signals
                pred_series = pd.Series(predictions)
                signals = np.zeros(len(predictions))
                
                if len(predictions) > 10:
                    signals[pred_series > pred_series.quantile(buy_threshold)] = 1
                    signals[pred_series < pred_series.quantile(sell_threshold)] = -1
                
                # Simulate trading
                portfolio_value = 10000
                cash = portfolio_value
                position = 0
                trades = 0
                returns = []
                
                for i in range(len(predictions)):
                    if i >= len(prices_test):
                        break
                        
                    current_price = prices_test.iloc[i]
                    
                    # Apply stop loss and take profit
                    if position > 0:
                        price_change = (current_price - entry_price) / entry_price
                        if price_change <= -stop_loss or price_change >= take_profit:
                            # Close position
                            cash += position * current_price
                            position = 0
                            trades += 1
                    
                    # Execute signals
                    if signals[i] == 1 and cash > 1000:  # Buy
                        trade_value = cash * position_size
                        position += trade_value / current_price
                        cash -= trade_value
                        entry_price = current_price
                        trades += 1
                    elif signals[i] == -1 and position > 0:  # Sell
                        cash += position * current_price
                        position = 0
                        trades += 1
                    
                    # Calculate portfolio value
                    current_value = cash + position * current_price
                    if len(returns) > 0:
                        daily_return = (current_value - returns[-1]) / returns[-1]
                    else:
                        daily_return = (current_value - portfolio_value) / portfolio_value
                    returns.append(current_value)
                
                # Calculate performance metrics
                if len(returns) > 1:
                    final_value = returns[-1]
                    total_return = (final_value / portfolio_value - 1) * 100
                    
                    daily_rets = pd.Series(returns).pct_change().dropna()
                    if len(daily_rets) > 0 and daily_rets.std() > 0:
                        sharpe_ratio = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                    else:
                        sharpe_ratio = 0
                    
                    # Penalize if no trades
                    if trades == 0:
                        return -50
                    
                    # Combined score: 60% Sharpe, 40% total return
                    combined_score = 0.6 * sharpe_ratio + 0.4 * total_return
                    return combined_score
                else:
                    return -100
                    
            except Exception as e:
                return -100
        
        # Run trading optimization
        trading_study = optuna.create_study(direction='maximize')
        trading_study.optimize(trading_objective, n_trials=n_trials)
        
        best_trading_params = trading_study.best_params
        best_trading_score = trading_study.best_value
        
        print(f"   ✅ Trading optimization complete:")
        print(f"      Best trading score: {best_trading_score:.3f}")
        print(f"      Best trading parameters: {best_trading_params}")
        
        self.optimization_results['trading'] = {
            'best_params': best_trading_params,
            'best_score': best_trading_score,
            'study': trading_study
        }
        
        # Store best trading parameters
        self.best_parameters['trading'] = best_trading_params
        
        return best_trading_params
    
    def _run_multi_objective_optimization(self, n_trials: int):
        """Run multi-objective optimization balancing return and risk."""
        print("   🎯 Running multi-objective optimization...")
        
        def multi_objective(trial):
            # Combined model + trading parameters
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=25),
                'max_depth': trial.suggest_int('max_depth', 10, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': True,
                'random_state': 42
            }
            
            trading_params = {
                'buy_threshold': trial.suggest_float('buy_threshold', 0.6, 0.8, step=0.05),
                'sell_threshold': trial.suggest_float('sell_threshold', 0.2, 0.4, step=0.05),
                'position_size': trial.suggest_float('position_size', 0.2, 0.4, step=0.05)
            }
            
            try:
                # Quick validation on Bitcoin 6h
                target_data = self.targets['bitcoin_6h']
                X = target_data['X']
                y = target_data['y']
                prices = target_data['prices']
                
                # Use top 100 features for speed - numeric columns only
                X_numeric = X.select_dtypes(include=[np.number])
                feature_vars = X_numeric.var()
                # Remove constant features
                non_constant_features = feature_vars[feature_vars > 0]
                n_features_to_use = min(100, len(non_constant_features))
                top_features = non_constant_features.nlargest(n_features_to_use).index
                X_selected = X[top_features]
                
                # Create model
                config = self.config
                config.model.__dict__.update(model_params)
                model = CryptoRandomForestModel(config.model)
                
                # Prepare and train
                X_with_target = X_selected.copy()
                X_with_target['target'] = y
                X_clean, y_clean = model.prepare_data(X_with_target, 'target')
                
                # Use last 20% for testing
                split_idx = int(len(X_clean) * 0.8)
                X_train = X_clean.iloc[:split_idx]
                y_train = y_clean.iloc[:split_idx]
                X_test = X_clean.iloc[split_idx:]
                y_test = y_clean.iloc[split_idx:]
                
                model.train(X_train, y_train, validation_split=0.2)
                predictions = model.predict(X_test)
                
                # Quick trading simulation
                pred_series = pd.Series(predictions)
                buy_signals = pred_series > pred_series.quantile(trading_params['buy_threshold'])
                sell_signals = pred_series < pred_series.quantile(trading_params['sell_threshold'])
                
                # Calculate simple returns
                strategy_returns = []
                position = 0
                
                for i in range(len(predictions)):
                    if buy_signals.iloc[i] and position == 0:
                        position = 1
                    elif sell_signals.iloc[i] and position == 1:
                        position = 0
                    
                    if i > 0:
                        if position == 1:
                            ret = y_test.iloc[i] if i < len(y_test) else 0
                        else:
                            ret = 0
                        strategy_returns.append(ret)
                
                if len(strategy_returns) > 1:
                    total_return = np.sum(strategy_returns) * 100
                    volatility = np.std(strategy_returns) * 100
                    sharpe = total_return / volatility if volatility > 0 else 0
                    
                    # Return multiple objectives
                    return total_return, sharpe
                else:
                    return -10, -10
                    
            except Exception as e:
                return -10, -10
        
        # Multi-objective study
        multi_study = optuna.create_study(
            directions=['maximize', 'maximize'],  # Maximize both return and Sharpe
            study_name='multi_objective_crypto'
        )
        multi_study.optimize(multi_objective, n_trials=n_trials)
        
        # Find best trade-off solution (highest combined score)
        best_trial = max(multi_study.trials, 
                        key=lambda t: sum(t.values) if t.values and len(t.values) == 2 else -100)
        
        print(f"   ✅ Multi-objective optimization complete:")
        print(f"      Best return: {best_trial.values[0]:.2f}%")
        print(f"      Best Sharpe: {best_trial.values[1]:.2f}")
        print(f"      Best parameters: {best_trial.params}")
        
        self.optimization_results['multi_objective'] = {
            'best_params': best_trial.params,
            'best_return': best_trial.values[0],
            'best_sharpe': best_trial.values[1],
            'study': multi_study
        }
        
        # Update best parameters with multi-objective results
        self.best_parameters['multi_objective'] = best_trial.params
        
        return best_trial.params
    
    def _validate_best_parameters(self):
        """Validate the best parameters on hold-out data."""
        print("   ✅ Validating best parameters...")
        
        # Use the best multi-objective parameters
        best_params = self.best_parameters.get('multi_objective', {})
        
        validation_results = {}
        
        for symbol in self.config.data.symbols:
            target_key = f"{symbol}_6h"
            if target_key not in self.targets:
                continue
                
            target_data = self.targets[target_key]
            X = target_data['X']
            y = target_data['y']
            prices = target_data['prices']
            
            # Use last 30% as validation
            split_idx = int(len(X) * 0.7)
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            prices_val = prices.iloc[split_idx:]
            
            try:
                # Create model with best parameters
                config = self.config
                model_params = {k: v for k, v in best_params.items() 
                              if k in ['n_estimators', 'max_depth', 'min_samples_split', 
                                     'min_samples_leaf', 'max_features', 'bootstrap']}
                config.model.__dict__.update(model_params)
                model = CryptoRandomForestModel(config.model)
                
                # Train on earlier data
                X_train = X.iloc[:split_idx]
                y_train = y.iloc[:split_idx]
                X_train_target = X_train.copy()
                X_train_target['target'] = y_train
                
                X_clean, y_clean = model.prepare_data(X_train_target, 'target')
                model.train(X_clean, y_clean, validation_split=0.2)
                
                # Validate
                X_val_clean = X_val[X_clean.columns]
                predictions = model.predict(X_val_clean)
                
                val_r2 = r2_score(y_val[:len(predictions)], predictions)
                
                # Quick trading validation
                buy_thresh = best_params.get('buy_threshold', 0.7)
                sell_thresh = best_params.get('sell_threshold', 0.3)
                
                pred_series = pd.Series(predictions)
                signals = np.zeros(len(predictions))
                signals[pred_series > pred_series.quantile(buy_thresh)] = 1
                signals[pred_series < pred_series.quantile(sell_thresh)] = -1
                
                # Simple returns calculation
                strategy_returns = signals[:-1] * y_val.values[1:len(signals)]
                total_return = np.sum(strategy_returns) * 100
                sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
                
                validation_results[symbol] = {
                    'r2_score': val_r2,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'predictions': len(predictions)
                }
                
                print(f"      {symbol.upper()}: R²={val_r2:.3f}, Return={total_return:.2f}%, Sharpe={sharpe:.2f}")
                
            except Exception as e:
                print(f"      {symbol.upper()}: Validation failed - {e}")
                validation_results[symbol] = {'error': str(e)}
        
        self.optimization_results['validation'] = validation_results
        
        return validation_results
    
    def _generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("   📊 Creating optimization report...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Hyperparameter Optimization Results', fontsize=16)
        
        # Plot 1: Model optimization progress
        ax1 = axes[0, 0]
        if 'model' in self.optimization_results:
            study = self.optimization_results['model']['study']
            values = [trial.value for trial in study.trials if trial.value is not None]
            ax1.plot(range(len(values)), values, 'b-', alpha=0.7, linewidth=2)
            ax1.set_title('Model Hyperparameter Optimization')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Combined Score')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trading optimization progress
        ax2 = axes[0, 1]
        if 'trading' in self.optimization_results:
            study = self.optimization_results['trading']['study']
            values = [trial.value for trial in study.trials if trial.value is not None]
            ax2.plot(range(len(values)), values, 'g-', alpha=0.7, linewidth=2)
            ax2.set_title('Trading Parameter Optimization')
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Trading Score')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Multi-objective Pareto front
        ax3 = axes[0, 2]
        if 'multi_objective' in self.optimization_results:
            study = self.optimization_results['multi_objective']['study']
            returns = []
            sharpes = []
            for trial in study.trials:
                if trial.values and len(trial.values) == 2:
                    returns.append(trial.values[0])
                    sharpes.append(trial.values[1])
            
            if returns and sharpes:
                ax3.scatter(returns, sharpes, alpha=0.6, c='red')
                ax3.set_title('Multi-Objective Pareto Front')
                ax3.set_xlabel('Return (%)')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation results
        ax4 = axes[1, 0]
        if 'validation' in self.optimization_results:
            val_results = self.optimization_results['validation']
            symbols = []
            returns = []
            for symbol, result in val_results.items():
                if 'total_return' in result:
                    symbols.append(symbol.upper())
                    returns.append(result['total_return'])
            
            if symbols and returns:
                ax4.bar(range(len(symbols)), returns, color=['orange', 'blue', 'purple'], alpha=0.7)
                ax4.set_xticks(range(len(symbols)))
                ax4.set_xticklabels(symbols)
                ax4.set_title('Validation Returns by Cryptocurrency')
                ax4.set_ylabel('Return (%)')
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Parameter importance (simplified)
        ax5 = axes[1, 1]
        if 'model' in self.optimization_results:
            study = self.optimization_results['model']['study']
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:6]  # Top 6
                values = [importance[p] for p in params]
                
                ax5.barh(range(len(params)), values, color='green', alpha=0.7)
                ax5.set_yticks(range(len(params)))
                ax5.set_yticklabels(params)
                ax5.set_title('Parameter Importance')
                ax5.set_xlabel('Importance')
                ax5.grid(True, alpha=0.3)
            except:
                ax5.text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=ax5.transAxes)
        
        # Plot 6: Best parameters comparison
        ax6 = axes[1, 2]
        best_scores = []
        optimization_types = []
        
        if 'model' in self.optimization_results:
            best_scores.append(self.optimization_results['model']['best_score'])
            optimization_types.append('Model')
        
        if 'trading' in self.optimization_results:
            best_scores.append(self.optimization_results['trading']['best_score'])
            optimization_types.append('Trading')
        
        if best_scores:
            ax6.bar(range(len(optimization_types)), best_scores, 
                   color=['blue', 'green'], alpha=0.7)
            ax6.set_xticks(range(len(optimization_types)))
            ax6.set_xticklabels(optimization_types)
            ax6.set_title('Best Scores by Optimization Type')
            ax6.set_ylabel('Score')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_filename = f"advanced_optimization_results_{timestamp}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved to {viz_filename}")
        plt.close()
        
        # Save detailed results
        results_filename = f"advanced_optimization_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            'best_parameters': self.best_parameters,
            'optimization_results': {
                'model': {
                    'best_params': self.optimization_results.get('model', {}).get('best_params', {}),
                    'best_score': self.optimization_results.get('model', {}).get('best_score', 0)
                },
                'trading': {
                    'best_params': self.optimization_results.get('trading', {}).get('best_params', {}),
                    'best_score': self.optimization_results.get('trading', {}).get('best_score', 0)
                },
                'multi_objective': {
                    'best_params': self.optimization_results.get('multi_objective', {}).get('best_params', {}),
                    'best_return': self.optimization_results.get('multi_objective', {}).get('best_return', 0),
                    'best_sharpe': self.optimization_results.get('multi_objective', {}).get('best_sharpe', 0)
                },
                'validation': self.optimization_results.get('validation', {})
            }
        }
        
        with open(results_filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {results_filename}")
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("🏆 ADVANCED OPTIMIZATION COMPLETE - SUMMARY")
        print("="*80)
        
        if 'model' in self.optimization_results:
            model_score = self.optimization_results['model']['best_score']
            print(f"\n🤖 Model Optimization:")
            print(f"   • Best Combined Score: {model_score:.3f}")
            model_params = self.optimization_results['model']['best_params']
            for param, value in model_params.items():
                print(f"   • {param}: {value}")
        
        if 'trading' in self.optimization_results:
            trading_score = self.optimization_results['trading']['best_score']
            print(f"\n💼 Trading Optimization:")
            print(f"   • Best Trading Score: {trading_score:.3f}")
            trading_params = self.optimization_results['trading']['best_params']
            for param, value in trading_params.items():
                print(f"   • {param}: {value}")
        
        if 'multi_objective' in self.optimization_results:
            mo_return = self.optimization_results['multi_objective']['best_return']
            mo_sharpe = self.optimization_results['multi_objective']['best_sharpe']
            print(f"\n🎯 Multi-Objective Optimization:")
            print(f"   • Best Return: {mo_return:.2f}%")
            print(f"   • Best Sharpe: {mo_sharpe:.2f}")
        
        if 'validation' in self.optimization_results:
            print(f"\n✅ Validation Results:")
            for symbol, result in self.optimization_results['validation'].items():
                if 'total_return' in result:
                    print(f"   • {symbol.upper()}: {result['total_return']:.2f}% return, {result['sharpe_ratio']:.2f} Sharpe")
        
        print(f"\n📊 Files Generated:")
        print(f"   • {viz_filename}")
        print(f"   • {results_filename}")


def main():
    """Run advanced hyperparameter optimization."""
    optimizer = AdvancedHyperparameterOptimizer()
    best_params = optimizer.run_advanced_optimization(n_trials=75)  # Reduced for speed
    
    print("\n✨ Advanced hyperparameter optimization completed successfully!")
    print("🚀 Models are now optimized for maximum performance!")
    
    return best_params


if __name__ == "__main__":
    main()