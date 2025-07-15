#!/usr/bin/env python3
"""
Ultra-optimized cryptocurrency prediction system with advanced ML techniques.
Combines multi-timeframe analysis, ultra-feature engineering, and risk-adjusted targets.
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import sklearn

# Core system imports
from utils.config import load_config, DataConfig, FeatureConfig, ModelConfig
from data.multi_timeframe_fetcher import UltraMultiTimeframeFetcher
from features.ultra_feature_engineering import UltraCryptoFeatureEngine
from models.ultra_target_engineering import UltraTargetEngineer
from models.random_forest_model import CryptoRandomForestModel, EnsembleRandomForestModel

warnings.filterwarnings('ignore')


class UltraOptimizedTradingSystem:
    """Ultra-optimized trading system with cutting-edge ML techniques."""
    
    def __init__(self, config_path: str):
        """Initialize the ultra-optimized trading system."""
        self.config = load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.data_config = self.config.data
        self.feature_config = self.config.features
        self.model_config = self.config.model
        
        # Ultra-advanced components
        self.multi_fetcher = None
        self.ultra_features = None
        self.ultra_targets = None
        self.model = None
        
        # Data storage
        self.raw_data = None
        self.feature_data = None
        self.target_data = None
        self.prepared_data = None
        
        self.logger.info("Initialized Ultra-Optimized Trading System")
        self.logger.info(f"Configuration: {config_path}")
        
    def setup_logging(self):
        """Setup enhanced logging."""
        log_level = getattr(logging, self.config.system.log_level)
        log_file = self.config.system.log_file
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_components(self):
        """Initialize all ultra-advanced components."""
        self.logger.info("Initializing ultra-advanced components...")
        
        # Multi-timeframe data fetcher
        if self.data_config.data_source == 'multi_timeframe':
            self.multi_fetcher = UltraMultiTimeframeFetcher(self.data_config)
        
        # Ultra feature engineering
        if self.data_config.use_ultra_features:
            self.ultra_features = UltraCryptoFeatureEngine(self.feature_config)
        
        # Ultra target engineering
        if self.data_config.use_ultra_targets:
            self.ultra_targets = UltraTargetEngineer(self.model_config)
        
        # Model selection
        if self.model_config.model_type == 'ensemble':
            self.model = EnsembleRandomForestModel(self.model_config)
        else:
            self.model = CryptoRandomForestModel(self.model_config)
        
        self.logger.info("Ultra-advanced components initialized successfully")
        
    def fetch_ultra_data(self) -> pd.DataFrame:
        """Fetch multi-timeframe data with ultra-advanced techniques."""
        self.logger.info("Fetching ultra-optimized multi-timeframe data...")
        start_time = time.time()
        
        if self.multi_fetcher:
            # Multi-timeframe approach
            self.raw_data = self.multi_fetcher.fetch_all_symbols_multi_timeframe()
        else:
            # Fallback to standard approach
            from data.yfinance_fetcher import YFinanceDataFetcher
            fetcher = YFinanceDataFetcher(self.data_config)
            self.raw_data = fetcher.fetch_all_symbols()
        
        fetch_time = time.time() - start_time
        self.logger.info(f"Data fetching completed in {fetch_time:.2f} seconds")
        self.logger.info(f"Raw data shape: {self.raw_data.shape}")
        self.logger.info(f"Raw data columns: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def engineer_ultra_features(self) -> pd.DataFrame:
        """Engineer ultra-advanced features."""
        if self.raw_data is None:
            raise ValueError("No raw data available. Run fetch_ultra_data() first.")
        
        self.logger.info("Engineering ultra-advanced features...")
        start_time = time.time()
        
        if self.ultra_features:
            # Ultra-advanced feature engineering
            self.feature_data = self.ultra_features.generate_ultra_features(self.raw_data)
        else:
            # Standard feature engineering
            from features.technical_indicators import TechnicalIndicators
            feature_engine = TechnicalIndicators(self.feature_config)
            self.feature_data = feature_engine.generate_all_features(self.raw_data)
        
        engineering_time = time.time() - start_time
        self.logger.info(f"Feature engineering completed in {engineering_time:.2f} seconds")
        self.logger.info(f"Feature data shape: {self.feature_data.shape}")
        
        return self.feature_data
    
    def create_ultra_targets(self) -> pd.DataFrame:
        """Create ultra-advanced target variables."""
        if self.feature_data is None:
            raise ValueError("No feature data available. Run engineer_ultra_features() first.")
        
        self.logger.info("Creating ultra-advanced target variables...")
        start_time = time.time()
        
        symbols = self.data_config.symbols
        
        if self.ultra_targets:
            # Ultra-advanced target engineering
            self.target_data = self.ultra_targets.create_ultra_targets(self.feature_data, symbols)
        else:
            # Standard target creation
            self.target_data = self.model.create_targets(self.feature_data, symbols)
        
        target_time = time.time() - start_time
        self.logger.info(f"Target creation completed in {target_time:.2f} seconds")
        self.logger.info(f"Target data shape: {self.target_data.shape}")
        self.logger.info(f"Target columns: {list(self.target_data.columns)}")
        
        return self.target_data
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Prepare data for ultra-optimized training."""
        if self.feature_data is None or self.target_data is None:
            raise ValueError("Feature and target data required")
        
        self.logger.info("Preparing ultra-optimized training data...")
        
        # Get primary symbol
        symbol = self.data_config.symbols[0]
        
        # Get primary target
        if self.ultra_targets:
            target_type = self.model_config.target_type
            try:
                primary_target_col = self.ultra_targets.get_primary_target(self.target_data, symbol, target_type)
                target_series, target_stats = self.ultra_targets.prepare_targets_for_training(
                    self.target_data, primary_target_col
                )
            except ValueError:
                # Fallback to basic target
                primary_target_col = f"{symbol}_return_{self.model_config.target_horizon}h"
                if primary_target_col not in self.target_data.columns:
                    # Use the first available target column
                    available_targets = [col for col in self.target_data.columns if col.startswith(f"{symbol}_")]
                    if available_targets:
                        primary_target_col = available_targets[0]
                    else:
                        raise ValueError(f"No target columns found for {symbol}")
                target_series = self.target_data[primary_target_col]
                target_stats = {}
        else:
            primary_target_col = f"{symbol}_target_{self.model_config.target_horizon}h"
            target_series = self.target_data[primary_target_col]
            target_stats = {}
        
        # Prepare features and targets
        # Create a combined dataframe with features and targets
        combined_data = self.feature_data.copy()
        combined_data[primary_target_col] = target_series
        
        features, targets = self.model.prepare_data(
            combined_data, 
            primary_target_col
        )
        
        self.prepared_data = {
            'features': features,
            'targets': targets,
            'feature_names': list(features.columns) if hasattr(features, 'columns') else []
        }
        
        self.logger.info(f"Training data prepared: {len(self.prepared_data['features'])} samples")
        self.logger.info(f"Primary target: {primary_target_col}")
        if target_stats:
            self.logger.info(f"Target statistics: {target_stats}")
        
        return self.prepared_data
    
    def ultra_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Perform ultra-advanced hyperparameter optimization."""
        if self.prepared_data is None:
            raise ValueError("Prepared data required for hyperparameter optimization")
        
        # Check if using ensemble model
        if isinstance(self.model, EnsembleRandomForestModel):
            # Respect the use_optuna_optimization setting for ensemble models too
            if self.model_config.use_optuna_optimization:
                self.logger.info("Using Optuna optimization for ensemble model")
                method = 'optuna'
            else:
                self.logger.info("Using standard hyperparameter tuning for ensemble model")
                method = 'grid_search'
            
            return self.model.hyperparameter_tuning(
                self.prepared_data['features'],
                self.prepared_data['targets'],
                method
            )
        
        if not self.model_config.use_optuna_optimization:
            self.logger.info("Optuna optimization disabled, using standard hyperparameter tuning")
            return self.model.hyperparameter_tuning(
                self.prepared_data['features'],
                self.prepared_data['targets']
            )
        
        self.logger.info("Starting ultra-advanced hyperparameter optimization with Optuna...")
        
        # Import Optuna for advanced optimization
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import MedianPruner
        except ImportError:
            self.logger.warning("Optuna not available, falling back to standard tuning")
            return self.model.hyperparameter_tuning(
                self.prepared_data['features'],
                self.prepared_data['targets']
            )
        
        def objective(trial):
            # Define search space based on config
            search_space = getattr(self.model_config, 'hyperparameter_search_space', {})
            
            params = {}
            if 'n_estimators' in search_space:
                params['n_estimators'] = trial.suggest_categorical('n_estimators', search_space['n_estimators'])
            if 'max_depth' in search_space:
                params['max_depth'] = trial.suggest_categorical('max_depth', search_space['max_depth'])
            if 'learning_rate' in search_space:
                params['learning_rate'] = trial.suggest_categorical('learning_rate', search_space['learning_rate'])
            
            # Create temporary model with trial parameters
            temp_config = self.model_config
            for key, value in params.items():
                setattr(temp_config, key, value)
            
            temp_model = CryptoRandomForestModel(temp_config)
            
            # Initialize the model by building it
            temp_model._build_model()
            
            # Evaluate using cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                temp_model.model,
                self.prepared_data['features'],
                self.prepared_data['targets'],
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            return np.mean(scores)
        
        # Create study
        sampler = TPESampler(seed=self.config.system.random_seed)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        n_trials = self.model_config.optuna_trials
        timeout = self.model_config.optuna_timeout
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best score: {study.best_value:.6f}")
        
        return best_params
    
    def train_ultra_model(self) -> Dict[str, Any]:
        """Train the ultra-optimized model."""
        if self.prepared_data is None:
            raise ValueError("Prepared data required for training")
        
        self.logger.info("Training ultra-optimized model...")
        start_time = time.time()
        
        # Hyperparameter optimization
        best_params = self.ultra_hyperparameter_optimization()
        if best_params:
            # Handle ensemble vs single model parameter setting
            if isinstance(self.model, EnsembleRandomForestModel):
                # For ensemble, the hyperparameter tuning already set the best_params
                # No need to set them again as they're already applied during tuning
                pass
            else:
                self.model.model.set_params(**best_params)
        
        # Train model
        self.model.train(
            self.prepared_data['features'],
            self.prepared_data['targets']
        )
        
        training_time = time.time() - start_time
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Feature importance analysis
        # Handle both ensemble and single model feature importance
        if isinstance(self.model, EnsembleRandomForestModel):
            # Aggregate feature importance across all ensemble models
            all_importances = []
            for model in self.model.models:
                if hasattr(model.model, 'feature_importances_'):
                    all_importances.append(model.model.feature_importances_)
            
            if all_importances:
                # Calculate mean importance across ensemble
                avg_importance = np.mean(all_importances, axis=0)
                importance_df = pd.DataFrame({
                    'feature': self.prepared_data['feature_names'],
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                
                self.logger.info("Top 10 most important ensemble features:")
                for _, row in importance_df.head(10).iterrows():
                    self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        elif hasattr(self.model.model, 'feature_importances_'):
            # Handle single model
            importance_df = pd.DataFrame({
                'feature': self.prepared_data['feature_names'],
                'importance': self.model.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info("Top 10 most important features:")
            for _, row in importance_df.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {"training_time": training_time, "best_params": best_params}
    
    def ultra_walk_forward_validation(self) -> Dict[str, Any]:
        """Perform ultra-advanced walk-forward validation."""
        if self.prepared_data is None:
            raise ValueError("Prepared data required for validation")
        
        self.logger.info("Performing ultra-advanced walk-forward validation...")
        
        # Use purged cross-validation if enabled
        if self.model_config.purged_cv:
            results = self._purged_walk_forward_validation()
        else:
            results = self.model.walk_forward_validation(
                self.prepared_data['features'],
                self.prepared_data['targets']
            )
        
        self.logger.info(f"Walk-forward validation completed")
        self.logger.info(f"Average score: {np.mean(results['scores']):.6f} (+/- {np.std(results['scores']):.6f})")
        
        return results
    
    def _purged_walk_forward_validation(self) -> Dict[str, Any]:
        """Implement purged walk-forward validation to avoid data leakage."""
        features = self.prepared_data['features']
        targets = self.prepared_data['targets']
        
        n_splits = self.model_config.walk_forward_splits
        embargo_periods = self.model_config.embargo_periods
        
        split_size = len(features) // n_splits
        scores = []
        predictions = []
        
        for i in range(n_splits - 1):
            # Training data (with embargo)
            train_end = (i + 1) * split_size - embargo_periods
            train_features = features.iloc[:train_end]
            train_targets = targets.iloc[:train_end]
            
            # Test data
            test_start = (i + 1) * split_size
            test_end = (i + 2) * split_size
            test_features = features.iloc[test_start:test_end]
            test_targets = targets.iloc[test_start:test_end]
            
            if len(train_features) == 0 or len(test_features) == 0:
                continue
            
            # Train and predict
            temp_model = CryptoRandomForestModel(self.model_config)
            temp_model.train(train_features, train_targets)
            
            pred = temp_model.predict(test_features)
            score = temp_model._calculate_score(test_targets, pred)
            
            scores.append(score)
            predictions.extend(pred)
        
        return {
            'scores': scores,
            'predictions': predictions,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    def run_ultra_backtest(self) -> Dict[str, Any]:
        """Run simplified backtesting to calculate basic performance metrics."""
        if not self.model.is_fitted:
            raise ValueError("Model must be trained before backtesting")
        
        self.logger.info("Running ultra-advanced backtest...")
        
        # Simple backtesting approach
        features = self.prepared_data['features']
        targets = self.prepared_data['targets']
        
        # Generate predictions
        predictions = self.model.predict(features)
        
        # Calculate basic metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Calculate directional accuracy
        target_direction = np.sign(targets)
        pred_direction = np.sign(predictions)
        directional_accuracy = np.mean(target_direction == pred_direction)
        
        # Calculate Sharpe-like metric for predictions
        pred_returns = pd.Series(predictions)
        pred_sharpe = pred_returns.mean() / pred_returns.std() if pred_returns.std() > 0 else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'directional_accuracy': directional_accuracy,
            'prediction_sharpe': pred_sharpe,
            'prediction_mean': np.mean(predictions),
            'prediction_std': np.std(predictions),
            'target_mean': np.mean(targets),
            'target_std': np.std(targets)
        }
        
        self.logger.info("Backtest Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete ultra-optimized pipeline."""
        self.logger.info("Starting ultra-optimized trading pipeline...")
        pipeline_start = time.time()
        
        results = {}
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Data pipeline
            self.fetch_ultra_data()
            self.engineer_ultra_features()
            self.create_ultra_targets()
            self.prepare_training_data()
            
            # Model pipeline
            training_results = self.train_ultra_model()
            validation_results = self.ultra_walk_forward_validation()
            backtest_results = self.run_ultra_backtest()
            
            # Compile results
            results = {
                'training': training_results,
                'validation': validation_results,
                'backtest': backtest_results,
                'data_shape': self.feature_data.shape,
                'feature_count': len(self.prepared_data['feature_names']),
                'model_type': self.model_config.model_type
            }
            
            pipeline_time = time.time() - pipeline_start
            results['total_time'] = pipeline_time
            
            self.logger.info(f"Ultra-optimized pipeline completed in {pipeline_time:.2f} seconds")
            self.logger.info(f"Final Sharpe Ratio: {backtest_results['metrics'].get('sharpe_ratio', 'N/A')}")
            
            # Save results if configured
            if self.config.backtest.save_results:
                self._save_results(results)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save ultra-optimized results."""
        results_dir = Path(self.config.backtest.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance metrics
        metrics_file = results_dir / f"ultra_metrics_{timestamp}.json"
        import json
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for key, value in results['backtest']['metrics'].items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_results[key] = value.item()
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Ultra-Optimized Crypto Trading System")
    parser.add_argument('--config', default='configs/ultra_btc_config.json',
                      help='Configuration file path')
    parser.add_argument('--mode', choices=['train', 'backtest', 'full'], default='full',
                      help='Execution mode')
    
    args = parser.parse_args()
    
    # Initialize system
    system = UltraOptimizedTradingSystem(args.config)
    
    try:
        if args.mode == 'full':
            results = system.run_full_pipeline()
            print(f"\n🎯 Ultra-Optimized Results:")
            print(f"📊 Features Generated: {results['feature_count']}")
            print(f"📈 Sharpe Ratio: {results['backtest']['metrics'].get('sharpe_ratio', 'N/A'):.4f}")
            print(f"📉 Max Drawdown: {results['backtest']['metrics'].get('max_drawdown', 'N/A'):.4f}")
            print(f"⏱️  Total Time: {results['total_time']:.2f} seconds")
            
        elif args.mode == 'train':
            system.initialize_components()
            system.fetch_ultra_data()
            system.engineer_ultra_features()
            system.create_ultra_targets()
            system.prepare_training_data()
            results = system.train_ultra_model()
            print(f"Training completed in {results['training_time']:.2f} seconds")
            
        elif args.mode == 'backtest':
            system.run_full_pipeline()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()