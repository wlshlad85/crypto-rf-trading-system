#!/usr/bin/env python3
"""
Main execution script for 6-month minute-by-minute Random Forest cryptocurrency backtesting.

This script orchestrates the complete pipeline:
1. Data fetching (6 months of minute-level data)
2. Feature engineering (100+ specialized features)
3. Model training (multi-horizon Random Forest models)
4. Strategy execution (high-frequency trading strategies)
5. Backtesting (ultra-fast minute-level backtesting)
6. Performance analysis (comprehensive analytics)
7. Visualization (charts, reports, and dashboards)

Usage:
    python run_minute_backtest.py [options]

Author: Claude (Anthropic)
Date: 2024
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from data.minute_data_manager import MinuteDataManager
from features.minute_feature_engineering import MinuteFeatureEngine
from models.minute_random_forest_model import MinuteRandomForestModel, EnsembleMinuteRandomForest
from strategies.minute_trading_strategies import MinuteStrategyEnsemble
from backtesting.minute_backtest_engine import MinuteBacktestEngine
from analytics.minute_performance_analytics import MinutePerformanceAnalytics
from visualization.minute_visualization import MinuteVisualizationSuite


class MinuteBacktestOrchestrator:
    """Main orchestrator for the 6-month minute-level backtesting pipeline."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_manager = None
        self.feature_engine = None
        self.model = None
        self.strategy = None
        self.backtest_engine = None
        self.analytics = None
        self.visualization = None
        
        # Results storage
        self.results = {}
        self.execution_log = []
        
        # Performance tracking
        self.start_time = None
        self.checkpoints = {}
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        
        default_config = {
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "MATIC-USD"],
            "interval": "1m",
            "period_months": 6,
            "initial_capital": 100000,
            
            "data": {
                "cache_dir": "data/cache_minute",
                "max_age_hours": 24,
                "validation_enabled": True
            },
            
            "features": {
                "enable_all_features": True,
                "high_frequency_features": True,
                "microstructure_features": True,
                "intraday_seasonality": True,
                "volume_profile": True,
                "order_flow_features": True,
                "memory_efficient": True
            },
            
            "model": {
                "type": "ensemble",  # or "single"
                "n_models": 3,
                "prediction_horizons": [1, 5, 15, 60],
                "retrain_frequency": 1440,  # minutes
                "online_learning": True
            },
            
            "strategy": {
                "ensemble_weights": [0.4, 0.3, 0.3],  # momentum, mean_reversion, breakout
                "rebalance_frequency": 5,  # minutes
                "max_total_exposure": 0.8,
                "max_single_position": 0.25,
                "risk_management": True
            },
            
            "backtest": {
                "commission": 0.001,
                "slippage": 0.0005,
                "parallel_processing": True,
                "batch_size": 1440,  # minutes per batch
                "memory_efficient": True
            },
            
            "analytics": {
                "risk_free_rate": 0.02,
                "confidence_levels": [0.95, 0.99],
                "rolling_windows": [60, 240, 1440],
                "generate_report": True
            },
            
            "visualization": {
                "create_charts": True,
                "interactive_dashboard": True,
                "pdf_report": True,
                "save_format": "png"
            },
            
            "execution": {
                "checkpoint_frequency": 60,  # minutes
                "max_memory_gb": 8,
                "verbose": True,
                "save_intermediate_results": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with defaults
                def merge_dicts(default, user):
                    result = default.copy()
                    for key, value in user.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = merge_dicts(result[key], value)
                        else:
                            result[key] = value
                    return result
                
                return merge_dicts(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger('MinuteBacktest')
        logger.setLevel(logging.INFO if self.config['execution']['verbose'] else logging.WARNING)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/minute_backtest_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.config['execution']['verbose'] else logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
    
    def run_complete_backtest(self) -> Dict[str, Any]:
        """Run the complete 6-month minute-level backtesting pipeline."""
        
        self.start_time = datetime.now()
        self.logger.info("="*80)
        self.logger.info("STARTING 6-MONTH MINUTE-LEVEL CRYPTOCURRENCY BACKTESTING")
        self.logger.info("="*80)
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info(f"Symbols: {self.config['symbols']}")
        self.logger.info(f"Period: {self.config['period_months']} months")
        self.logger.info(f"Initial capital: ${self.config['initial_capital']:,}")
        
        try:
            # Step 1: Data Collection
            self._log_step(1, "Data Collection")
            data_dict = self._fetch_minute_data()
            self._checkpoint("data_collection", {"data_points": sum(len(df) for df in data_dict.values())})
            
            # Step 2: Feature Engineering
            self._log_step(2, "Feature Engineering")
            features_dict, combined_features = self._generate_features(data_dict)
            self._checkpoint("feature_engineering", {"feature_count": combined_features.shape[1]})
            
            # Step 3: Model Training
            self._log_step(3, "Model Training")
            model, targets = self._train_models(combined_features, data_dict)
            self._checkpoint("model_training", {"models_trained": len(model.horizon_models) if hasattr(model, 'horizon_models') else 1})
            
            # Step 4: Strategy Setup
            self._log_step(4, "Strategy Configuration")
            strategy = self._setup_strategy()
            self._checkpoint("strategy_setup", {"strategy_type": "ensemble"})
            
            # Step 5: Backtesting
            self._log_step(5, "Backtesting Execution")
            backtest_results = self._run_backtest(data_dict, model, strategy)
            self._checkpoint("backtesting", {"final_value": backtest_results.get('final_value', 0)})
            
            # Step 6: Performance Analytics
            self._log_step(6, "Performance Analytics")
            analytics_results = self._analyze_performance(backtest_results)
            self._checkpoint("analytics", {"metrics_calculated": len(analytics_results.get('basic_metrics', {}))})
            
            # Step 7: Visualization and Reporting
            self._log_step(7, "Visualization and Reporting")
            visualization_results = self._create_visualizations(backtest_results, analytics_results, data_dict)
            self._checkpoint("visualization", {"charts_created": len(visualization_results)})
            
            # Step 8: Final Results Compilation
            self._log_step(8, "Results Compilation")
            final_results = self._compile_final_results(
                backtest_results, analytics_results, visualization_results
            )
            
            # Save complete results
            self._save_results(final_results)
            
            # Final summary
            self._log_completion_summary(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Critical error in backtesting pipeline: {e}")
            self.logger.exception("Full traceback:")
            return {"error": str(e), "checkpoints": self.checkpoints}
    
    def _fetch_minute_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch 6 months of minute-level data."""
        
        self.logger.info("Initializing minute data manager...")
        self.data_manager = MinuteDataManager(cache_dir=self.config['data']['cache_dir'])
        
        self.logger.info(f"Fetching {self.config['period_months']} months of {self.config['interval']} data...")
        self.logger.info(f"Symbols: {', '.join(self.config['symbols'])}")
        
        # Fetch data
        data_dict = self.data_manager.fetch_6_month_minute_data(
            symbols=self.config['symbols'],
            interval=self.config['interval']
        )
        
        if not data_dict:
            raise ValueError("No data was fetched. Check symbols and network connection.")
        
        # Log data summary
        total_points = sum(len(df) for df in data_dict.values())
        self.logger.info(f"Data fetching completed:")
        for symbol, df in data_dict.items():
            self.logger.info(f"  {symbol}: {len(df):,} data points ({df.index[0]} to {df.index[-1]})")
        self.logger.info(f"Total data points: {total_points:,}")
        
        # Generate data quality report
        cache_summary = self.data_manager.get_cache_summary()
        self.logger.info(f"Cache size: {cache_summary.get('cache_size_mb', 0):.2f} MB")
        
        return data_dict
    
    def _generate_features(self, data_dict: Dict[str, pd.DataFrame]) -> tuple:
        """Generate comprehensive features for all symbols."""
        
        self.logger.info("Initializing minute feature engine...")
        self.feature_engine = MinuteFeatureEngine(config=self.config['features'])
        
        # Generate features for each symbol
        features_dict = {}
        all_features = []
        
        for i, (symbol, data) in enumerate(data_dict.items()):
            self.logger.info(f"Generating features for {symbol} ({i+1}/{len(data_dict)})...")
            
            try:
                symbol_features = self.feature_engine.generate_minute_features(data, symbol)
                features_dict[symbol] = symbol_features
                
                # Add symbol prefix to features
                prefixed_features = symbol_features.copy()
                for col in prefixed_features.columns:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        prefixed_features.rename(columns={col: f"{symbol}_{col}"}, inplace=True)
                
                all_features.append(prefixed_features)
                
                self.logger.info(f"  Generated {symbol_features.shape[1]} features for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error generating features for {symbol}: {e}")
                continue
        
        # Combine all features
        if all_features:
            self.logger.info("Combining features from all symbols...")
            combined_features = pd.concat(all_features, axis=1, sort=True)
            combined_features = combined_features.fillna(method='ffill', limit=5)
            combined_features = combined_features.dropna()
            
            self.logger.info(f"Combined features shape: {combined_features.shape}")
            self.logger.info(f"Feature columns: {combined_features.shape[1]}")
            
            return features_dict, combined_features
        else:
            raise ValueError("No features were generated successfully.")
    
    def _train_models(self, features: pd.DataFrame, data_dict: Dict[str, pd.DataFrame]) -> tuple:
        """Train Random Forest models."""
        
        self.logger.info("Initializing Random Forest models...")
        
        if self.config['model']['type'] == 'ensemble':
            self.logger.info(f"Creating ensemble with {self.config['model']['n_models']} models")
            model = EnsembleMinuteRandomForest(n_models=self.config['model']['n_models'])
        else:
            self.logger.info("Creating single Random Forest model")
            model = MinuteRandomForestModel()
        
        # Create combined price data for targets
        self.logger.info("Preparing target variables...")
        combined_price_data = pd.DataFrame()
        
        for symbol, data in data_dict.items():
            combined_price_data[f"{symbol}_close"] = data['Close']
        
        # Create targets
        targets = model.prepare_multi_horizon_targets(combined_price_data, self.config['symbols'])
        
        self.logger.info(f"Target variables shape: {targets.shape}")
        
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        features_aligned = features.loc[common_index]
        targets_aligned = targets.loc[common_index]
        
        self.logger.info(f"Aligned data shape: {features_aligned.shape}")
        
        # Train models
        self.logger.info("Starting model training...")
        training_start = time.time()
        
        training_results = model.train_multi_horizon_models(
            features_aligned, targets_aligned, self.config['symbols']
        )
        
        training_time = time.time() - training_start
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Log training results
        if isinstance(training_results, dict):
            for symbol, symbol_results in training_results.items():
                self.logger.info(f"  {symbol} models:")
                for horizon, metrics in symbol_results.items():
                    if isinstance(metrics, dict) and 'r2_score' in metrics:
                        self.logger.info(f"    {horizon}min - R²: {metrics['r2_score']:.3f}, RMSE: {metrics['rmse']:.4f}")
        
        return model, targets_aligned
    
    def _setup_strategy(self) -> MinuteStrategyEnsemble:
        """Setup trading strategy ensemble."""
        
        self.logger.info("Initializing trading strategy ensemble...")
        
        strategy = MinuteStrategyEnsemble(weights=self.config['strategy']['ensemble_weights'])
        
        self.logger.info(f"Strategy ensemble created with weights: {strategy.weights}")
        
        return strategy
    
    def _run_backtest(self, data_dict: Dict[str, pd.DataFrame], 
                     model: Any, strategy: MinuteStrategyEnsemble) -> Dict[str, Any]:
        """Run the ultra-fast backtesting."""
        
        self.logger.info("Initializing backtest engine...")
        
        # Create backtest configuration
        from types import SimpleNamespace
        
        backtest_config = SimpleNamespace()
        backtest_config.backtest = SimpleNamespace()
        
        backtest_config.backtest.initial_capital = self.config['initial_capital']
        backtest_config.backtest.commission = self.config['backtest']['commission']
        backtest_config.backtest.slippage = self.config['backtest']['slippage']
        
        self.backtest_engine = MinuteBacktestEngine(backtest_config)
        
        # Run backtest
        self.logger.info("Starting ultra-fast backtesting...")
        backtest_start = time.time()
        
        if self.config['backtest']['parallel_processing']:
            self.logger.info("Using parallel processing for backtest")
            backtest_results = self.backtest_engine.run_parallel_backtest(
                data_dict, model, self.config['symbols']
            )
        else:
            self.logger.info("Using sequential processing for backtest")
            backtest_results = self.backtest_engine.run_ultra_fast_backtest(
                data_dict, model, self.config['symbols']
            )
        
        backtest_time = time.time() - backtest_start
        self.logger.info(f"Backtesting completed in {backtest_time:.2f} seconds")
        
        # Log backtest summary
        if 'final_value' in backtest_results:
            initial_value = self.config['initial_capital']
            final_value = backtest_results['final_value']
            total_return = (final_value / initial_value) - 1
            
            self.logger.info(f"Backtest Summary:")
            self.logger.info(f"  Initial Value: ${initial_value:,.2f}")
            self.logger.info(f"  Final Value: ${final_value:,.2f}")
            self.logger.info(f"  Total Return: {total_return:.2%}")
            self.logger.info(f"  Total Trades: {backtest_results.get('total_trades_executed', 0):,}")
        
        return backtest_results
    
    def _analyze_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive performance analytics."""
        
        self.logger.info("Initializing performance analytics...")
        self.analytics = MinutePerformanceAnalytics()
        
        # Extract portfolio history
        portfolio_history = None
        trades_history = None
        
        if 'portfolio_history' in backtest_results:
            portfolio_data = backtest_results['portfolio_history']
            if isinstance(portfolio_data, list) and portfolio_data:
                portfolio_history = pd.DataFrame(portfolio_data)
                if 'timestamp' in portfolio_history.columns:
                    portfolio_history.set_index('timestamp', inplace=True)
                    portfolio_history.index = pd.to_datetime(portfolio_history.index)
        
        if 'trades_history' in backtest_results:
            trades_data = backtest_results['trades_history']
            if isinstance(trades_data, list) and trades_data:
                trades_history = pd.DataFrame(trades_data)
                if 'timestamp' in trades_history.columns:
                    trades_history.set_index('timestamp', inplace=True)
                    trades_history.index = pd.to_datetime(trades_history.index)
        
        if portfolio_history is None or portfolio_history.empty:
            self.logger.warning("No portfolio history available for analysis")
            return {"error": "No portfolio history available"}
        
        # Run comprehensive analysis
        self.logger.info("Running comprehensive performance analysis...")
        analytics_results = self.analytics.analyze_portfolio_performance(
            portfolio_history, trades_history
        )
        
        # Log key metrics
        if 'basic_metrics' in analytics_results:
            metrics = analytics_results['basic_metrics']
            self.logger.info(f"Performance Metrics:")
            self.logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            self.logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            self.logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        
        if 'risk_metrics' in analytics_results:
            risk = analytics_results['risk_metrics']
            self.logger.info(f"Risk Metrics:")
            self.logger.info(f"  Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
            self.logger.info(f"  VaR (95%): {risk.get('var_95', 0):.4f}")
            self.logger.info(f"  Volatility: {risk.get('volatility', 0):.2%}")
        
        return analytics_results
    
    def _create_visualizations(self, backtest_results: Dict[str, Any],
                             analytics_results: Dict[str, Any],
                             data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create comprehensive visualizations and reports."""
        
        if not self.config['visualization']['create_charts']:
            self.logger.info("Visualization disabled in configuration")
            return {}
        
        self.logger.info("Initializing visualization suite...")
        self.visualization = MinuteVisualizationSuite()
        
        # Extract data for visualization
        portfolio_history = None
        trades_history = None
        
        if 'portfolio_history' in backtest_results:
            portfolio_data = backtest_results['portfolio_history']
            if isinstance(portfolio_data, list) and portfolio_data:
                portfolio_history = pd.DataFrame(portfolio_data)
                if 'timestamp' in portfolio_history.columns:
                    portfolio_history.set_index('timestamp', inplace=True)
                    portfolio_history.index = pd.to_datetime(portfolio_history.index)
        
        if 'trades_history' in backtest_results:
            trades_data = backtest_results['trades_history']
            if isinstance(trades_data, list) and trades_data:
                trades_history = pd.DataFrame(trades_data)
                if 'timestamp' in trades_history.columns:
                    trades_history.set_index('timestamp', inplace=True)
                    trades_history.index = pd.to_datetime(trades_history.index)
        
        if portfolio_history is None or portfolio_history.empty:
            self.logger.warning("No portfolio history available for visualization")
            return {"error": "No portfolio history available"}
        
        # Create comprehensive dashboard
        self.logger.info("Creating comprehensive dashboard...")
        dashboard_components = self.visualization.create_comprehensive_dashboard(
            portfolio_history, trades_history, data_dict, None, analytics_results
        )
        
        visualization_results = {'dashboard_components': dashboard_components}
        
        # Create interactive dashboard
        if self.config['visualization']['interactive_dashboard']:
            try:
                self.logger.info("Creating interactive dashboard...")
                interactive_path = self.visualization.create_interactive_dashboard(
                    portfolio_history, trades_history, data_dict
                )
                if interactive_path:
                    visualization_results['interactive_dashboard'] = interactive_path
                    self.logger.info(f"Interactive dashboard saved: {interactive_path}")
            except Exception as e:
                self.logger.warning(f"Could not create interactive dashboard: {e}")
        
        # Create PDF report
        if self.config['visualization']['pdf_report']:
            try:
                self.logger.info("Generating comprehensive PDF report...")
                report_path = self.visualization.generate_comprehensive_report(
                    portfolio_history, trades_history, data_dict, None, analytics_results
                )
                visualization_results['pdf_report'] = report_path
                self.logger.info(f"PDF report saved: {report_path}")
            except Exception as e:
                self.logger.warning(f"Could not create PDF report: {e}")
        
        return visualization_results
    
    def _compile_final_results(self, backtest_results: Dict[str, Any],
                             analytics_results: Dict[str, Any],
                             visualization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all results into final report."""
        
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        final_results = {
            'execution_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_execution_time_seconds': total_time,
                'total_execution_time_formatted': str(timedelta(seconds=int(total_time))),
                'configuration': self.config,
                'checkpoints': self.checkpoints
            },
            'backtest_results': backtest_results,
            'analytics_results': analytics_results,
            'visualization_results': visualization_results,
            'execution_log': self.execution_log
        }
        
        return final_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save complete results to files."""
        
        if not self.config['execution']['save_intermediate_results']:
            return
        
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'results/minute_backtest_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(results_dir, 'complete_results.json')
        
        try:
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(results)
            
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Complete results saved to: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Could not save JSON results: {e}")
        
        # Save configuration
        config_path = os.path.join(results_dir, 'configuration.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            self.logger.error(f"Could not save configuration: {e}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _log_step(self, step_num: int, step_name: str):
        """Log the start of a major step."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STEP {step_num}: {step_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        step_info = {
            'step': step_num,
            'name': step_name,
            'start_time': datetime.now().isoformat()
        }
        self.execution_log.append(step_info)
    
    def _checkpoint(self, name: str, data: Dict[str, Any]):
        """Create a checkpoint with timing and data."""
        checkpoint_time = datetime.now()
        elapsed = (checkpoint_time - self.start_time).total_seconds()
        
        self.checkpoints[name] = {
            'timestamp': checkpoint_time.isoformat(),
            'elapsed_seconds': elapsed,
            'data': data
        }
        
        self.logger.info(f"Checkpoint '{name}': {elapsed:.1f}s elapsed")
    
    def _log_completion_summary(self, results: Dict[str, Any]):
        """Log final completion summary."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("6-MONTH MINUTE-LEVEL BACKTESTING COMPLETED")
        self.logger.info("="*80)
        
        # Execution summary
        exec_info = results['execution_info']
        self.logger.info(f"Total execution time: {exec_info['total_execution_time_formatted']}")
        
        # Performance summary
        if 'analytics_results' in results and 'basic_metrics' in results['analytics_results']:
            metrics = results['analytics_results']['basic_metrics']
            self.logger.info(f"\nFINAL PERFORMANCE SUMMARY:")
            self.logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            self.logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            self.logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            if 'risk_metrics' in results['analytics_results']:
                risk = results['analytics_results']['risk_metrics']
                self.logger.info(f"  Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
                self.logger.info(f"  Volatility: {risk.get('volatility', 0):.2%}")
        
        # Data summary
        if 'backtest_results' in results:
            backtest = results['backtest_results']
            self.logger.info(f"\nDATA PROCESSED:")
            self.logger.info(f"  Data points: {backtest.get('total_minutes_processed', 0):,}")
            self.logger.info(f"  Trades executed: {backtest.get('total_trades_executed', 0):,}")
        
        self.logger.info("\n" + "="*80)


def main():
    """Main entry point for the script."""
    
    parser = argparse.ArgumentParser(
        description='6-Month Minute-Level Cryptocurrency Backtesting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python run_minute_backtest.py
    
    # Run with custom configuration
    python run_minute_backtest.py --config my_config.json
    
    # Run with specific symbols
    python run_minute_backtest.py --symbols BTC-USD ETH-USD SOL-USD
    
    # Run with reduced capital for testing
    python run_minute_backtest.py --capital 10000
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--symbols', nargs='+', 
                       help='Cryptocurrency symbols to trade (e.g., BTC-USD ETH-USD)')
    parser.add_argument('--capital', type=float, 
                       help='Initial capital for backtesting')
    parser.add_argument('--months', type=int, default=6,
                       help='Number of months of data to fetch (default: 6)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-charts', action='store_true',
                       help='Disable chart generation')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with reduced data')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MinuteBacktestOrchestrator(args.config)
    
    # Override configuration with command line arguments
    if args.symbols:
        orchestrator.config['symbols'] = args.symbols
    
    if args.capital:
        orchestrator.config['initial_capital'] = args.capital
    
    if args.months:
        orchestrator.config['period_months'] = args.months
    
    if args.verbose:
        orchestrator.config['execution']['verbose'] = True
    
    if args.no_charts:
        orchestrator.config['visualization']['create_charts'] = False
    
    if args.test_mode:
        # Reduce scope for testing
        orchestrator.config['symbols'] = orchestrator.config['symbols'][:2]  # Only first 2 symbols
        orchestrator.config['period_months'] = 1  # Only 1 month
        orchestrator.config['model']['n_models'] = 1  # Single model
        orchestrator.logger.info("Running in test mode with reduced scope")
    
    # Run the complete backtesting pipeline
    try:
        results = orchestrator.run_complete_backtest()
        
        if 'error' not in results:
            print(f"\n🎉 Backtesting completed successfully!")
            print(f"Results saved in logs and results directories.")
            
            # Print key metrics
            if 'analytics_results' in results and 'basic_metrics' in results['analytics_results']:
                metrics = results['analytics_results']['basic_metrics']
                print(f"\n📊 KEY RESULTS:")
                print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
                
                if 'risk_metrics' in results['analytics_results']:
                    risk = results['analytics_results']['risk_metrics']
                    print(f"   Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
        else:
            print(f"\n❌ Backtesting failed: {results['error']}")
            return 1
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Backtesting interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())