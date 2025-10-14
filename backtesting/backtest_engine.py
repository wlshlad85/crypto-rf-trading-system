"""Comprehensive backtesting engine for cryptocurrency trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import empyrical as emp
except ImportError:
    emp = None
    print("Warning: empyrical not installed. Some performance metrics will be unavailable.")

from utils.config import BacktestConfig, Config
from strategies.long_short_strategy import PortfolioManager
from models.random_forest_model import CryptoRandomForestModel


class CryptoBacktestEngine:
    """Comprehensive backtesting engine for cryptocurrency strategies."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.portfolio_history = None
        self.trades_history = None
        self.performance_metrics = {}
        
        # Benchmark data
        self.benchmark_returns = None
        
    def run_backtest(self, data: pd.DataFrame, features: pd.DataFrame, 
                    model: CryptoRandomForestModel, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive backtest."""
        self.logger.info("Starting backtest")
        
        # Prepare data
        backtest_data = self._prepare_backtest_data(data, features)
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(
            initial_capital=self.config.backtest.initial_capital,
            config=self.config.strategy
        )
        
        # Create targets for prediction
        targets = model.create_targets(data, symbols)
        
        # Time series split for walk-forward validation
        train_size = int(len(backtest_data) * 0.6)  # 60% for initial training

        # Initialize model cache for efficiency
        model_cache = {}
        cache_size = 100  # Keep models for last 100 training windows

        # Track predictions and signals
        all_predictions = []
        all_signals = []

        self.logger.info(f"Backtesting from {backtest_data.index[train_size]} to {backtest_data.index[-1]}")

        # Walk-forward backtest with model caching
        for i in range(train_size, len(backtest_data) - self.config.model.target_horizon):
            current_timestamp = backtest_data.index[i]

            # Create cache key for this training window
            cache_key = f"{i}_{train_size}"

            # Training window (rolling window for efficiency)
            train_start = max(0, i - 1000)  # Use last 1000 hours for training (reduced from 2000)
            train_end = i

            # Check if we have a cached model for this window
            if cache_key in model_cache:
                cached_models = model_cache[cache_key]
            else:
                # Get training data
                train_features = features.iloc[train_start:train_end]
                train_targets = targets.iloc[train_start:train_end]

                # Train models for each symbol (incremental training)
                cached_models = {}
                for symbol in symbols:
                    target_col = f"{symbol}_target"

                    if target_col not in train_targets.columns:
                        continue

                    # Prepare training data
                    X_train, y_train = model.prepare_data(train_features, target_col)

                    if len(X_train) < 100:  # Minimum training samples
                        continue

                    try:
                        # Use incremental training if possible
                        if symbol in model_cache and i > train_size + 50:
                            # Use previous model as starting point for incremental learning
                            symbol_model = model_cache[symbol][-1]  # Get last cached model for this symbol
                            # Incremental fit (if supported)
                            symbol_model.train_incremental(X_train.iloc[-100:], y_train.iloc[-100:])  # Train on recent data only
                        else:
                            # Clone model for this symbol
                            symbol_model = CryptoRandomForestModel(self.config.model)
                            symbol_model.train(X_train, y_train, validation_split=0.2)

                        cached_models[symbol] = symbol_model

                    except Exception as e:
                        self.logger.warning(f"Training failed for {symbol} at {current_timestamp}: {e}")
                        continue

                # Cache the models
                model_cache[cache_key] = cached_models

                # Manage cache size
                if len(model_cache) > cache_size:
                    # Remove oldest entries
                    oldest_keys = sorted(model_cache.keys())[:-cache_size]
                    for key in oldest_keys:
                        del model_cache[key]

            # Make predictions using cached models
            predictions = {}
            for symbol, symbol_model in cached_models.items():
                try:
                    # Predict for current timestamp
                    current_features = features.iloc[i:i+1]
                    if len(current_features) > 0:
                        pred = symbol_model.predict(current_features)[0]
                        predictions[f"{symbol}_target"] = pred
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {symbol} at {current_timestamp}: {e}")
                    continue

            # Store predictions
            if predictions:
                pred_series = pd.Series(predictions, name=current_timestamp)
                all_predictions.append(pred_series)
            
            # Generate signals
            if predictions:
                pred_df = pd.DataFrame([predictions], index=[current_timestamp])
                signals = portfolio_manager.strategy.generate_signals(
                    pred_df, backtest_data.iloc[i:i+1]
                )
                
                if not signals.empty:
                    signal_series = signals.iloc[0]
                    # Extract just the signal values (remove '_signal' suffix for symbol names)
                    clean_signals = pd.Series(dtype=float)
                    for col in signal_series.index:
                        if col.endswith('_signal'):
                            symbol = col.replace('_signal', '')
                            clean_signals[symbol] = signal_series[col]
                    
                    all_signals.append(clean_signals)
                    
                    # Update portfolio
                    current_prices = backtest_data.iloc[i]
                    portfolio_manager.update_portfolio(clean_signals, current_prices, current_timestamp)
            
            # Log progress
            if i % 500 == 0:
                current_value = portfolio_manager.strategy.total_value
                self.logger.info(f"Processed {i}/{len(backtest_data)} timestamps. "
                               f"Portfolio value: ${current_value:,.2f}")
        
        # Compile results
        self.portfolio_history = portfolio_manager.get_portfolio_dataframe()
        self.trades_history = portfolio_manager.get_trades_dataframe()
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(
            self.portfolio_history, backtest_data, symbols
        )
        
        # Create summary
        summary = portfolio_manager.get_summary()
        summary['performance_metrics'] = self.performance_metrics
        summary['backtest_period'] = {
            'start': backtest_data.index[train_size],
            'end': backtest_data.index[-1],
            'duration_hours': len(backtest_data) - train_size
        }
        
        self.results = summary
        
        self.logger.info("Backtest completed")
        self.logger.info(f"Final portfolio value: ${summary['current_value']:,.2f}")
        self.logger.info(f"Total return: {summary['total_return']:.2%}")
        
        return self.results
    
    def _prepare_backtest_data(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        # Filter data by backtest period
        start_date = pd.to_datetime(self.config.backtest.start_date)
        end_date = pd.to_datetime(self.config.backtest.end_date)
        
        # Filter both data and features
        data_filtered = data.loc[start_date:end_date]
        features_filtered = features.loc[start_date:end_date]
        
        # Ensure we have enough data
        if len(data_filtered) < 1000:
            self.logger.warning(f"Limited backtest data: {len(data_filtered)} hours")
        
        self.logger.info(f"Backtest period: {data_filtered.index[0]} to {data_filtered.index[-1]}")
        self.logger.info(f"Backtest data shape: {data_filtered.shape}")
        
        return data_filtered
    
    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame, 
                                     price_data: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if portfolio_df.empty:
            return {}
        
        # Portfolio returns
        portfolio_values = portfolio_df['total_value']
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        # Benchmark returns (Bitcoin)
        benchmark_col = f"{self.config.backtest.benchmark_symbol}_close"
        if benchmark_col in price_data.columns:
            benchmark_prices = price_data[benchmark_col].loc[portfolio_df.index]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Align returns
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]
        else:
            benchmark_returns = pd.Series()
        
        metrics = {}
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        metrics['total_return'] = total_return
        
        # Annualized return (assuming hourly data)
        n_periods = len(portfolio_returns)
        periods_per_year = 365.25 * 24  # Hours per year
        years = n_periods / periods_per_year
        metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        metrics['volatility'] = volatility
        
        # Sharpe ratio
        risk_free_rate = self.config.backtest.risk_free_rate
        excess_returns = portfolio_returns.mean() * periods_per_year - risk_free_rate
        metrics['sharpe_ratio'] = excess_returns / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdown.min())
        
        # Additional metrics if empyrical is available
        if emp is not None and len(portfolio_returns) > 0:
            try:
                metrics['sortino_ratio'] = emp.sortino_ratio(portfolio_returns, required_return=risk_free_rate/periods_per_year)
                metrics['calmar_ratio'] = emp.calmar_ratio(portfolio_returns, period='hourly')
                metrics['omega_ratio'] = emp.omega_ratio(portfolio_returns, risk_free=risk_free_rate/periods_per_year)
                metrics['tail_ratio'] = emp.tail_ratio(portfolio_returns)
                
                if len(benchmark_returns) > 0:
                    metrics['alpha'] = emp.alpha(portfolio_returns, benchmark_returns, risk_free=risk_free_rate/periods_per_year, period='hourly')
                    metrics['beta'] = emp.beta(portfolio_returns, benchmark_returns)
                    metrics['information_ratio'] = emp.excess_sharpe(portfolio_returns, benchmark_returns)
                
            except Exception as e:
                self.logger.warning(f"Error calculating empyrical metrics: {e}")
        
        # Trade-based metrics
        if hasattr(self, 'trades_history') and not self.trades_history.empty:
            trades = self.trades_history
            metrics['total_trades'] = len(trades)
            
            # Win rate (simplified - would need P&L calculation)
            # This is a placeholder - actual implementation would track trade P&L
            metrics['avg_trade_size'] = trades['value'].mean() if 'value' in trades.columns else 0
        
        return metrics
    
    def calculate_benchmark_comparison(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate performance vs benchmarks."""
        if self.portfolio_history is None:
            return {}
        
        portfolio_return = self.performance_metrics.get('total_return', 0)
        
        comparisons = {
            'portfolio_return': portfolio_return,
            'outperformance': {}
        }
        
        # Compare to buy-and-hold strategies
        for symbol in symbols[:3]:  # Top 3 for comparison
            try:
                # This would require price data for the full backtest period
                # Placeholder implementation
                comparisons['outperformance'][f'{symbol}_buyhold'] = 0.0
            except Exception as e:
                self.logger.warning(f"Could not calculate benchmark for {symbol}: {e}")
        
        return comparisons
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        if not self.results:
            return {"error": "No backtest results available"}
        
        report = {
            'summary': self.results,
            'performance_metrics': self.performance_metrics,
            'portfolio_statistics': self._calculate_portfolio_statistics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'trade_analysis': self._analyze_trades(),
            'monthly_returns': self._calculate_monthly_returns(),
        }
        
        return report
    
    def _calculate_portfolio_statistics(self) -> Dict[str, Any]:
        """Calculate portfolio-level statistics."""
        if self.portfolio_history is None:
            return {}
        
        df = self.portfolio_history
        
        stats = {
            'avg_num_positions': df['num_positions'].mean(),
            'max_num_positions': df['num_positions'].max(),
            'min_num_positions': df['num_positions'].min(),
            'portfolio_turnover': self._calculate_turnover(),
        }
        
        return stats
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-specific metrics."""
        if self.portfolio_history is None:
            return {}
        
        returns = self.portfolio_history['total_value'].pct_change().dropna()
        
        risk_metrics = {
            'var_95': returns.quantile(0.05),  # 95% VaR
            'var_99': returns.quantile(0.01),  # 99% VaR
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),  # Conditional VaR
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
        }
        
        return risk_metrics
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trading behavior."""
        if self.trades_history is None or self.trades_history.empty:
            return {}
        
        trades = self.trades_history
        
        analysis = {
            'total_trades': len(trades),
            'avg_trade_cost': trades['total_cost'].mean() if 'total_cost' in trades.columns else 0,
            'total_trading_costs': trades['total_cost'].sum() if 'total_cost' in trades.columns else 0,
        }
        
        return analysis
    
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly return breakdown."""
        if self.portfolio_history is None:
            return pd.DataFrame()
        
        monthly_values = self.portfolio_history['total_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        return monthly_returns.to_frame('monthly_return')
    
    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover rate."""
        # Simplified turnover calculation
        # In practice, this would track actual position changes
        return 0.0  # Placeholder
    
    def save_results(self, filepath: str):
        """Save backtest results to file."""
        import json
        
        # Prepare serializable results
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj


def run_walk_forward_backtest(data: pd.DataFrame, features: pd.DataFrame, 
                            config: Config, symbols: List[str]) -> Dict[str, Any]:
    """Run walk-forward backtest with multiple validation windows."""
    logger = logging.getLogger(__name__)
    logger.info("Starting walk-forward backtest")
    
    # Create multiple backtest engines for different periods
    results = []
    
    # Split data into multiple periods
    total_periods = 4  # Test on 4 different periods
    data_per_period = len(data) // total_periods
    
    for period in range(total_periods):
        start_idx = period * data_per_period
        end_idx = (period + 1) * data_per_period
        
        if end_idx > len(data):
            end_idx = len(data)
        
        period_data = data.iloc[start_idx:end_idx]
        period_features = features.iloc[start_idx:end_idx]
        
        # Create backtest config for this period
        period_config = config
        period_config.backtest.start_date = period_data.index[0].strftime('%Y-%m-%d')
        period_config.backtest.end_date = period_data.index[-1].strftime('%Y-%m-%d')
        
        # Run backtest
        engine = CryptoBacktestEngine(period_config)
        model = CryptoRandomForestModel(config.model)
        
        try:
            result = engine.run_backtest(period_data, period_features, model, symbols)
            result['period'] = period + 1
            result['period_start'] = period_data.index[0]
            result['period_end'] = period_data.index[-1]
            results.append(result)
            
            logger.info(f"Period {period + 1} completed. Return: {result['total_return']:.2%}")
            
        except Exception as e:
            logger.error(f"Error in period {period + 1}: {e}")
            continue
    
    # Aggregate results
    if results:
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['performance_metrics'].get('sharpe_ratio', 0) for r in results])
        
        summary = {
            'individual_periods': results,
            'aggregate_metrics': {
                'average_return': avg_return,
                'average_sharpe': avg_sharpe,
                'consistent_periods': sum(1 for r in results if r['total_return'] > 0),
                'total_periods': len(results)
            }
        }
        
        logger.info(f"Walk-forward backtest completed. Average return: {avg_return:.2%}")
        return summary
    else:
        return {"error": "No successful backtest periods"}