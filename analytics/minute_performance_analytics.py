"""Comprehensive performance analytics optimized for minute-level cryptocurrency trading."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import jarque_bera, normaltest
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization features will be limited.")

try:
    import empyrical as emp
except ImportError:
    emp = None

from utils.config import Config


class MinutePerformanceAnalytics:
    """Comprehensive performance analytics for minute-level trading."""
    
    def __init__(self, config: Config = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.risk_free_rate = 0.02  # Annual risk-free rate
        self.trading_days_per_year = 365.25
        self.minutes_per_year = self.trading_days_per_year * 24 * 60
        
        # Performance tracking
        self.performance_cache = {}
        self.benchmark_data = None
        
    def _get_default_config(self) -> Config:
        """Get default configuration."""
        from types import SimpleNamespace
        
        config = SimpleNamespace()
        config.analytics = SimpleNamespace()
        
        config.analytics.risk_free_rate = 0.02
        config.analytics.confidence_levels = [0.95, 0.99]
        config.analytics.rolling_windows = [60, 240, 1440]  # 1h, 4h, 24h
        config.analytics.benchmark_symbol = 'BTC-USD'
        
        return config
    
    def analyze_portfolio_performance(self, portfolio_history: pd.DataFrame,
                                    trades_history: pd.DataFrame = None,
                                    benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Comprehensive portfolio performance analysis."""
        
        self.logger.info("Starting comprehensive portfolio performance analysis")
        
        if portfolio_history.empty:
            return {"error": "Empty portfolio history provided"}
        
        # Ensure we have the required columns
        required_cols = ['total_value']
        if not all(col in portfolio_history.columns for col in required_cols):
            return {"error": f"Portfolio history missing required columns: {required_cols}"}
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_period': {
                'start': portfolio_history.index[0].isoformat(),
                'end': portfolio_history.index[-1].isoformat(),
                'duration_minutes': len(portfolio_history),
                'duration_days': len(portfolio_history) / (24 * 60)
            }
        }
        
        # Basic performance metrics
        results['basic_metrics'] = self._calculate_basic_metrics(portfolio_history)
        
        # Risk metrics
        results['risk_metrics'] = self._calculate_risk_metrics(portfolio_history)
        
        # High-frequency specific metrics
        results['hf_metrics'] = self._calculate_hf_metrics(portfolio_history)
        
        # Intraday analysis
        results['intraday_analysis'] = self._analyze_intraday_patterns(portfolio_history)
        
        # Drawdown analysis
        results['drawdown_analysis'] = self._analyze_drawdowns(portfolio_history)
        
        # Trade analysis (if available)
        if trades_history is not None and not trades_history.empty:
            results['trade_analysis'] = self._analyze_trades(trades_history)
        
        # Benchmark comparison (if available)
        if benchmark_data is not None and not benchmark_data.empty:
            results['benchmark_comparison'] = self._compare_to_benchmark(
                portfolio_history, benchmark_data
            )
        
        # Statistical analysis
        results['statistical_analysis'] = self._perform_statistical_analysis(portfolio_history)
        
        # Rolling performance analysis
        results['rolling_analysis'] = self._analyze_rolling_performance(portfolio_history)
        
        # Performance attribution
        results['attribution_analysis'] = self._analyze_performance_attribution(
            portfolio_history, trades_history
        )
        
        self.logger.info("Portfolio performance analysis completed")
        return results
    
    def _calculate_basic_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Total and annualized returns
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Calculate annualized return
        periods = len(returns)
        years = periods / self.minutes_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(self.minutes_per_year)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(self.minutes_per_year)
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Win rate
        positive_returns = (returns > 0).sum()
        win_rate = positive_returns / len(returns)
        
        # Average return per minute
        avg_return_per_minute = returns.mean()
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'win_rate': float(win_rate),
            'avg_return_per_minute': float(avg_return_per_minute),
            'total_periods': int(periods),
            'years_analyzed': float(years)
        }
    
    def _calculate_risk_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)  # 95% VaR
        var_99 = returns.quantile(0.01)  # 99% VaR
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = abs(returns.mean() * self.minutes_per_year) / max_drawdown if max_drawdown > 0 else 0
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio (95th percentile / 5th percentile)
        tail_ratio = abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
        
        # Beta vs benchmark (if available)
        beta = self._calculate_beta(returns) if hasattr(self, 'benchmark_returns') else np.nan
        
        return {
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'tail_ratio': float(tail_ratio),
            'beta': float(beta) if not np.isnan(beta) else None
        }
    
    def _calculate_hf_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Calculate high-frequency specific metrics."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Microstructure noise analysis
        autocorr_lag1 = returns.autocorr(lag=1) if len(returns) > 1 else 0
        autocorr_lag5 = returns.autocorr(lag=5) if len(returns) > 5 else 0
        
        # Realized volatility patterns
        realized_vol_1h = returns.rolling(60).std()
        realized_vol_4h = returns.rolling(240).std()
        realized_vol_24h = returns.rolling(1440).std()
        
        # Average realized volatilities
        avg_vol_1h = realized_vol_1h.mean() * np.sqrt(self.minutes_per_year)
        avg_vol_4h = realized_vol_4h.mean() * np.sqrt(self.minutes_per_year)
        avg_vol_24h = realized_vol_24h.mean() * np.sqrt(self.minutes_per_year)
        
        # Volatility clustering (ARCH effects)
        squared_returns = returns ** 2
        vol_clustering = squared_returns.autocorr(lag=1) if len(squared_returns) > 1 else 0
        
        # Jump detection
        return_threshold = returns.std() * 3  # 3 sigma threshold
        jumps = returns[abs(returns) > return_threshold]
        jump_frequency = len(jumps) / len(returns)
        
        # Market timing analysis
        positive_momentum_returns = returns[returns.shift(1) > 0]
        negative_momentum_returns = returns[returns.shift(1) < 0]
        
        momentum_persistence = {
            'positive_momentum_avg': positive_momentum_returns.mean() if len(positive_momentum_returns) > 0 else 0,
            'negative_momentum_avg': negative_momentum_returns.mean() if len(negative_momentum_returns) > 0 else 0
        }
        
        return {
            'autocorr_lag1': float(autocorr_lag1),
            'autocorr_lag5': float(autocorr_lag5),
            'volatility_clustering': float(vol_clustering),
            'jump_frequency': float(jump_frequency),
            'avg_realized_vol_1h': float(avg_vol_1h),
            'avg_realized_vol_4h': float(avg_vol_4h),
            'avg_realized_vol_24h': float(avg_vol_24h),
            'momentum_persistence': momentum_persistence
        }
    
    def _analyze_intraday_patterns(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze intraday performance patterns."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Add time components
        df = pd.DataFrame({'returns': returns})
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['minute'] = df.index.minute
        
        # Hourly patterns
        hourly_stats = df.groupby('hour')['returns'].agg(['mean', 'std', 'count'])
        best_hour = hourly_stats['mean'].idxmax()
        worst_hour = hourly_stats['mean'].idxmin()
        
        # Daily patterns
        daily_stats = df.groupby('day_of_week')['returns'].agg(['mean', 'std', 'count'])
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Intraday volatility patterns
        intraday_vol = df.groupby('hour')['returns'].std()
        vol_range = intraday_vol.max() - intraday_vol.min()
        
        # Market session analysis (crypto trades 24/7 but patterns exist)
        # Asian: 0-8 UTC, London: 8-16 UTC, NY: 16-24 UTC
        df['session'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], labels=['Asian', 'London', 'NY'], include_lowest=True)
        session_stats = df.groupby('session')['returns'].agg(['mean', 'std', 'count'])
        
        return {
            'hourly_patterns': {
                'best_hour': int(best_hour),
                'worst_hour': int(worst_hour),
                'best_hour_return': float(hourly_stats.loc[best_hour, 'mean']),
                'worst_hour_return': float(hourly_stats.loc[worst_hour, 'mean']),
                'volatility_range': float(vol_range)
            },
            'daily_patterns': {
                'daily_stats': {day_names[i]: {
                    'avg_return': float(daily_stats.iloc[i]['mean']),
                    'volatility': float(daily_stats.iloc[i]['std']),
                    'count': int(daily_stats.iloc[i]['count'])
                } for i in range(len(daily_stats))}
            },
            'session_analysis': {
                session: {
                    'avg_return': float(session_stats.loc[session, 'mean']),
                    'volatility': float(session_stats.loc[session, 'std']),
                    'count': int(session_stats.loc[session, 'count'])
                } for session in session_stats.index
            }
        }
    
    def _analyze_drawdowns(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Detailed drawdown analysis."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        if in_drawdown.any():
            # Group consecutive drawdown periods
            drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
            
            for group_id in drawdown_groups[in_drawdown].unique():
                group_mask = (drawdown_groups == group_id) & in_drawdown
                period_drawdown = drawdown[group_mask]
                
                if len(period_drawdown) > 0:
                    drawdown_periods.append({
                        'start': period_drawdown.index[0],
                        'end': period_drawdown.index[-1],
                        'duration_minutes': len(period_drawdown),
                        'max_drawdown': float(period_drawdown.min()),
                        'recovery_time': None  # Will be calculated if recovery occurs
                    })
        
        # Calculate recovery times
        for i, period in enumerate(drawdown_periods):
            end_time = period['end']
            end_value = cumulative[end_time]
            
            # Find when portfolio recovers to pre-drawdown level
            future_values = cumulative[cumulative.index > end_time]
            recovery_mask = future_values >= running_max[end_time]
            
            if recovery_mask.any():
                recovery_time = future_values[recovery_mask].index[0]
                period['recovery_time'] = (recovery_time - end_time).total_seconds() / 60
        
        # Summary statistics
        max_drawdown = abs(drawdown.min())
        avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0
        
        drawdown_stats = {
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'num_drawdown_periods': len(drawdown_periods),
            'drawdown_periods': drawdown_periods
        }
        
        if drawdown_periods:
            durations = [p['duration_minutes'] for p in drawdown_periods]
            recovery_times = [p['recovery_time'] for p in drawdown_periods if p['recovery_time'] is not None]
            
            drawdown_stats.update({
                'avg_drawdown_duration_minutes': float(np.mean(durations)),
                'max_drawdown_duration_minutes': float(np.max(durations)),
                'avg_recovery_time_minutes': float(np.mean(recovery_times)) if recovery_times else None,
                'max_recovery_time_minutes': float(np.max(recovery_times)) if recovery_times else None
            })
        
        return drawdown_stats
    
    def _analyze_trades(self, trades_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading behavior and performance."""
        
        if trades_history.empty:
            return {}
        
        # Ensure we have required columns
        if 'value' not in trades_history.columns:
            return {"error": "Trades history missing 'value' column"}
        
        # Basic trade statistics
        total_trades = len(trades_history)
        total_volume = abs(trades_history['value']).sum()
        avg_trade_size = abs(trades_history['value']).mean()
        
        # Trade direction analysis
        if 'quantity' in trades_history.columns:
            buy_trades = trades_history[trades_history['quantity'] > 0]
            sell_trades = trades_history[trades_history['quantity'] < 0]
            
            buy_count = len(buy_trades)
            sell_count = len(sell_trades)
            
            direction_stats = {
                'buy_trades': buy_count,
                'sell_trades': sell_count,
                'buy_ratio': buy_count / total_trades if total_trades > 0 else 0
            }
        else:
            direction_stats = {}
        
        # Transaction costs analysis
        if 'cost' in trades_history.columns:
            total_costs = trades_history['cost'].sum()
            avg_cost_per_trade = trades_history['cost'].mean()
            cost_as_pct_of_volume = total_costs / total_volume if total_volume > 0 else 0
            
            cost_stats = {
                'total_transaction_costs': float(total_costs),
                'avg_cost_per_trade': float(avg_cost_per_trade),
                'cost_as_pct_of_volume': float(cost_as_pct_of_volume)
            }
        else:
            cost_stats = {}
        
        # Trading frequency analysis
        if len(trades_history) > 1:
            time_between_trades = trades_history.index.to_series().diff().dt.total_seconds() / 60
            avg_time_between_trades = time_between_trades.mean()
            
            frequency_stats = {
                'avg_minutes_between_trades': float(avg_time_between_trades),
                'trades_per_hour': 60 / avg_time_between_trades if avg_time_between_trades > 0 else 0
            }
        else:
            frequency_stats = {}
        
        # Symbol analysis
        if 'symbol' in trades_history.columns:
            symbol_stats = trades_history.groupby('symbol').agg({
                'value': ['count', 'sum', 'mean'],
                'cost': 'sum' if 'cost' in trades_history.columns else lambda x: 0
            }).round(4)
            
            symbol_analysis = symbol_stats.to_dict()
        else:
            symbol_analysis = {}
        
        return {
            'basic_stats': {
                'total_trades': int(total_trades),
                'total_volume': float(total_volume),
                'avg_trade_size': float(avg_trade_size)
            },
            'direction_stats': direction_stats,
            'cost_stats': cost_stats,
            'frequency_stats': frequency_stats,
            'symbol_analysis': symbol_analysis
        }
    
    def _compare_to_benchmark(self, portfolio_history: pd.DataFrame, 
                            benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare portfolio performance to benchmark."""
        
        portfolio_values = portfolio_history['total_value']
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        # Align benchmark data
        if 'close' in benchmark_data.columns:
            benchmark_prices = benchmark_data['close']
        elif 'Close' in benchmark_data.columns:
            benchmark_prices = benchmark_data['Close']
        else:
            return {"error": "Benchmark data missing price column"}
        
        # Align time series
        common_index = portfolio_returns.index.intersection(benchmark_prices.index)
        if len(common_index) == 0:
            return {"error": "No common time periods between portfolio and benchmark"}
        
        portfolio_aligned = portfolio_returns.loc[common_index]
        benchmark_aligned = benchmark_prices.loc[common_index].pct_change().dropna()
        
        # Further align after pct_change
        common_index_final = portfolio_aligned.index.intersection(benchmark_aligned.index)
        portfolio_final = portfolio_aligned.loc[common_index_final]
        benchmark_final = benchmark_aligned.loc[common_index_final]
        
        if len(portfolio_final) == 0:
            return {"error": "No aligned returns available"}
        
        # Calculate comparison metrics
        portfolio_total_return = (1 + portfolio_final).prod() - 1
        benchmark_total_return = (1 + benchmark_final).prod() - 1
        
        # Alpha and beta
        if len(portfolio_final) > 10:  # Need sufficient data
            beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_final, portfolio_final)
            
            # Tracking error
            excess_returns = portfolio_final - benchmark_final
            tracking_error = excess_returns.std() * np.sqrt(self.minutes_per_year)
            
            # Information ratio
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # Correlation
            correlation = portfolio_final.corr(benchmark_final)
            
        else:
            beta = alpha = r_value = p_value = std_err = tracking_error = information_ratio = correlation = np.nan
        
        return {
            'comparison_period': {
                'start': common_index_final[0].isoformat(),
                'end': common_index_final[-1].isoformat(),
                'periods': len(portfolio_final)
            },
            'returns_comparison': {
                'portfolio_total_return': float(portfolio_total_return),
                'benchmark_total_return': float(benchmark_total_return),
                'excess_return': float(portfolio_total_return - benchmark_total_return)
            },
            'risk_adjusted_metrics': {
                'beta': float(beta) if not np.isnan(beta) else None,
                'alpha': float(alpha) if not np.isnan(alpha) else None,
                'correlation': float(correlation) if not np.isnan(correlation) else None,
                'tracking_error': float(tracking_error) if not np.isnan(tracking_error) else None,
                'information_ratio': float(information_ratio) if not np.isnan(information_ratio) else None,
                'r_squared': float(r_value**2) if not np.isnan(r_value) else None
            }
        }
    
    def _perform_statistical_analysis(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on returns."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) < 10:  # Need minimum data for statistical tests
            return {}
        
        # Normality tests
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
            normal_stat, normal_pvalue = normaltest(returns)
        except:
            jb_stat = jb_pvalue = normal_stat = normal_pvalue = np.nan
        
        # Autocorrelation tests
        autocorr_stats = {}
        for lag in [1, 5, 10, 30]:
            if len(returns) > lag:
                autocorr_stats[f'lag_{lag}'] = float(returns.autocorr(lag=lag))
        
        # Distribution statistics
        distribution_stats = {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'min': float(returns.min()),
            'max': float(returns.max()),
            'median': float(returns.median())
        }
        
        # Percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            percentiles[f'p{p}'] = float(returns.quantile(p/100))
        
        return {
            'normality_tests': {
                'jarque_bera_stat': float(jb_stat) if not np.isnan(jb_stat) else None,
                'jarque_bera_pvalue': float(jb_pvalue) if not np.isnan(jb_pvalue) else None,
                'dagostino_stat': float(normal_stat) if not np.isnan(normal_stat) else None,
                'dagostino_pvalue': float(normal_pvalue) if not np.isnan(normal_pvalue) else None
            },
            'autocorrelation': autocorr_stats,
            'distribution_stats': distribution_stats,
            'percentiles': percentiles
        }
    
    def _analyze_rolling_performance(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rolling performance metrics."""
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) < 60:  # Need minimum 1 hour of data
            return {}
        
        rolling_metrics = {}
        
        # Define rolling windows (in minutes)
        windows = [60, 240, 1440]  # 1h, 4h, 24h
        
        for window in windows:
            if len(returns) >= window:
                # Rolling returns
                rolling_returns = returns.rolling(window).sum()
                
                # Rolling volatility
                rolling_vol = returns.rolling(window).std() * np.sqrt(self.minutes_per_year / window)
                
                # Rolling Sharpe ratio
                rolling_sharpe = (rolling_returns.mean() - self.risk_free_rate / self.minutes_per_year * window) / rolling_vol
                
                # Rolling maximum drawdown
                rolling_cumulative = (1 + returns).rolling(window).apply(np.prod, raw=True)
                rolling_max = rolling_cumulative.rolling(window).max()
                rolling_dd = (rolling_cumulative - rolling_max) / rolling_max
                rolling_max_dd = rolling_dd.rolling(window).min()
                
                window_label = f'{window}min'
                rolling_metrics[window_label] = {
                    'avg_return': float(rolling_returns.mean()),
                    'avg_volatility': float(rolling_vol.mean()),
                    'avg_sharpe': float(rolling_sharpe.mean()) if not rolling_sharpe.isna().all() else None,
                    'avg_max_drawdown': float(abs(rolling_max_dd.mean())) if not rolling_max_dd.isna().all() else None,
                    'best_period_return': float(rolling_returns.max()),
                    'worst_period_return': float(rolling_returns.min())
                }
        
        return rolling_metrics
    
    def _analyze_performance_attribution(self, portfolio_history: pd.DataFrame,
                                       trades_history: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze sources of portfolio performance."""
        
        attribution = {}
        
        # Time-based attribution
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) > 0:
            # Hour-of-day attribution
            df = pd.DataFrame({'returns': returns})
            df['hour'] = df.index.hour
            
            hourly_contribution = df.groupby('hour')['returns'].sum()
            attribution['hourly_contribution'] = hourly_contribution.to_dict()
            
            # Day-of-week attribution
            df['day_of_week'] = df.index.dayofweek
            daily_contribution = df.groupby('day_of_week')['returns'].sum()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            attribution['daily_contribution'] = {
                day_names[i]: float(daily_contribution.iloc[i]) 
                for i in range(len(daily_contribution))
            }
        
        # Asset-based attribution (if trade data available)
        if trades_history is not None and not trades_history.empty and 'symbol' in trades_history.columns:
            symbol_pnl = {}
            
            for symbol in trades_history['symbol'].unique():
                symbol_trades = trades_history[trades_history['symbol'] == symbol]
                symbol_volume = abs(symbol_trades['value']).sum()
                symbol_costs = symbol_trades['cost'].sum() if 'cost' in symbol_trades.columns else 0
                
                symbol_pnl[symbol] = {
                    'total_volume': float(symbol_volume),
                    'total_costs': float(symbol_costs),
                    'trade_count': int(len(symbol_trades))
                }
            
            attribution['symbol_attribution'] = symbol_pnl
        
        return attribution
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta vs stored benchmark returns."""
        if not hasattr(self, 'benchmark_returns') or len(self.benchmark_returns) == 0:
            return np.nan
        
        # Align returns
        common_index = returns.index.intersection(self.benchmark_returns.index)
        if len(common_index) < 10:
            return np.nan
        
        portfolio_aligned = returns.loc[common_index]
        benchmark_aligned = self.benchmark_returns.loc[common_index]
        
        # Calculate beta using linear regression
        try:
            beta, _, _, _, _ = stats.linregress(benchmark_aligned, portfolio_aligned)
            return beta
        except:
            return np.nan
    
    def generate_performance_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive text report."""
        
        report = []
        report.append("=" * 80)
        report.append("MINUTE-LEVEL PORTFOLIO PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Timestamp: {analysis_results.get('analysis_timestamp', 'Unknown')}")
        
        # Analysis period
        if 'analysis_period' in analysis_results:
            period = analysis_results['analysis_period']
            report.append(f"\nAnalysis Period:")
            report.append(f"  Start: {period.get('start', 'Unknown')}")
            report.append(f"  End: {period.get('end', 'Unknown')}")
            report.append(f"  Duration: {period.get('duration_days', 0):.1f} days ({period.get('duration_minutes', 0):,} minutes)")
        
        # Basic metrics
        if 'basic_metrics' in analysis_results:
            metrics = analysis_results['basic_metrics']
            report.append(f"\nBASIC PERFORMANCE METRICS:")
            report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            report.append(f"  Volatility: {metrics.get('volatility', 0):.2%}")
            report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            report.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        
        # Risk metrics
        if 'risk_metrics' in analysis_results:
            risk = analysis_results['risk_metrics']
            report.append(f"\nRISK METRICS:")
            report.append(f"  Maximum Drawdown: {risk.get('max_drawdown', 0):.2%}")
            report.append(f"  VaR (95%): {risk.get('var_95', 0):.4f}")
            report.append(f"  VaR (99%): {risk.get('var_99', 0):.4f}")
            report.append(f"  CVaR (95%): {risk.get('cvar_95', 0):.4f}")
            report.append(f"  Calmar Ratio: {risk.get('calmar_ratio', 0):.2f}")
            report.append(f"  Skewness: {risk.get('skewness', 0):.2f}")
            report.append(f"  Kurtosis: {risk.get('kurtosis', 0):.2f}")
        
        # High-frequency metrics
        if 'hf_metrics' in analysis_results:
            hf = analysis_results['hf_metrics']
            report.append(f"\nHIGH-FREQUENCY METRICS:")
            report.append(f"  Autocorrelation (lag-1): {hf.get('autocorr_lag1', 0):.3f}")
            report.append(f"  Volatility Clustering: {hf.get('volatility_clustering', 0):.3f}")
            report.append(f"  Jump Frequency: {hf.get('jump_frequency', 0):.3%}")
            report.append(f"  Avg Realized Vol (1h): {hf.get('avg_realized_vol_1h', 0):.2%}")
            report.append(f"  Avg Realized Vol (24h): {hf.get('avg_realized_vol_24h', 0):.2%}")
        
        # Trade analysis
        if 'trade_analysis' in analysis_results and 'basic_stats' in analysis_results['trade_analysis']:
            trades = analysis_results['trade_analysis']['basic_stats']
            report.append(f"\nTRADE ANALYSIS:")
            report.append(f"  Total Trades: {trades.get('total_trades', 0):,}")
            report.append(f"  Total Volume: ${trades.get('total_volume', 0):,.2f}")
            report.append(f"  Average Trade Size: ${trades.get('avg_trade_size', 0):,.2f}")
        
        # Drawdown analysis
        if 'drawdown_analysis' in analysis_results:
            dd = analysis_results['drawdown_analysis']
            report.append(f"\nDRAWDOWN ANALYSIS:")
            report.append(f"  Maximum Drawdown: {dd.get('max_drawdown', 0):.2%}")
            report.append(f"  Average Drawdown: {dd.get('avg_drawdown', 0):.2%}")
            report.append(f"  Number of Drawdown Periods: {dd.get('num_drawdown_periods', 0)}")
            if dd.get('avg_drawdown_duration_minutes'):
                report.append(f"  Average Drawdown Duration: {dd['avg_drawdown_duration_minutes']:.0f} minutes")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Utility functions

def analyze_minute_portfolio(portfolio_history: pd.DataFrame,
                           trades_history: pd.DataFrame = None,
                           benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
    """Convenience function for complete portfolio analysis."""
    
    analyzer = MinutePerformanceAnalytics()
    return analyzer.analyze_portfolio_performance(portfolio_history, trades_history, benchmark_data)


def generate_performance_summary(analysis_results: Dict[str, Any]) -> Dict[str, float]:
    """Generate a summary of key performance metrics."""
    
    summary = {}
    
    if 'basic_metrics' in analysis_results:
        basic = analysis_results['basic_metrics']
        summary.update({
            'total_return': basic.get('total_return', 0),
            'annualized_return': basic.get('annualized_return', 0),
            'sharpe_ratio': basic.get('sharpe_ratio', 0),
            'win_rate': basic.get('win_rate', 0)
        })
    
    if 'risk_metrics' in analysis_results:
        risk = analysis_results['risk_metrics']
        summary.update({
            'max_drawdown': risk.get('max_drawdown', 0),
            'volatility': risk.get('volatility', 0),
            'var_95': risk.get('var_95', 0)
        })
    
    if 'trade_analysis' in analysis_results and 'basic_stats' in analysis_results['trade_analysis']:
        trades = analysis_results['trade_analysis']['basic_stats']
        summary.update({
            'total_trades': trades.get('total_trades', 0),
            'avg_trade_size': trades.get('avg_trade_size', 0)
        })
    
    return summary


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample portfolio data
    dates = pd.date_range('2024-01-01', '2024-01-02', freq='1T')
    np.random.seed(42)
    
    # Simulate portfolio performance
    initial_value = 100000
    returns = np.random.normal(0.0001, 0.01, len(dates))  # Small positive drift with volatility
    
    portfolio_values = [initial_value]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    portfolio_history = pd.DataFrame({
        'total_value': portfolio_values[1:],  # Skip initial value
        'cash': portfolio_values[1:] * 0.1,  # 10% cash
        'position_value': portfolio_values[1:] * 0.9,  # 90% invested
        'num_positions': np.random.randint(1, 6, len(dates))
    }, index=dates)
    
    # Create sample trades
    n_trades = 50
    trade_times = np.random.choice(dates, n_trades, replace=False)
    trades_history = pd.DataFrame({
        'symbol': np.random.choice(['BTC-USD', 'ETH-USD'], n_trades),
        'quantity': np.random.normal(0, 1, n_trades),
        'price': np.random.normal(50000, 5000, n_trades),
        'value': np.random.normal(0, 5000, n_trades),
        'cost': np.random.exponential(50, n_trades)
    }, index=trade_times)
    
    # Create benchmark data
    benchmark_returns = np.random.normal(0.00005, 0.008, len(dates))
    benchmark_prices = [50000]
    for ret in benchmark_returns:
        benchmark_prices.append(benchmark_prices[-1] * (1 + ret))
    
    benchmark_data = pd.DataFrame({
        'close': benchmark_prices[1:]
    }, index=dates)
    
    print(f"Sample data created:")
    print(f"  Portfolio: {len(portfolio_history)} minutes")
    print(f"  Trades: {len(trades_history)} trades")
    print(f"  Benchmark: {len(benchmark_data)} minutes")
    
    # Run analysis
    print("\nRunning comprehensive performance analysis...")
    analyzer = MinutePerformanceAnalytics()
    
    results = analyzer.analyze_portfolio_performance(
        portfolio_history, trades_history, benchmark_data
    )
    
    # Generate summary
    summary = generate_performance_summary(results)
    
    print("\nPERFORMANCE SUMMARY:")
    for metric, value in summary.items():
        if isinstance(value, (int, float)):
            if 'return' in metric or 'rate' in metric:
                print(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
            elif 'ratio' in metric:
                print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  {metric.replace('_', ' ').title()}: {value:,.2f}")
    
    # Generate full report
    print("\nGenerating detailed report...")
    report = analyzer.generate_performance_report(results)
    print(report)