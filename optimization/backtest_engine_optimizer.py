"""Optimized backtesting engine with vectorized operations and parallel processing."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from numba import jit, njit, prange, vectorize
import warnings
warnings.filterwarnings('ignore')

from utils.config import BacktestConfig


class OptimizedBacktestEngine:
    """Highly optimized backtesting engine using vectorization and parallel processing."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization settings
        self.use_parallel = True
        self.n_workers = mp.cpu_count() - 1
        self.chunk_size = 10000  # Process data in chunks
        
        # Pre-allocate arrays for performance
        self.position_history = []
        self.trade_history = []
        self.equity_curve = []
        
        # Vectorized commission and slippage
        self.commission_rate = config.commission_rate
        self.slippage_rate = config.slippage_rate
    
    def run_backtest_optimized(self, data: pd.DataFrame, signals: pd.DataFrame,
                               initial_capital: float = 100000) -> Dict[str, Any]:
        """Run optimized backtest using vectorized operations."""
        self.logger.info("Starting optimized backtest")
        start_time = datetime.now()
        
        # Ensure data is sorted by time
        data = data.sort_index()
        signals = signals.sort_index()
        
        # Align data and signals
        common_index = data.index.intersection(signals.index)
        data = data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Convert to numpy arrays for speed
        prices = data['close'].values.astype(np.float32)
        signal_values = signals.values.astype(np.float32) if len(signals.shape) == 1 else signals.values[:, 0].astype(np.float32)
        
        # Vectorized backtest
        results = self._run_vectorized_backtest(
            prices=prices,
            signals=signal_values,
            initial_capital=initial_capital,
            timestamps=data.index.values
        )
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics_vectorized(results)
        
        # Add timing information
        metrics['backtest_time'] = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Backtest completed in {metrics['backtest_time']:.2f} seconds")
        
        return {
            'metrics': metrics,
            'equity_curve': results['equity_curve'],
            'positions': results['positions'],
            'trades': results['trades'],
            'returns': results['returns']
        }
    
    @staticmethod
    @njit(parallel=True)
    def _run_vectorized_backtest(prices: np.ndarray, signals: np.ndarray,
                                initial_capital: float, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized backtest implementation using Numba."""
        n = len(prices)
        
        # Pre-allocate arrays
        positions = np.zeros(n, dtype=np.float32)
        cash = np.zeros(n, dtype=np.float32)
        equity = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        # Initialize
        cash[0] = initial_capital
        equity[0] = initial_capital
        
        # Trading parameters
        commission_rate = 0.001  # 0.1%
        slippage_rate = 0.0005   # 0.05%
        
        # Track trades
        trade_count = 0
        max_trades = n // 10  # Estimate max trades
        trade_entries = np.zeros(max_trades, dtype=np.int32)
        trade_exits = np.zeros(max_trades, dtype=np.int32)
        trade_returns = np.zeros(max_trades, dtype=np.float32)
        
        # Main backtest loop - vectorized where possible
        for i in prange(1, n):
            # Previous position
            prev_position = positions[i-1] if i > 0 else 0
            
            # Current signal
            target_position = signals[i]
            
            # Position change
            position_delta = target_position - prev_position
            
            # Calculate trade cost
            if position_delta != 0:
                trade_value = abs(position_delta * prices[i])
                commission = trade_value * commission_rate
                slippage = trade_value * slippage_rate
                total_cost = commission + slippage
            else:
                total_cost = 0
            
            # Update cash
            cash[i] = cash[i-1] - position_delta * prices[i] - total_cost
            
            # Update position
            positions[i] = target_position
            
            # Calculate equity
            equity[i] = cash[i] + positions[i] * prices[i]
            
            # Calculate returns
            if equity[i-1] > 0:
                returns[i] = (equity[i] - equity[i-1]) / equity[i-1]
            else:
                returns[i] = 0
            
            # Track trades
            if position_delta > 0 and trade_count < max_trades:
                trade_entries[trade_count] = i
                trade_count += 1
            elif position_delta < 0 and trade_count > 0:
                # Find matching entry
                for j in range(trade_count-1, -1, -1):
                    if trade_exits[j] == 0:
                        trade_exits[j] = i
                        entry_price = prices[trade_entries[j]]
                        exit_price = prices[i]
                        trade_returns[j] = (exit_price - entry_price) / entry_price
                        break
        
        # Prepare results
        results = {
            'equity_curve': equity,
            'positions': positions,
            'cash': cash,
            'returns': returns,
            'trade_entries': trade_entries[:trade_count],
            'trade_exits': trade_exits[:trade_count],
            'trade_returns': trade_returns[:trade_count]
        }
        
        return results
    
    def _calculate_performance_metrics_vectorized(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate performance metrics using vectorized operations."""
        equity = results['equity_curve']
        returns = results['returns']
        positions = results['positions']
        
        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Vectorized metrics calculation
        metrics = {
            'total_return': total_return,
            'annual_return': self._calculate_annual_return_vectorized(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio_vectorized(returns),
            'sortino_ratio': self._calculate_sortino_ratio_vectorized(returns),
            'max_drawdown': self._calculate_max_drawdown_vectorized(equity),
            'calmar_ratio': self._calculate_calmar_ratio_vectorized(returns, equity),
            'win_rate': self._calculate_win_rate_vectorized(results['trade_returns']),
            'profit_factor': self._calculate_profit_factor_vectorized(results['trade_returns']),
            'avg_trade_return': np.mean(results['trade_returns']) if len(results['trade_returns']) > 0 else 0,
            'total_trades': len(results['trade_returns']),
            'avg_position': np.mean(np.abs(positions)),
            'max_position': np.max(np.abs(positions))
        }
        
        return metrics
    
    @staticmethod
    @njit
    def _calculate_annual_return_vectorized(returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        # Compound returns
        cum_return = 1.0
        for r in returns:
            cum_return *= (1 + r)
        
        # Annualize (assuming hourly data, 24*365 hours per year)
        n_periods = len(returns)
        years = n_periods / (24 * 365)
        
        if years > 0:
            annual_return = (cum_return ** (1 / years)) - 1
            return annual_return
        else:
            return 0.0
    
    @staticmethod
    @njit
    def _calculate_sharpe_ratio_vectorized(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate / (24 * 365)  # Hourly risk-free rate
        
        # Calculate mean and std
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess > 0:
            # Annualize
            sharpe = mean_excess / std_excess * np.sqrt(24 * 365)
            return sharpe
        else:
            return 0.0
    
    @staticmethod
    @njit
    def _calculate_sortino_ratio_vectorized(returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate downside returns
        downside_returns = np.zeros(len(returns))
        for i in range(len(returns)):
            if returns[i] < target_return:
                downside_returns[i] = returns[i] - target_return
        
        # Calculate downside deviation
        downside_dev = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_dev > 0:
            mean_excess = np.mean(returns) - target_return
            sortino = mean_excess / downside_dev * np.sqrt(24 * 365)
            return sortino
        else:
            return 0.0
    
    @staticmethod
    @njit
    def _calculate_max_drawdown_vectorized(equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = np.zeros(len(equity))
        running_max[0] = equity[0]
        
        for i in range(1, len(equity)):
            running_max[i] = max(running_max[i-1], equity[i])
        
        # Calculate drawdowns
        drawdowns = (equity - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = np.min(drawdowns)
        
        return abs(max_dd)
    
    @staticmethod
    @njit
    def _calculate_calmar_ratio_vectorized(returns: np.ndarray, equity: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        annual_return = OptimizedBacktestEngine._calculate_annual_return_vectorized(returns)
        max_dd = OptimizedBacktestEngine._calculate_max_drawdown_vectorized(equity)
        
        if max_dd > 0:
            return annual_return / max_dd
        else:
            return 0.0
    
    @staticmethod
    @njit
    def _calculate_win_rate_vectorized(trade_returns: np.ndarray) -> float:
        """Calculate win rate."""
        if len(trade_returns) == 0:
            return 0.0
        
        wins = 0
        for ret in trade_returns:
            if ret > 0:
                wins += 1
        
        return wins / len(trade_returns)
    
    @staticmethod
    @njit
    def _calculate_profit_factor_vectorized(trade_returns: np.ndarray) -> float:
        """Calculate profit factor."""
        if len(trade_returns) == 0:
            return 0.0
        
        gross_profit = 0.0
        gross_loss = 0.0
        
        for ret in trade_returns:
            if ret > 0:
                gross_profit += ret
            else:
                gross_loss += abs(ret)
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        else:
            return gross_profit if gross_profit > 0 else 0.0
    
    def run_parallel_backtest_multiple_params(self, data: pd.DataFrame, 
                                            param_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run backtests for multiple parameter combinations in parallel."""
        self.logger.info(f"Running {len(param_combinations)} backtests in parallel")
        
        # Prepare data once
        data_prepared = data.copy()
        
        # Run backtests in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for i, params in enumerate(param_combinations):
                future = executor.submit(
                    self._run_single_param_backtest,
                    data_prepared,
                    params,
                    i
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Backtest failed: {e}")
        
        # Sort by performance
        results.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        
        return results
    
    def _run_single_param_backtest(self, data: pd.DataFrame, params: Dict[str, Any], 
                                  param_id: int) -> Dict[str, Any]:
        """Run backtest for single parameter combination."""
        # Generate signals based on parameters
        signals = self._generate_signals_from_params(data, params)
        
        # Run backtest
        result = self.run_backtest_optimized(data, signals)
        
        # Add parameter information
        result['params'] = params
        result['param_id'] = param_id
        
        return result
    
    def _generate_signals_from_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Generate trading signals from parameters."""
        # Example signal generation - replace with actual strategy
        fast_ma = data['close'].rolling(params.get('fast_period', 10)).mean()
        slow_ma = data['close'].rolling(params.get('slow_period', 30)).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals