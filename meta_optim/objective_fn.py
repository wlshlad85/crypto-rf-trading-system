#!/usr/bin/env python3
"""
Multi-Metric Objective Function for Random Forest Meta-Optimization

Implements sophisticated composite scoring that considers:
- Sharpe ratio (risk-adjusted returns)
- Profit factor (gross profits / gross losses)
- Maximum drawdown (worst peak-to-trough decline)
- Alpha persistence (rolling 4-week performance stability)
- Trade efficiency metrics

Usage: from objective_fn import MetaObjectiveFunction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MetaObjectiveFunction:
    """Sophisticated objective function for trading strategy optimization."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize with custom metric weights."""
        self.weights = weights or {
            'sharpe_ratio': 0.4,
            'profit_factor': 0.3,
            'max_drawdown': -0.2,  # Negative because lower is better
            'alpha_persistence': 0.1,
            'trade_efficiency': 0.05,
            'volatility_penalty': -0.05  # Penalty for excessive volatility
        }
        
        # Minimum thresholds for viable strategies
        self.min_thresholds = {
            'total_trades': 10,
            'sharpe_ratio': 0.5,
            'profit_factor': 1.1,
            'max_drawdown': -0.3  # Max 30% drawdown
        }
        
    def evaluate_strategy(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy performance using composite scoring."""
        
        # Extract performance data
        trades_df = pd.DataFrame(backtest_results.get('trades', []))
        portfolio_series = pd.Series(backtest_results.get('portfolio_values', []))
        
        if len(trades_df) == 0 or len(portfolio_series) == 0:
            return {'composite_score': -999.0, 'viable': False, 'reason': 'No trades or portfolio data'}
        
        # Calculate individual metrics
        metrics = {}
        
        try:
            # 1. Sharpe Ratio
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(portfolio_series)
            
            # 2. Profit Factor
            metrics['profit_factor'] = self._calculate_profit_factor(trades_df)
            
            # 3. Maximum Drawdown
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_series)
            
            # 4. Alpha Persistence
            metrics['alpha_persistence'] = self._calculate_alpha_persistence(portfolio_series)
            
            # 5. Trade Efficiency
            metrics['trade_efficiency'] = self._calculate_trade_efficiency(trades_df)
            
            # 6. Volatility Penalty
            metrics['volatility_penalty'] = self._calculate_volatility_penalty(portfolio_series)
            
            # 7. Additional risk metrics
            metrics.update(self._calculate_risk_metrics(portfolio_series, trades_df))
            
            # Check viability thresholds
            viability_check = self._check_viability(metrics, len(trades_df))
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(metrics)
            
            return {
                'composite_score': composite_score,
                'viable': viability_check['viable'],
                'reason': viability_check.get('reason', 'Viable strategy'),
                'individual_metrics': metrics,
                'weights_used': self.weights
            }
            
        except Exception as e:
            return {
                'composite_score': -999.0,
                'viable': False,
                'reason': f'Evaluation error: {str(e)}',
                'individual_metrics': {},
                'weights_used': self.weights
            }
    
    def _calculate_sharpe_ratio(self, portfolio_series: pd.Series) -> float:
        """Calculate Sharpe ratio (risk-adjusted returns)."""
        if len(portfolio_series) < 2:
            return 0.0
        
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        excess_return = returns.mean()
        volatility = returns.std()
        sharpe = (excess_return / volatility) * np.sqrt(252) if volatility > 0 else 0.0
        
        return np.clip(sharpe, -5.0, 5.0)  # Cap extreme values
    
    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        if 'pnl' not in trades_df.columns and 'profit' in trades_df.columns:
            trades_df['pnl'] = trades_df['profit']
        
        if 'pnl' not in trades_df.columns:
            # Calculate PnL from trade data
            if 'action' in trades_df.columns and 'value' in trades_df.columns:
                buy_value = trades_df[trades_df['action'] == 'BUY']['value'].sum()
                sell_value = trades_df[trades_df['action'] == 'SELL']['value'].sum()
                if buy_value > 0:
                    return sell_value / buy_value
            return 1.0
        
        profits = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        if losses == 0:
            return 10.0 if profits > 0 else 1.0
        
        return profits / losses if losses > 0 else 1.0
    
    def _calculate_max_drawdown(self, portfolio_series: pd.Series) -> float:
        """Calculate maximum drawdown (worst peak-to-trough decline)."""
        if len(portfolio_series) < 2:
            return 0.0
        
        # Calculate running maximum (peak)
        peak = portfolio_series.expanding().max()
        
        # Calculate drawdown as percentage decline from peak
        drawdown = (portfolio_series - peak) / peak
        
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def _calculate_alpha_persistence(self, portfolio_series: pd.Series, window: int = 28) -> float:
        """Calculate alpha persistence (rolling 4-week performance stability)."""
        if len(portfolio_series) < window * 2:
            return 0.0
        
        # Calculate rolling returns
        rolling_returns = portfolio_series.rolling(window).apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) if x.iloc[0] > 0 else 0
        ).dropna()
        
        if len(rolling_returns) < 2:
            return 0.0
        
        # Alpha persistence = consistency of positive rolling returns
        positive_periods = (rolling_returns > 0).sum()
        total_periods = len(rolling_returns)
        
        # Also consider stability (lower volatility of rolling returns)
        return_stability = 1 / (1 + rolling_returns.std()) if rolling_returns.std() > 0 else 1.0
        
        # Combined metric
        consistency = positive_periods / total_periods
        alpha_persistence = (consistency * 0.7) + (return_stability * 0.3)
        
        return np.clip(alpha_persistence, 0.0, 1.0)
    
    def _calculate_trade_efficiency(self, trades_df: pd.DataFrame) -> float:
        """Calculate trade efficiency metrics."""
        if len(trades_df) == 0:
            return 0.0
        
        # Win rate
        if 'pnl' in trades_df.columns:
            win_rate = (trades_df['pnl'] > 0).mean()
        else:
            # Simple approximation based on buy/sell balance
            buy_count = (trades_df['action'] == 'BUY').sum()
            sell_count = (trades_df['action'] == 'SELL').sum()
            win_rate = 0.6 if sell_count > 0 else 0.4  # Assume reasonable win rate
        
        # Trade frequency efficiency (not too many, not too few)
        trade_frequency = len(trades_df)
        frequency_score = 1.0 / (1.0 + abs(trade_frequency - 50) / 50)  # Optimal around 50 trades
        
        # Combined efficiency score
        efficiency = (win_rate * 0.7) + (frequency_score * 0.3)
        
        return np.clip(efficiency, 0.0, 1.0)
    
    def _calculate_volatility_penalty(self, portfolio_series: pd.Series) -> float:
        """Calculate penalty for excessive volatility."""
        if len(portfolio_series) < 2:
            return 0.0
        
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Penalty increases with volatility above 20%
        target_volatility = 0.20
        excess_volatility = max(0, volatility - target_volatility)
        
        # Exponential penalty for high volatility
        penalty = np.exp(excess_volatility * 5) - 1
        
        return -min(penalty, 2.0)  # Cap penalty at -2.0
    
    def _calculate_risk_metrics(self, portfolio_series: pd.Series, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional risk and performance metrics."""
        metrics = {}
        
        # Sortino ratio (downside deviation focus)
        if len(portfolio_series) >= 2:
            returns = portfolio_series.pct_change().dropna()
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std() * np.sqrt(252)
                metrics['sortino_ratio'] = (returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
            else:
                metrics['sortino_ratio'] = 5.0  # No negative returns
        else:
            metrics['sortino_ratio'] = 0.0
        
        # Calmar ratio (annual return / max drawdown)
        if len(portfolio_series) >= 2:
            annual_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) ** (252 / len(portfolio_series)) - 1
            max_dd = abs(self._calculate_max_drawdown(portfolio_series))
            metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0.01 else 0
        else:
            metrics['calmar_ratio'] = 0.0
        
        # Recovery factor (total return / max drawdown)
        if len(portfolio_series) >= 2:
            total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
            max_dd = abs(self._calculate_max_drawdown(portfolio_series))
            metrics['recovery_factor'] = total_return / max_dd if max_dd > 0.01 else 0
        else:
            metrics['recovery_factor'] = 0.0
        
        # Trade consistency (standard deviation of trade returns)
        if 'pnl' in trades_df.columns and len(trades_df) > 1:
            trade_consistency = 1 / (1 + trades_df['pnl'].std()) if trades_df['pnl'].std() > 0 else 1.0
            metrics['trade_consistency'] = trade_consistency
        else:
            metrics['trade_consistency'] = 0.5
        
        return metrics
    
    def _check_viability(self, metrics: Dict[str, float], num_trades: int) -> Dict[str, Any]:
        """Check if strategy meets minimum viability thresholds."""
        
        # Check minimum number of trades
        if num_trades < self.min_thresholds['total_trades']:
            return {
                'viable': False,
                'reason': f'Insufficient trades: {num_trades} < {self.min_thresholds["total_trades"]}'
            }
        
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < self.min_thresholds['sharpe_ratio']:
            return {
                'viable': False,
                'reason': f'Low Sharpe ratio: {metrics.get("sharpe_ratio", 0):.2f} < {self.min_thresholds["sharpe_ratio"]}'
            }
        
        # Check profit factor
        if metrics.get('profit_factor', 0) < self.min_thresholds['profit_factor']:
            return {
                'viable': False,
                'reason': f'Low profit factor: {metrics.get("profit_factor", 0):.2f} < {self.min_thresholds["profit_factor"]}'
            }
        
        # Check maximum drawdown
        if metrics.get('max_drawdown', 0) < self.min_thresholds['max_drawdown']:
            return {
                'viable': False,
                'reason': f'Excessive drawdown: {metrics.get("max_drawdown", 0):.2f} < {self.min_thresholds["max_drawdown"]}'
            }
        
        return {'viable': True, 'reason': 'All thresholds met'}
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        score = 0.0
        
        for metric_name, weight in self.weights.items():
            metric_value = metrics.get(metric_name, 0.0)
            
            # Normalize some metrics to 0-1 range
            if metric_name == 'sharpe_ratio':
                # Normalize Sharpe ratio (0-3 range typical)
                normalized_value = np.clip(metric_value / 3.0, 0.0, 1.0)
            elif metric_name == 'profit_factor':
                # Normalize profit factor (1-3 range)
                normalized_value = np.clip((metric_value - 1.0) / 2.0, 0.0, 1.0)
            elif metric_name == 'max_drawdown':
                # Drawdown is negative, convert to 0-1 (0% = 1.0, -30% = 0.0)
                normalized_value = np.clip(1.0 + (metric_value / 0.3), 0.0, 1.0)
            elif metric_name in ['alpha_persistence', 'trade_efficiency']:
                # Already normalized to 0-1
                normalized_value = np.clip(metric_value, 0.0, 1.0)
            elif metric_name == 'volatility_penalty':
                # Already a penalty value
                normalized_value = metric_value
            else:
                # Default normalization
                normalized_value = np.clip(metric_value, 0.0, 1.0)
            
            score += weight * normalized_value
        
        return score
    
    def compare_strategies(self, strategy_a: Dict, strategy_b: Dict) -> Dict[str, Any]:
        """Compare two strategies and determine which is better."""
        
        score_a = strategy_a['composite_score']
        score_b = strategy_b['composite_score']
        
        # Require significant improvement (5% threshold)
        improvement_threshold = 0.05
        
        is_better = (score_b - score_a) > improvement_threshold
        
        return {
            'strategy_b_better': is_better,
            'score_improvement': score_b - score_a,
            'improvement_percent': ((score_b - score_a) / abs(score_a)) * 100 if score_a != 0 else 0,
            'meets_threshold': is_better,
            'detailed_comparison': {
                'strategy_a_score': score_a,
                'strategy_b_score': score_b,
                'strategy_a_viable': strategy_a['viable'],
                'strategy_b_viable': strategy_b['viable']
            }
        }
    
    def get_metric_explanations(self) -> Dict[str, str]:
        """Get explanations for each metric."""
        return {
            'sharpe_ratio': 'Risk-adjusted returns (higher = better, >1.0 good, >2.0 excellent)',
            'profit_factor': 'Gross profits / gross losses (>1.1 profitable, >2.0 excellent)',
            'max_drawdown': 'Worst peak-to-trough decline (closer to 0 better, <-20% concerning)',
            'alpha_persistence': 'Consistency of positive rolling returns (0-1, higher = more stable)',
            'trade_efficiency': 'Combination of win rate and trade frequency (0-1, higher = better)',
            'volatility_penalty': 'Penalty for excessive volatility (negative values)',
            'composite_score': 'Weighted combination of all metrics (higher = better strategy)'
        }

def main():
    """Test the objective function with sample data."""
    
    # Create sample backtest results
    sample_results = {
        'trades': [
            {'action': 'BUY', 'value': 1000, 'pnl': 0},
            {'action': 'SELL', 'value': 1100, 'pnl': 100},
            {'action': 'BUY', 'value': 1100, 'pnl': 0},
            {'action': 'SELL', 'value': 1080, 'pnl': -20},
            {'action': 'BUY', 'value': 1080, 'pnl': 0},
            {'action': 'SELL', 'value': 1200, 'pnl': 120}
        ],
        'portfolio_values': [100000] + list(np.random.randn(100).cumsum() * 1000 + 102000)
    }
    
    # Test objective function
    objective_fn = MetaObjectiveFunction()
    
    results = objective_fn.evaluate_strategy(sample_results)
    
    print("🎯 Meta-Objective Function Test Results")
    print("=" * 50)
    print(f"Composite Score: {results['composite_score']:.3f}")
    print(f"Strategy Viable: {results['viable']}")
    print(f"Reason: {results['reason']}")
    
    print("\n📊 Individual Metrics:")
    for metric, value in results['individual_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n📚 Metric Explanations:")
    explanations = objective_fn.get_metric_explanations()
    for metric, explanation in explanations.items():
        print(f"  {metric}: {explanation}")

if __name__ == "__main__":
    main()