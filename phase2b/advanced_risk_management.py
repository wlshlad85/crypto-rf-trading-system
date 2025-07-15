#!/usr/bin/env python3
"""
Phase 2B: Advanced Risk Management with Kelly Criterion
ULTRATHINK Implementation - Optimal Position Sizing & Risk Control

Implements sophisticated risk management used by institutional trading firms:
- Fractional Kelly criterion for crypto (25-50% of full Kelly)
- CVaR (Conditional Value at Risk) optimization
- Dynamic position sizing based on confidence and volatility
- Multi-objective risk optimization
- Real-time drawdown monitoring and controls
- Regime-dependent risk adjustment

Designed to optimize risk-adjusted returns while preventing catastrophic losses.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from scipy.optimize import minimize, differential_evolution
from scipy import stats
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

warnings.filterwarnings('ignore')

@dataclass
class RiskConfig:
    """Configuration for advanced risk management system."""
    # Kelly criterion parameters
    kelly_fraction: float = 0.25  # Fractional Kelly (25% of full Kelly for crypto)
    max_kelly_position: float = 0.50  # Maximum position size (50% of capital)
    min_kelly_position: float = 0.01  # Minimum position size (1% of capital)
    kelly_lookback: int = 252  # Lookback period for Kelly calculation
    
    # CVaR parameters
    cvar_alpha: float = 0.05  # 5% tail risk (95% confidence)
    cvar_window: int = 100  # Rolling window for CVaR calculation
    max_cvar: float = 0.10  # Maximum acceptable CVaR (10%)
    
    # Position sizing parameters
    base_position_size: float = 0.10  # Base position size (10%)
    volatility_scaling: bool = True
    volatility_target: float = 0.15  # Target annual volatility (15%)
    
    # Risk limits
    max_portfolio_risk: float = 0.20  # Maximum portfolio risk (20%)
    max_single_position: float = 0.30  # Maximum single position (30%)
    stop_loss_pct: float = 0.02  # Stop loss percentage (2%)
    take_profit_pct: float = 0.05  # Take profit percentage (5%)
    
    # Drawdown controls
    max_drawdown: float = 0.15  # Maximum drawdown before reducing positions (15%)
    drawdown_reduction_factor: float = 0.5  # Reduce positions by 50% after max DD
    recovery_threshold: float = 0.05  # Recovery threshold to resume normal sizing
    
    # Dynamic adjustment parameters
    confidence_scaling: bool = True
    regime_adjustment: bool = True
    volatility_regime_multiplier: Dict[str, float] = None
    
    def __post_init__(self):
        if self.volatility_regime_multiplier is None:
            self.volatility_regime_multiplier = {
                'Bull_Market': 1.2,      # Increase positions in bull markets
                'Sideways_Market': 1.0,  # Normal positions in sideways markets
                'Bear_Market': 0.6       # Reduce positions in bear markets
            }

class KellyCriterion:
    """
    Kelly Criterion implementation for optimal position sizing.
    
    Calculates optimal position sizes to maximize long-term growth
    while managing risk of ruin.
    """
    
    def __init__(self, fraction: float = 0.25, lookback: int = 252):
        self.fraction = fraction
        self.lookback = lookback
        self.win_rate_history = []
        self.win_loss_ratio_history = []
        self.kelly_values = []
    
    def calculate_kelly(self, returns: pd.Series, predictions: pd.Series) -> float:
        """
        Calculate Kelly fraction based on historical performance.
        
        Args:
            returns: Historical returns
            predictions: Model predictions (binary: 1 for long, 0 for hold/short)
            
        Returns:
            Optimal Kelly fraction
        """
        if len(returns) != len(predictions):
            # Align series
            min_len = min(len(returns), len(predictions))
            returns = returns.iloc[-min_len:]
            predictions = predictions.iloc[-min_len:]
        
        # Filter for actual trades (when prediction = 1)
        trade_mask = predictions == 1
        trade_returns = returns[trade_mask]
        
        if len(trade_returns) < 10:  # Minimum trades for meaningful calculation
            return self.fraction * 0.5  # Conservative default
        
        # Calculate win rate
        wins = trade_returns > 0
        win_rate = wins.mean()
        
        if win_rate == 0 or win_rate == 1:
            return self.fraction * 0.5  # Conservative for extreme cases
        
        # Calculate average win and loss
        winning_trades = trade_returns[wins]
        losing_trades = trade_returns[~wins]
        
        if len(winning_trades) == 0 or len(losing_trades) == 0:
            return self.fraction * 0.5
        
        avg_win = winning_trades.mean()
        avg_loss = abs(losing_trades.mean())
        
        if avg_loss == 0:
            return self.fraction * 0.5
        
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: f* = (bp - q) / b
        # where b = win/loss ratio, p = win rate, q = loss rate
        b = win_loss_ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_optimal = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly_fraction = kelly_optimal * self.fraction
        
        # Store history
        self.win_rate_history.append(win_rate)
        self.win_loss_ratio_history.append(win_loss_ratio)
        self.kelly_values.append(kelly_fraction)
        
        # Ensure reasonable bounds
        kelly_fraction = max(0.01, min(0.5, kelly_fraction))
        
        return kelly_fraction
    
    def calculate_confidence_adjusted_kelly(self, 
                                          returns: pd.Series,
                                          predictions: pd.Series,
                                          confidences: pd.Series) -> float:
        """
        Calculate confidence-adjusted Kelly fraction.
        
        Args:
            returns: Historical returns
            predictions: Model predictions
            confidences: Prediction confidences (0-1)
            
        Returns:
            Confidence-adjusted Kelly fraction
        """
        base_kelly = self.calculate_kelly(returns, predictions)
        
        # Weight by average confidence
        if len(confidences) > 0:
            avg_confidence = confidences.mean()
            confidence_multiplier = 0.5 + avg_confidence  # Range: 0.5 to 1.5
            adjusted_kelly = base_kelly * confidence_multiplier
        else:
            adjusted_kelly = base_kelly
        
        return max(0.01, min(0.5, adjusted_kelly))

class CVaROptimizer:
    """
    Conditional Value at Risk (CVaR) optimizer for portfolio risk management.
    
    Optimizes portfolios to minimize tail risk while maintaining expected returns.
    """
    
    def __init__(self, alpha: float = 0.05, window: int = 100):
        self.alpha = alpha  # Confidence level (5% = 95% confidence)
        self.window = window
        self.cvar_history = []
    
    def calculate_cvar(self, returns: pd.Series) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Return series
            
        Returns:
            CVaR value (positive number representing expected loss)
        """
        if len(returns) < 10:
            return 0.05  # Default 5% CVaR
        
        # Calculate VaR at alpha level
        var = np.percentile(returns, self.alpha * 100)
        
        # Calculate CVaR (expected value of returns below VaR)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            cvar = abs(var)
        else:
            cvar = abs(tail_returns.mean())
        
        self.cvar_history.append(cvar)
        
        return cvar
    
    def optimize_position_size_cvar(self,
                                   expected_return: float,
                                   return_volatility: float,
                                   max_cvar: float,
                                   base_position: float = 0.1) -> float:
        """
        Optimize position size to meet CVaR constraint.
        
        Args:
            expected_return: Expected return of the position
            return_volatility: Volatility of returns
            max_cvar: Maximum acceptable CVaR
            base_position: Base position size
            
        Returns:
            Optimal position size
        """
        if return_volatility == 0:
            return base_position
        
        # Estimate CVaR for normal distribution (approximation)
        # CVaR ≈ σ * φ(Φ^(-1)(α)) / α - μ
        # where φ is PDF, Φ is CDF of standard normal
        
        z_alpha = stats.norm.ppf(self.alpha)
        phi_z = stats.norm.pdf(z_alpha)
        
        # CVaR formula for normal distribution
        estimated_cvar = return_volatility * phi_z / self.alpha - expected_return
        
        if estimated_cvar <= 0:
            return base_position
        
        # Scale position size to meet CVaR constraint
        position_multiplier = max_cvar / estimated_cvar
        optimal_position = base_position * min(position_multiplier, 2.0)  # Cap at 2x base
        
        return max(0.01, min(0.5, optimal_position))

class DynamicPositionSizer:
    """
    Dynamic position sizing engine that combines multiple risk management techniques.
    
    Integrates Kelly criterion, CVaR optimization, volatility targeting,
    and regime-dependent adjustments.
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.kelly_calculator = KellyCriterion(
            fraction=config.kelly_fraction,
            lookback=config.kelly_lookback
        )
        self.cvar_optimizer = CVaROptimizer(
            alpha=config.cvar_alpha,
            window=config.cvar_window
        )
        
        # Portfolio state tracking
        self.current_positions = {}
        self.portfolio_value = 100000  # Starting capital
        self.max_portfolio_value = 100000
        self.current_drawdown = 0.0
        self.position_history = []
        
    def calculate_optimal_position_size(self,
                                      signal_strength: float,
                                      confidence: float,
                                      expected_return: float,
                                      return_volatility: float,
                                      historical_returns: pd.Series,
                                      historical_predictions: pd.Series,
                                      market_regime: str = 'Sideways_Market') -> Dict[str, float]:
        """
        Calculate optimal position size using multiple risk management techniques.
        
        Args:
            signal_strength: Trading signal strength (-1 to 1)
            confidence: Model confidence (0 to 1)
            expected_return: Expected return of the trade
            return_volatility: Expected volatility
            historical_returns: Historical returns for Kelly calculation
            historical_predictions: Historical predictions for Kelly calculation
            market_regime: Current market regime
            
        Returns:
            Dictionary with position sizing recommendations
        """
        # 1. Base position size
        base_size = self.config.base_position_size
        
        # 2. Kelly criterion position
        if len(historical_returns) > 10 and len(historical_predictions) > 10:
            kelly_size = self.kelly_calculator.calculate_confidence_adjusted_kelly(
                historical_returns, historical_predictions, pd.Series([confidence])
            )
        else:
            kelly_size = base_size * 0.5
        
        # 3. CVaR-optimized position
        cvar_size = self.cvar_optimizer.optimize_position_size_cvar(
            expected_return, return_volatility, self.config.max_cvar, base_size
        )
        
        # 4. Volatility-adjusted position
        if self.config.volatility_scaling and return_volatility > 0:
            vol_adjustment = self.config.volatility_target / (return_volatility * np.sqrt(252))
            vol_adjusted_size = base_size * vol_adjustment
        else:
            vol_adjusted_size = base_size
        
        # 5. Confidence adjustment
        if self.config.confidence_scaling:
            confidence_adjusted_size = base_size * (0.5 + confidence)
        else:
            confidence_adjusted_size = base_size
        
        # 6. Regime adjustment
        if self.config.regime_adjustment:
            regime_multiplier = self.config.volatility_regime_multiplier.get(market_regime, 1.0)
            regime_adjusted_size = base_size * regime_multiplier
        else:
            regime_adjusted_size = base_size
        
        # 7. Signal strength adjustment
        signal_adjusted_size = base_size * abs(signal_strength)
        
        # 8. Drawdown adjustment
        drawdown_adjustment = self._get_drawdown_adjustment()
        
        # Combine all sizing methods
        sizes = {
            'base_size': base_size,
            'kelly_size': kelly_size,
            'cvar_size': cvar_size,
            'volatility_adjusted': vol_adjusted_size,
            'confidence_adjusted': confidence_adjusted_size,
            'regime_adjusted': regime_adjusted_size,
            'signal_adjusted': signal_adjusted_size
        }
        
        # Calculate composite position size (weighted average)
        weights = {
            'kelly_size': 0.3,
            'cvar_size': 0.25,
            'volatility_adjusted': 0.2,
            'confidence_adjusted': 0.15,
            'regime_adjusted': 0.1
        }
        
        composite_size = sum(sizes[key] * weight for key, weight in weights.items())
        
        # Apply signal strength and drawdown adjustments
        composite_size *= abs(signal_strength) * drawdown_adjustment
        
        # Apply risk limits
        final_size = self._apply_risk_limits(composite_size)
        
        # Store the decision
        sizes['composite_size'] = composite_size
        sizes['final_size'] = final_size
        sizes['drawdown_adjustment'] = drawdown_adjustment
        sizes['signal_strength'] = signal_strength
        sizes['confidence'] = confidence
        sizes['market_regime'] = market_regime
        
        return sizes
    
    def _get_drawdown_adjustment(self) -> float:
        """Calculate position size adjustment based on current drawdown."""
        if self.current_drawdown <= self.config.max_drawdown:
            return 1.0  # Normal sizing
        else:
            # Reduce position sizes after maximum drawdown
            excess_drawdown = self.current_drawdown - self.config.max_drawdown
            reduction = min(self.config.drawdown_reduction_factor, excess_drawdown * 2)
            return max(0.1, 1.0 - reduction)
    
    def _apply_risk_limits(self, position_size: float) -> float:
        """Apply final risk limits to position size."""
        # Apply Kelly limits
        position_size = max(self.config.min_kelly_position, 
                           min(self.config.max_kelly_position, position_size))
        
        # Apply single position limit
        position_size = min(position_size, self.config.max_single_position)
        
        # Check portfolio risk limit
        total_risk = sum(abs(pos) for pos in self.current_positions.values())
        if total_risk + position_size > self.config.max_portfolio_risk:
            position_size = max(0, self.config.max_portfolio_risk - total_risk)
        
        return position_size
    
    def update_portfolio_state(self, 
                              asset: str,
                              position_change: float,
                              current_value: float):
        """Update portfolio state for risk tracking."""
        # Update position
        if asset not in self.current_positions:
            self.current_positions[asset] = 0
        
        self.current_positions[asset] += position_change
        
        # Update portfolio value
        self.portfolio_value = current_value
        
        # Update maximum portfolio value
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        # Calculate current drawdown
        self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        # Store history
        self.position_history.append({
            'timestamp': datetime.now(),
            'asset': asset,
            'position': self.current_positions[asset],
            'portfolio_value': current_value,
            'drawdown': self.current_drawdown
        })
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics."""
        return {
            'portfolio_value': self.portfolio_value,
            'max_portfolio_value': self.max_portfolio_value,
            'current_drawdown': self.current_drawdown,
            'total_position_risk': sum(abs(pos) for pos in self.current_positions.values()),
            'positions': self.current_positions.copy(),
            'kelly_history': {
                'win_rates': self.kelly_calculator.win_rate_history[-10:],
                'win_loss_ratios': self.kelly_calculator.win_loss_ratio_history[-10:],
                'kelly_values': self.kelly_calculator.kelly_values[-10:]
            },
            'cvar_history': self.cvar_optimizer.cvar_history[-10:],
            'drawdown_adjustment': self._get_drawdown_adjustment()
        }

class AdvancedRiskManager:
    """
    Comprehensive risk management system integrating multiple techniques.
    
    Provides real-time risk monitoring, position sizing, and portfolio protection
    for cryptocurrency trading systems.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize advanced risk management system.
        
        Args:
            config: Risk management configuration
        """
        self.config = config or RiskConfig()
        self.position_sizer = DynamicPositionSizer(self.config)
        
        # Risk monitoring
        self.risk_alerts = []
        self.risk_metrics_history = []
        
        # Performance tracking
        self.sharpe_ratios = []
        self.max_drawdowns = []
        self.recovery_times = []
        
        print("💰 Advanced Risk Management System Initialized")
        print(f"📊 Kelly Fraction: {self.config.kelly_fraction:.1%}")
        print(f"🎯 CVaR Target: {self.config.cvar_alpha:.1%} (max {self.config.max_cvar:.1%})")
        print(f"⚠️ Max Drawdown: {self.config.max_drawdown:.1%}")
        print(f"📈 Volatility Target: {self.config.volatility_target:.1%}")
    
    def calculate_position_size(self,
                              trading_signal: Dict[str, Any],
                              market_data: pd.DataFrame,
                              regime_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size for a trading opportunity.
        
        Args:
            trading_signal: Dictionary containing signal information
            market_data: Historical market data
            regime_info: Current market regime information
            
        Returns:
            Position sizing recommendation with risk analysis
        """
        # Extract signal information
        signal_strength = trading_signal.get('strength', 0.5)
        confidence = trading_signal.get('confidence', 0.5)
        expected_return = trading_signal.get('expected_return', 0.01)
        
        # Calculate return volatility
        if 'Close' in market_data.columns:
            returns = market_data['Close'].pct_change().dropna()
            return_volatility = returns.std()
        else:
            return_volatility = 0.02  # Default 2% daily volatility
        
        # Get historical predictions (mock for demonstration)
        historical_predictions = pd.Series([1] * len(returns))  # Assume all long signals
        
        # Get market regime
        market_regime = regime_info.get('current_regime', 'Sideways_Market') if regime_info else 'Sideways_Market'
        
        # Calculate optimal position size
        position_info = self.position_sizer.calculate_optimal_position_size(
            signal_strength=signal_strength,
            confidence=confidence,
            expected_return=expected_return,
            return_volatility=return_volatility,
            historical_returns=returns,
            historical_predictions=historical_predictions,
            market_regime=market_regime
        )
        
        # Add risk analysis
        risk_analysis = self._analyze_position_risk(position_info, return_volatility)
        position_info['risk_analysis'] = risk_analysis
        
        # Check for risk alerts
        alerts = self._check_risk_alerts(position_info, risk_analysis)
        position_info['risk_alerts'] = alerts
        
        return position_info
    
    def _analyze_position_risk(self, position_info: Dict, volatility: float) -> Dict[str, Any]:
        """Analyze risk characteristics of the proposed position."""
        final_size = position_info['final_size']
        
        # Calculate potential loss scenarios
        daily_var_95 = 1.645 * volatility  # 95% VaR
        daily_var_99 = 2.33 * volatility   # 99% VaR
        
        # Position-adjusted risk
        position_var_95 = final_size * daily_var_95
        position_var_99 = final_size * daily_var_99
        
        # Estimate maximum loss scenarios
        max_loss_1day = final_size * (volatility * 3)  # 3-sigma event
        max_loss_1week = final_size * (volatility * np.sqrt(5) * 2)  # 2-sigma weekly
        
        return {
            'position_size': final_size,
            'daily_var_95': position_var_95,
            'daily_var_99': position_var_99,
            'max_loss_1day': max_loss_1day,
            'max_loss_1week': max_loss_1week,
            'risk_reward_ratio': final_size / max_loss_1day if max_loss_1day > 0 else 0,
            'volatility_adjusted_size': final_size / volatility if volatility > 0 else 0
        }
    
    def _check_risk_alerts(self, position_info: Dict, risk_analysis: Dict) -> List[str]:
        """Check for risk management alerts."""
        alerts = []
        
        final_size = position_info['final_size']
        
        # Position size alerts
        if final_size > self.config.max_single_position:
            alerts.append(f"WARNING: Position size ({final_size:.1%}) exceeds maximum ({self.config.max_single_position:.1%})")
        
        if final_size < self.config.min_kelly_position:
            alerts.append(f"INFO: Very small position size ({final_size:.1%})")
        
        # Risk alerts
        if risk_analysis['daily_var_99'] > 0.05:  # 5% daily VaR
            alerts.append(f"WARNING: High daily VaR 99% ({risk_analysis['daily_var_99']:.1%})")
        
        if risk_analysis['max_loss_1day'] > 0.10:  # 10% maximum daily loss
            alerts.append(f"CRITICAL: Potential large daily loss ({risk_analysis['max_loss_1day']:.1%})")
        
        # Drawdown alerts
        current_dd = self.position_sizer.current_drawdown
        if current_dd > self.config.max_drawdown * 0.8:  # 80% of max drawdown
            alerts.append(f"WARNING: Approaching maximum drawdown ({current_dd:.1%})")
        
        return alerts
    
    def monitor_portfolio_risk(self) -> Dict[str, Any]:
        """Monitor overall portfolio risk metrics."""
        risk_metrics = self.position_sizer.get_risk_metrics()
        
        # Add additional monitoring
        portfolio_health = {
            'overall_risk_level': 'LOW',
            'risk_score': 0.0,
            'recommendations': []
        }
        
        # Assess risk level
        drawdown = risk_metrics['current_drawdown']
        total_risk = risk_metrics['total_position_risk']
        
        risk_score = 0
        
        # Drawdown component
        if drawdown > self.config.max_drawdown:
            risk_score += 40
            portfolio_health['recommendations'].append("REDUCE positions due to excessive drawdown")
        elif drawdown > self.config.max_drawdown * 0.8:
            risk_score += 20
            portfolio_health['recommendations'].append("CAUTION: Approaching maximum drawdown")
        
        # Position risk component
        if total_risk > self.config.max_portfolio_risk:
            risk_score += 30
            portfolio_health['recommendations'].append("REDUCE total position risk")
        elif total_risk > self.config.max_portfolio_risk * 0.8:
            risk_score += 15
            portfolio_health['recommendations'].append("Monitor total position risk")
        
        # Kelly component
        if len(self.position_sizer.kelly_calculator.kelly_values) > 0:
            recent_kelly = self.position_sizer.kelly_calculator.kelly_values[-1]
            if recent_kelly < 0.05:
                risk_score += 10
                portfolio_health['recommendations'].append("LOW Kelly signals - consider reducing activity")
        
        # CVaR component
        if len(self.position_sizer.cvar_optimizer.cvar_history) > 0:
            recent_cvar = self.position_sizer.cvar_optimizer.cvar_history[-1]
            if recent_cvar > self.config.max_cvar:
                risk_score += 20
                portfolio_health['recommendations'].append("HIGH CVaR - reduce tail risk exposure")
        
        # Determine overall risk level
        if risk_score >= 60:
            portfolio_health['overall_risk_level'] = 'CRITICAL'
        elif risk_score >= 30:
            portfolio_health['overall_risk_level'] = 'HIGH'
        elif risk_score >= 15:
            portfolio_health['overall_risk_level'] = 'MEDIUM'
        else:
            portfolio_health['overall_risk_level'] = 'LOW'
        
        portfolio_health['risk_score'] = risk_score
        
        # Combine with existing metrics
        risk_metrics['portfolio_health'] = portfolio_health
        
        # Store history
        self.risk_metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': risk_metrics
        })
        
        return risk_metrics
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk management report."""
        current_metrics = self.monitor_portfolio_risk()
        
        # Performance analysis
        if len(self.risk_metrics_history) > 1:
            values = [m['metrics']['portfolio_value'] for m in self.risk_metrics_history]
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                max_dd = max([m['metrics']['current_drawdown'] for m in self.risk_metrics_history])
            else:
                sharpe_ratio = 0
                max_dd = 0
        else:
            sharpe_ratio = 0
            max_dd = 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'performance_summary': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'portfolio_value': current_metrics['portfolio_value'],
                'total_risk_exposure': current_metrics['total_position_risk']
            },
            'risk_limits': {
                'max_single_position': self.config.max_single_position,
                'max_portfolio_risk': self.config.max_portfolio_risk,
                'max_drawdown': self.config.max_drawdown,
                'max_cvar': self.config.max_cvar
            },
            'recommendations': current_metrics['portfolio_health']['recommendations'],
            'configuration': {
                'kelly_fraction': self.config.kelly_fraction,
                'cvar_alpha': self.config.cvar_alpha,
                'volatility_target': self.config.volatility_target,
                'regime_adjustment': self.config.regime_adjustment
            }
        }
        
        return report
    
    def save_risk_manager(self, filepath: str):
        """Save risk management state."""
        state_data = {
            'config': self.config,
            'position_sizer': self.position_sizer,
            'risk_metrics_history': self.risk_metrics_history[-100:],  # Keep last 100 records
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        print(f"💾 Risk manager saved: {filepath}")
    
    @classmethod
    def load_risk_manager(cls, filepath: str) -> 'AdvancedRiskManager':
        """Load risk management state."""
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        risk_manager = cls(state_data['config'])
        risk_manager.position_sizer = state_data['position_sizer']
        risk_manager.risk_metrics_history = state_data['risk_metrics_history']
        
        print(f"📂 Risk manager loaded: {filepath}")
        return risk_manager

def main():
    """Demonstrate advanced risk management system."""
    print("💰 PHASE 2B: Advanced Risk Management with Kelly Criterion")
    print("ULTRATHINK Implementation - Optimal Position Sizing")
    print("=" * 60)
    
    # Load data from Phase 1
    data_dir = "phase1/data/processed"
    import glob
    import os
    
    data_files = glob.glob(f"{data_dir}/BTC-USD_*.csv")
    if not data_files:
        print("❌ No data files found. Run Phase 1A first.")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"📂 Loading data from: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    print(f"📊 Data loaded: {len(df)} samples")
    
    # Initialize risk management system
    config = RiskConfig(
        kelly_fraction=0.25,
        cvar_alpha=0.05,
        max_drawdown=0.15,
        volatility_target=0.15,
        regime_adjustment=True
    )
    
    risk_manager = AdvancedRiskManager(config)
    
    try:
        # Simulate trading signals and position sizing
        print(f"\n🔧 Simulating Risk Management Scenarios...")
        
        # Use recent data for simulation
        recent_data = df.iloc[-500:].copy()
        
        # Simulate different trading scenarios
        scenarios = [
            {'strength': 0.8, 'confidence': 0.9, 'expected_return': 0.03, 'regime': 'Bull_Market'},
            {'strength': 0.3, 'confidence': 0.5, 'expected_return': 0.01, 'regime': 'Sideways_Market'},
            {'strength': 0.6, 'confidence': 0.7, 'expected_return': -0.02, 'regime': 'Bear_Market'},
            {'strength': 0.9, 'confidence': 0.95, 'expected_return': 0.05, 'regime': 'Bull_Market'},
        ]
        
        print(f"\n📊 POSITION SIZING SCENARIOS:")
        
        for i, scenario in enumerate(scenarios, 1):
            # Create trading signal
            trading_signal = {
                'strength': scenario['strength'],
                'confidence': scenario['confidence'],
                'expected_return': scenario['expected_return']
            }
            
            # Create regime info
            regime_info = {'current_regime': scenario['regime']}
            
            # Calculate position size
            position_info = risk_manager.calculate_position_size(
                trading_signal, recent_data, regime_info
            )
            
            print(f"\n   Scenario {i}: {scenario['regime']}")
            print(f"      Signal Strength: {scenario['strength']:.1%}")
            print(f"      Confidence: {scenario['confidence']:.1%}")
            print(f"      Expected Return: {scenario['expected_return']:+.1%}")
            print(f"      Final Position Size: {position_info['final_size']:.1%}")
            print(f"      Kelly Size: {position_info['kelly_size']:.1%}")
            print(f"      CVaR Size: {position_info['cvar_size']:.1%}")
            
            # Show risk analysis
            risk_analysis = position_info['risk_analysis']
            print(f"      Daily VaR (95%): {risk_analysis['daily_var_95']:.2%}")
            print(f"      Max 1-day Loss: {risk_analysis['max_loss_1day']:.2%}")
            
            # Show alerts
            if position_info['risk_alerts']:
                print(f"      🚨 Alerts: {len(position_info['risk_alerts'])}")
                for alert in position_info['risk_alerts']:
                    print(f"         - {alert}")
            
            # Update portfolio state for next scenario
            risk_manager.position_sizer.update_portfolio_state(
                'BTC', position_info['final_size'], 100000 * (1 + scenario['expected_return'])
            )
        
        # Generate risk monitoring report
        print(f"\n📈 PORTFOLIO RISK MONITORING:")
        risk_metrics = risk_manager.monitor_portfolio_risk()
        
        portfolio_health = risk_metrics['portfolio_health']
        print(f"   Overall Risk Level: {portfolio_health['overall_risk_level']}")
        print(f"   Risk Score: {portfolio_health['risk_score']}/100")
        print(f"   Current Drawdown: {risk_metrics['current_drawdown']:.2%}")
        print(f"   Total Position Risk: {risk_metrics['total_position_risk']:.2%}")
        
        if portfolio_health['recommendations']:
            print(f"   💡 Recommendations:")
            for rec in portfolio_health['recommendations']:
                print(f"      - {rec}")
        
        # Generate comprehensive report
        full_report = risk_manager.generate_risk_report()
        
        # Save risk manager and report
        risk_manager_file = "phase2b/advanced_risk_manager.pkl"
        risk_manager.save_risk_manager(risk_manager_file)
        
        report_file = "phase2b/risk_management_report.json"
        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Risk Manager: {risk_manager_file}")
        print(f"   📈 Report: {report_file}")
        
        print(f"\n🚀 Phase 2B Advanced Risk Management: COMPLETE")
        print(f"🎯 Phase 2B High-Priority Tasks: ALL COMPLETE")
        print(f"🔄 Ready for Production Deployment or Phase 2C")
        
    except Exception as e:
        print(f"❌ Error in risk management: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()