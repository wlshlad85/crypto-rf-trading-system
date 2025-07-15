"""High-frequency trading strategies optimized for minute-level cryptocurrency trading."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from utils.config import StrategyConfig
from models.minute_random_forest_model import MinuteRandomForestModel


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class MinuteMomentumStrategy:
    """Momentum strategy optimized for minute-level trading using Random Forest predictions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.lookback_periods = [5, 15, 30, 60]  # minutes
        self.momentum_threshold = 0.02  # 2% momentum threshold
        self.prediction_weight = 0.7    # 70% weight to RF predictions
        self.technical_weight = 0.3     # 30% weight to technical indicators
        
        # Risk management
        self.stop_loss = 0.015          # 1.5% stop loss
        self.take_profit = 0.025        # 2.5% take profit
        self.trailing_stop = 0.01       # 1% trailing stop
        self.max_position_size = 0.2    # Max 20% per position
        
        # Position tracking
        self.current_positions = {}
        self.entry_prices = {}
        self.highest_prices = {}  # For trailing stops
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default strategy configuration."""
        return {
            'name': 'MinuteMomentum',
            'horizons': [1, 5, 15],  # minutes
            'rebalance_frequency': 5,  # rebalance every 5 minutes
            'min_prediction_confidence': 0.6,
            'momentum_lookback': 30,
            'volatility_adjustment': True,
            'regime_awareness': True,
        }
    
    def generate_signals(self, predictions: pd.DataFrame, market_data: pd.DataFrame,
                        features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Generate trading signals based on RF predictions and momentum indicators."""
        
        signals = pd.DataFrame(index=market_data.index)
        
        for symbol in symbols:
            try:
                # Get predictions for multiple horizons
                pred_signals = self._get_prediction_signals(predictions, symbol)
                
                # Get momentum signals
                momentum_signals = self._get_momentum_signals(market_data, features, symbol)
                
                # Get volatility adjustment
                vol_adjustment = self._get_volatility_adjustment(market_data, symbol)
                
                # Combine signals
                combined_signal = (
                    pred_signals * self.prediction_weight +
                    momentum_signals * self.technical_weight
                ) * vol_adjustment
                
                # Apply risk management
                final_signal = self._apply_risk_management(
                    combined_signal, market_data, symbol
                )
                
                signals[f'{symbol}_signal'] = final_signal
                
            except Exception as e:
                self.logger.warning(f"Error generating signals for {symbol}: {e}")
                signals[f'{symbol}_signal'] = 0
        
        return signals
    
    def _get_prediction_signals(self, predictions: pd.DataFrame, symbol: str) -> pd.Series:
        """Extract signals from Random Forest predictions."""
        
        signal = pd.Series(0.0, index=predictions.index)
        
        # Multi-horizon prediction analysis
        horizons = [1, 5, 15]  # minutes
        horizon_weights = [0.5, 0.3, 0.2]  # Short-term bias
        
        for horizon, weight in zip(horizons, horizon_weights):
            pred_col = f'{symbol}_{horizon}min_pred'
            
            if pred_col in predictions.columns:
                pred_values = predictions[pred_col]
                
                # Convert predictions to signals using dynamic thresholds
                rolling_std = pred_values.rolling(window=60, min_periods=10).std()
                upper_threshold = pred_values.rolling(window=60, min_periods=10).quantile(0.75)
                lower_threshold = pred_values.rolling(window=60, min_periods=10).quantile(0.25)
                
                # Generate signals
                horizon_signals = pd.Series(0.0, index=pred_values.index)
                horizon_signals[pred_values > upper_threshold] = 1
                horizon_signals[pred_values < lower_threshold] = -1
                
                # Weight by prediction confidence (based on distance from mean)
                confidence = np.abs(pred_values - pred_values.rolling(60).mean()) / (rolling_std + 1e-8)
                confidence = np.clip(confidence, 0, 2)  # Cap confidence
                
                signal += horizon_signals * weight * confidence
        
        # Normalize signals to [-1, 1]
        signal = np.clip(signal, -1, 1)
        
        return signal
    
    def _get_momentum_signals(self, market_data: pd.DataFrame, features: pd.DataFrame, 
                            symbol: str) -> pd.Series:
        """Generate momentum-based signals."""
        
        signal = pd.Series(0.0, index=market_data.index)
        
        close_col = f'{symbol}_close'
        if close_col not in market_data.columns:
            return signal
        
        prices = market_data[close_col]
        
        # Multi-timeframe momentum
        momentum_signals = []
        
        for period in self.lookback_periods:
            # Price momentum
            price_momentum = prices.pct_change(period)
            
            # Volume-weighted momentum (if volume available)
            volume_col = f'{symbol}_volume'
            if volume_col in market_data.columns:
                volume = market_data[volume_col]
                volume_ma = volume.rolling(period).mean()
                volume_ratio = volume / (volume_ma + 1e-8)
                price_momentum *= np.log1p(volume_ratio)  # Volume weighting
            
            # Normalize momentum
            momentum_std = price_momentum.rolling(60).std()
            normalized_momentum = price_momentum / (momentum_std + 1e-8)
            
            momentum_signals.append(normalized_momentum)
        
        # Combine momentum signals
        if momentum_signals:
            # Weight shorter periods more heavily
            weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(momentum_signals)]
            weights = weights / weights.sum()
            
            for mom_signal, weight in zip(momentum_signals, weights):
                signal += mom_signal * weight
        
        # Apply momentum threshold
        signal = np.where(np.abs(signal) > self.momentum_threshold, signal, 0)
        
        return signal
    
    def _get_volatility_adjustment(self, market_data: pd.DataFrame, symbol: str) -> pd.Series:
        """Get volatility-based signal adjustment."""
        
        close_col = f'{symbol}_close'
        if close_col not in market_data.columns:
            return pd.Series(1.0, index=market_data.index)
        
        prices = market_data[close_col]
        returns = prices.pct_change()
        
        # Calculate rolling volatility
        short_vol = returns.rolling(15).std()  # 15-minute volatility
        long_vol = returns.rolling(60).std()   # 1-hour volatility
        
        # Volatility ratio (current vs historical)
        vol_ratio = short_vol / (long_vol + 1e-8)
        
        # Adjust signals based on volatility regime
        adjustment = pd.Series(1.0, index=market_data.index)
        
        if self.config['volatility_adjustment']:
            # Reduce signal strength in high volatility periods
            high_vol_mask = vol_ratio > 1.5
            low_vol_mask = vol_ratio < 0.7
            
            adjustment[high_vol_mask] = 0.7  # Reduce signals by 30% in high vol
            adjustment[low_vol_mask] = 1.2   # Increase signals by 20% in low vol
            
            # Cap adjustment
            adjustment = np.clip(adjustment, 0.5, 1.5)
        
        return adjustment
    
    def _apply_risk_management(self, signal: pd.Series, market_data: pd.DataFrame, 
                             symbol: str) -> pd.Series:
        """Apply risk management rules to signals."""
        
        final_signal = signal.copy()
        
        close_col = f'{symbol}_close'
        if close_col not in market_data.columns:
            return final_signal
        
        current_prices = market_data[close_col]
        
        # Apply position sizing
        position_limit = self.max_position_size
        final_signal = final_signal * position_limit
        
        # Stop loss and take profit logic
        current_position = self.current_positions.get(symbol, 0)
        
        if current_position != 0:
            entry_price = self.entry_prices.get(symbol, current_prices.iloc[0])
            highest_price = self.highest_prices.get(symbol, entry_price)
            
            for timestamp in current_prices.index:
                current_price = current_prices[timestamp]
                
                if pd.isna(current_price):
                    continue
                
                # Update highest price for trailing stop
                if current_position > 0 and current_price > highest_price:
                    highest_price = current_price
                    self.highest_prices[symbol] = highest_price
                
                # Check stop loss
                if current_position > 0:  # Long position
                    loss = (current_price - entry_price) / entry_price
                    trail_loss = (current_price - highest_price) / highest_price
                    
                    if loss <= -self.stop_loss or trail_loss <= -self.trailing_stop:
                        final_signal[timestamp] = -1  # Force exit
                    elif loss >= self.take_profit:
                        final_signal[timestamp] = -0.5  # Partial profit taking
                
                elif current_position < 0:  # Short position
                    loss = (entry_price - current_price) / entry_price
                    
                    if loss <= -self.stop_loss:
                        final_signal[timestamp] = 1  # Force exit
                    elif loss >= self.take_profit:
                        final_signal[timestamp] = 0.5  # Partial profit taking
        
        return final_signal
    
    def update_positions(self, symbol: str, new_position: float, entry_price: float):
        """Update position tracking."""
        self.current_positions[symbol] = new_position
        if new_position != 0:
            self.entry_prices[symbol] = entry_price
            self.highest_prices[symbol] = entry_price
        else:
            # Clear tracking when position is closed
            self.entry_prices.pop(symbol, None)
            self.highest_prices.pop(symbol, None)


class MinuteMeanReversionStrategy:
    """Mean reversion strategy for minute-level trading."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.lookback_period = 60        # 1 hour lookback
        self.std_threshold = 2.0         # 2 standard deviations
        self.prediction_weight = 0.8     # High weight on RF predictions
        self.mean_reversion_weight = 0.2
        
        # Risk management
        self.stop_loss = 0.01           # 1% stop loss (tighter for mean reversion)
        self.take_profit = 0.015        # 1.5% take profit
        self.max_holding_period = 120   # Maximum 2 hours holding
        
        # Position tracking
        self.position_entries = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default mean reversion configuration."""
        return {
            'name': 'MinuteMeanReversion',
            'mean_lookback': 60,
            'volatility_lookback': 30,
            'reversion_threshold': 2.0,
            'max_holding_minutes': 120,
        }
    
    def generate_signals(self, predictions: pd.DataFrame, market_data: pd.DataFrame,
                        features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Generate mean reversion signals."""
        
        signals = pd.DataFrame(index=market_data.index)
        
        for symbol in symbols:
            try:
                # Get mean reversion signals
                reversion_signals = self._get_mean_reversion_signals(market_data, symbol)
                
                # Get prediction signals (contrarian interpretation)
                pred_signals = self._get_contrarian_prediction_signals(predictions, symbol)
                
                # Combine signals
                combined_signal = (
                    reversion_signals * self.mean_reversion_weight +
                    pred_signals * self.prediction_weight
                )
                
                # Apply time-based exit rules
                final_signal = self._apply_time_based_exits(
                    combined_signal, market_data.index, symbol
                )
                
                signals[f'{symbol}_signal'] = final_signal
                
            except Exception as e:
                self.logger.warning(f"Error generating mean reversion signals for {symbol}: {e}")
                signals[f'{symbol}_signal'] = 0
        
        return signals
    
    def _get_mean_reversion_signals(self, market_data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate mean reversion signals based on price deviations."""
        
        close_col = f'{symbol}_close'
        if close_col not in market_data.columns:
            return pd.Series(0.0, index=market_data.index)
        
        prices = market_data[close_col]
        
        # Calculate rolling mean and standard deviation
        rolling_mean = prices.rolling(self.lookback_period).mean()
        rolling_std = prices.rolling(self.lookback_period).std()
        
        # Z-score (price deviation from mean)
        z_score = (prices - rolling_mean) / (rolling_std + 1e-8)
        
        # Generate signals
        signal = pd.Series(0.0, index=prices.index)
        
        # Buy when price is significantly below mean (oversold)
        signal[z_score < -self.std_threshold] = 1
        
        # Sell when price is significantly above mean (overbought)
        signal[z_score > self.std_threshold] = -1
        
        # Gradual signal strength based on z-score magnitude
        signal = signal * np.minimum(np.abs(z_score) / self.std_threshold, 2.0)
        
        return signal
    
    def _get_contrarian_prediction_signals(self, predictions: pd.DataFrame, symbol: str) -> pd.Series:
        """Interpret RF predictions in a contrarian manner for mean reversion."""
        
        signal = pd.Series(0.0, index=predictions.index)
        
        # Use shortest horizon for mean reversion
        pred_col = f'{symbol}_1min_pred'
        
        if pred_col in predictions.columns:
            pred_values = predictions[pred_col]
            
            # Contrarian interpretation: fade extreme predictions
            pred_mean = pred_values.rolling(60).mean()
            pred_std = pred_values.rolling(60).std()
            pred_z = (pred_values - pred_mean) / (pred_std + 1e-8)
            
            # Fade extreme positive predictions (expect reversion)
            signal[pred_z > 1.5] = -1
            
            # Fade extreme negative predictions (expect bounce)
            signal[pred_z < -1.5] = 1
            
            # Scale by prediction extremeness
            signal = signal * np.minimum(np.abs(pred_z) / 1.5, 2.0)
        
        return signal


class MinuteBreakoutStrategy:
    """Breakout strategy for minute-level trading."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.breakout_periods = [15, 30, 60]  # minutes
        self.volume_threshold = 1.5           # 50% above average volume
        self.prediction_confirmation = True    # Require RF confirmation
        
        # Risk management
        self.stop_loss = 0.02               # 2% stop loss
        self.take_profit = 0.04             # 4% take profit
        self.trailing_stop = 0.015          # 1.5% trailing stop
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default breakout configuration."""
        return {
            'name': 'MinuteBreakout',
            'breakout_periods': [15, 30, 60],
            'volume_confirmation': True,
            'prediction_confirmation': True,
            'min_breakout_strength': 0.5,
        }
    
    def generate_signals(self, predictions: pd.DataFrame, market_data: pd.DataFrame,
                        features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Generate breakout signals."""
        
        signals = pd.DataFrame(index=market_data.index)
        
        for symbol in symbols:
            try:
                # Get breakout signals
                breakout_signals = self._get_breakout_signals(market_data, symbol)
                
                # Get volume confirmation
                volume_signals = self._get_volume_confirmation(market_data, symbol)
                
                # Get prediction confirmation
                pred_confirmation = self._get_prediction_confirmation(predictions, symbol)
                
                # Combine signals (all must align for breakout)
                combined_signal = breakout_signals * volume_signals
                
                if self.prediction_confirmation:
                    combined_signal *= pred_confirmation
                
                signals[f'{symbol}_signal'] = combined_signal
                
            except Exception as e:
                self.logger.warning(f"Error generating breakout signals for {symbol}: {e}")
                signals[f'{symbol}_signal'] = 0
        
        return signals
    
    def _get_breakout_signals(self, market_data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate signals based on price breakouts."""
        
        close_col = f'{symbol}_close'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        
        if not all(col in market_data.columns for col in [close_col, high_col, low_col]):
            return pd.Series(0.0, index=market_data.index)
        
        prices = market_data[close_col]
        highs = market_data[high_col]
        lows = market_data[low_col]
        
        signal = pd.Series(0.0, index=prices.index)
        
        for period in self.breakout_periods:
            # Calculate support and resistance levels
            resistance = highs.rolling(period).max()
            support = lows.rolling(period).min()
            
            # Breakout above resistance
            upward_breakout = prices > resistance.shift(1)
            
            # Breakdown below support
            downward_breakout = prices < support.shift(1)
            
            # Weight shorter periods more heavily
            weight = 1.0 / (1 + np.log(period))
            
            signal[upward_breakout] += weight
            signal[downward_breakout] -= weight
        
        # Normalize signals
        signal = np.clip(signal, -1, 1)
        
        return signal
    
    def _get_volume_confirmation(self, market_data: pd.DataFrame, symbol: str) -> pd.Series:
        """Get volume confirmation for breakouts."""
        
        volume_col = f'{symbol}_volume'
        if volume_col not in market_data.columns:
            return pd.Series(1.0, index=market_data.index)
        
        volume = market_data[volume_col]
        volume_ma = volume.rolling(60).mean()
        volume_ratio = volume / (volume_ma + 1e-8)
        
        # Confirm signals only when volume is above threshold
        confirmation = pd.Series(0.0, index=volume.index)
        confirmation[volume_ratio > self.volume_threshold] = 1.0
        confirmation[volume_ratio <= self.volume_threshold] = 0.3  # Weak confirmation
        
        return confirmation
    
    def _get_prediction_confirmation(self, predictions: pd.DataFrame, symbol: str) -> pd.Series:
        """Get Random Forest prediction confirmation."""
        
        confirmation = pd.Series(1.0, index=predictions.index)
        
        # Use 5-minute horizon for breakout confirmation
        pred_col = f'{symbol}_5min_pred'
        
        if pred_col in predictions.columns:
            pred_values = predictions[pred_col]
            
            # Strong predictions confirm breakouts
            pred_abs = np.abs(pred_values)
            pred_threshold = pred_abs.rolling(60).quantile(0.7)
            
            # Confirm when predictions are strong
            confirmation = pd.Series(0.5, index=pred_values.index)  # Base confirmation
            confirmation[pred_abs > pred_threshold] = 1.0           # Strong confirmation
            
            # Direction must align with prediction
            pred_direction = np.sign(pred_values)
            confirmation = confirmation * pred_direction
        
        return confirmation


class MinuteStrategyEnsemble:
    """Ensemble of minute-level trading strategies."""
    
    def __init__(self, strategies: List[Any] = None, weights: List[float] = None):
        self.strategies = strategies or [
            MinuteMomentumStrategy(),
            MinuteMeanReversionStrategy(),
            MinuteBreakoutStrategy()
        ]
        
        self.weights = weights or [0.4, 0.3, 0.3]  # Default equal-ish weighting
        self.logger = logging.getLogger(__name__)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Strategy performance tracking
        self.strategy_performance = {}
        self.rebalance_frequency = 60  # Rebalance weights every hour
        
    def generate_signals(self, predictions: pd.DataFrame, market_data: pd.DataFrame,
                        features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Generate ensemble signals from all strategies."""
        
        ensemble_signals = pd.DataFrame(index=market_data.index)
        
        # Get signals from each strategy
        strategy_signals = []
        
        for i, strategy in enumerate(self.strategies):
            try:
                signals = strategy.generate_signals(predictions, market_data, features, symbols)
                strategy_signals.append(signals)
                
            except Exception as e:
                self.logger.warning(f"Error in strategy {i}: {e}")
                # Create zero signals as fallback
                zero_signals = pd.DataFrame(0, index=market_data.index, 
                                          columns=[f'{symbol}_signal' for symbol in symbols])
                strategy_signals.append(zero_signals)
        
        # Combine signals using weights
        for symbol in symbols:
            signal_col = f'{symbol}_signal'
            combined_signal = pd.Series(0.0, index=market_data.index)
            
            for signals, weight in zip(strategy_signals, self.weights):
                if signal_col in signals.columns:
                    combined_signal += signals[signal_col] * weight
            
            ensemble_signals[signal_col] = combined_signal
        
        # Apply ensemble-level risk management
        ensemble_signals = self._apply_ensemble_risk_management(
            ensemble_signals, market_data, symbols
        )
        
        return ensemble_signals
    
    def _apply_ensemble_risk_management(self, signals: pd.DataFrame, 
                                      market_data: pd.DataFrame, 
                                      symbols: List[str]) -> pd.DataFrame:
        """Apply risk management at the ensemble level."""
        
        # Portfolio-level position limits
        max_total_exposure = 0.8  # Max 80% total exposure
        max_single_position = 0.25  # Max 25% per position
        
        adjusted_signals = signals.copy()
        
        for timestamp in signals.index:
            signal_row = signals.loc[timestamp]
            
            # Calculate total exposure
            total_exposure = sum(abs(signal_row[f'{symbol}_signal']) 
                               for symbol in symbols 
                               if f'{symbol}_signal' in signal_row.index)
            
            # Scale down if total exposure exceeds limit
            if total_exposure > max_total_exposure:
                scale_factor = max_total_exposure / total_exposure
                for symbol in symbols:
                    signal_col = f'{symbol}_signal'
                    if signal_col in signal_row.index:
                        adjusted_signals.loc[timestamp, signal_col] *= scale_factor
            
            # Apply individual position limits
            for symbol in symbols:
                signal_col = f'{symbol}_signal'
                if signal_col in signal_row.index:
                    signal_value = adjusted_signals.loc[timestamp, signal_col]
                    if abs(signal_value) > max_single_position:
                        adjusted_signals.loc[timestamp, signal_col] = (
                            np.sign(signal_value) * max_single_position
                        )
        
        return adjusted_signals
    
    def update_strategy_weights(self, performance_data: Dict[str, float]):
        """Dynamically update strategy weights based on performance."""
        
        if len(performance_data) != len(self.strategies):
            return
        
        # Convert performance to weights (better performance = higher weight)
        performances = list(performance_data.values())
        
        # Normalize performances to [0, 1] range
        min_perf = min(performances)
        max_perf = max(performances)
        
        if max_perf > min_perf:
            normalized_perfs = [(p - min_perf) / (max_perf - min_perf) for p in performances]
            # Add base weight to prevent zero weights
            normalized_perfs = [p + 0.1 for p in normalized_perfs]
            
            # Normalize to sum to 1
            total_weight = sum(normalized_perfs)
            self.weights = [w / total_weight for w in normalized_perfs]
            
            self.logger.info(f"Updated strategy weights: {self.weights}")


# Utility functions

def create_minute_momentum_strategy(symbols: List[str] = None) -> MinuteMomentumStrategy:
    """Create a configured momentum strategy."""
    config = {
        'symbols': symbols or ['BTC-USD', 'ETH-USD', 'SOL-USD'],
        'rebalance_frequency': 5,
        'momentum_lookback': 30,
        'volatility_adjustment': True,
    }
    return MinuteMomentumStrategy(config)


def create_minute_strategy_ensemble(symbols: List[str] = None) -> MinuteStrategyEnsemble:
    """Create a configured strategy ensemble."""
    strategies = [
        MinuteMomentumStrategy(),
        MinuteMeanReversionStrategy(),
        MinuteBreakoutStrategy()
    ]
    weights = [0.4, 0.3, 0.3]  # Momentum bias
    
    return MinuteStrategyEnsemble(strategies, weights)


def backtest_strategy_performance(strategy: Any, predictions: pd.DataFrame,
                                market_data: pd.DataFrame, features: pd.DataFrame,
                                symbols: List[str]) -> Dict[str, float]:
    """Quick performance evaluation of a strategy."""
    
    signals = strategy.generate_signals(predictions, market_data, features, symbols)
    
    # Simple return calculation (without transaction costs)
    returns = {}
    
    for symbol in symbols:
        signal_col = f'{symbol}_signal'
        price_col = f'{symbol}_close'
        
        if signal_col in signals.columns and price_col in market_data.columns:
            prices = market_data[price_col]
            strategy_signals = signals[signal_col]
            
            # Calculate strategy returns
            price_returns = prices.pct_change()
            strategy_returns = strategy_signals.shift(1) * price_returns
            
            # Performance metrics
            total_return = (1 + strategy_returns.dropna()).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(365 * 24 * 60)  # Annualized
            sharpe = total_return / volatility if volatility > 0 else 0
            
            returns[symbol] = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'n_trades': (strategy_signals.diff() != 0).sum()
            }
    
    return returns


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-01-02', freq='1T')
    symbols = ['BTC-USD', 'ETH-USD']
    np.random.seed(42)
    
    # Sample market data
    market_data = pd.DataFrame(index=dates)
    for symbol in symbols:
        price = 50000 + np.random.randn(len(dates)).cumsum() * 100
        market_data[f'{symbol}_close'] = price
        market_data[f'{symbol}_high'] = price + np.random.rand(len(dates)) * 50
        market_data[f'{symbol}_low'] = price - np.random.rand(len(dates)) * 50
        market_data[f'{symbol}_volume'] = np.random.exponential(1000, len(dates))
    
    # Sample predictions
    predictions = pd.DataFrame(index=dates)
    for symbol in symbols:
        for horizon in [1, 5, 15]:
            predictions[f'{symbol}_{horizon}min_pred'] = np.random.randn(len(dates)) * 0.01
    
    # Sample features
    features = pd.DataFrame(
        np.random.randn(len(dates), 20),
        index=dates,
        columns=[f'feature_{i}' for i in range(20)]
    )
    
    print(f"Sample data created: {len(dates)} minutes, {len(symbols)} symbols")
    
    # Test strategies
    print("\nTesting individual strategies...")
    
    # Momentum strategy
    momentum_strategy = MinuteMomentumStrategy()
    momentum_signals = momentum_strategy.generate_signals(predictions, market_data, features, symbols)
    print(f"Momentum signals shape: {momentum_signals.shape}")
    print(f"Momentum signal range: {momentum_signals.min().min():.3f} to {momentum_signals.max().max():.3f}")
    
    # Mean reversion strategy
    mean_rev_strategy = MinuteMeanReversionStrategy()
    mean_rev_signals = mean_rev_strategy.generate_signals(predictions, market_data, features, symbols)
    print(f"Mean reversion signals shape: {mean_rev_signals.shape}")
    
    # Breakout strategy
    breakout_strategy = MinuteBreakoutStrategy()
    breakout_signals = breakout_strategy.generate_signals(predictions, market_data, features, symbols)
    print(f"Breakout signals shape: {breakout_signals.shape}")
    
    # Ensemble strategy
    print("\nTesting ensemble strategy...")
    ensemble = MinuteStrategyEnsemble()
    ensemble_signals = ensemble.generate_signals(predictions, market_data, features, symbols)
    print(f"Ensemble signals shape: {ensemble_signals.shape}")
    print(f"Ensemble signal range: {ensemble_signals.min().min():.3f} to {ensemble_signals.max().max():.3f}")
    
    # Quick performance test
    print("\nQuick performance evaluation...")
    perf = backtest_strategy_performance(ensemble, predictions, market_data, features, symbols)
    for symbol, metrics in perf.items():
        print(f"{symbol}: Return={metrics['total_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}, Trades={metrics['n_trades']}")