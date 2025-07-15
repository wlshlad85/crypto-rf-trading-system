"""Long/short trading strategy implementation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from utils.config import StrategyConfig


class LongShortStrategy:
    """Long/short cryptocurrency trading strategy."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.positions = {}
        self.cash = 0.0
        self.total_value = 0.0
        
        # Trading history
        self.trades = []
        self.portfolio_history = []
        
        # Risk management
        self.max_drawdown_reached = False
        self.peak_value = 0.0
        
    def generate_signals(self, predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on model predictions."""
        signals = pd.DataFrame(index=predictions.index)
        
        # Get symbols from prediction columns
        symbols = [col.replace('_target', '') for col in predictions.columns if col.endswith('_target')]
        
        for timestamp in predictions.index:
            timestamp_predictions = predictions.loc[timestamp]
            
            # Calculate percentiles for long/short thresholds
            valid_predictions = timestamp_predictions.dropna()
            
            if len(valid_predictions) == 0:
                continue
            
            # Calculate thresholds
            long_threshold = valid_predictions.quantile(self.config.long_threshold)
            short_threshold = valid_predictions.quantile(self.config.short_threshold)
            
            # Generate signals for each symbol
            for symbol in symbols:
                target_col = f"{symbol}_target"
                
                if target_col not in valid_predictions.index:
                    signals.loc[timestamp, f"{symbol}_signal"] = 0
                    continue
                
                prediction = valid_predictions[target_col]
                
                # Generate signal
                if prediction >= long_threshold:
                    signal = 1  # Long
                elif prediction <= short_threshold:
                    signal = -1  # Short
                else:
                    signal = 0  # Hold
                
                signals.loc[timestamp, f"{symbol}_signal"] = signal
        
        # Forward fill signals to handle missing values
        signals = signals.fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Generated signals for {len(symbols)} symbols")
        
        return signals
    
    def calculate_position_sizes(self, signals: pd.Series, prices: pd.Series, 
                               portfolio_value: float) -> pd.Series:
        """Calculate position sizes based on signals and risk management."""
        position_sizes = pd.Series(index=signals.index, data=0.0)
        
        # Get active signals (non-zero)
        active_signals = signals[signals != 0]
        
        if len(active_signals) == 0:
            return position_sizes
        
        if self.config.position_sizing_method == "equal_weight":
            # Equal weight allocation
            base_size = 1.0 / len(active_signals)
            
            for symbol in active_signals.index:
                signal = active_signals[symbol]
                size = base_size * abs(signal)  # Use absolute value for short positions
                
                # Apply position size limits
                size = np.clip(size, self.config.min_position_size, self.config.max_position_size)
                
                position_sizes[symbol] = size * np.sign(signal)  # Maintain sign for direction
        
        elif self.config.position_sizing_method == "volatility_adjusted":
            # Volatility-adjusted position sizing
            volatilities = {}
            
            for symbol in active_signals.index:
                symbol_prices = prices[f"{symbol}_close"]
                if len(symbol_prices) > 20:
                    vol = symbol_prices.pct_change().rolling(20).std().iloc[-1]
                    volatilities[symbol] = vol if not pd.isna(vol) else 0.01
                else:
                    volatilities[symbol] = 0.01
            
            # Inverse volatility weighting
            inv_vols = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            
            for symbol in active_signals.index:
                signal = active_signals[symbol]
                weight = inv_vols[symbol] / total_inv_vol
                size = weight * abs(signal)
                
                # Apply position size limits
                size = np.clip(size, self.config.min_position_size, self.config.max_position_size)
                
                position_sizes[symbol] = size * np.sign(signal)
        
        # Ensure total leverage doesn't exceed limit
        total_leverage = position_sizes.abs().sum()
        if total_leverage > self.config.max_portfolio_leverage:
            position_sizes = position_sizes * (self.config.max_portfolio_leverage / total_leverage)
        
        return position_sizes
    
    def execute_trades(self, current_positions: Dict[str, float], target_positions: Dict[str, float],
                      prices: Dict[str, float], timestamp: datetime, portfolio_value: float) -> List[Dict]:
        """Execute trades to reach target positions."""
        trades = []
        
        # Get all symbols (current and target)
        all_symbols = set(list(current_positions.keys()) + list(target_positions.keys()))
        
        for symbol in all_symbols:
            current_pos = current_positions.get(symbol, 0.0)
            target_pos = target_positions.get(symbol, 0.0)
            
            # Calculate trade size
            trade_size = target_pos - current_pos
            
            if abs(trade_size) < 0.01:  # Minimum trade threshold
                continue
            
            # Get current price
            price = prices.get(f"{symbol}_close", 0.0)
            if price <= 0:
                continue
            
            # Calculate trade value
            trade_value = abs(trade_size) * portfolio_value * price
            
            # Apply transaction costs
            transaction_cost = trade_value * self.config.transaction_cost_pct
            slippage_cost = trade_value * self.config.slippage_pct
            total_cost = transaction_cost + slippage_cost
            
            # Create trade record
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'buy' if trade_size > 0 else 'sell',
                'size': abs(trade_size),
                'price': price,
                'value': trade_value,
                'transaction_cost': transaction_cost,
                'slippage_cost': slippage_cost,
                'total_cost': total_cost,
                'current_position': current_pos,
                'target_position': target_pos
            }
            
            trades.append(trade)
        
        return trades
    
    def apply_risk_management(self, positions: Dict[str, float], prices: Dict[str, float],
                            portfolio_value: float, timestamp: datetime) -> Tuple[Dict[str, float], List[Dict]]:
        """Apply risk management rules."""
        risk_trades = []
        adjusted_positions = positions.copy()
        
        for symbol, position in positions.items():
            if position == 0:
                continue
            
            price_col = f"{symbol}_close"
            if price_col not in prices:
                continue
            
            current_price = prices[price_col]
            
            # Get position entry data (simplified - in practice, track entry prices)
            # For now, we'll check against recent price movements
            # This would be enhanced with proper position tracking
            
            # Example: Check for stop loss based on position P&L
            # In a real implementation, you'd track entry prices and P&L per position
            
            # Placeholder for stop loss logic
            # if position_pnl_pct < -self.config.stop_loss_pct:
            #     # Close position
            #     adjusted_positions[symbol] = 0
            #     risk_trades.append({...})
        
        return adjusted_positions, risk_trades
    
    def should_rebalance(self, last_rebalance: datetime, current_time: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if self.config.rebalancing_frequency == "monthly":
            return (current_time.month != last_rebalance.month or 
                   current_time.year != last_rebalance.year)
        elif self.config.rebalancing_frequency == "weekly":
            return (current_time - last_rebalance).days >= 7
        elif self.config.rebalancing_frequency == "daily":
            return current_time.date() != last_rebalance.date()
        else:
            return False
    
    def calculate_portfolio_value(self, positions: Dict[str, float], prices: Dict[str, float],
                                cash: float) -> float:
        """Calculate total portfolio value."""
        total_value = cash
        
        for symbol, position in positions.items():
            price_col = f"{symbol}_close"
            if price_col in prices and prices[price_col] > 0:
                # For simplicity, assuming position is in percentage of portfolio
                # In practice, this would be number of shares * price
                total_value += position * prices[price_col]
        
        return total_value
    
    def calculate_drawdown(self, portfolio_value: float) -> float:
        """Calculate current drawdown."""
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            drawdown = 0.0
        
        return drawdown
    
    def check_risk_limits(self, portfolio_value: float) -> bool:
        """Check if risk limits are breached."""
        drawdown = self.calculate_drawdown(portfolio_value)
        
        if drawdown > self.config.max_drawdown_pct:
            self.max_drawdown_reached = True
            self.logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
            return False
        
        return True
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'peak_value': self.peak_value,
            'current_drawdown': self.calculate_drawdown(self.total_value),
            'max_drawdown_reached': self.max_drawdown_reached,
            'total_trades': len(self.trades)
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate basic performance metrics."""
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        returns = portfolio_df['total_value'].pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (portfolio_df['total_value'].iloc[-1] / portfolio_df['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (365.25 * 24 / len(returns)) - 1  # Assuming hourly data
        volatility = returns.std() * np.sqrt(365.25 * 24)  # Annualized volatility
        
        # Sharpe ratio (assuming risk-free rate from config or 0)
        risk_free_rate = getattr(self.config, 'risk_free_rate', 0.02)
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate (assuming we have trade data)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            if 'pnl' in trades_df.columns:
                win_rate = (trades_df['pnl'] > 0).mean()
            else:
                win_rate = 0.5  # Placeholder
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'total_trades': len(self.trades)
        }


class PortfolioManager:
    """Portfolio management for long/short strategy."""
    
    def __init__(self, initial_capital: float, config: StrategyConfig):
        self.initial_capital = initial_capital
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> position_size (as fraction of portfolio)
        self.portfolio_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.last_rebalance = None
        
        # Strategy instance
        self.strategy = LongShortStrategy(config)
    
    def update_portfolio(self, signals: pd.Series, prices: pd.Series, timestamp: datetime):
        """Update portfolio based on new signals."""
        # Calculate current portfolio value
        current_value = self.strategy.calculate_portfolio_value(self.positions, prices.to_dict(), self.cash)
        
        # Check risk limits
        if not self.strategy.check_risk_limits(current_value):
            # Emergency liquidation if risk limits breached
            self.positions = {}
            self.logger.warning("Risk limits breached - liquidating all positions")
            return
        
        # Check if rebalancing is needed
        should_rebalance = (
            self.last_rebalance is None or 
            self.strategy.should_rebalance(self.last_rebalance, timestamp)
        )
        
        if should_rebalance:
            # Calculate target positions
            target_positions = self.strategy.calculate_position_sizes(
                signals, prices, current_value
            ).to_dict()
            
            # Execute trades
            trades = self.strategy.execute_trades(
                self.positions, target_positions, prices.to_dict(), timestamp, current_value
            )
            
            # Update positions
            self.positions = target_positions
            self.total_trades += len(trades)
            self.last_rebalance = timestamp
            
            # Store trades
            self.strategy.trades.extend(trades)
        
        # Apply risk management
        self.positions, risk_trades = self.strategy.apply_risk_management(
            self.positions, prices.to_dict(), current_value, timestamp
        )
        
        if risk_trades:
            self.strategy.trades.extend(risk_trades)
        
        # Update portfolio history
        final_value = self.strategy.calculate_portfolio_value(self.positions, prices.to_dict(), self.cash)
        
        portfolio_record = {
            'timestamp': timestamp,
            'total_value': final_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'num_positions': len([p for p in self.positions.values() if abs(p) > 0.01])
        }
        
        self.portfolio_history.append(portfolio_record)
        self.strategy.portfolio_history.append(portfolio_record)
        
        # Update strategy state
        self.strategy.total_value = final_value
        self.strategy.positions = self.positions
        self.strategy.cash = self.cash
    
    def get_portfolio_dataframe(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades history as DataFrame."""
        if not self.strategy.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.strategy.trades)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        portfolio_summary = self.strategy.get_portfolio_summary()
        performance_metrics = self.strategy.get_performance_metrics()
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': portfolio_summary.get('total_value', 0),
            'total_return': (portfolio_summary.get('total_value', 0) / self.initial_capital) - 1,
            'total_trades': self.total_trades,
            'active_positions': len([p for p in self.positions.values() if abs(p) > 0.01]),
            'performance_metrics': performance_metrics
        }