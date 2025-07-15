#!/usr/bin/env python3
"""
Real-time paper trading system for minute-level Random Forest cryptocurrency trading.

This system:
1. Fetches real-time minute data
2. Generates features and predictions
3. Executes trades with simulated money
4. Tracks performance in real-time
5. Provides live dashboard and reporting

Usage:
    python minute_paper_trader.py [--config config.json] [--capital 100000]
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.minute_data_manager import MinuteDataManager
from features.minute_feature_engineering import MinuteFeatureEngine
from models.minute_random_forest_model import MinuteRandomForestModel
from strategies.minute_trading_strategies import MinuteStrategyEnsemble
from analytics.minute_performance_analytics import MinutePerformanceAnalytics
from visualization.minute_visualization import MinuteVisualizationSuite


class PaperTradingAccount:
    """Simulated trading account with realistic constraints."""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.commission = commission
        
        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.pending_orders = {}
        
        # Performance tracking
        self.total_commission_paid = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Risk limits
        self.max_position_size = 0.25  # Max 25% in one position
        self.max_daily_loss = 0.05     # Max 5% daily loss
        self.daily_loss = 0
        self.last_reset_date = datetime.now().date()
        
    def execute_order(self, symbol: str, quantity: float, price: float, 
                     order_type: str = 'market') -> Dict[str, Any]:
        """Execute a trading order."""
        
        # Reset daily loss if new day
        if datetime.now().date() != self.last_reset_date:
            self.daily_loss = 0
            self.last_reset_date = datetime.now().date()
        
        # Check risk limits
        if self.daily_loss >= self.max_daily_loss:
            return {
                'status': 'rejected',
                'reason': 'Daily loss limit reached',
                'symbol': symbol,
                'quantity': quantity
            }
        
        # Calculate trade value and commission
        trade_value = abs(quantity * price)
        commission = trade_value * self.commission
        
        # Check if we have enough cash for buying
        if quantity > 0 and (trade_value + commission) > self.cash:
            # Adjust quantity to available cash
            available_cash = self.cash - commission
            if available_cash > 0:
                quantity = available_cash / price
                trade_value = quantity * price
            else:
                return {
                    'status': 'rejected',
                    'reason': 'Insufficient funds',
                    'symbol': symbol,
                    'requested_value': trade_value
                }
        
        # Check position size limit
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        position_value = abs(new_position * price)
        total_value = self.get_total_value(current_prices={symbol: price})
        
        if position_value > total_value * self.max_position_size:
            return {
                'status': 'rejected', 
                'reason': 'Position size limit exceeded',
                'symbol': symbol,
                'position_value': position_value,
                'limit': total_value * self.max_position_size
            }
        
        # Execute trade
        if quantity > 0:  # Buy
            self.cash -= (trade_value + commission)
        else:  # Sell
            self.cash += (abs(trade_value) - commission)
        
        # Update position
        self.positions[symbol] = new_position
        if abs(self.positions[symbol]) < 1e-8:  # Clean up tiny positions
            del self.positions[symbol]
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'commission': commission,
            'order_type': order_type,
            'cash_after': self.cash,
            'position_after': self.positions.get(symbol, 0)
        }
        
        self.trades.append(trade_record)
        self.total_commission_paid += commission
        self.total_trades += 1
        
        return {
            'status': 'executed',
            'trade': trade_record
        }
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                position_value += quantity * current_prices[symbol]
        
        return self.cash + position_value
    
    def get_positions_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get summary of current positions."""
        positions_data = []
        total_position_value = 0
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity != 0:
                current_price = current_prices[symbol]
                position_value = quantity * current_price
                total_position_value += position_value
                
                # Find average entry price
                symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
                if symbol_trades:
                    # Weighted average entry price
                    total_quantity = 0
                    total_cost = 0
                    for trade in symbol_trades:
                        if trade['quantity'] > 0:  # Buys only
                            total_quantity += trade['quantity']
                            total_cost += trade['quantity'] * trade['price']
                    
                    avg_entry_price = total_cost / total_quantity if total_quantity > 0 else current_price
                    pnl = (current_price - avg_entry_price) * quantity
                    pnl_pct = pnl / (avg_entry_price * abs(quantity)) if quantity != 0 else 0
                else:
                    avg_entry_price = current_price
                    pnl = 0
                    pnl_pct = 0
                
                positions_data.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': current_price,
                    'position_value': position_value,
                    'avg_entry_price': avg_entry_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
        
        total_value = self.cash + total_position_value
        
        return {
            'positions': positions_data,
            'cash': self.cash,
            'total_position_value': total_position_value,
            'total_value': total_value,
            'total_return': (total_value / self.initial_capital) - 1,
            'total_commission_paid': self.total_commission_paid,
            'total_trades': self.total_trades
        }
    
    def record_portfolio_snapshot(self, current_prices: Dict[str, float]):
        """Record current portfolio state."""
        total_value = self.get_total_value(current_prices)
        
        snapshot = {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.cash,
            'position_value': total_value - self.cash,
            'num_positions': len([p for p in self.positions.values() if p != 0]),
            'return': (total_value / self.initial_capital) - 1
        }
        
        self.portfolio_history.append(snapshot)
        
        # Update daily loss tracking
        daily_return = (total_value / self.initial_capital) - 1
        self.daily_loss = min(0, daily_return)


class MinutePaperTrader:
    """Real-time paper trading system for minute-level strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Trading components
        self.account = PaperTradingAccount(
            initial_capital=self.config['initial_capital'],
            commission=self.config['commission']
        )
        
        # Model and strategy
        self.model = None
        self.strategy = None
        self.feature_engine = MinuteFeatureEngine()
        
        # Market data
        self.symbols = self.config['symbols']
        self.market_data = {}  # symbol -> recent data
        self.current_prices = {}
        
        # Trading state
        self.is_running = False
        self.last_trade_time = {}
        self.min_trade_interval = 60  # Minimum seconds between trades per symbol
        
        # Performance tracking
        self.start_time = None
        self.trade_signals = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
            'initial_capital': 100000,
            'commission': 0.001,
            'update_interval': 60,  # seconds
            'lookback_minutes': 1440,  # 24 hours
            'model_path': None,
            'retrain_hours': 24,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'max_positions': 3,
            'position_sizing': 'equal_weight',
            'log_trades': True,
            'save_snapshots': True
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for paper trading."""
        logger = logging.getLogger('MinutePaperTrader')
        logger.setLevel(logging.INFO)
        
        # File handler
        os.makedirs('logs/paper_trading', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'logs/paper_trading/paper_trade_{timestamp}.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def initialize_model(self):
        """Initialize or load the trading model."""
        self.logger.info("Initializing trading model...")
        
        if self.config['model_path'] and os.path.exists(self.config['model_path']):
            # Load pre-trained model
            self.logger.info(f"Loading model from {self.config['model_path']}")
            self.model = MinuteRandomForestModel()
            self.model.load_models(self.config['model_path'])
        else:
            # Train new model with recent data
            self.logger.info("Training new model with recent data...")
            self.model = self._train_new_model()
        
        # Initialize strategy
        self.strategy = MinuteStrategyEnsemble()
        self.logger.info("Model and strategy initialized")
    
    def _train_new_model(self) -> MinuteRandomForestModel:
        """Train a new model with recent historical data."""
        # Fetch training data (last 7 days - yfinance limit for minute data)
        self.logger.info("Fetching training data...")
        
        training_data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="7d", interval="1m")
                if not data.empty:
                    # Standardize column names to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    training_data[symbol] = data
                    self.logger.info(f"  {symbol}: {len(data)} training samples")
            except Exception as e:
                self.logger.error(f"Error fetching training data for {symbol}: {e}")
        
        if not training_data:
            raise ValueError("No training data available")
        
        # Generate features
        all_features = []
        all_targets = []
        
        for symbol, data in training_data.items():
            features = self.feature_engine.generate_minute_features(data, symbol)
            
            # Simple targets (5-minute returns)
            target = data['close'].pct_change(5).shift(-5)
            
            # Align data
            common_idx = features.index.intersection(target.index)
            features_aligned = features.loc[common_idx]
            target_aligned = target.loc[common_idx]
            
            # Remove NaN
            valid_idx = ~(features_aligned.isna().any(axis=1) | target_aligned.isna())
            
            all_features.append(features_aligned[valid_idx])
            all_targets.append(target_aligned[valid_idx])
        
        # Combine and train
        if all_features:
            combined_features = pd.concat(all_features)
            combined_targets = pd.concat(all_targets)
            
            # Simple model for demo
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(combined_features)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_scaled, combined_targets)
            
            # Wrap in our model class
            class SimpleModelWrapper:
                def __init__(self, model, scaler):
                    self.model = model
                    self.scaler = scaler
                    self.is_fitted = True
                
                def predict_multi_horizon(self, features, symbols):
                    predictions = pd.DataFrame(index=features.index)
                    X_scaled = self.scaler.transform(features.fillna(0))
                    pred = self.model.predict(X_scaled)
                    
                    for symbol in symbols:
                        predictions[f'{symbol}_1min_pred'] = pred
                    
                    return predictions
            
            return SimpleModelWrapper(model, scaler)
        else:
            raise ValueError("No valid training data")
    
    def fetch_latest_data(self):
        """Fetch the latest minute data for all symbols."""
        self.logger.debug("Fetching latest market data...")
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get recent data (last 2 hours for features)
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    # Standardize column names to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    self.market_data[symbol] = data
                    self.current_prices[symbol] = data['close'].iloc[-1]
                    self.logger.debug(f"  {symbol}: ${self.current_prices[symbol]:,.2f}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
    
    def generate_trading_signals(self) -> Dict[str, float]:
        """Generate trading signals based on latest data."""
        if not self.model or not self.market_data:
            return {}
        
        try:
            # Combine recent data for features
            combined_features = []
            
            for symbol in self.symbols:
                if symbol in self.market_data:
                    data = self.market_data[symbol]
                    if len(data) > 30:  # Need minimum data for features
                        features = self.feature_engine.generate_minute_features(data.tail(100), symbol)
                        if not features.empty:
                            combined_features.append(features.iloc[-1:])  # Latest features only
            
            if not combined_features:
                return {}
            
            # Get predictions
            latest_features = pd.concat(combined_features)
            predictions = self.model.predict_multi_horizon(latest_features, self.symbols)
            
            # Generate signals
            signals = {}
            for symbol in self.symbols:
                pred_col = f'{symbol}_1min_pred'
                if pred_col in predictions.columns:
                    pred_value = predictions[pred_col].iloc[-1]
                    
                    # Simple threshold strategy
                    if pred_value > 0.002:  # 0.2% expected return
                        signals[symbol] = 1  # Buy
                    elif pred_value < -0.002:
                        signals[symbol] = -1  # Sell
                    else:
                        signals[symbol] = 0  # Hold
                else:
                    signals[symbol] = 0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {}
    
    def execute_trading_signals(self, signals: Dict[str, float]):
        """Execute trading signals with risk management."""
        
        for symbol, signal in signals.items():
            if signal == 0 or symbol not in self.current_prices:
                continue
            
            # Check minimum trade interval
            if symbol in self.last_trade_time:
                time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
                if time_since_last < self.min_trade_interval:
                    continue
            
            current_price = self.current_prices[symbol]
            current_position = self.account.positions.get(symbol, 0)
            
            # Position sizing
            total_value = self.account.get_total_value(self.current_prices)
            max_position_value = total_value * 0.2  # Max 20% per position
            
            if signal > 0 and current_position <= 0:  # Buy signal
                # Calculate position size
                position_value = min(max_position_value, self.account.cash * 0.5)
                quantity = position_value / current_price
                
                if quantity * current_price > 100:  # Minimum trade size
                    result = self.account.execute_order(symbol, quantity, current_price)
                    
                    if result['status'] == 'executed':
                        self.logger.info(f"BUY {symbol}: {quantity:.4f} @ ${current_price:,.2f}")
                        self.last_trade_time[symbol] = datetime.now()
                    else:
                        self.logger.warning(f"Buy order rejected for {symbol}: {result['reason']}")
            
            elif signal < 0 and current_position > 0:  # Sell signal
                # Sell entire position
                result = self.account.execute_order(symbol, -current_position, current_price)
                
                if result['status'] == 'executed':
                    self.logger.info(f"SELL {symbol}: {current_position:.4f} @ ${current_price:,.2f}")
                    self.last_trade_time[symbol] = datetime.now()
                else:
                    self.logger.warning(f"Sell order rejected for {symbol}: {result['reason']}")
            
            # Record signal
            self.trade_signals.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'price': current_price,
                'position': current_position
            })
    
    def apply_risk_management(self):
        """Apply stop-loss and take-profit rules."""
        positions_summary = self.account.get_positions_summary(self.current_prices)
        
        for position in positions_summary['positions']:
            symbol = position['symbol']
            pnl_pct = position['pnl_pct']
            
            # Stop loss
            if pnl_pct <= -self.config['stop_loss']:
                self.logger.warning(f"STOP LOSS triggered for {symbol} at {pnl_pct:.2%} loss")
                result = self.account.execute_order(
                    symbol, 
                    -position['quantity'], 
                    position['current_price'],
                    order_type='stop_loss'
                )
                if result['status'] == 'executed':
                    self.logger.info(f"Stop loss executed for {symbol}")
            
            # Take profit
            elif pnl_pct >= self.config['take_profit']:
                self.logger.info(f"TAKE PROFIT triggered for {symbol} at {pnl_pct:.2%} gain")
                result = self.account.execute_order(
                    symbol,
                    -position['quantity'] * 0.5,  # Sell half
                    position['current_price'],
                    order_type='take_profit'
                )
                if result['status'] == 'executed':
                    self.logger.info(f"Partial take profit executed for {symbol}")
    
    def print_portfolio_status(self):
        """Print current portfolio status."""
        summary = self.account.get_positions_summary(self.current_prices)
        
        print("\n" + "="*60)
        print(f"PORTFOLIO STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        print(f"\nAccount Value: ${summary['total_value']:,.2f}")
        print(f"Total Return: {summary['total_return']:.2%}")
        print(f"Cash: ${summary['cash']:,.2f}")
        print(f"Positions Value: ${summary['total_position_value']:,.2f}")
        
        if summary['positions']:
            print("\nOpen Positions:")
            print("-"*60)
            print(f"{'Symbol':<10} {'Quantity':>10} {'Price':>10} {'Value':>12} {'P&L':>10} {'P&L%':>8}")
            print("-"*60)
            
            for pos in summary['positions']:
                print(f"{pos['symbol']:<10} {pos['quantity']:>10.4f} "
                      f"${pos['current_price']:>9.2f} ${pos['position_value']:>11.2f} "
                      f"${pos['pnl']:>9.2f} {pos['pnl_pct']:>7.2%}")
        
        print(f"\nTotal Trades: {summary['total_trades']}")
        print(f"Commission Paid: ${summary['total_commission_paid']:,.2f}")
        print("="*60)
    
    def save_snapshot(self):
        """Save current trading snapshot."""
        if not self.config['save_snapshots']:
            return
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': self.account.get_positions_summary(self.current_prices),
            'recent_trades': self.account.trades[-10:] if self.account.trades else [],
            'current_prices': self.current_prices,
            'config': self.config
        }
        
        # Save to file
        os.makedirs('snapshots', exist_ok=True)
        filename = f"snapshots/paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
    
    def run_trading_loop(self):
        """Main trading loop."""
        self.logger.info("Starting paper trading...")
        self.is_running = True
        self.start_time = datetime.now()
        
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                
                # Fetch latest data
                self.fetch_latest_data()
                
                # Record portfolio snapshot
                self.account.record_portfolio_snapshot(self.current_prices)
                
                # Generate and execute signals
                signals = self.generate_trading_signals()
                if signals:
                    self.execute_trading_signals(signals)
                
                # Apply risk management
                self.apply_risk_management()
                
                # Print status every 5 iterations
                if iteration % 5 == 0:
                    self.print_portfolio_status()
                    self.save_snapshot()
                
                # Wait for next update
                self.logger.debug(f"Iteration {iteration} complete. Waiting {self.config['update_interval']}s...")
                time.sleep(self.config['update_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading stopped by user")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
        finally:
            self.is_running = False
            self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final trading report."""
        self.logger.info("Generating final report...")
        
        # Portfolio performance
        summary = self.account.get_positions_summary(self.current_prices)
        
        # Create performance dataframe
        if self.account.portfolio_history:
            portfolio_df = pd.DataFrame(self.account.portfolio_history)
            portfolio_df.set_index('timestamp', inplace=True)
            
            # Calculate metrics
            returns = portfolio_df['total_value'].pct_change().dropna()
            
            total_return = summary['total_return']
            volatility = returns.std() * np.sqrt(365 * 24 * 60)  # Annualized
            
            # Sharpe ratio
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 365 * 24 * 60 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Win rate
            if self.account.trades:
                trades_df = pd.DataFrame(self.account.trades)
                # Group by symbol to calculate P&L
                winning_trades = 0
                losing_trades = 0
                
                for symbol in trades_df['symbol'].unique():
                    symbol_trades = trades_df[trades_df['symbol'] == symbol]
                    net_position = symbol_trades['quantity'].sum()
                    
                    if net_position < 0:  # Closed position
                        avg_buy = symbol_trades[symbol_trades['quantity'] > 0]['price'].mean()
                        avg_sell = symbol_trades[symbol_trades['quantity'] < 0]['price'].mean()
                        
                        if avg_sell > avg_buy:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                
                win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
            else:
                win_rate = 0
            
            # Trading duration
            duration = datetime.now() - self.start_time
            
            # Print report
            print("\n" + "="*80)
            print("PAPER TRADING FINAL REPORT")
            print("="*80)
            
            print(f"\nTrading Period:")
            print(f"  Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Duration: {duration}")
            
            print(f"\nPerformance Summary:")
            print(f"  Initial Capital: ${self.account.initial_capital:,.2f}")
            print(f"  Final Value: ${summary['total_value']:,.2f}")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Annualized Return: {(total_return / (duration.total_seconds() / (365 * 24 * 3600))):.2%}")
            
            print(f"\nRisk Metrics:")
            print(f"  Volatility (Annual): {volatility:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")
            
            print(f"\nTrading Statistics:")
            print(f"  Total Trades: {summary['total_trades']}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Commission Paid: ${summary['total_commission_paid']:,.2f}")
            print(f"  Commission Impact: {(summary['total_commission_paid'] / self.account.initial_capital):.2%}")
            
            if summary['positions']:
                print(f"\nFinal Positions:")
                for pos in summary['positions']:
                    print(f"  {pos['symbol']}: {pos['quantity']:.4f} units @ ${pos['current_price']:,.2f} (P&L: {pos['pnl_pct']:.2%})")
            
            print("\n" + "="*80)
            
            # Save detailed report
            self._save_detailed_report(portfolio_df, summary, {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'duration': str(duration)
            })
            
            # Create visualizations
            self._create_performance_charts(portfolio_df)
            
        else:
            print("No trading history available for report")
    
    def _save_detailed_report(self, portfolio_df: pd.DataFrame, summary: Dict, metrics: Dict):
        """Save detailed report to file."""
        report = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'symbols': self.symbols,
                'initial_capital': self.account.initial_capital
            },
            'performance_metrics': metrics,
            'final_portfolio': summary,
            'trades': [self._serialize_trade(t) for t in self.account.trades],
            'portfolio_history': portfolio_df.to_dict('records'),
            'config': self.config
        }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        filename = f"reports/paper_trade_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Detailed report saved to {filename}")
    
    def _serialize_trade(self, trade: Dict) -> Dict:
        """Serialize trade record for JSON."""
        serialized = {}
        for key, value in trade.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        return serialized
    
    def _create_performance_charts(self, portfolio_df: pd.DataFrame):
        """Create performance visualization charts."""
        try:
            viz_suite = MinuteVisualizationSuite()
            
            # Create simple performance chart
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Portfolio value
            ax1.plot(portfolio_df.index, portfolio_df['total_value'], linewidth=2)
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # Returns
            returns = portfolio_df['total_value'].pct_change() * 100
            ax2.plot(portfolio_df.index, returns, linewidth=1, alpha=0.7)
            ax2.set_title('Minute Returns (%)')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            os.makedirs('charts', exist_ok=True)
            chart_path = f"charts/paper_trade_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Performance chart saved to {chart_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating charts: {e}")


def main():
    """Main entry point for paper trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Minute-Level Cryptocurrency Paper Trading')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--symbols', nargs='+', default=['BTC-USD', 'ETH-USD'], 
                       help='Symbols to trade')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Update interval in seconds')
    parser.add_argument('--model', type=str, help='Pre-trained model path')
    
    args = parser.parse_args()
    
    # Load or create config
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments and ensure all required keys exist
    config['initial_capital'] = args.capital
    config['symbols'] = args.symbols
    config['update_interval'] = args.interval
    if args.model:
        config['model_path'] = args.model
    
    # Ensure all required config keys exist
    if 'commission' not in config:
        config['commission'] = 0.001
    if 'lookback_minutes' not in config:
        config['lookback_minutes'] = 1440
    if 'model_path' not in config:
        config['model_path'] = None
    if 'retrain_hours' not in config:
        config['retrain_hours'] = 24
    if 'stop_loss' not in config:
        config['stop_loss'] = 0.02
    if 'take_profit' not in config:
        config['take_profit'] = 0.05
    if 'max_positions' not in config:
        config['max_positions'] = 3
    if 'position_sizing' not in config:
        config['position_sizing'] = 'equal_weight'
    if 'log_trades' not in config:
        config['log_trades'] = True
    if 'save_snapshots' not in config:
        config['save_snapshots'] = True
    
    # Create and run paper trader
    print("🚀 Starting Minute-Level Cryptocurrency Paper Trading")
    print("="*60)
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Update Interval: {args.interval} seconds")
    print("="*60)
    print("\nPress Ctrl+C to stop trading and generate report\n")
    
    trader = MinutePaperTrader(config)
    
    try:
        # Initialize model
        trader.initialize_model()
        
        # Run trading loop
        trader.run_trading_loop()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        trader.logger.error(f"Fatal error: {e}")
    
    print("\n✅ Paper trading session completed!")


if __name__ == "__main__":
    main()