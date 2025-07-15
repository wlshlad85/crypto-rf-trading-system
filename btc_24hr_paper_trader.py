#!/usr/bin/env python3
"""
24-Hour BTC-USD Paper Trading Session

This runs a continuous 24-hour paper trading session for Bitcoin with:
- Real-time minute data fetching
- Advanced trading strategies
- Risk management
- Continuous monitoring and logging
- Performance tracking

Usage: python3 btc_24hr_paper_trader.py
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class BTC24HourTrader:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.btc_position = 0
        self.current_price = 0
        self.trades = []
        self.portfolio_history = []
        self.price_history = []
        self.iteration = 0
        self.start_time = datetime.now()
        
        # Trading parameters
        self.min_trade_amount = 1000  # $1000 minimum trade
        self.max_position_pct = 0.8   # Max 80% in BTC
        self.stop_loss_pct = 0.03     # 3% stop loss
        self.take_profit_pct = 0.05   # 5% take profit
        
        # Strategy parameters
        self.last_signals = []
        self.position_entry_price = 0
        
        # Create logs directory
        os.makedirs('logs/24hr_trading', exist_ok=True)
        self.log_file = f'logs/24hr_trading/btc_24hr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        self.log_message("🚀 24-Hour BTC Paper Trading Session Initialized")
        self.log_message(f"💰 Initial Capital: ${initial_capital:,.2f}")
        self.log_message(f"📝 Log file: {self.log_file}")
        print("🚀 24-Hour BTC Paper Trading Session Starting!")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"📝 Logs: {self.log_file}")
        print("=" * 60)
    
    def log_message(self, message):
        """Log message to file and print to console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
        
        # Print to console
        print(log_entry)
    
    def fetch_btc_data(self):
        """Fetch latest BTC data and price history."""
        try:
            ticker = yf.Ticker("BTC-USD")
            
            # Get recent 1-hour data for strategy
            hist = ticker.history(period="2d", interval="5m")
            if hist.empty:
                return False
            
            self.current_price = hist['Close'].iloc[-1]
            
            # Store price history for analysis
            self.price_history.append({
                'timestamp': datetime.now(),
                'price': self.current_price,
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            })
            
            # Keep only last 1000 price points (about 83 hours at 5min intervals)
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
            
            return hist
            
        except Exception as e:
            self.log_message(f"⚠️ Error fetching BTC data: {e}")
            return False
    
    def calculate_technical_indicators(self, hist):
        """Calculate technical indicators for trading signals."""
        try:
            # Moving averages
            ma_short = hist['Close'].tail(12).mean()  # 1-hour MA (12 * 5min)
            ma_long = hist['Close'].tail(48).mean()   # 4-hour MA (48 * 5min)
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = hist['Close'].tail(bb_period).mean()
            bb_std_val = hist['Close'].tail(bb_period).std()
            bb_upper = bb_ma + (bb_std_val * bb_std)
            bb_lower = bb_ma - (bb_std_val * bb_std)
            
            # Volatility
            returns = hist['Close'].pct_change().tail(48)
            volatility = returns.std() * np.sqrt(288)  # Annualized (288 5-min periods per day)
            
            return {
                'ma_short': ma_short,
                'ma_long': ma_long,
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_ma': bb_ma,
                'volatility': volatility,
                'current_price': self.current_price
            }
            
        except Exception as e:
            self.log_message(f"⚠️ Error calculating indicators: {e}")
            return None
    
    def generate_trading_signal(self, indicators):
        """Generate trading signal based on technical analysis."""
        if not indicators:
            return 0
        
        signals = []
        
        # Signal 1: Moving Average Crossover
        if indicators['ma_short'] > indicators['ma_long']:
            signals.append(1)  # Bullish
        else:
            signals.append(-1)  # Bearish
        
        # Signal 2: RSI
        if indicators['rsi'] < 30:
            signals.append(1)  # Oversold - Buy
        elif indicators['rsi'] > 70:
            signals.append(-1)  # Overbought - Sell
        else:
            signals.append(0)  # Neutral
        
        # Signal 3: Bollinger Bands
        if self.current_price < indicators['bb_lower']:
            signals.append(1)  # Below lower band - Buy
        elif self.current_price > indicators['bb_upper']:
            signals.append(-1)  # Above upper band - Sell
        else:
            signals.append(0)  # Between bands
        
        # Signal 4: Volatility filter
        if indicators['volatility'] > 0.5:  # High volatility
            signals = [s * 0.5 for s in signals]  # Reduce signal strength
        
        # Combine signals
        combined_signal = sum(signals) / len(signals)
        
        # Store for trend analysis
        self.last_signals.append(combined_signal)
        if len(self.last_signals) > 10:
            self.last_signals = self.last_signals[-10:]
        
        # Return discrete signal
        if combined_signal > 0.3:
            return 1  # Buy
        elif combined_signal < -0.3:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def apply_risk_management(self):
        """Apply stop-loss and take-profit rules."""
        if self.btc_position == 0 or self.position_entry_price == 0:
            return False
        
        current_value = self.btc_position * self.current_price
        entry_value = self.btc_position * self.position_entry_price
        pnl_pct = (current_value - entry_value) / entry_value
        
        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            self.log_message(f"🛑 STOP LOSS triggered at {pnl_pct:.2%} loss")
            self.execute_trade(-1, "STOP_LOSS")
            return True
        
        # Take profit
        elif pnl_pct >= self.take_profit_pct:
            self.log_message(f"🎯 TAKE PROFIT triggered at {pnl_pct:.2%} gain")
            # Sell 50% of position
            self.execute_trade(-0.5, "TAKE_PROFIT")
            return True
        
        return False
    
    def execute_trade(self, signal, reason="SIGNAL"):
        """Execute trading order based on signal."""
        if signal == 0:
            return
        
        total_value = self.get_total_value()
        
        if signal > 0:  # Buy signal
            # Calculate position size
            if self.btc_position == 0:
                # New position - use percentage of capital
                max_trade_value = total_value * 0.3  # 30% of capital
                trade_value = min(max_trade_value, self.cash)
            else:
                # Add to position - smaller amount
                trade_value = min(total_value * 0.1, self.cash)
            
            if trade_value >= self.min_trade_amount and trade_value <= self.cash:
                btc_quantity = trade_value / self.current_price
                
                # Execute buy
                self.btc_position += btc_quantity
                self.cash -= trade_value
                
                # Update entry price (weighted average)
                if self.position_entry_price == 0:
                    self.position_entry_price = self.current_price
                else:
                    total_cost = (self.btc_position - btc_quantity) * self.position_entry_price + trade_value
                    self.position_entry_price = total_cost / self.btc_position
                
                self.trades.append({
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'quantity': btc_quantity,
                    'price': self.current_price,
                    'value': trade_value,
                    'reason': reason,
                    'total_btc': self.btc_position,
                    'cash_after': self.cash
                })
                
                self.log_message(f"🟢 BUY {btc_quantity:.6f} BTC @ ${self.current_price:,.2f} "
                               f"(${trade_value:,.2f}) - {reason}")
        
        elif signal < 0:  # Sell signal
            if self.btc_position > 0:
                # Determine sell quantity
                if abs(signal) >= 1:
                    sell_quantity = self.btc_position  # Sell all
                else:
                    sell_quantity = self.btc_position * abs(signal)  # Partial sell
                
                trade_value = sell_quantity * self.current_price
                
                # Execute sell
                self.btc_position -= sell_quantity
                self.cash += trade_value
                
                # Reset entry price if fully sold
                if self.btc_position < 1e-8:
                    self.btc_position = 0
                    self.position_entry_price = 0
                
                self.trades.append({
                    'timestamp': datetime.now(),
                    'action': 'SELL',
                    'quantity': sell_quantity,
                    'price': self.current_price,
                    'value': trade_value,
                    'reason': reason,
                    'total_btc': self.btc_position,
                    'cash_after': self.cash
                })
                
                self.log_message(f"🔴 SELL {sell_quantity:.6f} BTC @ ${self.current_price:,.2f} "
                               f"(${trade_value:,.2f}) - {reason}")
    
    def get_total_value(self):
        """Calculate total portfolio value."""
        btc_value = self.btc_position * self.current_price
        return self.cash + btc_value
    
    def record_portfolio_snapshot(self):
        """Record current portfolio state."""
        total_value = self.get_total_value()
        btc_value = self.btc_position * self.current_price
        
        snapshot = {
            'timestamp': datetime.now(),
            'iteration': self.iteration,
            'total_value': total_value,
            'cash': self.cash,
            'btc_position': self.btc_position,
            'btc_value': btc_value,
            'btc_price': self.current_price,
            'return_pct': (total_value / self.initial_capital - 1) * 100,
            'hours_elapsed': (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
        self.portfolio_history.append(snapshot)
    
    def print_status(self):
        """Print current portfolio status."""
        total_value = self.get_total_value()
        btc_value = self.btc_position * self.current_price
        total_return = (total_value / self.initial_capital - 1) * 100
        hours_elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Calculate P&L if we have a position
        position_pnl = ""
        if self.btc_position > 0 and self.position_entry_price > 0:
            pnl_pct = (self.current_price / self.position_entry_price - 1) * 100
            position_pnl = f"Position P&L: {pnl_pct:+.2f}%"
        
        self.log_message(f"\n📊 PORTFOLIO STATUS - Hour {hours_elapsed:.1f}/24")
        self.log_message(f"💰 Total Value: ${total_value:,.2f}")
        self.log_message(f"📈 Return: {total_return:+.2f}%")
        self.log_message(f"💵 Cash: ${self.cash:,.2f}")
        self.log_message(f"₿ BTC Position: {self.btc_position:.6f} BTC (${btc_value:,.2f})")
        self.log_message(f"💲 BTC Price: ${self.current_price:,.2f}")
        if position_pnl:
            self.log_message(f"📊 {position_pnl}")
        self.log_message(f"📝 Total Trades: {len(self.trades)}")
        self.log_message("-" * 60)
    
    def save_session_data(self):
        """Save current session data to file."""
        session_data = {
            'session_info': {
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat(),
                'hours_elapsed': (datetime.now() - self.start_time).total_seconds() / 3600,
                'initial_capital': self.initial_capital,
                'iteration': self.iteration
            },
            'current_portfolio': {
                'total_value': self.get_total_value(),
                'cash': self.cash,
                'btc_position': self.btc_position,
                'btc_price': self.current_price,
                'return_pct': (self.get_total_value() / self.initial_capital - 1) * 100
            },
            'trades': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    'action': trade['action'],
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'value': trade['value'],
                    'reason': trade['reason']
                } for trade in self.trades
            ],
            'portfolio_history': [
                {
                    'timestamp': snapshot['timestamp'].isoformat(),
                    'total_value': snapshot['total_value'],
                    'return_pct': snapshot['return_pct'],
                    'btc_price': snapshot['btc_price'],
                    'hours_elapsed': snapshot['hours_elapsed']
                } for snapshot in self.portfolio_history
            ]
        }
        
        # Save to JSON file
        filename = f"btc_24hr_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return filename
    
    def run_24_hour_session(self, update_interval=300):  # 5-minute updates
        """Run the 24-hour trading session."""
        end_time = self.start_time + timedelta(hours=24)
        
        self.log_message(f"🕐 Session will run until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"⏱️ Update interval: {update_interval} seconds")
        self.log_message("Press Ctrl+C to stop early and generate report\n")
        
        try:
            while datetime.now() < end_time:
                self.iteration += 1
                
                # Fetch latest BTC data
                hist = self.fetch_btc_data()
                if hist is False or (isinstance(hist, pd.DataFrame) and hist.empty):
                    self.log_message("❌ Failed to fetch data, skipping iteration")
                    time.sleep(update_interval)
                    continue
                
                # Calculate technical indicators
                indicators = self.calculate_technical_indicators(hist)
                
                # Apply risk management first
                risk_action = self.apply_risk_management()
                
                # Generate and execute trading signal (if no risk action taken)
                if not risk_action and indicators:
                    signal = self.generate_trading_signal(indicators)
                    if signal != 0:
                        self.execute_trade(signal)
                
                # Record portfolio state
                self.record_portfolio_snapshot()
                
                # Print status every hour (12 iterations at 5-min intervals)
                if self.iteration % 12 == 0:
                    self.print_status()
                
                # Save session data every 4 hours
                if self.iteration % 48 == 0:
                    filename = self.save_session_data()
                    self.log_message(f"💾 Session data saved to: {filename}")
                
                # Wait for next update
                remaining_hours = (end_time - datetime.now()).total_seconds() / 3600
                self.log_message(f"⏳ Next update in {update_interval}s... "
                               f"({remaining_hours:.1f} hours remaining)")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.log_message("\n⏹️ Session stopped by user")
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        final_value = self.get_total_value()
        total_return = (final_value / self.initial_capital - 1) * 100
        
        self.log_message("\n" + "=" * 80)
        self.log_message("📋 24-HOUR BTC PAPER TRADING FINAL REPORT")
        self.log_message("=" * 80)
        
        # Session summary
        self.log_message(f"\n🕐 SESSION SUMMARY:")
        self.log_message(f"  Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"  End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"  Duration: {duration}")
        self.log_message(f"  Total Iterations: {self.iteration}")
        
        # Financial performance
        self.log_message(f"\n💰 FINANCIAL PERFORMANCE:")
        self.log_message(f"  Initial Capital: ${self.initial_capital:,.2f}")
        self.log_message(f"  Final Value: ${final_value:,.2f}")
        self.log_message(f"  Total Return: {total_return:+.2f}%")
        self.log_message(f"  Profit/Loss: ${final_value - self.initial_capital:+,.2f}")
        
        # Trading activity
        self.log_message(f"\n📊 TRADING ACTIVITY:")
        self.log_message(f"  Total Trades: {len(self.trades)}")
        
        if self.trades:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            self.log_message(f"  Buy Orders: {len(buy_trades)}")
            self.log_message(f"  Sell Orders: {len(sell_trades)}")
            
            # Trade reasons analysis
            reasons = {}
            for trade in self.trades:
                reason = trade['reason']
                if reason not in reasons:
                    reasons[reason] = 0
                reasons[reason] += 1
            
            self.log_message(f"  Trade Reasons:")
            for reason, count in reasons.items():
                self.log_message(f"    {reason}: {count}")
        
        # Final position
        btc_value = self.btc_position * self.current_price
        self.log_message(f"\n📍 FINAL POSITION:")
        self.log_message(f"  Cash: ${self.cash:,.2f}")
        self.log_message(f"  BTC Position: {self.btc_position:.6f} BTC")
        self.log_message(f"  BTC Value: ${btc_value:,.2f}")
        self.log_message(f"  Final BTC Price: ${self.current_price:,.2f}")
        
        # Performance metrics
        if len(self.portfolio_history) > 1:
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_val = self.portfolio_history[i-1]['total_value']
                curr_val = self.portfolio_history[i]['total_value']
                if prev_val > 0:
                    ret = (curr_val / prev_val - 1) * 100
                    returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                volatility = np.std(returns)
                max_return = max(returns)
                min_return = min(returns)
                
                self.log_message(f"\n📈 PERFORMANCE METRICS:")
                self.log_message(f"  Average Period Return: {avg_return:.4f}%")
                self.log_message(f"  Volatility: {volatility:.4f}%")
                self.log_message(f"  Best Period: {max_return:+.4f}%")
                self.log_message(f"  Worst Period: {min_return:+.4f}%")
                
                if volatility > 0:
                    sharpe = avg_return / volatility
                    self.log_message(f"  Sharpe-like Ratio: {sharpe:.3f}")
        
        # Save final session data
        filename = self.save_session_data()
        self.log_message(f"\n💾 Complete session data saved to: {filename}")
        
        self.log_message("=" * 80)
        self.log_message("✅ 24-Hour BTC Paper Trading Session Completed!")
        
        return filename

def main():
    """Main function to run 24-hour BTC paper trading."""
    print("🚀 Starting 24-Hour BTC-USD Paper Trading Session")
    print("=" * 60)
    
    # Create trader
    trader = BTC24HourTrader(initial_capital=100000)
    
    # Run 24-hour session with 5-minute updates
    trader.run_24_hour_session(update_interval=300)

if __name__ == "__main__":
    main()