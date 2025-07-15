#!/usr/bin/env python3
"""
Quick BTC Paper Trading Demo - 30 minutes with real-time updates

This demonstrates the full paper trading system in a condensed timeframe:
- Real BTC price fetching every 30 seconds
- Advanced technical analysis and signal generation
- Live portfolio tracking and trade execution
- Risk management and performance analysis

Usage: python3 btc_quick_demo.py
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class BTCQuickDemo:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.btc_position = 0
        self.current_price = 0
        self.trades = []
        self.portfolio_history = []
        self.iteration = 0
        self.start_time = datetime.now()
        
        # Trading parameters
        self.min_trade_amount = 1000
        self.position_entry_price = 0
        
        print("🚀 BTC Paper Trading Quick Demo")
        print("=" * 50)
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"⏱️ Demo Duration: 30 minutes")
        print(f"🔄 Update Frequency: 30 seconds")
        print("=" * 50)
    
    def fetch_btc_data(self):
        """Fetch latest BTC data."""
        try:
            ticker = yf.Ticker("BTC-USD")
            hist = ticker.history(period="1d", interval="5m")
            
            if hist.empty:
                return None
            
            self.current_price = hist['Close'].iloc[-1]
            return hist
            
        except Exception as e:
            print(f"⚠️ Error fetching BTC data: {e}")
            return None
    
    def calculate_signals(self, hist):
        """Calculate trading signals using technical analysis."""
        if len(hist) < 50:
            return 0
        
        try:
            # Moving averages
            ma_short = hist['Close'].tail(12).mean()  # 1-hour
            ma_long = hist['Close'].tail(48).mean()   # 4-hour
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Price momentum
            price_change_1h = (self.current_price / hist['Close'].iloc[-12] - 1) * 100
            
            # Signal calculation
            signals = []
            
            # MA signal
            if ma_short > ma_long:
                signals.append(1)
            else:
                signals.append(-1)
            
            # RSI signal
            if rsi < 30:
                signals.append(1)  # Oversold
            elif rsi > 70:
                signals.append(-1)  # Overbought
            else:
                signals.append(0)
            
            # Momentum signal
            if price_change_1h > 2:
                signals.append(1)
            elif price_change_1h < -2:
                signals.append(-1)
            else:
                signals.append(0)
            
            combined_signal = sum(signals) / len(signals)
            
            # Return signal with analysis
            signal_strength = "WEAK"
            if abs(combined_signal) > 0.6:
                signal_strength = "STRONG"
            elif abs(combined_signal) > 0.3:
                signal_strength = "MEDIUM"
            
            analysis = {
                'signal': combined_signal,
                'strength': signal_strength,
                'ma_short': ma_short,
                'ma_long': ma_long,
                'rsi': rsi,
                'price_change_1h': price_change_1h,
                'current_price': self.current_price
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ Error calculating signals: {e}")
            return 0
    
    def execute_trade(self, analysis):
        """Execute trades based on signal analysis."""
        if not analysis or analysis == 0:
            return
        
        signal = analysis['signal']
        
        if signal > 0.3:  # Strong buy signal
            if self.btc_position == 0:
                # New position
                trade_amount = min(self.cash * 0.4, 40000)  # 40% or $40k max
            else:
                # Add to position
                trade_amount = min(self.cash * 0.2, 20000)  # 20% or $20k max
            
            if trade_amount >= self.min_trade_amount:
                btc_quantity = trade_amount / self.current_price
                
                self.btc_position += btc_quantity
                self.cash -= trade_amount
                
                if self.position_entry_price == 0:
                    self.position_entry_price = self.current_price
                else:
                    # Weighted average
                    total_cost = (self.btc_position - btc_quantity) * self.position_entry_price + trade_amount
                    self.position_entry_price = total_cost / self.btc_position
                
                self.trades.append({
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'quantity': btc_quantity,
                    'price': self.current_price,
                    'value': trade_amount,
                    'signal_strength': analysis['strength'],
                    'rsi': analysis['rsi']
                })
                
                print(f"🟢 BUY: {btc_quantity:.6f} BTC @ ${self.current_price:,.2f} "
                      f"(${trade_amount:,.0f}) - {analysis['strength']} signal")
        
        elif signal < -0.3 and self.btc_position > 0:  # Strong sell signal
            # Sell portion of position
            sell_quantity = self.btc_position * 0.5
            trade_value = sell_quantity * self.current_price
            
            self.btc_position -= sell_quantity
            self.cash += trade_value
            
            if self.btc_position < 1e-8:
                self.btc_position = 0
                self.position_entry_price = 0
            
            self.trades.append({
                'timestamp': datetime.now(),
                'action': 'SELL',
                'quantity': sell_quantity,
                'price': self.current_price,
                'value': trade_value,
                'signal_strength': analysis['strength'],
                'rsi': analysis['rsi']
            })
            
            print(f"🔴 SELL: {sell_quantity:.6f} BTC @ ${self.current_price:,.2f} "
                  f"(${trade_value:,.0f}) - {analysis['strength']} signal")
    
    def apply_risk_management(self):
        """Apply stop-loss and take-profit."""
        if self.btc_position == 0 or self.position_entry_price == 0:
            return False
        
        pnl_pct = (self.current_price / self.position_entry_price - 1) * 100
        
        # Stop loss at -3%
        if pnl_pct <= -3:
            trade_value = self.btc_position * self.current_price
            self.cash += trade_value
            
            self.trades.append({
                'timestamp': datetime.now(),
                'action': 'STOP_LOSS',
                'quantity': self.btc_position,
                'price': self.current_price,
                'value': trade_value,
                'pnl_pct': pnl_pct
            })
            
            print(f"🛑 STOP LOSS: {self.btc_position:.6f} BTC @ ${self.current_price:,.2f} "
                  f"({pnl_pct:.1f}% loss)")
            
            self.btc_position = 0
            self.position_entry_price = 0
            return True
        
        # Take profit at +5%
        elif pnl_pct >= 5:
            sell_quantity = self.btc_position * 0.5  # Sell half
            trade_value = sell_quantity * self.current_price
            
            self.btc_position -= sell_quantity
            self.cash += trade_value
            
            self.trades.append({
                'timestamp': datetime.now(),
                'action': 'TAKE_PROFIT',
                'quantity': sell_quantity,
                'price': self.current_price,
                'value': trade_value,
                'pnl_pct': pnl_pct
            })
            
            print(f"🎯 TAKE PROFIT: {sell_quantity:.6f} BTC @ ${self.current_price:,.2f} "
                  f"({pnl_pct:.1f}% gain)")
            return True
        
        return False
    
    def get_total_value(self):
        """Calculate total portfolio value."""
        return self.cash + (self.btc_position * self.current_price)
    
    def record_portfolio_snapshot(self):
        """Record portfolio state."""
        total_value = self.get_total_value()
        
        snapshot = {
            'timestamp': datetime.now(),
            'iteration': self.iteration,
            'total_value': total_value,
            'cash': self.cash,
            'btc_position': self.btc_position,
            'btc_price': self.current_price,
            'return_pct': (total_value / self.initial_capital - 1) * 100
        }
        
        self.portfolio_history.append(snapshot)
    
    def print_status(self, analysis=None):
        """Print current status."""
        total_value = self.get_total_value()
        btc_value = self.btc_position * self.current_price
        total_return = (total_value / self.initial_capital - 1) * 100
        elapsed_min = (datetime.now() - self.start_time).total_seconds() / 60
        
        print(f"\n📊 STATUS - Minute {elapsed_min:.1f}/30")
        print(f"💰 Portfolio: ${total_value:,.2f} ({total_return:+.2f}%)")
        print(f"💵 Cash: ${self.cash:,.2f}")
        print(f"₿ BTC: {self.btc_position:.6f} (${btc_value:,.2f})")
        print(f"💲 Price: ${self.current_price:,.2f}")
        
        if self.btc_position > 0 and self.position_entry_price > 0:
            pnl = (self.current_price / self.position_entry_price - 1) * 100
            print(f"📈 Position P&L: {pnl:+.2f}%")
        
        if analysis and analysis != 0:
            print(f"🔍 Signal: {analysis['signal']:.2f} ({analysis['strength']})")
            print(f"📊 RSI: {analysis['rsi']:.1f}")
            print(f"📈 1h Change: {analysis['price_change_1h']:+.2f}%")
        
        print(f"📝 Trades: {len(self.trades)}")
        print("-" * 50)
    
    def run_demo(self, duration_minutes=30, update_seconds=30):
        """Run the quick demo."""
        end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        print(f"\n🚀 Starting {duration_minutes}-minute BTC paper trading demo")
        print(f"⏰ Will run until: {end_time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop early\n")
        
        try:
            while datetime.now() < end_time:
                self.iteration += 1
                
                # Fetch BTC data
                hist = self.fetch_btc_data()
                if hist is None:
                    print("❌ Failed to fetch data, retrying...")
                    time.sleep(update_seconds)
                    continue
                
                # Calculate signals
                analysis = self.calculate_signals(hist)
                
                # Apply risk management first
                risk_action = self.apply_risk_management()
                
                # Execute trades if no risk action
                if not risk_action and analysis != 0:
                    self.execute_trade(analysis)
                
                # Record state
                self.record_portfolio_snapshot()
                
                # Print status every few iterations
                if self.iteration % 3 == 0:  # Every ~90 seconds
                    self.print_status(analysis)
                
                # Wait for next update
                remaining = (end_time - datetime.now()).total_seconds()
                if remaining > 0:
                    print(f"⏳ Next update in {update_seconds}s... ({remaining/60:.1f} min remaining)")
                    time.sleep(update_seconds)
                
        except KeyboardInterrupt:
            print("\n⏹️ Demo stopped by user")
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final demo report."""
        final_value = self.get_total_value()
        total_return = (final_value / self.initial_capital - 1) * 100
        duration = datetime.now() - self.start_time
        
        print("\n" + "=" * 60)
        print("📋 BTC PAPER TRADING DEMO RESULTS")
        print("=" * 60)
        
        print(f"\n⏱️ SESSION INFO:")
        print(f"  Duration: {duration}")
        print(f"  Updates: {self.iteration}")
        print(f"  Start Time: {self.start_time.strftime('%H:%M:%S')}")
        print(f"  End Time: {datetime.now().strftime('%H:%M:%S')}")
        
        print(f"\n💰 FINANCIAL RESULTS:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Profit/Loss: ${final_value - self.initial_capital:+,.2f}")
        
        print(f"\n📊 TRADING ACTIVITY:")
        print(f"  Total Trades: {len(self.trades)}")
        
        if self.trades:
            buy_trades = sum(1 for t in self.trades if t['action'] == 'BUY')
            sell_trades = sum(1 for t in self.trades if t['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])
            
            print(f"  Buy Orders: {buy_trades}")
            print(f"  Sell Orders: {sell_trades}")
            
            print(f"\n📝 TRADE HISTORY:")
            for i, trade in enumerate(self.trades[-5:], 1):  # Last 5 trades
                action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"  {i}. {action_emoji} {trade['action']}: {trade['quantity']:.4f} BTC @ "
                      f"${trade['price']:,.2f} ({trade['timestamp'].strftime('%H:%M:%S')})")
        
        print(f"\n📍 FINAL POSITION:")
        btc_value = self.btc_position * self.current_price
        print(f"  Cash: ${self.cash:,.2f} ({self.cash/final_value:.1%})")
        print(f"  BTC: {self.btc_position:.6f} BTC (${btc_value:,.2f}) ({btc_value/final_value:.1%})")
        print(f"  Final BTC Price: ${self.current_price:,.2f}")
        
        # Performance metrics
        if len(self.portfolio_history) > 1:
            values = [s['total_value'] for s in self.portfolio_history]
            returns = [(values[i]/values[i-1]-1)*100 for i in range(1, len(values))]
            
            if returns:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"  Best Period: {max(returns):+.3f}%")
                print(f"  Worst Period: {min(returns):+.3f}%")
                print(f"  Avg Period Return: {np.mean(returns):+.3f}%")
                print(f"  Volatility: {np.std(returns):.3f}%")
        
        # Save results
        results = {
            'session_info': {
                'start_time': self.start_time.isoformat(),
                'duration_minutes': duration.total_seconds() / 60,
                'iterations': self.iteration
            },
            'performance': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return_pct': total_return,
                'profit_loss': final_value - self.initial_capital
            },
            'trades': len(self.trades),
            'final_position': {
                'cash': self.cash,
                'btc_position': self.btc_position,
                'btc_price': self.current_price
            }
        }
        
        filename = f"btc_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        print("=" * 60)
        print("✅ BTC Paper Trading Demo Completed!")

def main():
    """Run the BTC paper trading demo."""
    demo = BTCQuickDemo(initial_capital=100000)
    demo.run_demo(duration_minutes=30, update_seconds=30)

if __name__ == "__main__":
    main()