#!/usr/bin/env python3
"""
Simple Paper Trading Demo for Cryptocurrency Trading System

This demonstrates real-time paper trading functionality with:
- Live minute data fetching
- Simple moving average strategy
- Risk management
- Real-time portfolio tracking

Usage: python3 simple_paper_trader.py
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimplePaperTrader:
    def __init__(self, initial_capital=100000, symbols=['BTC-USD', 'ETH-USD']):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.symbols = symbols
        self.positions = {symbol: 0 for symbol in symbols}
        self.current_prices = {}
        self.trades = []
        self.portfolio_history = []
        self.iteration = 0
        
        print("📊 Simple Paper Trading System Initialized")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"🪙 Trading Symbols: {', '.join(symbols)}")
        print("=" * 50)
    
    def fetch_latest_prices(self):
        """Fetch latest prices for all symbols."""
        try:
            for symbol in self.symbols:
                ticker = yf.Ticker(symbol)
                # Get most recent price
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    self.current_prices[symbol] = hist['Close'].iloc[-1]
            return True
        except Exception as e:
            print(f"⚠️ Error fetching prices: {e}")
            return False
    
    def calculate_signals(self):
        """Generate simple trading signals based on moving averages."""
        signals = {}
        
        try:
            for symbol in self.symbols:
                if symbol not in self.current_prices:
                    signals[symbol] = 0
                    continue
                
                # Get recent data for moving averages
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="5m")
                
                if len(hist) >= 20:
                    # Simple strategy: 5-period vs 20-period moving average
                    ma5 = hist['Close'].tail(5).mean()
                    ma20 = hist['Close'].tail(20).mean()
                    current_price = hist['Close'].iloc[-1]
                    
                    # Buy signal: MA5 > MA20 and price above MA5
                    if ma5 > ma20 and current_price > ma5:
                        signals[symbol] = 1  # Buy
                    # Sell signal: MA5 < MA20 and we have position
                    elif ma5 < ma20 and self.positions[symbol] > 0:
                        signals[symbol] = -1  # Sell
                    else:
                        signals[symbol] = 0  # Hold
                else:
                    signals[symbol] = 0
                    
        except Exception as e:
            print(f"⚠️ Error calculating signals: {e}")
            signals = {symbol: 0 for symbol in self.symbols}
        
        return signals
    
    def execute_trades(self, signals):
        """Execute trades based on signals."""
        total_value = self.get_total_value()
        
        for symbol, signal in signals.items():
            if signal == 0 or symbol not in self.current_prices:
                continue
                
            current_price = self.current_prices[symbol]
            
            if signal == 1:  # Buy signal
                # Use 20% of available cash
                trade_amount = self.cash * 0.2
                if trade_amount > 1000:  # Minimum trade size
                    quantity = trade_amount / current_price
                    
                    # Execute buy
                    self.positions[symbol] += quantity
                    self.cash -= trade_amount
                    
                    self.trades.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': current_price,
                        'value': trade_amount
                    })
                    
                    print(f"🟢 BUY {symbol}: {quantity:.4f} @ ${current_price:,.2f} (${trade_amount:,.2f})")
            
            elif signal == -1 and self.positions[symbol] > 0:  # Sell signal
                # Sell 50% of position
                quantity_to_sell = self.positions[symbol] * 0.5
                trade_value = quantity_to_sell * current_price
                
                # Execute sell
                self.positions[symbol] -= quantity_to_sell
                self.cash += trade_value
                
                self.trades.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity_to_sell,
                    'price': current_price,
                    'value': trade_value
                })
                
                print(f"🔴 SELL {symbol}: {quantity_to_sell:.4f} @ ${current_price:,.2f} (${trade_value:,.2f})")
    
    def get_total_value(self):
        """Calculate total portfolio value."""
        position_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in self.current_prices:
                position_value += quantity * self.current_prices[symbol]
        return self.cash + position_value
    
    def record_portfolio_snapshot(self):
        """Record current portfolio state."""
        total_value = self.get_total_value()
        position_value = total_value - self.cash
        
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.cash,
            'position_value': position_value,
            'return': (total_value / self.initial_capital - 1) * 100
        })
    
    def print_status(self):
        """Print current portfolio status."""
        total_value = self.get_total_value()
        total_return = (total_value / self.initial_capital - 1) * 100
        
        print(f"\n📊 PORTFOLIO STATUS - Iteration #{self.iteration}")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"💰 Total Value: ${total_value:,.2f}")
        print(f"📈 Return: {total_return:+.2f}%")
        print(f"💵 Cash: ${self.cash:,.2f}")
        
        print(f"\n📍 Current Positions:")
        for symbol in self.symbols:
            if symbol in self.current_prices:
                price = self.current_prices[symbol]
                quantity = self.positions[symbol]
                value = quantity * price
                print(f"  {symbol}: {quantity:.4f} units @ ${price:,.2f} = ${value:,.2f}")
        
        print(f"\n📊 Recent Trades: {len(self.trades)}")
        if self.trades:
            for trade in self.trades[-3:]:  # Show last 3 trades
                print(f"  {trade['timestamp'].strftime('%H:%M')} {trade['action']} {trade['symbol']} "
                      f"{trade['quantity']:.4f} @ ${trade['price']:,.2f}")
        
        print("-" * 50)
    
    def run_trading_session(self, duration_minutes=10, update_interval=60):
        """Run a paper trading session."""
        print(f"🚀 Starting {duration_minutes}-minute trading session")
        print(f"⏱️ Updates every {update_interval} seconds")
        print("Press Ctrl+C to stop early\n")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now() < end_time:
                self.iteration += 1
                
                # Fetch latest market data
                if not self.fetch_latest_prices():
                    print("❌ Failed to fetch prices, skipping iteration")
                    time.sleep(update_interval)
                    continue
                
                # Record portfolio state
                self.record_portfolio_snapshot()
                
                # Generate and execute signals
                signals = self.calculate_signals()
                self.execute_trades(signals)
                
                # Print status every few iterations
                if self.iteration % 2 == 0:  # Every other iteration
                    self.print_status()
                
                # Wait for next update
                print(f"⏳ Waiting {update_interval}s for next update...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Trading session stopped by user")
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final trading report."""
        print("\n" + "=" * 60)
        print("📋 FINAL TRADING REPORT")
        print("=" * 60)
        
        final_value = self.get_total_value()
        total_return = (final_value / self.initial_capital - 1) * 100
        
        print(f"\n💰 FINANCIAL SUMMARY:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Profit/Loss: ${final_value - self.initial_capital:+,.2f}")
        
        print(f"\n📊 TRADING ACTIVITY:")
        print(f"  Total Trades: {len(self.trades)}")
        print(f"  Trading Sessions: {self.iteration}")
        
        if self.trades:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            print(f"  Buy Orders: {len(buy_trades)}")
            print(f"  Sell Orders: {len(sell_trades)}")
        
        print(f"\n📍 FINAL POSITIONS:")
        for symbol in self.symbols:
            if symbol in self.current_prices:
                quantity = self.positions[symbol]
                if quantity > 0:
                    value = quantity * self.current_prices[symbol]
                    print(f"  {symbol}: {quantity:.4f} units (${value:,.2f})")
        
        print(f"\n💵 Cash Remaining: ${self.cash:,.2f}")
        
        # Performance metrics
        if len(self.portfolio_history) > 1:
            returns = []
            for i, snapshot in enumerate(self.portfolio_history[1:], 1):
                prev_value = self.portfolio_history[i-1]['total_value']
                curr_value = snapshot['total_value']
                if prev_value > 0:
                    ret = (curr_value / prev_value - 1) * 100
                    returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                volatility = np.std(returns)
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"  Average Period Return: {avg_return:.3f}%")
                print(f"  Volatility: {volatility:.3f}%")
                if volatility > 0:
                    sharpe = avg_return / volatility
                    print(f"  Sharpe-like Ratio: {sharpe:.2f}")
        
        print("=" * 60)
        print("✅ Paper trading session completed!")

def main():
    """Main function to run paper trading demo."""
    print("🚀 Simple Cryptocurrency Paper Trading Demo")
    print("=" * 50)
    
    # Create trader
    trader = SimplePaperTrader(
        initial_capital=100000,
        symbols=['BTC-USD', 'ETH-USD']
    )
    
    # Run trading session
    trader.run_trading_session(
        duration_minutes=5,  # 5-minute demo
        update_interval=30   # Update every 30 seconds
    )

if __name__ == "__main__":
    main()