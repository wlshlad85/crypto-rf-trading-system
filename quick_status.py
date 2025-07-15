#!/usr/bin/env python3
"""
Quick Trading Session Status Check
"""

import os
import re
from datetime import datetime

def parse_latest_status():
    """Parse and display current trading session status."""
    log_path = "logs/24hr_trading/btc_24hr_20250713_140540.log"
    
    if not os.path.exists(log_path):
        print("❌ Log file not found")
        return
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Parse key data
    trades = []
    latest_portfolio = {}
    session_start = None
    
    for line in lines:
        # Session start
        if '24-Hour BTC Paper Trading Session Initialized' in line:
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                session_start = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        
        # Trades
        trade_match = re.search(r'\[([^\]]+)\] (🟢|🔴) (BUY|SELL|STOP_LOSS|TAKE_PROFIT) ([0-9.]+) BTC @ \$([0-9,]+\.[0-9]+) \(\$([0-9,]+\.[0-9]+)\)', line)
        if trade_match:
            timestamp_str, emoji, action, quantity, price, value = trade_match.groups()
            trades.append({
                'time': timestamp_str,
                'action': action,
                'quantity': float(quantity),
                'price': float(price.replace(',', '')),
                'value': float(value.replace(',', ''))
            })
        
        # Portfolio status
        if 'Total Value: $' in line:
            value_match = re.search(r'Total Value: \$([0-9,]+\.[0-9]+)', line)
            if value_match:
                latest_portfolio['total_value'] = float(value_match.group(1).replace(',', ''))
        
        if 'Return: ' in line:
            return_match = re.search(r'Return: ([+-][0-9.]+)%', line)
            if return_match:
                latest_portfolio['return_pct'] = float(return_match.group(1))
        
        if 'BTC Position: ' in line:
            btc_match = re.search(r'BTC Position: ([0-9.]+) BTC', line)
            if btc_match:
                latest_portfolio['btc_quantity'] = float(btc_match.group(1))
    
    # Display results
    print("🚀 BTC 24-Hour Paper Trading Status")
    print("=" * 50)
    
    if session_start:
        elapsed = datetime.now() - session_start
        print(f"⏰ Session Running: {elapsed}")
        print(f"📅 Started: {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 PORTFOLIO:")
    if latest_portfolio:
        total_value = latest_portfolio.get('total_value', 100000)
        return_pct = latest_portfolio.get('return_pct', 0)
        profit = total_value - 100000
        
        print(f"   💰 Total Value: ${total_value:,.2f}")
        print(f"   📈 Return: {return_pct:+.2f}%")
        print(f"   💵 Profit/Loss: ${profit:+,.2f}")
        
        if 'btc_quantity' in latest_portfolio:
            print(f"   ₿ BTC Position: {latest_portfolio['btc_quantity']:.6f}")
    
    print(f"\n📝 TRADING ACTIVITY:")
    print(f"   Total Trades: {len(trades)}")
    
    if trades:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']]
        
        print(f"   🟢 Buy Orders: {len(buy_trades)}")
        print(f"   🔴 Sell Orders: {len(sell_trades)}")
        
        # Recent trades
        print(f"\n📋 RECENT TRADES (Last 5):")
        for trade in trades[-5:]:
            action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
            time_str = trade['time'].split(' ')[1]  # Just time part
            print(f"   {action_emoji} {time_str} {trade['action']} {trade['quantity']:.4f} BTC @ ${trade['price']:,.2f}")
    
    # Check if still running
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        is_running = 'btc_24hr_paper_trader.py' in result.stdout
        
        print(f"\n🔄 SYSTEM STATUS:")
        if is_running:
            print("   ✅ Trading bot: ACTIVE")
        else:
            print("   ❌ Trading bot: STOPPED")
        
        # File age
        file_age = (datetime.now().timestamp() - os.path.getmtime(log_path)) / 60
        print(f"   📝 Last update: {file_age:.1f} minutes ago")
        
    except:
        print("   ⚠️ Status check failed")
    
    print("=" * 50)

if __name__ == "__main__":
    parse_latest_status()