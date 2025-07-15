#!/usr/bin/env python3
"""
24-Hour BTC Trading Session Monitor

This script monitors the ongoing 24-hour trading session and provides status updates.
"""

import os
import json
import time
from datetime import datetime, timedelta
import subprocess

def get_trading_process_status():
    """Check if the trading process is still running."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'btc_24hr_paper_trader.py' in result.stdout:
            lines = [line for line in result.stdout.split('\n') if 'btc_24hr_paper_trader.py' in line]
            if lines:
                parts = lines[0].split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                return {'running': True, 'pid': pid, 'cpu': cpu, 'memory': mem}
        return {'running': False}
    except:
        return {'running': False}

def read_latest_log():
    """Read the latest trading log file."""
    log_dir = "logs/24hr_trading"
    if not os.path.exists(log_dir):
        return None
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('btc_24hr_')]
    if not log_files:
        return None
    
    latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(log_dir, f)))
    log_path = os.path.join(log_dir, latest_log)
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        return {'path': log_path, 'lines': lines}
    except:
        return None

def find_latest_session_data():
    """Find the latest session data JSON file."""
    json_files = [f for f in os.listdir('.') if f.startswith('btc_24hr_session_') and f.endswith('.json')]
    if not json_files:
        return None
    
    latest_json = max(json_files, key=lambda f: os.path.getctime(f))
    try:
        with open(latest_json, 'r') as f:
            data = json.load(f)
        return data
    except:
        return None

def parse_log_for_status(log_data):
    """Parse log file for trading status."""
    if not log_data:
        return None
    
    lines = log_data['lines']
    status = {
        'start_time': None,
        'end_time': None,
        'trades': 0,
        'latest_price': None,
        'portfolio_value': None,
        'return_pct': None,
        'last_update': None
    }
    
    for line in lines:
        line = line.strip()
        
        # Parse start time
        if '24-Hour BTC Paper Trading Session Initialized' in line:
            try:
                timestamp_str = line.split(']')[0][1:]
                status['start_time'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Parse end time
        if 'Session will run until:' in line:
            try:
                end_str = line.split('until: ')[1]
                status['end_time'] = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Parse trades
        if 'BUY:' in line or 'SELL:' in line:
            status['trades'] += 1
        
        # Parse latest portfolio status
        if 'Portfolio: $' in line:
            try:
                # Extract portfolio value and return
                parts = line.split('Portfolio: $')[1]
                value_part = parts.split(' ')[0].replace(',', '')
                status['portfolio_value'] = float(value_part)
                
                if '(' in parts and '%)' in parts:
                    return_str = parts.split('(')[1].split('%)')[0]
                    status['return_pct'] = float(return_str.replace('%', ''))
            except:
                pass
        
        # Parse BTC price
        if 'Price: $' in line:
            try:
                price_str = line.split('Price: $')[1].split(' ')[0].replace(',', '')
                status['latest_price'] = float(price_str)
            except:
                pass
        
        # Track last update
        if '[2025-' in line:
            try:
                timestamp_str = line.split(']')[0][1:]
                status['last_update'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass
    
    return status

def print_session_status():
    """Print comprehensive session status."""
    print("🔍 24-Hour BTC Paper Trading Session Monitor")
    print("=" * 60)
    
    # Check process status
    process_status = get_trading_process_status()
    if process_status['running']:
        print(f"✅ Trading Process: RUNNING (PID: {process_status['pid']})")
        print(f"💻 CPU Usage: {process_status['cpu']}%")
        print(f"🧠 Memory Usage: {process_status['memory']}%")
    else:
        print("❌ Trading Process: NOT RUNNING")
        return
    
    # Read log data
    log_data = read_latest_log()
    if not log_data:
        print("⚠️ No log file found")
        return
    
    print(f"📝 Log File: {log_data['path']}")
    
    # Parse status from log
    status = parse_log_for_status(log_data)
    if not status:
        print("⚠️ Could not parse log data")
        return
    
    current_time = datetime.now()
    
    if status['start_time']:
        elapsed = current_time - status['start_time']
        print(f"⏰ Session Started: {status['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Elapsed Time: {elapsed}")
        
        if status['end_time']:
            remaining = status['end_time'] - current_time
            if remaining.total_seconds() > 0:
                hours = remaining.total_seconds() / 3600
                print(f"⏳ Remaining Time: {remaining} ({hours:.1f} hours)")
            else:
                print("✅ Session Complete!")
    
    if status['last_update']:
        time_since_update = current_time - status['last_update']
        print(f"🔄 Last Update: {status['last_update'].strftime('%H:%M:%S')} ({time_since_update} ago)")
    
    print(f"\n💰 TRADING STATUS:")
    print(f"   Total Trades: {status['trades']}")
    
    if status['portfolio_value']:
        print(f"   Portfolio Value: ${status['portfolio_value']:,.2f}")
    
    if status['return_pct'] is not None:
        print(f"   Return: {status['return_pct']:+.2f}%")
    
    if status['latest_price']:
        print(f"   Latest BTC Price: ${status['latest_price']:,.2f}")
    
    # Check for session data file
    session_data = find_latest_session_data()
    if session_data:
        print(f"\n💾 Latest Session Data Available:")
        print(f"   Iterations: {session_data.get('session_info', {}).get('iteration', 'N/A')}")
        current_portfolio = session_data.get('current_portfolio', {})
        if current_portfolio:
            print(f"   Data Portfolio: ${current_portfolio.get('total_value', 0):,.2f}")
            print(f"   Data Return: {current_portfolio.get('return_pct', 0):+.2f}%")
    
    # Show recent log entries
    if log_data['lines']:
        print(f"\n📋 RECENT ACTIVITY (Last 5 entries):")
        recent_lines = [line.strip() for line in log_data['lines'][-5:] if line.strip()]
        for i, line in enumerate(recent_lines, 1):
            print(f"   {i}. {line}")
    
    print("=" * 60)

def main():
    """Main monitoring function."""
    print_session_status()

if __name__ == "__main__":
    main()