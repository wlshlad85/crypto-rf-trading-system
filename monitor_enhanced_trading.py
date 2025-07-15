#!/usr/bin/env python3
"""
Enhanced Trading Session Monitor

Monitor the enhanced Random Forest paper trading session in real-time.
"""

import os
import time
from datetime import datetime

def monitor_enhanced_session():
    """Monitor the enhanced trading session."""
    log_dir = "logs/enhanced_24hr_trading"
    
    # Find the latest log file
    if not os.path.exists(log_dir):
        print("❌ No enhanced trading session found")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('enhanced_btc_24hr_')]
    if not log_files:
        print("❌ No enhanced trading log files found")
        return
    
    latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(log_dir, f)))
    log_path = os.path.join(log_dir, latest_log)
    
    print("🚀 Enhanced Random Forest Trading Session Monitor")
    print("=" * 60)
    print(f"📝 Monitoring: {latest_log}")
    print("=" * 60)
    
    # Read and display the log
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            print(content)
        
        # Show file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_path))
        print(f"\n📅 Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if session is still running
        lines = content.strip().split('\n')
        if lines:
            last_line = lines[-1]
            if 'Next update in' in last_line:
                print("✅ Session is actively running")
            elif 'SESSION COMPLETE' in last_line:
                print("🎉 Session completed successfully")
            else:
                print("⚠️ Session status unclear")
        
    except Exception as e:
        print(f"❌ Error reading log: {e}")

if __name__ == "__main__":
    monitor_enhanced_session()