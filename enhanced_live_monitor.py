#!/usr/bin/env python3
"""
Enhanced Real-Time BTC Trading Monitor

Provides comprehensive live monitoring of the 24-hour trading session with:
- Real-time trade parsing and display
- Live portfolio tracking with P&L
- Performance metrics dashboard
- Auto-refreshing rich terminal UI
- Strategy analysis and trade breakdown

Usage: python3 enhanced_live_monitor.py
"""

import os
import re
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import subprocess

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.align import Align
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Install with: pip install rich")

class TradingDataParser:
    """Advanced parser for trading log data with regex patterns."""
    
    def __init__(self):
        # Compiled regex patterns for efficient parsing
        self.patterns = {
            'trade': re.compile(r'\[([^\]]+)\] (🟢|🔴) (BUY|SELL|STOP_LOSS|TAKE_PROFIT) ([0-9.]+) BTC @ \$([0-9,]+\.[0-9]+) \(\$([0-9,]+\.[0-9]+)\) - (.+)'),
            'portfolio_status': re.compile(r'\[([^\]]+)\] 📊 PORTFOLIO STATUS - Hour ([0-9.]+)/24'),
            'total_value': re.compile(r'\[([^\]]+)\] 💰 Total Value: \$([0-9,]+\.[0-9]+)'),
            'return_pct': re.compile(r'\[([^\]]+)\] 📈 Return: ([+-][0-9.]+)%'),
            'cash': re.compile(r'\[([^\]]+)\] 💵 Cash: \$([0-9,]+\.[0-9]+)'),
            'btc_position': re.compile(r'\[([^\]]+)\] ₿ BTC Position: ([0-9.]+) BTC \(\$([0-9,]+\.[0-9]+)\)'),
            'btc_price': re.compile(r'\[([^\]]+)\] 💲 BTC Price: \$([0-9,]+\.[0-9]+)'),
            'position_pnl': re.compile(r'\[([^\]]+)\] 📊 Position P&L: ([+-][0-9.]+)%'),
            'total_trades': re.compile(r'\[([^\]]+)\] 📝 Total Trades: ([0-9]+)'),
            'session_init': re.compile(r'\[([^\]]+)\] 🚀 24-Hour BTC Paper Trading Session Initialized'),
            'session_end': re.compile(r'\[([^\]]+)\] 🕐 Session will run until: ([0-9-]+ [0-9:]+)')
        }
    
    def parse_trade(self, line: str) -> Optional[Dict]:
        """Parse a trade line and extract structured data."""
        match = self.patterns['trade'].search(line)
        if not match:
            return None
        
        timestamp_str, emoji, action, quantity, price, value, reason = match.groups()
        
        return {
            'timestamp': datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'),
            'action': action,
            'quantity': float(quantity),
            'price': float(price.replace(',', '')),
            'value': float(value.replace(',', '')),
            'reason': reason.strip(),
            'emoji': emoji
        }
    
    def parse_portfolio_status(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a complete portfolio status block."""
        status = {}
        
        for i in range(start_idx, min(start_idx + 10, len(lines))):
            line = lines[i]
            
            # Portfolio status header
            match = self.patterns['portfolio_status'].search(line)
            if match:
                timestamp_str, hour = match.groups()
                status['timestamp'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                status['hour'] = float(hour)
                continue
            
            # Total value
            match = self.patterns['total_value'].search(line)
            if match:
                status['total_value'] = float(match.group(2).replace(',', ''))
                continue
            
            # Return percentage
            match = self.patterns['return_pct'].search(line)
            if match:
                status['return_pct'] = float(match.group(2))
                continue
            
            # Cash
            match = self.patterns['cash'].search(line)
            if match:
                status['cash'] = float(match.group(2).replace(',', ''))
                continue
            
            # BTC position
            match = self.patterns['btc_position'].search(line)
            if match:
                status['btc_quantity'] = float(match.group(2))
                status['btc_value'] = float(match.group(3).replace(',', ''))
                continue
            
            # BTC price
            match = self.patterns['btc_price'].search(line)
            if match:
                status['btc_price'] = float(match.group(2).replace(',', ''))
                continue
            
            # Position P&L
            match = self.patterns['position_pnl'].search(line)
            if match:
                status['position_pnl'] = float(match.group(2))
                continue
            
            # Total trades
            match = self.patterns['total_trades'].search(line)
            if match:
                status['total_trades'] = int(match.group(2))
                continue
        
        return status if len(status) > 2 else None

class TradingSessionState:
    """Maintains the current state of the trading session."""
    
    def __init__(self):
        self.session_start = None
        self.session_end = None
        self.trades = []
        self.portfolio_snapshots = []
        self.latest_portfolio = {}
        self.latest_btc_price = 0
        self.initial_capital = 100000
        self.parser = TradingDataParser()
    
    def update_from_log(self, log_path: str):
        """Update state by parsing the complete log file."""
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Clear existing data for fresh parse
            trades = []
            portfolio_snapshots = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Parse session initialization
                if '24-Hour BTC Paper Trading Session Initialized' in line:
                    match = re.search(r'\[([^\]]+)\]', line)
                    if match:
                        self.session_start = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                
                # Parse session end time
                elif 'Session will run until:' in line:
                    match = re.search(r'until: ([0-9-]+ [0-9:]+)', line)
                    if match:
                        self.session_end = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                
                # Parse trades
                trade = self.parser.parse_trade(line)
                if trade:
                    trades.append(trade)
                
                # Parse portfolio status blocks
                if 'PORTFOLIO STATUS' in line:
                    portfolio = self.parser.parse_portfolio_status(lines, i)
                    if portfolio:
                        portfolio_snapshots.append(portfolio)
                        self.latest_portfolio = portfolio
                
                # Parse latest BTC price from any line
                btc_price_match = self.parser.patterns['btc_price'].search(line)
                if btc_price_match:
                    self.latest_btc_price = float(btc_price_match.group(2).replace(',', ''))
                
                i += 1
            
            self.trades = trades
            self.portfolio_snapshots = portfolio_snapshots
            
        except Exception as e:
            print(f"Error parsing log: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Calculate current performance metrics."""
        if not self.portfolio_snapshots:
            return {}
        
        latest = self.latest_portfolio
        
        # Calculate time-based metrics
        elapsed_hours = 0
        remaining_hours = 24
        
        if self.session_start:
            elapsed = datetime.now() - self.session_start
            elapsed_hours = elapsed.total_seconds() / 3600
            
            if self.session_end:
                remaining = self.session_end - datetime.now()
                remaining_hours = max(0, remaining.total_seconds() / 3600)
        
        # Trade analysis
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']]
        
        # Calculate win rate (simplified)
        total_value = latest.get('total_value', self.initial_capital)
        
        return {
            'total_value': total_value,
            'total_return': latest.get('return_pct', 0),
            'cash': latest.get('cash', 0),
            'btc_quantity': latest.get('btc_quantity', 0),
            'btc_value': latest.get('btc_value', 0),
            'position_pnl': latest.get('position_pnl', 0),
            'btc_price': self.latest_btc_price,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'elapsed_hours': elapsed_hours,
            'remaining_hours': remaining_hours,
            'profit_loss': total_value - self.initial_capital,
            'hourly_rate': (total_value - self.initial_capital) / max(elapsed_hours, 0.1)
        }

class EnhancedTradingMonitor:
    """Rich terminal-based trading monitor with live updates."""
    
    def __init__(self, log_path: str):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for enhanced monitor")
        
        self.log_path = log_path
        self.console = Console()
        self.session_state = TradingSessionState()
        self.last_update = 0
        self.update_interval = 5  # seconds
        
    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        layout["left"].split_column(
            Layout(name="trades", ratio=2),
            Layout(name="portfolio", ratio=1)
        )
        
        layout["right"].split_column(
            Layout(name="metrics", ratio=1),
            Layout(name="status", ratio=1)
        )
        
        return layout
    
    def create_header_panel(self) -> Panel:
        """Create header with session info."""
        metrics = self.session_state.get_current_metrics()
        
        title = "[bold blue]🚀 LIVE BTC TRADING MONITOR[/bold blue]"
        
        if self.session_state.session_start:
            elapsed = f"{metrics['elapsed_hours']:.1f}h"
            remaining = f"{metrics['remaining_hours']:.1f}h"
            subtitle = f"Started: {self.session_state.session_start.strftime('%H:%M:%S')} | Elapsed: {elapsed} | Remaining: {remaining}"
        else:
            subtitle = "Loading session data..."
        
        header_text = f"{title}\n{subtitle}"
        
        return Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE
        )
    
    def create_trades_panel(self) -> Panel:
        """Create trades table panel."""
        table = Table(title="Recent Trades", show_header=True, header_style="bold magenta")
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Action", style="white", width=8)
        table.add_column("Quantity", style="yellow", width=10)
        table.add_column("Price", style="green", width=12)
        table.add_column("Value", style="blue", width=12)
        table.add_column("Reason", style="white", width=15)
        
        # Show last 10 trades
        recent_trades = self.session_state.trades[-10:] if self.session_state.trades else []
        
        for trade in reversed(recent_trades):  # Most recent first
            action_color = "green" if trade['action'] == 'BUY' else "red"
            
            table.add_row(
                trade['timestamp'].strftime('%H:%M:%S'),
                f"[{action_color}]{trade['emoji']} {trade['action']}[/{action_color}]",
                f"{trade['quantity']:.6f}",
                f"${trade['price']:,.2f}",
                f"${trade['value']:,.2f}",
                trade['reason'][:12] + "..." if len(trade['reason']) > 12 else trade['reason']
            )
        
        return Panel(table, title="[bold]Trading Activity[/bold]", border_style="blue")
    
    def create_portfolio_panel(self) -> Panel:
        """Create portfolio status panel."""
        metrics = self.session_state.get_current_metrics()
        
        if not metrics:
            return Panel("No portfolio data available", title="Portfolio Status")
        
        # Portfolio composition chart (simple text-based)
        total_value = metrics['total_value']
        cash_pct = (metrics['cash'] / total_value) * 100 if total_value > 0 else 0
        btc_pct = 100 - cash_pct
        
        # Create portfolio info table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="white", width=15)
        table.add_column("Details", style="dim", width=20)
        
        # Portfolio value with color coding
        value_color = "green" if metrics['profit_loss'] >= 0 else "red"
        return_sign = "+" if metrics['total_return'] >= 0 else ""
        
        table.add_row(
            "Portfolio Value",
            f"[{value_color}]${total_value:,.2f}[/{value_color}]",
            f"{return_sign}{metrics['total_return']:.2f}%"
        )
        
        table.add_row(
            "Profit/Loss",
            f"[{value_color}]{'+' if metrics['profit_loss'] >= 0 else ''}${metrics['profit_loss']:,.2f}[/{value_color}]",
            f"${metrics['hourly_rate']:,.0f}/hour"
        )
        
        table.add_row(
            "Cash",
            f"${metrics['cash']:,.2f}",
            f"{cash_pct:.1f}% of portfolio"
        )
        
        table.add_row(
            "BTC Position",
            f"{metrics['btc_quantity']:.6f} BTC",
            f"{btc_pct:.1f}% of portfolio"
        )
        
        table.add_row(
            "BTC Value",
            f"${metrics['btc_value']:,.2f}",
            f"@ ${metrics['btc_price']:,.2f}"
        )
        
        if metrics['position_pnl'] != 0:
            pnl_color = "green" if metrics['position_pnl'] >= 0 else "red"
            table.add_row(
                "Position P&L",
                f"[{pnl_color}]{'+' if metrics['position_pnl'] >= 0 else ''}{metrics['position_pnl']:.2f}%[/{pnl_color}]",
                "Current position"
            )
        
        return Panel(table, title="[bold]Portfolio Status[/bold]", border_style="green")
    
    def create_metrics_panel(self) -> Panel:
        """Create performance metrics panel."""
        metrics = self.session_state.get_current_metrics()
        
        if not metrics:
            return Panel("No metrics available", title="Performance Metrics")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="yellow", width=12)
        table.add_column("Value", style="white", width=12)
        
        table.add_row("Total Trades", str(metrics['total_trades']))
        table.add_row("Buy Orders", str(metrics['buy_trades']))
        table.add_row("Sell Orders", str(metrics['sell_trades']))
        
        # Win rate calculation (simplified)
        if metrics['total_trades'] > 0:
            # This is a simplified win rate based on overall profit
            win_rate = 100 if metrics['profit_loss'] > 0 else 0
            table.add_row("Win Rate", f"{win_rate:.0f}%")
        
        # Trading frequency
        if metrics['elapsed_hours'] > 0:
            trades_per_hour = metrics['total_trades'] / metrics['elapsed_hours']
            table.add_row("Trade Freq", f"{trades_per_hour:.1f}/hour")
        
        return Panel(table, title="[bold]Metrics[/bold]", border_style="yellow")
    
    def create_status_panel(self) -> Panel:
        """Create system status panel."""
        # Check if trading process is running
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            process_running = 'btc_24hr_paper_trader.py' in result.stdout
        except:
            process_running = False
        
        # File update status
        try:
            file_age = time.time() - os.path.getmtime(self.log_path)
        except:
            file_age = float('inf')
        
        table = Table(show_header=False, box=None)
        table.add_column("Status", style="cyan", width=12)
        table.add_column("Value", style="white", width=12)
        
        # Process status
        status_color = "green" if process_running else "red"
        status_text = "RUNNING" if process_running else "STOPPED"
        table.add_row("Process", f"[{status_color}]{status_text}[/{status_color}]")
        
        # File freshness
        if file_age < 300:  # 5 minutes
            freshness_color = "green"
            freshness_text = "FRESH"
        elif file_age < 600:  # 10 minutes
            freshness_color = "yellow"
            freshness_text = "STALE"
        else:
            freshness_color = "red"
            freshness_text = "OLD"
        
        table.add_row("Data", f"[{freshness_color}]{freshness_text}[/{freshness_color}]")
        
        # Last update
        table.add_row("Updated", f"{file_age:.0f}s ago")
        
        # Current time
        table.add_row("Time", datetime.now().strftime('%H:%M:%S'))
        
        return Panel(table, title="[bold]System Status[/bold]", border_style="cyan")
    
    def create_footer_panel(self) -> Panel:
        """Create footer with controls and info."""
        metrics = self.session_state.get_current_metrics()
        
        if metrics:
            btc_price = metrics['btc_price']
            profit = metrics['profit_loss']
            return_pct = metrics['total_return']
            
            profit_color = "green" if profit >= 0 else "red"
            profit_sign = "+" if profit >= 0 else ""
            
            footer_text = (
                f"[bold]BTC Price: ${btc_price:,.2f}[/bold] | "
                f"[{profit_color}]P&L: {profit_sign}${profit:.2f} ({return_pct:+.2f}%)[/{profit_color}] | "
                f"Press Ctrl+C to exit | Auto-refresh: {self.update_interval}s"
            )
        else:
            footer_text = "Loading data... | Press Ctrl+C to exit"
        
        return Panel(
            Align.center(footer_text),
            style="dim",
            box=box.ROUNDED
        )
    
    def update_display(self, layout: Layout):
        """Update all dashboard panels."""
        # Update session state from log
        self.session_state.update_from_log(self.log_path)
        
        # Update layout panels
        layout["header"].update(self.create_header_panel())
        layout["trades"].update(self.create_trades_panel())
        layout["portfolio"].update(self.create_portfolio_panel())
        layout["metrics"].update(self.create_metrics_panel())
        layout["status"].update(self.create_status_panel())
        layout["footer"].update(self.create_footer_panel())
    
    async def run_monitor(self):
        """Run the live monitoring loop."""
        layout = self.create_layout()
        
        with Live(layout, refresh_per_second=1, screen=True) as live:
            try:
                while True:
                    self.update_display(layout)
                    live.update(layout)
                    await asyncio.sleep(self.update_interval)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Monitor stopped by user[/yellow]")
                
                # Show final summary
                metrics = self.session_state.get_current_metrics()
                if metrics:
                    self.console.print(f"\n[bold]Final Summary:[/bold]")
                    self.console.print(f"  Total Trades: {metrics['total_trades']}")
                    self.console.print(f"  Portfolio Value: ${metrics['total_value']:,.2f}")
                    self.console.print(f"  Profit/Loss: ${metrics['profit_loss']:+,.2f}")
                    self.console.print(f"  Return: {metrics['total_return']:+.2f}%")

def find_latest_log():
    """Find the latest trading log file."""
    log_dir = "logs/24hr_trading"
    if not os.path.exists(log_dir):
        return None
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('btc_24hr_')]
    if not log_files:
        return None
    
    latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(log_dir, f)))
    return os.path.join(log_dir, latest_log)

async def main():
    """Main function to run the enhanced monitor."""
    if not RICH_AVAILABLE:
        print("❌ Enhanced monitor requires Rich library")
        print("Install with: pip install rich")
        return
    
    log_path = find_latest_log()
    if not log_path:
        print("❌ No trading log file found")
        print("Make sure the 24-hour trading session is running")
        return
    
    print(f"🔍 Monitoring trading session: {log_path}")
    print("🚀 Starting enhanced live monitor...")
    time.sleep(2)
    
    monitor = EnhancedTradingMonitor(log_path)
    await monitor.run_monitor()

if __name__ == "__main__":
    asyncio.run(main())