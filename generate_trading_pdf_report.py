#!/usr/bin/env python3
"""
Professional PDF Trading Report Generator

Generates a comprehensive PDF report of the 24-hour BTC trading session including:
- Executive summary with profits
- Complete trade log
- Performance charts
- Strategic analysis

Usage: python3 generate_trading_pdf_report.py
"""

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not installed. Install with: pip install reportlab")

class TradingReportGenerator:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.trades = []
        self.portfolio_snapshots = []
        self.session_info = {}
        self.initial_capital = 100000
        
    def parse_log_file(self):
        """Parse the trading log file to extract all data."""
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
        
        # Reset data
        self.trades = []
        self.portfolio_snapshots = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Parse session start
            if '24-Hour BTC Paper Trading Session Initialized' in line:
                match = re.search(r'\[([^\]]+)\]', line)
                if match:
                    self.session_info['start_time'] = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
            
            # Parse trades
            trade_match = re.search(
                r'\[([^\]]+)\] (🟢|🔴) (BUY|SELL|STOP_LOSS|TAKE_PROFIT) ([0-9.]+) BTC @ \$([0-9,]+\.[0-9]+) \(\$([0-9,]+\.[0-9]+)\) - (.+)',
                line
            )
            if trade_match:
                timestamp_str, emoji, action, quantity, price, value, reason = trade_match.groups()
                self.trades.append({
                    'timestamp': datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'),
                    'action': action,
                    'quantity': float(quantity),
                    'price': float(price.replace(',', '')),
                    'value': float(value.replace(',', '')),
                    'reason': reason.strip(),
                    'emoji': emoji
                })
            
            # Parse portfolio status
            if 'PORTFOLIO STATUS - Hour' in line:
                portfolio_data = self._parse_portfolio_block(lines, i)
                if portfolio_data:
                    self.portfolio_snapshots.append(portfolio_data)
    
    def _parse_portfolio_block(self, lines: List[str], start_idx: int) -> Dict:
        """Parse a portfolio status block."""
        data = {}
        
        for j in range(start_idx, min(start_idx + 10, len(lines))):
            line = lines[j]
            
            # Hour
            hour_match = re.search(r'Hour ([0-9.]+)/24', line)
            if hour_match:
                data['hour'] = float(hour_match.group(1))
            
            # Total value
            value_match = re.search(r'Total Value: \$([0-9,]+\.[0-9]+)', line)
            if value_match:
                data['total_value'] = float(value_match.group(1).replace(',', ''))
            
            # Return
            return_match = re.search(r'Return: ([+-][0-9.]+)%', line)
            if return_match:
                data['return_pct'] = float(return_match.group(1))
            
            # Cash
            cash_match = re.search(r'Cash: \$([0-9,]+\.[0-9]+)', line)
            if cash_match:
                data['cash'] = float(cash_match.group(1).replace(',', ''))
            
            # BTC position
            btc_match = re.search(r'BTC Position: ([0-9.]+) BTC \(\$([0-9,]+\.[0-9]+)\)', line)
            if btc_match:
                data['btc_quantity'] = float(btc_match.group(1))
                data['btc_value'] = float(btc_match.group(2).replace(',', ''))
            
            # BTC price
            price_match = re.search(r'BTC Price: \$([0-9,]+\.[0-9]+)', line)
            if price_match:
                data['btc_price'] = float(price_match.group(1).replace(',', ''))
            
            # Timestamp
            timestamp_match = re.search(r'\[([^\]]+)\]', line)
            if timestamp_match and 'timestamp' not in data:
                data['timestamp'] = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
        
        return data if 'total_value' in data else None
    
    def generate_charts(self):
        """Generate all charts for the report."""
        charts = {}
        
        # 1. Portfolio Value Over Time
        plt.figure(figsize=(10, 6))
        
        # Extract data
        if self.portfolio_snapshots:
            hours = [s['hour'] for s in self.portfolio_snapshots]
            values = [s['total_value'] for s in self.portfolio_snapshots]
            returns = [s['return_pct'] for s in self.portfolio_snapshots]
            
            # Portfolio value line
            plt.subplot(2, 1, 1)
            plt.plot(hours, values, 'b-', linewidth=2, marker='o', markersize=5)
            plt.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            plt.fill_between(hours, self.initial_capital, values, 
                           where=np.array(values) >= self.initial_capital, 
                           color='green', alpha=0.3, label='Profit')
            plt.fill_between(hours, self.initial_capital, values, 
                           where=np.array(values) < self.initial_capital, 
                           color='red', alpha=0.3, label='Loss')
            
            plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Hours Elapsed')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format y-axis
            ax = plt.gca()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Return percentage
            plt.subplot(2, 1, 2)
            plt.plot(hours, returns, 'g-', linewidth=2, marker='o', markersize=5)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.fill_between(hours, 0, returns, 
                           where=np.array(returns) >= 0, 
                           color='green', alpha=0.3)
            plt.fill_between(hours, 0, returns, 
                           where=np.array(returns) < 0, 
                           color='red', alpha=0.3)
            
            plt.title('Return Percentage', fontsize=12, fontweight='bold')
            plt.xlabel('Hours Elapsed')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
            charts['portfolio_performance'] = 'portfolio_performance.png'
            plt.close()
        
        # 2. Trade Distribution Chart
        plt.figure(figsize=(10, 6))
        
        if self.trades:
            # Trade timeline
            trade_times = [(t['timestamp'] - self.session_info['start_time']).total_seconds() / 3600 for t in self.trades]
            trade_prices = [t['price'] for t in self.trades]
            trade_types = [t['action'] for t in self.trades]
            
            # Create scatter plot
            for i, (time, price, action) in enumerate(zip(trade_times, trade_prices, trade_types)):
                if action == 'BUY':
                    plt.scatter(time, price, color='green', s=100, marker='^', alpha=0.7)
                else:
                    plt.scatter(time, price, color='red', s=100, marker='v', alpha=0.7)
            
            # Add price line
            if self.portfolio_snapshots:
                snap_hours = [s['hour'] for s in self.portfolio_snapshots if 'btc_price' in s]
                snap_prices = [s['btc_price'] for s in self.portfolio_snapshots if 'btc_price' in s]
                if snap_hours and snap_prices:
                    plt.plot(snap_hours, snap_prices, 'b-', alpha=0.5, linewidth=1, label='BTC Price')
            
            plt.title('Trade Execution Timeline', fontsize=14, fontweight='bold')
            plt.xlabel('Hours Elapsed')
            plt.ylabel('BTC Price ($)')
            plt.grid(True, alpha=0.3)
            
            # Custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Buy'),
                Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=10, label='Sell'),
                Line2D([0], [0], color='b', linewidth=2, label='BTC Price')
            ]
            plt.legend(handles=legend_elements)
            
            # Format y-axis
            ax = plt.gca()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            plt.savefig('trade_timeline.png', dpi=300, bbox_inches='tight')
            charts['trade_timeline'] = 'trade_timeline.png'
            plt.close()
        
        # 3. Asset Allocation Pie Chart
        if self.portfolio_snapshots and len(self.portfolio_snapshots) > 0:
            latest = self.portfolio_snapshots[-1]
            
            plt.figure(figsize=(8, 8))
            
            cash = latest.get('cash', 0)
            btc_value = latest.get('btc_value', 0)
            
            if cash + btc_value > 0:
                sizes = [cash, btc_value]
                labels = [f'Cash\n${cash:,.2f}', f'BTC\n${btc_value:,.2f}']
                colors = ['#4CAF50', '#FF9800']
                explode = (0.05, 0.05)
                
                plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
                plt.title('Current Asset Allocation', fontsize=14, fontweight='bold')
                plt.axis('equal')
                
                plt.tight_layout()
                plt.savefig('asset_allocation.png', dpi=300, bbox_inches='tight')
                charts['asset_allocation'] = 'asset_allocation.png'
                plt.close()
        
        return charts
    
    def generate_pdf(self, output_filename='trading_report.pdf'):
        """Generate the PDF report."""
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab is required to generate PDF. Install with: pip install reportlab")
            return None
        
        # Parse data
        self.parse_log_file()
        
        # Generate charts
        charts = self.generate_charts()
        
        # Create PDF
        doc = SimpleDocTemplate(output_filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E7D32'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph("24-Hour BTC Trading Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        # Calculate summary metrics
        current_value = self.portfolio_snapshots[-1]['total_value'] if self.portfolio_snapshots else self.initial_capital
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        profit_loss = current_value - self.initial_capital
        
        summary_data = [
            ['Metric', 'Value'],
            ['Session Start', self.session_info.get('start_time', 'N/A').strftime('%Y-%m-%d %H:%M:%S') if 'start_time' in self.session_info else 'N/A'],
            ['Initial Capital', f'${self.initial_capital:,.2f}'],
            ['Current Portfolio Value', f'${current_value:,.2f}'],
            ['Total Profit/Loss', f'${profit_loss:+,.2f}'],
            ['Return Percentage', f'{total_return:+.2f}%'],
            ['Total Trades', str(len(self.trades))],
            ['Buy Orders', str(len([t for t in self.trades if t['action'] == 'BUY']))],
            ['Sell Orders', str(len([t for t in self.trades if t['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']]))]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Portfolio Performance Chart
        if 'portfolio_performance' in charts:
            story.append(Paragraph("Portfolio Performance", heading_style))
            img = Image(charts['portfolio_performance'], width=6*inch, height=3.6*inch)
            story.append(img)
            story.append(PageBreak())
        
        # Trade Log
        story.append(Paragraph("Complete Trade Log", heading_style))
        
        trade_data = [['Time', 'Action', 'Quantity', 'Price', 'Value', 'Reason']]
        
        for trade in self.trades:
            trade_data.append([
                trade['timestamp'].strftime('%H:%M:%S'),
                trade['action'],
                f"{trade['quantity']:.6f}",
                f"${trade['price']:,.2f}",
                f"${trade['value']:,.2f}",
                trade['reason'][:20] + '...' if len(trade['reason']) > 20 else trade['reason']
            ])
        
        # Split into chunks if too many trades
        chunk_size = 25
        for i in range(0, len(trade_data)-1, chunk_size):
            chunk = [trade_data[0]] + trade_data[i+1:i+chunk_size+1]
            
            trade_table = Table(chunk, colWidths=[0.8*inch, 0.8*inch, 1*inch, 1.2*inch, 1.2*inch, 1.5*inch])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            # Color code buy/sell
            for j in range(1, len(chunk)):
                if chunk[j][1] == 'BUY':
                    trade_table.setStyle(TableStyle([('TEXTCOLOR', (1, j), (1, j), colors.green)]))
                else:
                    trade_table.setStyle(TableStyle([('TEXTCOLOR', (1, j), (1, j), colors.red)]))
            
            story.append(trade_table)
            story.append(Spacer(1, 0.2*inch))
            
            if i + chunk_size < len(trade_data) - 1:
                story.append(PageBreak())
        
        # Trade Timeline Chart
        if 'trade_timeline' in charts:
            story.append(PageBreak())
            story.append(Paragraph("Trade Execution Timeline", heading_style))
            img = Image(charts['trade_timeline'], width=6*inch, height=3.6*inch)
            story.append(img)
        
        # Asset Allocation
        if 'asset_allocation' in charts:
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Current Asset Allocation", heading_style))
            img = Image(charts['asset_allocation'], width=4*inch, height=4*inch)
            story.append(img)
        
        # Portfolio Snapshots
        story.append(PageBreak())
        story.append(Paragraph("Hourly Portfolio Snapshots", heading_style))
        
        snapshot_data = [['Hour', 'Portfolio Value', 'Return %', 'Cash', 'BTC Value', 'BTC Qty']]
        
        for snap in self.portfolio_snapshots:
            snapshot_data.append([
                f"{snap.get('hour', 0):.1f}",
                f"${snap.get('total_value', 0):,.2f}",
                f"{snap.get('return_pct', 0):+.2f}%",
                f"${snap.get('cash', 0):,.2f}",
                f"${snap.get('btc_value', 0):,.2f}",
                f"{snap.get('btc_quantity', 0):.6f}"
            ])
        
        snapshot_table = Table(snapshot_data, colWidths=[0.8*inch, 1.5*inch, 1*inch, 1.2*inch, 1.2*inch, 1*inch])
        snapshot_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(snapshot_table)
        
        # Trading Strategy Summary
        story.append(PageBreak())
        story.append(Paragraph("Trading Strategy Analysis", heading_style))
        
        # Identify trading cycles
        cycles = []
        current_cycle = {'buys': [], 'sells': []}
        
        for trade in self.trades:
            if trade['action'] == 'BUY':
                current_cycle['buys'].append(trade)
            else:
                current_cycle['sells'].append(trade)
                if current_cycle['buys']:
                    cycles.append(current_cycle)
                    current_cycle = {'buys': [], 'sells': []}
        
        if current_cycle['buys']:
            cycles.append(current_cycle)
        
        cycle_text = f"<b>Trading Cycles Identified:</b> {len(cycles)}<br/><br/>"
        
        for i, cycle in enumerate(cycles, 1):
            total_buy_value = sum(t['value'] for t in cycle['buys'])
            total_sell_value = sum(t['value'] for t in cycle['sells'])
            cycle_profit = total_sell_value - total_buy_value if cycle['sells'] else 0
            
            cycle_text += f"<b>Cycle {i}:</b><br/>"
            cycle_text += f"• Buy Orders: {len(cycle['buys'])}<br/>"
            cycle_text += f"• Total Buy Value: ${total_buy_value:,.2f}<br/>"
            
            if cycle['sells']:
                cycle_text += f"• Sell Orders: {len(cycle['sells'])}<br/>"
                cycle_text += f"• Total Sell Value: ${total_sell_value:,.2f}<br/>"
                cycle_text += f"• Cycle Profit: ${cycle_profit:+,.2f}<br/>"
            else:
                cycle_text += f"• Status: Currently holding position<br/>"
            
            cycle_text += "<br/>"
        
        story.append(Paragraph(cycle_text, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Clean up chart files
        for chart_file in charts.values():
            if os.path.exists(chart_file):
                os.remove(chart_file)
        
        return output_filename

def main():
    """Generate the trading report PDF."""
    # Find the latest log file
    log_dir = "logs/24hr_trading"
    if not os.path.exists(log_dir):
        print("❌ No trading logs found")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('btc_24hr_')]
    if not log_files:
        print("❌ No trading log files found")
        return
    
    latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(log_dir, f)))
    log_path = os.path.join(log_dir, latest_log)
    
    print(f"📊 Generating PDF report from: {log_path}")
    
    # Generate report
    generator = TradingReportGenerator(log_path)
    output_file = f"BTC_Trading_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    result = generator.generate_pdf(output_file)
    
    if result:
        print(f"✅ PDF report generated successfully: {result}")
        print(f"📄 File size: {os.path.getsize(result) / 1024:.1f} KB")
    else:
        print("❌ Failed to generate PDF report")

if __name__ == "__main__":
    main()