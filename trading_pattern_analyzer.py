#!/usr/bin/env python3
"""
Trading Pattern Analyzer - Extract Successful Patterns from Live Trading Data

Analyzes the 78-trade BTC session to identify high-profit patterns for Random Forest enhancement.
Extracts momentum indicators, position sizing rules, and exit timing signals.

Usage: python3 trading_pattern_analyzer.py
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class TradingPatternAnalyzer:
    """Analyzes successful trading patterns from live session data."""
    
    def __init__(self, log_path: str = "logs/24hr_trading/btc_24hr_20250713_140540.log"):
        self.log_path = log_path
        self.trades = []
        self.portfolio_snapshots = []
        self.trading_cycles = []
        self.pattern_insights = {}
        
    def parse_trading_data(self) -> pd.DataFrame:
        """Parse the complete trading log for pattern analysis."""
        print("📊 Parsing successful trading session data...")
        
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
        
        # Parse trades with enhanced data
        trade_pattern = re.compile(
            r'\[([^\]]+)\] (🟢|🔴) (BUY|SELL|STOP_LOSS|TAKE_PROFIT) ([0-9.]+) BTC @ \$([0-9,]+\.[0-9]+) \(\$([0-9,]+\.[0-9]+)\)(?:\s*-\s*(.+))?'
        )
        
        trades_data = []
        for line in lines:
            match = trade_pattern.search(line)
            if match:
                timestamp_str, emoji, action, quantity, price, value, reason = match.groups()
                
                trade = {
                    'timestamp': datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'),
                    'action': action,
                    'quantity': float(quantity),
                    'price': float(price.replace(',', '')),
                    'value': float(value.replace(',', '')),
                    'reason': reason.strip() if reason else 'SIGNAL',
                    'emoji': emoji,
                    'trade_id': len(trades_data) + 1
                }
                trades_data.append(trade)
        
        self.trades_df = pd.DataFrame(trades_data)
        self.trades_df.set_index('timestamp', inplace=True)
        
        print(f"✅ Parsed {len(self.trades_df)} trades for pattern analysis")
        return self.trades_df
    
    def identify_trading_cycles(self) -> List[Dict]:
        """Identify complete buy-accumulate-sell cycles."""
        print("🔄 Identifying trading cycles...")
        
        cycles = []
        current_cycle = {'buys': [], 'sell': None, 'btc_accumulated': 0}
        
        for idx, trade in self.trades_df.iterrows():
            if trade['action'] == 'BUY':
                current_cycle['buys'].append({
                    'timestamp': idx,
                    'price': trade['price'],
                    'quantity': trade['quantity'],
                    'value': trade['value']
                })
                current_cycle['btc_accumulated'] += trade['quantity']
                
            elif trade['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']:
                if current_cycle['buys']:  # Complete cycle
                    current_cycle['sell'] = {
                        'timestamp': idx,
                        'price': trade['price'],
                        'quantity': trade['quantity'],
                        'value': trade['value']
                    }
                    
                    # Calculate cycle metrics
                    cycle_start = current_cycle['buys'][0]['timestamp']
                    cycle_end = idx
                    cycle_duration = (cycle_end - cycle_start).total_seconds() / 3600  # hours
                    
                    # Calculate weighted average buy price
                    total_value = sum(buy['value'] for buy in current_cycle['buys'])
                    total_quantity = sum(buy['quantity'] for buy in current_cycle['buys'])
                    avg_buy_price = total_value / total_quantity if total_quantity > 0 else 0
                    
                    # Calculate cycle profit
                    sell_value = current_cycle['sell']['value']
                    cycle_profit = sell_value - total_value
                    cycle_return = (cycle_profit / total_value) * 100 if total_value > 0 else 0
                    
                    cycle_data = {
                        'cycle_id': len(cycles) + 1,
                        'start_time': cycle_start,
                        'end_time': cycle_end,
                        'duration_hours': cycle_duration,
                        'num_buys': len(current_cycle['buys']),
                        'total_btc': total_quantity,
                        'avg_buy_price': avg_buy_price,
                        'sell_price': current_cycle['sell']['price'],
                        'total_invested': total_value,
                        'sell_value': sell_value,
                        'profit': cycle_profit,
                        'return_pct': cycle_return,
                        'buys': current_cycle['buys'],
                        'sell': current_cycle['sell']
                    }
                    
                    cycles.append(cycle_data)
                    
                    # Reset for next cycle
                    current_cycle = {'buys': [], 'sell': None, 'btc_accumulated': 0}
        
        self.trading_cycles = cycles
        print(f"✅ Identified {len(cycles)} complete trading cycles")
        return cycles
    
    def analyze_profitable_patterns(self) -> Dict:
        """Analyze patterns from profitable cycles."""
        print("🎯 Analyzing profitable patterns...")
        
        if not self.trading_cycles:
            self.identify_trading_cycles()
        
        cycles_df = pd.DataFrame(self.trading_cycles)
        
        # Separate profitable vs unprofitable cycles
        profitable_cycles = cycles_df[cycles_df['return_pct'] > 0]
        unprofitable_cycles = cycles_df[cycles_df['return_pct'] <= 0]
        
        print(f"📈 Profitable cycles: {len(profitable_cycles)}")
        print(f"📉 Unprofitable cycles: {len(unprofitable_cycles)}")
        
        # Analyze profitable cycle characteristics
        patterns = {
            'profitable_stats': {
                'count': len(profitable_cycles),
                'avg_return': profitable_cycles['return_pct'].mean(),
                'avg_duration': profitable_cycles['duration_hours'].mean(),
                'avg_num_buys': profitable_cycles['num_buys'].mean(),
                'avg_btc_size': profitable_cycles['total_btc'].mean(),
                'best_return': profitable_cycles['return_pct'].max(),
                'worst_return': profitable_cycles['return_pct'].min()
            },
            'unprofitable_stats': {
                'count': len(unprofitable_cycles),
                'avg_return': unprofitable_cycles['return_pct'].mean() if len(unprofitable_cycles) > 0 else 0,
                'avg_duration': unprofitable_cycles['duration_hours'].mean() if len(unprofitable_cycles) > 0 else 0
            },
            'optimal_ranges': self._find_optimal_ranges(profitable_cycles),
            'price_momentum_patterns': self._analyze_price_momentum(profitable_cycles),
            'position_sizing_patterns': self._analyze_position_sizing(profitable_cycles),
            'timing_patterns': self._analyze_timing_patterns(profitable_cycles)
        }
        
        self.pattern_insights = patterns
        return patterns
    
    def _find_optimal_ranges(self, profitable_cycles: pd.DataFrame) -> Dict:
        """Find optimal ranges for key parameters."""
        return {
            'optimal_duration': {
                'min': profitable_cycles['duration_hours'].quantile(0.25),
                'max': profitable_cycles['duration_hours'].quantile(0.75),
                'median': profitable_cycles['duration_hours'].median()
            },
            'optimal_position_size': {
                'min': profitable_cycles['total_btc'].quantile(0.25),
                'max': profitable_cycles['total_btc'].quantile(0.75),
                'median': profitable_cycles['total_btc'].median()
            },
            'optimal_num_buys': {
                'min': profitable_cycles['num_buys'].quantile(0.25),
                'max': profitable_cycles['num_buys'].quantile(0.75),
                'median': profitable_cycles['num_buys'].median()
            }
        }
    
    def _analyze_price_momentum(self, profitable_cycles: pd.DataFrame) -> Dict:
        """Analyze price momentum patterns in profitable cycles."""
        momentum_patterns = []
        
        for _, cycle in profitable_cycles.iterrows():
            # Calculate price momentum during cycle
            start_price = cycle['buys'][0]['price']
            end_price = cycle['sell_price']
            price_change = (end_price - start_price) / start_price * 100
            
            # Calculate momentum strength
            max_buy_price = max(buy['price'] for buy in cycle['buys'])
            min_buy_price = min(buy['price'] for buy in cycle['buys'])
            price_range = (max_buy_price - min_buy_price) / min_buy_price * 100
            
            momentum_patterns.append({
                'cycle_id': cycle['cycle_id'],
                'price_change': price_change,
                'price_range': price_range,
                'momentum_strength': price_change / cycle['duration_hours'] if cycle['duration_hours'] > 0 else 0,
                'return_pct': cycle['return_pct']
            })
        
        momentum_df = pd.DataFrame(momentum_patterns)
        
        return {
            'avg_price_change': momentum_df['price_change'].mean(),
            'avg_momentum_strength': momentum_df['momentum_strength'].mean(),
            'high_momentum_threshold': momentum_df['momentum_strength'].quantile(0.75),
            'optimal_price_change_range': {
                'min': momentum_df['price_change'].quantile(0.25),
                'max': momentum_df['price_change'].quantile(0.75)
            }
        }
    
    def _analyze_position_sizing(self, profitable_cycles: pd.DataFrame) -> Dict:
        """Analyze position sizing patterns."""
        sizing_patterns = []
        
        for _, cycle in profitable_cycles.iterrows():
            # Analyze buy order progression
            buy_sizes = [buy['value'] for buy in cycle['buys']]
            
            sizing_patterns.append({
                'cycle_id': cycle['cycle_id'],
                'total_size': sum(buy_sizes),
                'avg_buy_size': np.mean(buy_sizes),
                'size_consistency': np.std(buy_sizes) / np.mean(buy_sizes) if np.mean(buy_sizes) > 0 else 0,
                'size_trend': 'increasing' if len(buy_sizes) > 1 and buy_sizes[-1] > buy_sizes[0] else 'consistent',
                'return_pct': cycle['return_pct']
            })
        
        sizing_df = pd.DataFrame(sizing_patterns)
        
        return {
            'optimal_total_size': {
                'min': sizing_df['total_size'].quantile(0.25),
                'max': sizing_df['total_size'].quantile(0.75),
                'median': sizing_df['total_size'].median()
            },
            'optimal_avg_buy_size': sizing_df['avg_buy_size'].median(),
            'size_consistency_factor': sizing_df['size_consistency'].median()
        }
    
    def _analyze_timing_patterns(self, profitable_cycles: pd.DataFrame) -> Dict:
        """Analyze timing patterns in profitable trades."""
        timing_patterns = []
        
        for _, cycle in profitable_cycles.iterrows():
            # Hour of day analysis
            start_hour = cycle['start_time'].hour
            end_hour = cycle['end_time'].hour
            
            # Day of week analysis
            start_dow = cycle['start_time'].weekday()
            
            timing_patterns.append({
                'cycle_id': cycle['cycle_id'],
                'start_hour': start_hour,
                'end_hour': end_hour,
                'start_dow': start_dow,
                'duration_hours': cycle['duration_hours'],
                'return_pct': cycle['return_pct']
            })
        
        timing_df = pd.DataFrame(timing_patterns)
        
        # Find best performing hours
        hour_performance = timing_df.groupby('start_hour')['return_pct'].agg(['mean', 'count'])
        best_hours = hour_performance[hour_performance['count'] >= 2].sort_values('mean', ascending=False)
        
        return {
            'best_start_hours': best_hours.head(5).index.tolist(),
            'optimal_duration': timing_df['duration_hours'].median(),
            'hour_performance': hour_performance.to_dict(),
            'weekend_performance': timing_df[timing_df['start_dow'].isin([5, 6])]['return_pct'].mean()
        }
    
    def extract_ml_features(self) -> pd.DataFrame:
        """Extract features for Random Forest training based on pattern analysis."""
        print("🤖 Extracting ML features from pattern analysis...")
        
        if not self.pattern_insights:
            self.analyze_profitable_patterns()
        
        ml_features = []
        
        for trade_idx, trade in self.trades_df.iterrows():
            # Time-based features
            hour = trade_idx.hour
            dow = trade_idx.weekday()
            is_weekend = dow >= 5
            
            # Price momentum features (using rolling windows)
            price_window = self.trades_df.loc[:trade_idx, 'price'].tail(10)
            if len(price_window) >= 2:
                price_momentum_1h = (price_window.iloc[-1] - price_window.iloc[-2]) / price_window.iloc[-2] * 100
                price_momentum_5t = (price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0] * 100 if len(price_window) >= 5 else 0
                price_volatility = price_window.std() / price_window.mean() * 100
            else:
                price_momentum_1h = price_momentum_5t = price_volatility = 0
            
            # Position tracking features
            recent_trades = self.trades_df.loc[:trade_idx].tail(20)
            current_position = recent_trades[recent_trades['action'] == 'BUY']['quantity'].sum() - \
                             recent_trades[recent_trades['action'].isin(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])]['quantity'].sum()
            
            # Pattern-based features
            is_optimal_hour = hour in self.pattern_insights['timing_patterns']['best_start_hours']
            is_momentum_favorable = price_momentum_1h > self.pattern_insights['price_momentum_patterns']['avg_momentum_strength']
            is_position_optimal = current_position <= self.pattern_insights['optimal_ranges']['optimal_position_size']['max']
            
            # Target variables (future price movement)
            future_trades = self.trades_df.loc[trade_idx:].iloc[1:6]  # Next 5 trades
            if len(future_trades) > 0:
                future_price_max = future_trades['price'].max()
                future_price_min = future_trades['price'].min()
                price_up_1h = future_price_max > trade['price']
                price_movement = (future_price_max - trade['price']) / trade['price'] * 100
            else:
                price_up_1h = False
                price_movement = 0
            
            feature_row = {
                'timestamp': trade_idx,
                'price': trade['price'],
                'action': trade['action'],
                'quantity': trade['quantity'],
                'hour': hour,
                'dow': dow,
                'is_weekend': is_weekend,
                'price_momentum_1h': price_momentum_1h,
                'price_momentum_5t': price_momentum_5t,
                'price_volatility': price_volatility,
                'current_position': current_position,
                'is_optimal_hour': is_optimal_hour,
                'is_momentum_favorable': is_momentum_favorable,
                'is_position_optimal': is_position_optimal,
                'price_up_1h': price_up_1h,
                'price_movement': price_movement
            }
            
            ml_features.append(feature_row)
        
        features_df = pd.DataFrame(ml_features)
        print(f"✅ Extracted {len(features_df)} feature rows with {len(features_df.columns)} features")
        
        return features_df
    
    def generate_pattern_report(self) -> str:
        """Generate comprehensive pattern analysis report."""
        print("📋 Generating pattern analysis report...")
        
        if not self.pattern_insights:
            self.analyze_profitable_patterns()
        
        report = f"""
# 🎯 Trading Pattern Analysis Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Performance
- **Total Trades:** {len(self.trades_df)}
- **Complete Cycles:** {len(self.trading_cycles)}
- **Profitable Cycles:** {self.pattern_insights['profitable_stats']['count']}
- **Success Rate:** {(self.pattern_insights['profitable_stats']['count'] / len(self.trading_cycles) * 100):.1f}%

## 🏆 Profitable Cycle Characteristics
- **Average Return:** {self.pattern_insights['profitable_stats']['avg_return']:.2f}%
- **Average Duration:** {self.pattern_insights['profitable_stats']['avg_duration']:.1f} hours
- **Average Position Size:** {self.pattern_insights['profitable_stats']['avg_btc_size']:.4f} BTC
- **Average Buy Orders:** {self.pattern_insights['profitable_stats']['avg_num_buys']:.1f}

## 🎯 Optimal Ranges for Random Forest
### Position Size
- **Optimal Range:** {self.pattern_insights['optimal_ranges']['optimal_position_size']['min']:.3f} - {self.pattern_insights['optimal_ranges']['optimal_position_size']['max']:.3f} BTC
- **Median:** {self.pattern_insights['optimal_ranges']['optimal_position_size']['median']:.3f} BTC

### Duration
- **Optimal Range:** {self.pattern_insights['optimal_ranges']['optimal_duration']['min']:.1f} - {self.pattern_insights['optimal_ranges']['optimal_duration']['max']:.1f} hours
- **Median:** {self.pattern_insights['optimal_ranges']['optimal_duration']['median']:.1f} hours

## 🚀 Momentum Patterns
- **Average Price Change:** {self.pattern_insights['price_momentum_patterns']['avg_price_change']:.2f}%
- **Momentum Strength:** {self.pattern_insights['price_momentum_patterns']['avg_momentum_strength']:.3f}%/hour
- **High Momentum Threshold:** {self.pattern_insights['price_momentum_patterns']['high_momentum_threshold']:.3f}%/hour

## ⏰ Timing Insights
- **Best Start Hours:** {', '.join(map(str, self.pattern_insights['timing_patterns']['best_start_hours']))}
- **Optimal Duration:** {self.pattern_insights['timing_patterns']['optimal_duration']:.1f} hours

## 🎲 Key Features for Random Forest Enhancement
1. **Momentum Detection:** price_momentum_1h, price_momentum_5t
2. **Position Management:** current_position, is_position_optimal
3. **Timing Optimization:** is_optimal_hour, hour, dow
4. **Volatility Assessment:** price_volatility
5. **Pattern Recognition:** is_momentum_favorable

## 🎯 Recommended Model Improvements
1. **Separate models for:** Entry timing, Position sizing, Exit signals
2. **Enhanced features:** Multi-timeframe momentum, Support/resistance levels
3. **Target optimization:** Multi-horizon profit targets (1h, 4h, 8h)
4. **Risk management:** Cycle duration limits, Position size controls
"""
        
        return report
    
    def save_analysis(self, output_dir: str = "analysis") -> Dict[str, str]:
        """Save all analysis results."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save pattern insights
        insights_path = os.path.join(output_dir, "pattern_insights.json")
        with open(insights_path, 'w') as f:
            json.dump(self.pattern_insights, f, indent=2, default=str)
        
        # Save ML features
        features_df = self.extract_ml_features()
        features_path = os.path.join(output_dir, "ml_features.csv")
        features_df.to_csv(features_path, index=False)
        
        # Save cycles data
        cycles_path = os.path.join(output_dir, "trading_cycles.csv")
        pd.DataFrame(self.trading_cycles).to_csv(cycles_path, index=False)
        
        # Save pattern report
        report = self.generate_pattern_report()
        report_path = os.path.join(output_dir, "pattern_analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Analysis saved to {output_dir}/")
        
        return {
            'insights': insights_path,
            'features': features_path,
            'cycles': cycles_path,
            'report': report_path
        }

def main():
    """Main function to run pattern analysis."""
    analyzer = TradingPatternAnalyzer()
    
    try:
        # Run complete analysis
        print("🚀 Starting Trading Pattern Analysis")
        print("=" * 50)
        
        analyzer.parse_trading_data()
        analyzer.identify_trading_cycles()
        patterns = analyzer.analyze_profitable_patterns()
        
        # Generate and display report
        report = analyzer.generate_pattern_report()
        print(report)
        
        # Save results
        saved_files = analyzer.save_analysis()
        
        print("=" * 50)
        print("✅ Pattern analysis complete!")
        print(f"📁 Results saved to: {list(saved_files.values())}")
        
    except Exception as e:
        print(f"❌ Error in pattern analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()