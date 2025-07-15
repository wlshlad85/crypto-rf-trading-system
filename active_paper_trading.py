"""Active paper trading with enhanced trading logic and debugging."""

import asyncio
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config, Config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel, EnsembleRandomForestModel

# Setup colorful logging
import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ActiveCryptoPaperTrader:
    """Active paper trading system with enhanced trading logic."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.portfolio = {
            'cash': 10000.0,  # Starting with $10,000
            'positions': {},  # symbol -> quantity
            'position_values': {},  # symbol -> current value
            'total_value': 10000.0,
            'trades': [],
            'returns': [],
            'timestamps': [],
            'predictions': [],
            'signals': []
        }
        self.model = None
        self.feature_engine = None
        self.current_prices = {}
        self.trading_log = []
        self.prediction_history = []
        
    async def run_active_trading(self, trading_days: int = 5):
        """Run active paper trading simulation."""
        print("\n" + "="*80)
        print("🚀 ACTIVE CRYPTOCURRENCY PAPER TRADING")
        print("="*80 + "\n")
        
        print(f"💰 Starting portfolio: ${self.portfolio['cash']:,.2f}")
        print(f"📅 Trading period: {trading_days} days")
        print(f"🎯 Strategy: Active trading with moderate thresholds")
        
        # Initialize system
        await self._initialize_trading_system()
        
        # Run trading simulation
        await self._run_active_simulation(trading_days)
        
        # Generate performance report
        self._generate_enhanced_report()
        
    async def _initialize_trading_system(self):
        """Initialize the trading system."""
        print("\n🔧 Initializing Active Trading System...")
        
        # Use 3 cryptocurrencies for diversification
        symbols = ['bitcoin', 'ethereum', 'solana']
        self.config.data.symbols = symbols
        self.config.data.days = 60  # 2 months of training data
        
        # Fetch training data
        print("   📊 Fetching training data...")
        fetcher = YFinanceCryptoFetcher(self.config.data)
        data_dict = fetcher.fetch_all_symbols(symbols)
        
        # Get latest prices
        self.current_prices = fetcher.get_latest_prices(symbols)
        print("   💰 Current prices:")
        for symbol, price in self.current_prices.items():
            print(f"      {symbol.upper()}: ${price:,.2f}")
        
        # Prepare data and features
        combined_data = fetcher.combine_data(data_dict)
        clean_data = fetcher.get_clean_data(combined_data)
        
        # Generate features
        print("   🔧 Generating features...")
        self.feature_engine = CryptoFeatureEngine(self.config.features)
        features = self.feature_engine.generate_features(clean_data)
        
        # Train model for each symbol
        print("   🤖 Training models for each cryptocurrency...")
        self.models = {}
        
        for symbol in symbols:
            print(f"      Training {symbol.upper()} model...")
            
            # Create symbol-specific model
            model = CryptoRandomForestModel(self.config.model)
            
            # Create target (6-hour returns)
            target = clean_data[f'{symbol}_close'].pct_change(6).shift(-6).dropna()
            
            # Prepare training data
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index].copy()
            y = target.loc[common_index]
            X['target'] = y
            
            # Train the model
            X_clean, y_clean = model.prepare_data(X, 'target')
            model.train(X_clean, y_clean)
            
            self.models[symbol] = model
            print(f"         ✅ {symbol.upper()} model trained on {len(X_clean)} samples")
        
        # Store current features for prediction
        self.current_features = features.iloc[-1:].copy()
        
        print(f"   ✅ System initialized with {len(self.models)} symbol-specific models")
        
    async def _run_active_simulation(self, trading_days: int):
        """Run the active trading simulation."""
        print(f"\n🚀 Starting {trading_days}-day active trading simulation...\n")
        
        # Use more moderate thresholds for active trading
        buy_thresh = 0.65  # Buy top 35%
        sell_thresh = 0.35  # Sell bottom 35%
        
        print(f"📊 Trading thresholds: Buy>{buy_thresh:.1%}, Sell<{sell_thresh:.1%}")
        
        # Initialize positions with equal allocation
        initial_allocation = self.portfolio['cash'] / len(self.models)
        
        # Start with small positions in each crypto
        for symbol in self.models.keys():
            price = self.current_prices[symbol]
            quantity = (initial_allocation * 0.3) / price  # 30% initial allocation
            self.portfolio['positions'][symbol] = quantity
            self.portfolio['cash'] -= quantity * price
            print(f"   🎯 Initial position: {quantity:.6f} {symbol.upper()} @ ${price:,.2f}")
        
        # Simulate trading over multiple days
        for day in range(trading_days):
            print(f"\n📅 Day {day + 1}/{trading_days}")
            print("-" * 50)
            
            # Multiple trading sessions per day
            for session in range(4):  # 4 sessions per day (6-hour intervals)
                await self._execute_active_session(day, session, buy_thresh, sell_thresh)
                await asyncio.sleep(0.1)  # Brief pause
            
            # Daily summary
            self._print_enhanced_summary(day + 1)
            
    async def _execute_active_session(self, day: int, session: int, buy_thresh: float, sell_thresh: float):
        """Execute an active trading session."""
        timestamp = datetime.now() + timedelta(days=day, hours=session*6)
        
        # Simulate more realistic price movements
        for symbol in self.current_prices:
            # Add trend and volatility
            trend = np.random.normal(0, 0.001)  # Small trend
            volatility = np.random.normal(0, 0.015)  # 1.5% volatility
            change = trend + volatility
            self.current_prices[symbol] *= (1 + change)
        
        # Update portfolio values
        self._update_portfolio_values()
        
        # Make trading decisions for each symbol
        session_trades = []
        predictions = {}
        
        for symbol in self.models.keys():
            # Get prediction
            prediction = self._make_symbol_prediction(symbol)
            predictions[symbol] = prediction
            
            if prediction is not None:
                # Convert to signal
                signal = self._convert_to_signal(prediction, buy_thresh, sell_thresh)
                
                # Execute trade if signal is strong
                if signal != 0:
                    trade_result = await self._execute_symbol_trade(symbol, signal, timestamp)
                    if trade_result:
                        session_trades.append(trade_result)
        
        # Store session data
        self.portfolio['returns'].append(self.portfolio['total_value'])
        self.portfolio['timestamps'].append(timestamp)
        self.portfolio['predictions'].append(predictions)
        self.portfolio['signals'].append({symbol: self._convert_to_signal(predictions.get(symbol, 0), buy_thresh, sell_thresh) for symbol in self.models.keys()})
        
        # Log session activity
        if session_trades:
            print(f"   📊 Session {session + 1}: {len(session_trades)} trades executed")
            for trade in session_trades:
                self.trading_log.append(f"      {trade['action']} {trade['quantity']:.6f} {trade['symbol'].upper()} @ ${trade['price']:,.2f}")
        
    def _make_symbol_prediction(self, symbol: str) -> float:
        """Make a prediction for a specific symbol."""
        try:
            # Use current features
            features = self.current_features.copy()
            
            # Remove target column if present
            if 'target' in features.columns:
                features = features.drop('target', axis=1)
            
            # Keep only numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_cols]
            
            # Make prediction
            prediction = self.models[symbol].predict(features)[0]
            
            # Store prediction for analysis
            self.prediction_history.append({
                'symbol': symbol,
                'prediction': prediction,
                'timestamp': datetime.now()
            })
            
            return prediction
            
        except Exception as e:
            print(f"   ⚠️  Prediction failed for {symbol}: {e}")
            return None
    
    def _convert_to_signal(self, prediction: float, buy_thresh: float, sell_thresh: float) -> int:
        """Convert prediction to trading signal."""
        if prediction is None:
            return 0
        
        # Use recent predictions to establish thresholds
        recent_predictions = [p['prediction'] for p in self.prediction_history[-50:] if p['prediction'] is not None]
        
        if len(recent_predictions) < 10:
            # Not enough data, use simple thresholds
            if prediction > 0.002:  # 0.2% threshold
                return 1
            elif prediction < -0.002:
                return -1
            else:
                return 0
        
        # Use percentile-based thresholds
        if prediction > np.percentile(recent_predictions, buy_thresh * 100):
            return 1  # Buy
        elif prediction < np.percentile(recent_predictions, sell_thresh * 100):
            return -1  # Sell
        else:
            return 0  # Hold
    
    async def _execute_symbol_trade(self, symbol: str, signal: int, timestamp: datetime):
        """Execute a trade for a specific symbol."""
        price = self.current_prices[symbol]
        current_position = self.portfolio['positions'].get(symbol, 0)
        
        if signal == 1:  # Buy signal
            # Use 20% of available cash
            cash_to_use = self.portfolio['cash'] * 0.2
            if cash_to_use > 50:  # Minimum trade size
                quantity = cash_to_use / price
                
                # Execute buy
                self.portfolio['cash'] -= cash_to_use
                self.portfolio['positions'][symbol] = self.portfolio['positions'].get(symbol, 0) + quantity
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'value': cash_to_use
                }
                self.portfolio['trades'].append(trade)
                return trade
                
        elif signal == -1 and current_position > 0:  # Sell signal
            # Sell 40% of position
            quantity_to_sell = current_position * 0.4
            sell_value = quantity_to_sell * price
            
            if sell_value > 50:  # Minimum trade size
                # Execute sell
                self.portfolio['cash'] += sell_value
                self.portfolio['positions'][symbol] -= quantity_to_sell
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity_to_sell,
                    'price': price,
                    'value': sell_value
                }
                self.portfolio['trades'].append(trade)
                return trade
        
        return None
    
    def _update_portfolio_values(self):
        """Update portfolio values."""
        total_value = self.portfolio['cash']
        
        for symbol, quantity in self.portfolio['positions'].items():
            if quantity > 0:
                current_value = quantity * self.current_prices[symbol]
                self.portfolio['position_values'][symbol] = current_value
                total_value += current_value
            else:
                self.portfolio['position_values'][symbol] = 0
        
        self.portfolio['total_value'] = total_value
    
    def _print_enhanced_summary(self, day: int):
        """Print enhanced daily summary."""
        total_return = (self.portfolio['total_value'] / 10000 - 1) * 100
        
        print(f"\n📊 Day {day} Enhanced Summary:")
        print(f"   💰 Portfolio Value: ${self.portfolio['total_value']:,.2f}")
        print(f"   📈 Total Return: {total_return:+.2f}%")
        print(f"   💵 Cash: ${self.portfolio['cash']:,.2f}")
        
        # Show positions with P&L
        print(f"   🪙 Positions:")
        for symbol, quantity in self.portfolio['positions'].items():
            if quantity > 0:
                value = self.portfolio['position_values'][symbol]
                price = self.current_prices[symbol]
                print(f"      {symbol.upper()}: {quantity:.6f} @ ${price:,.2f} (${value:,.2f})")
        
        # Show recent predictions
        recent_preds = [p for p in self.prediction_history if p['timestamp'].date() == (datetime.now() + timedelta(days=day-1)).date()]
        if recent_preds:
            print(f"   🔮 Recent Predictions:")
            for pred in recent_preds[-3:]:  # Show last 3
                print(f"      {pred['symbol'].upper()}: {pred['prediction']:+.4f}")
        
        # Show today's trades
        today_trades = [t for t in self.portfolio['trades'] if t['timestamp'].date() == (datetime.now() + timedelta(days=day-1)).date()]
        if today_trades:
            print(f"   📋 Today's Trades: {len(today_trades)}")
            for trade in today_trades[-3:]:  # Show last 3
                print(f"      {trade['action']} {trade['quantity']:.6f} {trade['symbol'].upper()} @ ${trade['price']:,.2f}")
    
    def _generate_enhanced_report(self):
        """Generate comprehensive trading report."""
        print("\n" + "="*80)
        print("📊 ACTIVE PAPER TRADING PERFORMANCE REPORT")
        print("="*80 + "\n")
        
        # Calculate performance metrics
        initial_value = 10000
        final_value = self.portfolio['total_value']
        total_return = (final_value / initial_value - 1) * 100
        
        print(f"📈 Performance Summary:")
        print(f"   💰 Initial Value: ${initial_value:,.2f}")
        print(f"   💰 Final Value: ${final_value:,.2f}")
        print(f"   📊 Total Return: {total_return:+.2f}%")
        
        # Calculate more detailed metrics
        if len(self.portfolio['returns']) > 1:
            returns = pd.Series(self.portfolio['returns'])
            daily_returns = returns.pct_change().dropna()
            
            if len(daily_returns) > 0:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
                max_drawdown = self._calculate_max_drawdown(returns)
                win_rate = len([r for r in daily_returns if r > 0]) / len(daily_returns)
                
                print(f"   ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
                print(f"   📉 Max Drawdown: {max_drawdown:.2f}%")
                print(f"   🎯 Win Rate: {win_rate:.2%}")
        
        # Trading statistics
        total_trades = len(self.portfolio['trades'])
        buy_trades = len([t for t in self.portfolio['trades'] if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.portfolio['trades'] if t['action'] == 'SELL'])
        
        print(f"\n📋 Trading Statistics:")
        print(f"   🔄 Total Trades: {total_trades}")
        print(f"   🟢 Buy Orders: {buy_trades}")
        print(f"   🔴 Sell Orders: {sell_trades}")
        
        # Symbol-specific performance
        print(f"\n💼 Current Holdings:")
        print(f"   💵 Cash: ${self.portfolio['cash']:,.2f}")
        for symbol, quantity in self.portfolio['positions'].items():
            if quantity > 0:
                value = self.portfolio['position_values'][symbol]
                price = self.current_prices[symbol]
                print(f"   🪙 {symbol.upper()}: {quantity:.6f} @ ${price:,.2f} (${value:,.2f})")
        
        # Create enhanced visualization
        self._create_enhanced_visualization()
        
        # Show prediction accuracy
        if self.prediction_history:
            print(f"\n🔮 Prediction Analysis:")
            predictions = [p['prediction'] for p in self.prediction_history]
            print(f"   📊 Total Predictions: {len(predictions)}")
            print(f"   📈 Avg Prediction: {np.mean(predictions):+.4f}")
            print(f"   📊 Prediction Std: {np.std(predictions):.4f}")
            
            # Show symbol-specific stats
            for symbol in self.models.keys():
                symbol_preds = [p['prediction'] for p in self.prediction_history if p['symbol'] == symbol]
                if symbol_preds:
                    print(f"   🪙 {symbol.upper()}: {len(symbol_preds)} predictions, avg {np.mean(symbol_preds):+.4f}")
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = returns / returns.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def _create_enhanced_visualization(self):
        """Create enhanced visualization."""
        if len(self.portfolio['returns']) < 2:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Active Cryptocurrency Paper Trading Performance', fontsize=16)
        
        # Portfolio value over time
        ax1 = axes[0, 0]
        timestamps = self.portfolio['timestamps']
        values = self.portfolio['returns']
        ax1.plot(timestamps, values, 'b-', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        ax2 = axes[0, 1]
        returns = pd.Series(values).pct_change().dropna() * 100
        ax2.hist(returns, bins=15, alpha=0.7, color='green')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax3 = axes[0, 2]
        cumulative_returns = (pd.Series(values) / values[0] - 1) * 100
        ax3.plot(timestamps, cumulative_returns, 'r-', linewidth=2)
        ax3.set_title('Cumulative Returns')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Trade distribution
        ax4 = axes[1, 0]
        if self.portfolio['trades']:
            trade_actions = [t['action'] for t in self.portfolio['trades']]
            trade_counts = pd.Series(trade_actions).value_counts()
            ax4.bar(trade_counts.index, trade_counts.values, color=['green', 'red'])
            ax4.set_title('Trade Distribution')
            ax4.set_ylabel('Number of Trades')
            ax4.grid(True, alpha=0.3)
        
        # Position values
        ax5 = axes[1, 1]
        if self.portfolio['position_values']:
            symbols = list(self.portfolio['position_values'].keys())
            values = list(self.portfolio['position_values'].values())
            ax5.bar(symbols, values, color='blue', alpha=0.7)
            ax5.set_title('Current Position Values')
            ax5.set_ylabel('Value ($)')
            ax5.grid(True, alpha=0.3)
        
        # Prediction trends
        ax6 = axes[1, 2]
        if self.prediction_history:
            pred_df = pd.DataFrame(self.prediction_history)
            for symbol in self.models.keys():
                symbol_data = pred_df[pred_df['symbol'] == symbol]
                if len(symbol_data) > 0:
                    ax6.plot(symbol_data['timestamp'], symbol_data['prediction'], 
                            label=symbol.upper(), alpha=0.7)
            ax6.set_title('Prediction Trends')
            ax6.set_ylabel('Prediction')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_filename = f"active_trading_results_{timestamp}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Enhanced visualization saved to {viz_filename}")
        plt.close()
        
        # Save detailed results
        results = {
            'portfolio': {
                'initial_value': 10000,
                'final_value': self.portfolio['total_value'],
                'total_return': (self.portfolio['total_value'] / 10000 - 1) * 100,
                'cash': self.portfolio['cash'],
                'positions': self.portfolio['positions'],
                'position_values': self.portfolio['position_values']
            },
            'trades': self.portfolio['trades'],
            'predictions': self.prediction_history,
            'performance': {
                'total_trades': len(self.portfolio['trades']),
                'total_predictions': len(self.prediction_history)
            }
        }
        
        results_filename = f"active_trading_results_{timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Detailed results saved to {results_filename}")


async def main():
    """Run the active paper trading simulation."""
    trader = ActiveCryptoPaperTrader()
    await trader.run_active_trading(trading_days=5)
    
    print("\n✨ Active paper trading simulation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())