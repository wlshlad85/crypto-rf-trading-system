"""Paper trading implementation using optimized Random Forest parameters."""

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


class CryptoPaperTrader:
    """Paper trading system using optimized Random Forest model."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.portfolio = {
            'cash': 10000.0,  # Starting with $10,000
            'positions': {},  # symbol -> quantity
            'position_values': {},  # symbol -> current value
            'total_value': 10000.0,
            'trades': [],
            'returns': [],
            'timestamps': []
        }
        self.model = None
        self.feature_engine = None
        self.current_prices = {}
        self.trading_log = []
        
        # Load optimized parameters from latest results
        self._load_optimized_params()
        
    def _load_optimized_params(self):
        """Load the best parameters from optimization results."""
        try:
            # Find latest optimization results
            result_files = list(Path('.').glob('fast_optimization_results_*.json'))
            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                # Apply optimized hyperparameters
                if 'hyperparameters' in results:
                    best_params = results['hyperparameters']['best_params']
                    print(f"📊 Loading optimized hyperparameters: {best_params}")
                    self.config.model.__dict__.update(best_params)
                
                # Store optimization results for reference
                self.optimization_results = results
                
        except Exception as e:
            print(f"⚠️  Could not load optimization results: {e}")
            print("   Using default parameters")
    
    async def run_paper_trading(self, trading_days: int = 7):
        """Run paper trading simulation."""
        print("\n" + "="*80)
        print("📈 CRYPTOCURRENCY PAPER TRADING - LIVE SIMULATION")
        print("="*80 + "\n")
        
        print(f"💰 Starting portfolio: ${self.portfolio['cash']:,.2f}")
        print(f"📅 Trading period: {trading_days} days")
        print(f"🎯 Strategy: Long/Short with optimized thresholds")
        
        # Initialize system
        await self._initialize_trading_system()
        
        # Run trading simulation
        await self._run_trading_simulation(trading_days)
        
        # Generate performance report
        self._generate_trading_report()
        
    async def _initialize_trading_system(self):
        """Initialize the trading system with trained model."""
        print("\n🔧 Initializing Trading System...")
        
        # Use optimized symbols (top 3 for faster execution)
        symbols = ['bitcoin', 'ethereum', 'solana']
        self.config.data.symbols = symbols
        self.config.data.days = 30  # 1 month of training data
        
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
        
        # Train ensemble model with optimized parameters
        print("   🤖 Training optimized ensemble model...")
        ensemble_size = self.optimization_results.get('ensemble', {}).get('best_size', 3)
        self.model = EnsembleRandomForestModel(self.config.model, n_models=ensemble_size)
        
        # Create target (12-hour returns based on optimization)
        target_symbol = 'bitcoin'  # Primary trading symbol
        target = clean_data[f'{target_symbol}_close'].pct_change(12).shift(-12).dropna()
        
        # Prepare training data
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index].copy()
        y = target.loc[common_index]
        X['target'] = y
        
        # Train the ensemble
        X_clean, y_clean = self.model.models[0].prepare_data(X, 'target')
        self.model.train(X_clean, y_clean)
        
        # Store current features for prediction
        self.current_features = features.iloc[-1:].copy()  # Latest features
        
        print(f"   ✅ System initialized with {ensemble_size}-model ensemble")
        print(f"   📊 Trained on {len(X_clean)} samples with {X_clean.shape[1]} features")
        
    async def _run_trading_simulation(self, trading_days: int):
        """Run the paper trading simulation."""
        print(f"\n🚀 Starting {trading_days}-day paper trading simulation...\n")
        
        # Get optimized trading thresholds
        thresholds = self.optimization_results.get('strategy', {}).get('best_thresholds', (0.8, 0.2))
        if isinstance(thresholds, str):
            # Parse from string format "0.8_0.2"
            buy_thresh, sell_thresh = map(float, thresholds.split('_'))
        else:
            buy_thresh, sell_thresh = thresholds
        
        print(f"📊 Trading thresholds: Buy>{buy_thresh:.1%}, Sell<{sell_thresh:.1%}")
        
        # Simulate trading over multiple days
        for day in range(trading_days):
            print(f"\n📅 Day {day + 1}/{trading_days}")
            print("-" * 40)
            
            # Simulate multiple trading sessions per day (every 4 hours)
            for session in range(6):  # 6 sessions per day (4-hour intervals)
                await self._execute_trading_session(day, session, buy_thresh, sell_thresh)
                
                # Small delay to simulate real-time trading
                await asyncio.sleep(0.1)
            
            # Daily summary
            self._print_daily_summary(day + 1)
            
    async def _execute_trading_session(self, day: int, session: int, buy_thresh: float, sell_thresh: float):
        """Execute a single trading session."""
        timestamp = datetime.now() + timedelta(days=day, hours=session*4)
        
        # Simulate price movements (±2% random walk)
        for symbol in self.current_prices:
            change = np.random.normal(0, 0.02)  # 2% volatility
            self.current_prices[symbol] *= (1 + change)
            
        # Update portfolio values
        self._update_portfolio_values()
        
        # Make trading decision
        prediction = self._make_trading_prediction()
        
        if prediction is not None:
            # Convert prediction to trading signal
            signal = self._convert_prediction_to_signal(prediction, buy_thresh, sell_thresh)
            
            # Execute trades
            if signal != 0:
                await self._execute_trades(signal, timestamp)
        
        # Record portfolio state
        self.portfolio['returns'].append(self.portfolio['total_value'])
        self.portfolio['timestamps'].append(timestamp)
        
    def _make_trading_prediction(self):
        """Make a prediction using the trained model."""
        try:
            # Use the latest features (in practice, this would be updated with new data)
            features = self.current_features.iloc[-1:].copy()
            
            # Remove target column if present
            if 'target' in features.columns:
                features = features.drop('target', axis=1)
            
            # Remove non-numeric columns like 'symbol' that weren't used in training
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_cols]
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            return prediction
            
        except Exception as e:
            print(f"   ⚠️  Prediction failed: {e}")
            return None
    
    def _convert_prediction_to_signal(self, prediction: float, buy_thresh: float, sell_thresh: float):
        """Convert model prediction to trading signal."""
        # Create a dummy distribution for threshold calculation
        # In practice, this would use historical predictions
        predictions = np.random.normal(prediction, abs(prediction) * 0.1, 100)
        
        if prediction > np.quantile(predictions, buy_thresh):
            return 1  # Buy signal
        elif prediction < np.quantile(predictions, sell_thresh):
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    async def _execute_trades(self, signal: int, timestamp: datetime):
        """Execute trades based on signal."""
        primary_symbol = 'bitcoin'  # Primary trading symbol
        price = self.current_prices[primary_symbol]
        
        if signal == 1:  # Buy
            # Use 30% of available cash
            cash_to_use = self.portfolio['cash'] * 0.3
            if cash_to_use > 100:  # Minimum trade size
                quantity = cash_to_use / price
                
                # Execute buy order
                self.portfolio['cash'] -= cash_to_use
                self.portfolio['positions'][primary_symbol] = self.portfolio['positions'].get(primary_symbol, 0) + quantity
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': primary_symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'value': cash_to_use
                }
                self.portfolio['trades'].append(trade)
                self.trading_log.append(f"   🟢 BUY {quantity:.6f} {primary_symbol.upper()} @ ${price:,.2f}")
                
        elif signal == -1:  # Sell
            # Sell 50% of position if we have any
            if primary_symbol in self.portfolio['positions'] and self.portfolio['positions'][primary_symbol] > 0:
                quantity_to_sell = self.portfolio['positions'][primary_symbol] * 0.5
                sell_value = quantity_to_sell * price
                
                # Execute sell order
                self.portfolio['cash'] += sell_value
                self.portfolio['positions'][primary_symbol] -= quantity_to_sell
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': primary_symbol,
                    'action': 'SELL',
                    'quantity': quantity_to_sell,
                    'price': price,
                    'value': sell_value
                }
                self.portfolio['trades'].append(trade)
                self.trading_log.append(f"   🔴 SELL {quantity_to_sell:.6f} {primary_symbol.upper()} @ ${price:,.2f}")
    
    def _update_portfolio_values(self):
        """Update portfolio values based on current prices."""
        total_value = self.portfolio['cash']
        
        for symbol, quantity in self.portfolio['positions'].items():
            if quantity > 0:
                current_value = quantity * self.current_prices[symbol]
                self.portfolio['position_values'][symbol] = current_value
                total_value += current_value
            else:
                self.portfolio['position_values'][symbol] = 0
        
        self.portfolio['total_value'] = total_value
    
    def _print_daily_summary(self, day: int):
        """Print daily trading summary."""
        total_return = (self.portfolio['total_value'] / 10000 - 1) * 100
        
        print(f"\n📊 Day {day} Summary:")
        print(f"   💰 Portfolio Value: ${self.portfolio['total_value']:,.2f}")
        print(f"   📈 Total Return: {total_return:+.2f}%")
        print(f"   💵 Cash: ${self.portfolio['cash']:,.2f}")
        
        # Show positions
        for symbol, quantity in self.portfolio['positions'].items():
            if quantity > 0:
                value = self.portfolio['position_values'][symbol]
                print(f"   🪙 {symbol.upper()}: {quantity:.6f} (${value:,.2f})")
        
        # Show recent trades
        recent_trades = [t for t in self.portfolio['trades'] if t['timestamp'].date() == (datetime.now() + timedelta(days=day-1)).date()]
        if recent_trades:
            print(f"   📋 Today's Trades: {len(recent_trades)}")
            for trade in recent_trades[-3:]:  # Show last 3 trades
                print(f"      {trade['action']} {trade['quantity']:.6f} {trade['symbol'].upper()} @ ${trade['price']:,.2f}")
    
    def _generate_trading_report(self):
        """Generate comprehensive trading performance report."""
        print("\n" + "="*80)
        print("📊 PAPER TRADING PERFORMANCE REPORT")
        print("="*80 + "\n")
        
        # Calculate performance metrics
        initial_value = 10000
        final_value = self.portfolio['total_value']
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate daily returns
        if len(self.portfolio['returns']) > 1:
            returns = pd.Series(self.portfolio['returns'])
            daily_returns = returns.pct_change().dropna()
            
            # Performance metrics
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = len([r for r in daily_returns if r > 0]) / len(daily_returns) if len(daily_returns) > 0 else 0
            
            print(f"📈 Performance Metrics:")
            print(f"   💰 Initial Value: ${initial_value:,.2f}")
            print(f"   💰 Final Value: ${final_value:,.2f}")
            print(f"   📊 Total Return: {total_return:+.2f}%")
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
            
            # Current positions
            print(f"\n💼 Current Positions:")
            print(f"   💵 Cash: ${self.portfolio['cash']:,.2f}")
            for symbol, quantity in self.portfolio['positions'].items():
                if quantity > 0:
                    value = self.portfolio['position_values'][symbol]
                    print(f"   🪙 {symbol.upper()}: {quantity:.6f} (${value:,.2f})")
            
            # Create visualization
            self._create_trading_visualization()
        
        # Show recent trading log
        print(f"\n📝 Recent Trading Activity:")
        for log_entry in self.trading_log[-10:]:  # Show last 10 entries
            print(log_entry)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def _create_trading_visualization(self):
        """Create trading performance visualization."""
        if len(self.portfolio['returns']) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cryptocurrency Paper Trading Performance', fontsize=16)
        
        # Portfolio value over time
        ax1 = axes[0, 0]
        timestamps = self.portfolio['timestamps']
        values = self.portfolio['returns']
        ax1.plot(timestamps, values, 'b-', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Daily returns
        ax2 = axes[0, 1]
        returns = pd.Series(values).pct_change().dropna() * 100
        ax2.hist(returns, bins=20, alpha=0.7, color='green')
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax3 = axes[1, 0]
        cumulative_returns = (pd.Series(values) / values[0] - 1) * 100
        ax3.plot(timestamps, cumulative_returns, 'r-', linewidth=2)
        ax3.set_title('Cumulative Returns')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Trade distribution
        ax4 = axes[1, 1]
        if self.portfolio['trades']:
            trade_actions = [t['action'] for t in self.portfolio['trades']]
            trade_counts = pd.Series(trade_actions).value_counts()
            ax4.bar(trade_counts.index, trade_counts.values, color=['green', 'red'])
            ax4.set_title('Trade Distribution')
            ax4.set_ylabel('Number of Trades')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"paper_trading_results_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Trading visualization saved to {filename}")
        plt.close()
        
        # Save results to JSON
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
            'performance': {
                'total_trades': len(self.portfolio['trades']),
                'win_rate': len([r for r in pd.Series(values).pct_change().dropna() if r > 0]) / len(pd.Series(values).pct_change().dropna()) if len(values) > 1 else 0
            }
        }
        
        results_filename = f"paper_trading_results_{timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Trading results saved to {results_filename}")


async def main():
    """Run the paper trading simulation."""
    trader = CryptoPaperTrader()
    await trader.run_paper_trading(trading_days=7)
    
    print("\n✨ Paper trading simulation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())