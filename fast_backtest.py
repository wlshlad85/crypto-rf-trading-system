"""Fast backtesting for Cryptocurrency Random Forest Trading System."""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastBacktester:
    """Fast backtesting for crypto Random Forest strategy."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.results = {}
        
        # Load optimized parameters
        self._load_optimized_params()
        
    def _load_optimized_params(self):
        """Load optimized parameters."""
        try:
            result_files = list(Path('.').glob('fast_optimization_results_*.json'))
            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                if 'hyperparameters' in results:
                    best_params = results['hyperparameters']['best_params']
                    self.config.model.__dict__.update(best_params)
                    logger.info(f"Loaded optimized parameters from {latest_file.name}")
                    
        except Exception as e:
            logger.warning(f"Could not load optimization results: {e}")
    
    def run_fast_backtest(self):
        """Run fast backtesting."""
        print("\n" + "="*80)
        print("📊 FAST BACKTESTING - CRYPTO RANDOM FOREST STRATEGY")
        print("="*80 + "\n")
        
        # Step 1: Prepare data
        print("📊 Step 1: Preparing Data...")
        self._prepare_data()
        
        # Step 2: Train model
        print("\n🤖 Step 2: Training Model...")
        model = self._train_model()
        
        # Step 3: Run backtest
        print("\n📈 Step 3: Running Backtest...")
        backtest_results = self._run_backtest(model)
        
        # Step 4: Calculate metrics
        print("\n📊 Step 4: Calculating Performance Metrics...")
        metrics = self._calculate_metrics(backtest_results)
        
        # Step 5: Compare benchmarks
        print("\n🎯 Step 5: Comparing Against Benchmarks...")
        benchmarks = self._compare_benchmarks()
        
        # Step 6: Generate report
        print("\n📊 Step 6: Generating Report...")
        self._generate_report(metrics, benchmarks, backtest_results)
        
        return self.results
    
    def _prepare_data(self):
        """Prepare data for backtesting."""
        # Use 3 months for faster processing
        self.config.data.days = 90
        self.config.data.symbols = ['bitcoin', 'ethereum', 'solana']
        
        print(f"   📊 Fetching 3 months of data...")
        fetcher = YFinanceCryptoFetcher(self.config.data)
        data_dict = fetcher.fetch_all_symbols(self.config.data.symbols)
        
        # Get latest prices
        prices = fetcher.get_latest_prices(self.config.data.symbols)
        print("   💰 Current prices:")
        for symbol, price in prices.items():
            print(f"      {symbol.upper()}: ${price:,.2f}")
        
        # Combine and clean data
        self.raw_data = fetcher.combine_data(data_dict)
        self.clean_data = fetcher.get_clean_data(self.raw_data)
        
        # Generate features
        print(f"   🔧 Generating features...")
        feature_engine = CryptoFeatureEngine(self.config.features)
        self.features = feature_engine.generate_features(self.clean_data)
        
        # Split data
        split_idx = int(len(self.clean_data) * 0.7)
        self.train_data = self.clean_data.iloc[:split_idx]
        self.test_data = self.clean_data.iloc[split_idx:]
        self.train_features = self.features.iloc[:split_idx]
        self.test_features = self.features.iloc[split_idx:]
        
        print(f"   ✅ Data prepared: Train={len(self.train_data)}, Test={len(self.test_data)}")
    
    def _train_model(self):
        """Train the model."""
        print("   🤖 Training Random Forest model...")
        
        # Use single model for speed
        model = CryptoRandomForestModel(self.config.model)
        
        # Prepare training data
        target = self.train_data['bitcoin_close'].pct_change(6).shift(-6).dropna()
        
        common_index = self.train_features.index.intersection(target.index)
        X = self.train_features.loc[common_index].copy()
        y = target.loc[common_index]
        X['target'] = y
        
        X_clean, y_clean = model.prepare_data(X, 'target')
        train_results = model.train(X_clean, y_clean)
        
        print(f"   ✅ Model trained with R²: {train_results['validation']['r2']:.4f}")
        
        return model
    
    def _run_backtest(self, model):
        """Run the backtest simulation."""
        print("   📈 Running backtest simulation...")
        
        portfolio = {
            'cash': 10000.0,
            'positions': {},
            'equity_curve': [10000.0],
            'trades': [],
            'returns': []
        }
        
        # Trading parameters
        buy_thresh = 0.65
        sell_thresh = 0.35
        position_size = 0.3
        
        # Simulate trading
        for i in range(1, len(self.test_data)):
            current_idx = self.test_data.index[i]
            
            # Current prices
            btc_price = self.test_data.loc[current_idx, 'bitcoin_close']
            eth_price = self.test_data.loc[current_idx, 'ethereum_close']
            sol_price = self.test_data.loc[current_idx, 'solana_close']
            
            # Make prediction
            features_row = self.test_features.iloc[i:i+1]
            numeric_cols = features_row.select_dtypes(include=[np.number]).columns
            features_clean = features_row[numeric_cols]
            
            try:
                prediction = model.predict(features_clean)[0]
                
                # Simple signal generation
                if prediction > 0.001:  # Bullish
                    signal = 1
                elif prediction < -0.001:  # Bearish
                    signal = -1
                else:
                    signal = 0
                
                # Execute trades (simplified)
                if signal == 1 and portfolio['cash'] > 1000:
                    # Buy Bitcoin
                    trade_value = portfolio['cash'] * position_size
                    quantity = trade_value / btc_price
                    
                    portfolio['cash'] -= trade_value
                    portfolio['positions']['bitcoin'] = portfolio['positions'].get('bitcoin', 0) + quantity
                    
                    portfolio['trades'].append({
                        'timestamp': current_idx,
                        'action': 'BUY',
                        'symbol': 'bitcoin',
                        'quantity': quantity,
                        'price': btc_price
                    })
                    
                elif signal == -1 and 'bitcoin' in portfolio['positions']:
                    # Sell half position
                    quantity = portfolio['positions']['bitcoin'] * 0.5
                    trade_value = quantity * btc_price
                    
                    portfolio['cash'] += trade_value
                    portfolio['positions']['bitcoin'] -= quantity
                    
                    portfolio['trades'].append({
                        'timestamp': current_idx,
                        'action': 'SELL',
                        'symbol': 'bitcoin',
                        'quantity': quantity,
                        'price': btc_price
                    })
                    
            except:
                pass  # Skip failed predictions
            
            # Calculate portfolio value
            total_value = portfolio['cash']
            if 'bitcoin' in portfolio['positions']:
                total_value += portfolio['positions']['bitcoin'] * btc_price
            
            portfolio['equity_curve'].append(total_value)
            
            # Calculate return
            if len(portfolio['equity_curve']) > 1:
                daily_return = (portfolio['equity_curve'][-1] / portfolio['equity_curve'][-2] - 1)
                portfolio['returns'].append(daily_return)
        
        print(f"   ✅ Backtest complete: {len(portfolio['trades'])} trades executed")
        
        return portfolio
    
    def _calculate_metrics(self, backtest_results):
        """Calculate performance metrics."""
        equity_curve = pd.Series(backtest_results['equity_curve'])
        returns = pd.Series(backtest_results['returns'])
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        
        # Risk metrics
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252 * 24)
            else:
                sortino_ratio = 0
        else:
            sharpe_ratio = sortino_ratio = 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        if len(cumulative) > 0:
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Trade statistics
        trades = backtest_results['trades']
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t['action'] == 'BUY'])
        sell_trades = len([t for t in trades if t['action'] == 'SELL'])
        
        # Win rate
        positive_returns = len(returns[returns > 0])
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'avg_return': returns.mean() if len(returns) > 0 else 0,
            'volatility': returns.std() if len(returns) > 0 else 0
        }
        
        print(f"   ✅ Performance metrics calculated")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"      Max Drawdown: {max_drawdown:.2%}")
        print(f"      Win Rate: {win_rate:.2%}")
        
        self.results['metrics'] = metrics
        return metrics
    
    def _compare_benchmarks(self):
        """Compare against benchmarks."""
        # Buy and Hold Bitcoin
        btc_start = self.test_data['bitcoin_close'].iloc[0]
        btc_end = self.test_data['bitcoin_close'].iloc[-1]
        btc_return = (btc_end / btc_start - 1)
        
        # Buy and Hold Equal Weight
        eth_start = self.test_data['ethereum_close'].iloc[0]
        eth_end = self.test_data['ethereum_close'].iloc[-1]
        sol_start = self.test_data['solana_close'].iloc[0]
        sol_end = self.test_data['solana_close'].iloc[-1]
        
        equal_weight_return = (
            (btc_end / btc_start + eth_end / eth_start + sol_end / sol_start) / 3 - 1
        )
        
        benchmarks = {
            'buy_hold_btc': btc_return,
            'buy_hold_eth': (eth_end / eth_start - 1),
            'buy_hold_sol': (sol_end / sol_start - 1),
            'equal_weight': equal_weight_return
        }
        
        print(f"   ✅ Benchmark comparison:")
        print(f"      Strategy: {self.results['metrics']['total_return']:.2%}")
        print(f"      Buy & Hold BTC: {btc_return:.2%}")
        print(f"      Equal Weight: {equal_weight_return:.2%}")
        
        self.results['benchmarks'] = benchmarks
        return benchmarks
    
    def _generate_report(self, metrics, benchmarks, backtest_results):
        """Generate backtesting report."""
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cryptocurrency Random Forest Strategy - Fast Backtest Results', fontsize=16)
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        equity_curve = backtest_results['equity_curve']
        ax1.plot(range(len(equity_curve)), equity_curve, 'b-', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Distribution
        ax2 = axes[0, 1]
        if backtest_results['returns']:
            returns = pd.Series(backtest_results['returns']) * 100
            ax2.hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Returns Distribution')
            ax2.set_xlabel('Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # 3. Strategy vs Benchmarks
        ax3 = axes[1, 0]
        strategies = ['RF Strategy', 'BTC B&H', 'ETH B&H', 'SOL B&H', 'Equal Weight']
        returns = [
            metrics['total_return'] * 100,
            benchmarks['buy_hold_btc'] * 100,
            benchmarks['buy_hold_eth'] * 100,
            benchmarks['buy_hold_sol'] * 100,
            benchmarks['equal_weight'] * 100
        ]
        colors = ['green', 'orange', 'blue', 'purple', 'red']
        ax3.bar(strategies, returns, color=colors, alpha=0.7)
        ax3.set_title('Strategy vs Benchmarks')
        ax3.set_ylabel('Total Return (%)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Performance Metrics
        ax4 = axes[1, 1]
        metric_names = ['Sharpe', 'Sortino', 'Win Rate']
        metric_values = [
            metrics['sharpe_ratio'],
            metrics['sortino_ratio'],
            metrics['win_rate'] * 100
        ]
        ax4.bar(metric_names, metric_values, color=['blue', 'green', 'orange'], alpha=0.7)
        ax4.set_title('Risk-Adjusted Performance')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_filename = f"fast_backtest_results_{timestamp}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved to {viz_filename}")
        plt.close()
        
        # Save results
        results_filename = f"fast_backtest_results_{timestamp}.json"
        json_results = {
            'metrics': metrics,
            'benchmarks': benchmarks,
            'trade_count': len(backtest_results['trades']),
            'test_period': f"{self.test_data.index[0]} to {self.test_data.index[-1]}",
            'final_value': backtest_results['equity_curve'][-1]
        }
        
        with open(results_filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {results_filename}")
        
        # Print summary
        print("\n" + "="*80)
        print("🏆 BACKTEST SUMMARY")
        print("="*80)
        print(f"\n📊 Performance:")
        print(f"   • Total Return: {metrics['total_return']:.2%}")
        print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   • Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   • Win Rate: {metrics['win_rate']:.2%}")
        print(f"   • Total Trades: {metrics['total_trades']}")
        
        if metrics['total_return'] > benchmarks['buy_hold_btc']:
            print(f"\n✅ Strategy OUTPERFORMED Buy & Hold BTC!")
        else:
            print(f"\n⚠️  Strategy underperformed Buy & Hold BTC")
        
        print(f"\n📊 Files Generated:")
        print(f"   • {viz_filename}")
        print(f"   • {results_filename}")


def main():
    """Run fast backtesting."""
    backtester = FastBacktester()
    results = backtester.run_fast_backtest()
    
    print("\n✨ Fast backtesting completed successfully!")
    return results


if __name__ == "__main__":
    main()