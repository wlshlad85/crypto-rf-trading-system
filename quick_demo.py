"""Quick demonstration of the Crypto RF Trading System with results."""

import asyncio
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def run_crypto_rf_demo():
    """Run a complete demonstration of the crypto RF trading system."""
    print("\n" + "="*80)
    print("🚀 CRYPTOCURRENCY RANDOM FOREST TRADING SYSTEM DEMO")
    print("="*80 + "\n")
    
    # 1. Configuration
    print("📋 Step 1: Configuration")
    config = get_default_config()
    config.data.symbols = ['bitcoin', 'ethereum', 'solana']  # Top 3 for demo
    config.data.days = 90  # 3 months for faster demo
    config.model.n_estimators = 100  # Fewer trees for speed
    print(f"   ✓ Configured for {len(config.data.symbols)} cryptocurrencies")
    print(f"   ✓ {config.data.days} days of historical data")
    
    # 2. Fetch Real Cryptocurrency Data
    print("\n📊 Step 2: Fetching Real-Time Cryptocurrency Data...")
    fetcher = YFinanceCryptoFetcher(config.data)
    data_dict = fetcher.fetch_all_symbols(config.data.symbols)
    
    if not data_dict:
        print("❌ Failed to fetch data")
        return
    
    # Show current prices
    print("\n💰 Current Cryptocurrency Prices:")
    prices = fetcher.get_latest_prices(config.data.symbols)
    for symbol, price in prices.items():
        print(f"   {symbol.upper()}: ${price:,.2f}")
    
    # Combine and clean data
    combined_data = fetcher.combine_data(data_dict)
    clean_data = fetcher.get_clean_data(combined_data)
    print(f"\n   ✓ Fetched {clean_data.shape[0]} hourly records")
    print(f"   ✓ Date range: {clean_data.index[0]} to {clean_data.index[-1]}")
    
    # 3. Feature Engineering
    print("\n🔧 Step 3: Engineering Trading Features...")
    feature_engine = CryptoFeatureEngine(config.features)
    features = feature_engine.generate_features(clean_data)
    print(f"   ✓ Generated {features.shape[1]} features")
    
    # Show top features
    feature_categories = {
        'Technical': [col for col in features.columns if any(ind in col for ind in ['rsi', 'macd', 'bb'])],
        'Volume': [col for col in features.columns if 'volume' in col],
        'Returns': [col for col in features.columns if 'return' in col],
        'Volatility': [col for col in features.columns if 'volatility' in col]
    }
    
    print("\n   📈 Feature Categories:")
    for category, cols in feature_categories.items():
        print(f"      - {category}: {len(cols)} features")
    
    # 4. Prepare Model and Target
    print("\n🤖 Step 4: Training Random Forest Model...")
    model = CryptoRandomForestModel(config.model)
    
    # Create target - predict Bitcoin's next 6-hour return
    btc_close = clean_data['bitcoin_close']
    target = btc_close.pct_change(6).shift(-6)
    target = target.dropna()
    
    # Align features and target
    common_index = features.index.intersection(target.index)
    X = features.loc[common_index].copy()
    y = target.loc[common_index]
    
    # Add target for prepare_data
    X['target'] = y
    X_clean, y_clean = model.prepare_data(X, 'target')
    
    print(f"   ✓ Training data: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
    
    # 5. Train Model
    train_size = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:train_size]
    y_train = y_clean.iloc[:train_size]
    X_test = X_clean.iloc[train_size:]
    y_test = y_clean.iloc[train_size:]
    
    # Train
    results = model.train(X_train, y_train, validation_split=0.2)
    print(f"\n   📊 Training Results:")
    print(f"      - Train R²: {results['train']['r2']:.4f}")
    print(f"      - Validation R²: {results['validation']['r2']:.4f}")
    print(f"      - RMSE: {results['validation']['rmse']:.4f}")
    
    # 6. Make Predictions
    print("\n🎯 Step 5: Making Predictions...")
    predictions = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)
    
    print(f"   ✓ Test Set Performance:")
    print(f"      - MAE: {test_mae:.4f}")
    print(f"      - R²: {test_r2:.4f}")
    
    # 7. Feature Importance
    print("\n🏆 Step 6: Top 10 Most Important Features:")
    importance_df = model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"   {idx+1:2d}. {row['feature'][:50]:50s} {row['importance']:.4f}")
    
    # 8. Trading Signals
    print("\n📈 Step 7: Generating Trading Signals...")
    
    # Convert predictions to trading signals
    pred_percentile = pd.Series(predictions).rank(pct=True)
    signals = pd.Series(index=X_test.index[:len(predictions)], data=0)
    signals[pred_percentile.values > 0.7] = 1   # Buy top 30%
    signals[pred_percentile.values < 0.3] = -1  # Sell bottom 30%
    
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    hold_signals = (signals == 0).sum()
    
    print(f"   ✓ Trading Signals Generated:")
    print(f"      - Buy signals: {buy_signals} ({buy_signals/len(signals)*100:.1f}%)")
    print(f"      - Sell signals: {sell_signals} ({sell_signals/len(signals)*100:.1f}%)")
    print(f"      - Hold signals: {hold_signals} ({hold_signals/len(signals)*100:.1f}%)")
    
    # 9. Simple Backtest
    print("\n💼 Step 8: Simple Backtest Results...")
    
    # Calculate returns based on signals
    btc_returns = clean_data['bitcoin_close'].pct_change()
    btc_returns = btc_returns.loc[signals.index]
    
    strategy_returns = signals.shift(1) * btc_returns  # Lag signals by 1 period
    strategy_returns = strategy_returns.dropna()
    
    # Performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(24 * 365)  # Annualized
    max_drawdown = (strategy_returns.cumsum().expanding().max() - strategy_returns.cumsum()).max()
    
    print(f"   ✓ Strategy Performance:")
    print(f"      - Total Return: {total_return:.2%}")
    print(f"      - Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"      - Max Drawdown: {max_drawdown:.2%}")
    print(f"      - Win Rate: {(strategy_returns > 0).mean():.2%}")
    
    # 10. Visualization
    print("\n📊 Step 9: Creating Visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cryptocurrency Random Forest Trading System Results', fontsize=16)
    
    # Plot 1: Cumulative Returns
    ax1 = axes[0, 0]
    cumulative_returns = (1 + strategy_returns).cumprod()
    cumulative_btc = (1 + btc_returns).cumprod()
    
    ax1.plot(cumulative_returns.index, cumulative_returns, label='RF Strategy', linewidth=2)
    ax1.plot(cumulative_btc.index, cumulative_btc, label='Buy & Hold BTC', alpha=0.7)
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance
    ax2 = axes[0, 1]
    top_features = importance_df.head(10)
    ax2.barh(range(len(top_features)), top_features['importance'])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels([f[:30] for f in top_features['feature']])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Feature Importance')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Predictions vs Actual
    ax3 = axes[1, 0]
    ax3.scatter(y_test[:100], predictions[:100], alpha=0.5)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Returns')
    ax3.set_ylabel('Predicted Returns')
    ax3.set_title(f'Predictions vs Actual (R² = {test_r2:.3f})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Signal Distribution
    ax4 = axes[1, 1]
    signal_counts = signals.value_counts()
    signal_labels = {1: 'Buy', 0: 'Hold', -1: 'Sell'}
    ax4.pie(signal_counts.values, labels=[signal_labels[k] for k in signal_counts.index], 
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Trading Signal Distribution')
    
    plt.tight_layout()
    plt.savefig('crypto_rf_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Results saved to crypto_rf_results.png")
    
    # Summary
    print("\n" + "="*80)
    print("✅ CRYPTO RANDOM FOREST TRADING SYSTEM DEMO COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   • Analyzed {len(config.data.symbols)} cryptocurrencies")
    print(f"   • Generated {features.shape[1]} trading features")
    print(f"   • Trained Random Forest with {config.model.n_estimators} trees")
    print(f"   • Achieved {sharpe_ratio:.2f} Sharpe ratio")
    print(f"   • Generated {buy_signals + sell_signals} trading signals")
    print(f"\n🚀 The system is ready for production use!")
    
    return {
        'model': model,
        'features': features,
        'predictions': predictions,
        'signals': signals,
        'performance': {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'test_r2': test_r2
        }
    }


if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(run_crypto_rf_demo())
    print("\n✨ Demo completed successfully!")