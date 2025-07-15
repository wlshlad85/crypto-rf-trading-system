"""Simple test script to verify crypto RF trading system functionality."""

import asyncio
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel

logging.basicConfig(level=logging.INFO)


async def test_simple_pipeline():
    """Test basic pipeline functionality."""
    print("\n=== Testing Crypto RF Trading System ===\n")
    
    # 1. Configuration
    config = get_default_config()
    config.data.symbols = ['bitcoin', 'ethereum', 'solana']  # Test with 3 symbols
    config.data.days = 30  # Just 1 month for quick test
    
    # 2. Fetch Data
    print("1. Fetching cryptocurrency data...")
    fetcher = YFinanceCryptoFetcher(config.data)
    data_dict = fetcher.fetch_all_symbols(config.data.symbols)
    
    if not data_dict:
        print("❌ Failed to fetch data")
        return
    
    combined_data = fetcher.combine_data(data_dict)
    clean_data = fetcher.get_clean_data(combined_data)
    print(f"✅ Fetched data: {clean_data.shape}")
    
    # 3. Feature Engineering
    print("\n2. Engineering features...")
    feature_engine = CryptoFeatureEngine(config.features)
    features = feature_engine.generate_features(clean_data)
    print(f"✅ Generated features: {features.shape}")
    
    # 4. Create Model
    print("\n3. Creating Random Forest model...")
    model = CryptoRandomForestModel(config.model)
    
    # Create simple target - predict next hour's return for Bitcoin
    print("\n4. Creating target variable...")
    btc_close = clean_data['bitcoin_close']
    target = btc_close.pct_change(6).shift(-6)  # 6-hour ahead returns
    target = target.dropna()
    
    # Align features and target
    common_index = features.index.intersection(target.index)
    X = features.loc[common_index]
    y = target.loc[common_index]
    
    print(f"✅ Training data prepared: X={X.shape}, y={len(y)}")
    
    # 5. Simple training (no hyperparameter tuning)
    print("\n5. Training model...")
    # Add target to X temporarily for prepare_data
    X_with_target = X.copy()
    X_with_target['target'] = y
    X_clean, y_clean = model.prepare_data(X_with_target, 'target')
    
    # Use only first 1000 samples for quick test
    if len(X_clean) > 1000:
        X_clean = X_clean.iloc[:1000]
        y_clean = y_clean.iloc[:1000]
    
    results = model.train(X_clean, y_clean, validation_split=0.2)
    print(f"✅ Model trained!")
    print(f"   Train R²: {results['train']['r2']:.3f}")
    print(f"   Val R²: {results['validation']['r2']:.3f}")
    
    # 6. Feature importance
    print("\n6. Top 10 Feature Importance:")
    importance_df = model.get_feature_importance(top_n=10)
    print(importance_df.to_string())
    
    # 7. Make predictions
    print("\n7. Making predictions on test data...")
    test_X = X_clean.iloc[-10:]  # Last 10 samples
    predictions = model.predict(test_X)
    
    print(f"✅ Predictions made: {len(predictions)} samples")
    print(f"   Predicted returns range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # 8. Latest prices
    print("\n8. Current cryptocurrency prices:")
    prices = fetcher.get_latest_prices(config.data.symbols)
    for symbol, price in prices.items():
        print(f"   {symbol.upper()}: ${price:,.2f}")
    
    print("\n✅ Simple test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_simple_pipeline())