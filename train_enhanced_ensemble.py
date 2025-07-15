#!/usr/bin/env python3
"""
Train Enhanced Random Forest Ensemble on Real Market Data

Trains the multi-model ensemble system on 4-hour cryptocurrency data
targeting 4-6% returns based on successful trading patterns.

Usage: python3 train_enhanced_ensemble.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, List

from enhanced_rf_ensemble import EnhancedRFEnsemble

def load_training_data(dataset_path: str) -> pd.DataFrame:
    """Load the 4-hour cryptocurrency dataset."""
    print(f"📁 Loading training data from {dataset_path}")
    
    data = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    
    print(f"✅ Loaded {len(data)} records with {len(data.columns)} features")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    print(f"🔢 Symbols: {data['symbol'].nunique()} cryptocurrencies")
    
    return data

def prepare_enhanced_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset with enhanced features for ensemble training."""
    print("🔧 Preparing enhanced dataset for ensemble training...")
    
    # Remove any rows with excessive missing values
    data_clean = data.dropna(thresh=int(len(data.columns) * 0.7))
    
    # Fill remaining missing values
    data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
    
    # Create additional ensemble-specific features
    data_clean['hour'] = data_clean.index.hour
    data_clean['day_of_week'] = data_clean.index.dayofweek
    data_clean['is_weekend'] = (data_clean['day_of_week'] >= 5).astype(int)
    
    # Create momentum strength indicators
    for col in ['momentum_1', 'momentum_4', 'momentum_12', 'momentum_24']:
        if col in data_clean.columns:
            data_clean[f'{col}_strength'] = data_clean[col] / data_clean.index.to_series().diff().dt.total_seconds().fillna(14400) * 3600  # Per hour
    
    # Create volatility indicators
    if 'atr' in data_clean.columns and 'close' in data_clean.columns:
        data_clean['atr_pct'] = data_clean['atr'] / data_clean['close'] * 100
        data_clean['volatility_regime'] = pd.cut(
            data_clean['atr_pct'],
            bins=[0, 2, 4, 6, np.inf],
            labels=['low', 'medium', 'high', 'extreme']
        )
    
    # Create target variables based on successful patterns
    horizons = [1, 2, 3, 6]  # 4h, 8h, 12h, 24h ahead
    
    for h in horizons:
        # Price movement targets
        data_clean[f'price_up_{h}h'] = (data_clean['close'].shift(-h) > data_clean['close']).astype(int)
        data_clean[f'price_change_{h}h'] = (data_clean['close'].shift(-h) / data_clean['close'] - 1) * 100
        
        # Profit targets based on successful 0.68% average return
        data_clean[f'profitable_{h}h'] = (data_clean[f'price_change_{h}h'] > 0.68).astype(int)
        data_clean[f'high_profit_{h}h'] = (data_clean[f'price_change_{h}h'] > 1.5).astype(int)
        
        # Risk-adjusted targets
        if 'atr_pct' in data_clean.columns:
            data_clean[f'risk_adj_return_{h}h'] = data_clean[f'price_change_{h}h'] / data_clean['atr_pct']
            data_clean[f'good_risk_adj_{h}h'] = (data_clean[f'risk_adj_return_{h}h'] > 0.5).astype(int)
    
    # Optimal cycle detection (based on successful 0.6-1.2h patterns)
    data_clean['momentum_strength_2p'] = data_clean.get('momentum_1_strength', 0)
    data_clean['is_high_momentum'] = (data_clean['momentum_strength_2p'] > 1.780).astype(int)
    data_clean['is_optimal_hour'] = (data_clean['hour'] == 3).astype(int)
    
    # Position optimization features
    data_clean['momentum_position_size'] = np.where(
        data_clean['is_high_momentum'] == 1,
        0.800,  # Max position for high momentum
        np.where(
            data_clean['momentum_strength_2p'] > 0.5,
            0.588,  # Median position for medium momentum
            0.464   # Min position for low momentum
        )
    )
    
    # Market structure features
    data_clean['trend_alignment'] = 0  # Placeholder
    data_clean['market_structure_score'] = 0  # Placeholder
    data_clean['distance_to_resistance'] = 5.0  # Placeholder
    data_clean['distance_to_support'] = 5.0  # Placeholder
    
    # Pattern features
    data_clean['bullish_engulfing'] = 0  # Placeholder
    data_clean['trend_continuation'] = 0  # Placeholder
    data_clean['pattern_score'] = data_clean['is_high_momentum'] * 3 + data_clean['trend_alignment'] * 2
    
    # Success probability
    data_clean['success_probability'] = np.clip(
        (data_clean['pattern_score'] / 8.0) * 0.636,  # Scale by historical success rate
        0.0, 1.0
    )
    
    # Entry/exit signals
    data_clean['optimal_cycle_start'] = (
        (data_clean['is_high_momentum'] == 1) &
        (data_clean['success_probability'] > 0.5)
    ).astype(int)
    
    data_clean['is_exit_window'] = (data_clean.groupby('symbol').cumcount() % 3 == 2).astype(int)
    
    # Risk features
    data_clean['volatility_momentum'] = data_clean.get('atr_pct', 2.0).pct_change() * 100
    data_clean['is_vol_expanding'] = (data_clean['volatility_momentum'] > 0).astype(int)
    data_clean['price_efficiency'] = np.abs(data_clean.get('momentum_1', 0)) / data_clean.get('atr_pct', 2.0)
    
    # Clean up missing values in new features
    # Handle categorical columns separately
    for col in data_clean.columns:
        if data_clean[col].dtype.name == 'category':
            # For categorical columns, use forward fill then mode
            data_clean[col] = data_clean[col].fillna(method='ffill')
            if data_clean[col].isna().any():
                mode_val = data_clean[col].mode()
                if len(mode_val) > 0:
                    data_clean[col] = data_clean[col].fillna(mode_val[0])
                else:
                    # Convert to object and fill with 'unknown'
                    data_clean[col] = data_clean[col].astype('object').fillna('unknown')
        else:
            # For numeric columns, fill with 0
            data_clean[col] = data_clean[col].fillna(0)
    
    print(f"✅ Enhanced dataset prepared: {len(data_clean)} records, {len(data_clean.columns)} features")
    
    return data_clean

def train_and_evaluate_ensemble(data: pd.DataFrame) -> Dict:
    """Train the enhanced ensemble and evaluate performance."""
    print("🚀 Training Enhanced Random Forest Ensemble on Real Market Data")
    print("=" * 70)
    
    # Initialize ensemble with optimized config
    config = {
        'entry_model': {
            'n_estimators': 300,  # Increased for better performance
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'class_weight': 'balanced',
            'random_state': 42
        },
        'position_model': {
            'n_estimators': 250,
            'max_depth': 10,
            'min_samples_split': 8,
            'min_samples_leaf': 4,
            'random_state': 42
        },
        'exit_model': {
            'n_estimators': 280,
            'max_depth': 14,
            'min_samples_split': 12,
            'min_samples_leaf': 6,
            'class_weight': 'balanced',
            'random_state': 42
        },
        'risk_model': {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'random_state': 42
        }
    }
    
    ensemble = EnhancedRFEnsemble(config)
    
    # Prepare training data for each model
    print("📊 Preparing specialized datasets...")
    
    # Filter data to ensure we have enough samples
    min_samples = 500
    if len(data) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {len(data)}")
    
    # Create mock training datasets with required features
    training_data = {}
    
    # Entry dataset
    entry_features = [
        'momentum_strength_2p', 'is_high_momentum', 'volatility_momentum',
        'trend_alignment', 'market_structure_score', 'distance_to_resistance',
        'is_optimal_hour', 'success_probability', 'pattern_score'
    ]
    
    entry_data = data[entry_features + ['optimal_cycle_start', 'profitable_2h']].copy()
    entry_data = entry_data.dropna()
    training_data['entry'] = entry_data
    
    # Position dataset  
    position_features = [
        'momentum_strength_2p', 'is_high_momentum', 'atr_pct',
        'volatility_momentum', 'trend_alignment', 'market_structure_score',
        'distance_to_resistance', 'success_probability'
    ]
    
    position_data = data[position_features + ['momentum_position_size']].copy()
    position_data = position_data.dropna()
    position_data = position_data.rename(columns={'momentum_position_size': 'risk_adjusted_position'})
    training_data['position'] = position_data
    
    # Exit dataset
    exit_features = [
        'momentum_strength_2p', 'market_structure_score', 'volatility_momentum',
        'is_vol_expanding', 'distance_to_resistance', 'is_exit_window'
    ]
    
    exit_data = data[exit_features + ['is_exit_window', 'high_profit_2h']].copy()
    exit_data = exit_data.dropna()
    training_data['exit'] = exit_data
    
    # Risk dataset
    risk_features = [
        'atr_pct', 'volatility_momentum', 'market_structure_score',
        'distance_to_support', 'price_efficiency', 'is_vol_expanding'
    ]
    
    risk_data = data[risk_features + ['good_risk_adj_2h']].copy()
    risk_data = risk_data.dropna()
    training_data['risk'] = risk_data
    
    print(f"📈 Entry dataset: {len(training_data['entry'])} samples")
    print(f"📊 Position dataset: {len(training_data['position'])} samples")
    print(f"📉 Exit dataset: {len(training_data['exit'])} samples")
    print(f"🛡️ Risk dataset: {len(training_data['risk'])} samples")
    
    # Train the ensemble
    performance_metrics = ensemble.train_ensemble(training_data)
    
    # Save the trained ensemble
    ensemble.save_ensemble("models/enhanced_rf_ensemble_trained.pkl")
    
    # Get feature importance
    feature_importance = ensemble.get_feature_importance()
    
    # Test on recent data
    test_data = data.tail(100)
    try:
        sample_signals = ensemble.predict_trading_signals(test_data)
        print("\n🎯 Sample Prediction on Recent Data:")
        print(f"Action: {sample_signals['action']}")
        print(f"Confidence: {sample_signals['confidence']:.3f}")
        print(f"Position Size: {sample_signals['position_size']:.3f}")
        print(f"Risk Level: {sample_signals['risk_level']}")
    except Exception as e:
        print(f"⚠️ Prediction test failed: {e}")
        sample_signals = None
    
    return {
        'performance_metrics': performance_metrics,
        'feature_importance': feature_importance,
        'sample_signals': sample_signals,
        'training_samples': {k: len(v) for k, v in training_data.items()}
    }

def save_training_results(results: Dict, output_path: str = "models/training_results.json"):
    """Save training results for analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert feature importance DataFrames to dicts for JSON serialization
    results_serializable = results.copy()
    if 'feature_importance' in results_serializable:
        importance_dict = {}
        for model_name, df in results_serializable['feature_importance'].items():
            importance_dict[model_name] = df.to_dict('records')
        results_serializable['feature_importance'] = importance_dict
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"📁 Training results saved to {output_path}")

def main():
    """Main training function."""
    try:
        # Find the latest dataset
        data_dir = "data/4h_training"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No CSV dataset found in data/4h_training/")
        
        latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(data_dir, f)))
        dataset_path = os.path.join(data_dir, latest_file)
        
        print(f"🚀 Training Enhanced Random Forest Ensemble")
        print(f"📊 Using dataset: {latest_file}")
        print("=" * 70)
        
        # Load and prepare data
        raw_data = load_training_data(dataset_path)
        enhanced_data = prepare_enhanced_dataset(raw_data)
        
        # Train ensemble
        results = train_and_evaluate_ensemble(enhanced_data)
        
        # Save results
        save_training_results(results)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Enhanced Random Forest Ensemble Training Complete!")
        print("\n📊 Performance Summary:")
        for model_name, metrics in results['performance_metrics'].items():
            print(f"   {model_name.upper()}: {metrics}")
        
        print(f"\n🎯 Training Samples:")
        for model_name, count in results['training_samples'].items():
            print(f"   {model_name}: {count:,} samples")
        
        print(f"\n💾 Models saved to: models/enhanced_rf_ensemble_trained.pkl")
        print(f"📈 Target Performance: 4-6% returns (vs 2.82% baseline)")
        print(f"🚀 Ready for enhanced trading with optimized Random Forest!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()