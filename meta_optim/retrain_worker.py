#!/usr/bin/env python3
"""
Automated Model Retraining and Evaluation Worker

Handles model training, backtesting, and evaluation for the meta-optimization layer.
Designed to work with parameter configurations from Hyperband runner.

Usage: from retrain_worker import RetrainWorker
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RetrainWorker:
    """Automated model retraining and evaluation for meta-optimization."""
    
    def __init__(self, data_path: str = None):
        """Initialize retrain worker."""
        self.data_path = data_path or "data/4h_training/crypto_4h_dataset_20250714_130201.csv"
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.training_data = None
        self.validation_data = None
        
        # Load training data
        self._load_training_data()
    
    def _load_training_data(self):
        """Load and prepare training data."""
        try:
            if not os.path.exists(self.data_path):
                print(f"⚠️ Data file not found: {self.data_path}")
                self._create_synthetic_data()
                return
            
            print(f"📊 Loading training data from {self.data_path}")
            data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            
            # Basic data preparation
            data = self._prepare_training_data(data)
            
            # Split into training and validation (temporal split)
            split_date = data.index[int(len(data) * 0.8)]
            self.training_data = data[data.index <= split_date]
            self.validation_data = data[data.index > split_date]
            
            print(f"✅ Training data loaded: {len(self.training_data)} train, {len(self.validation_data)} validation")
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic data for testing when real data unavailable."""
        print("🔄 Creating synthetic training data for testing...")
        
        # Generate synthetic crypto data
        np.random.seed(42)
        n_samples = 2000
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='4H')
        
        # Base price series with trend and volatility
        price_base = 50000
        price_trend = np.cumsum(np.random.randn(n_samples) * 0.02)
        price_noise = np.random.randn(n_samples) * 0.05
        
        synthetic_data = pd.DataFrame({
            'open': price_base * (1 + price_trend + price_noise),
            'high': price_base * (1 + price_trend + price_noise + np.abs(np.random.randn(n_samples) * 0.02)),
            'low': price_base * (1 + price_trend + price_noise - np.abs(np.random.randn(n_samples) * 0.02)),
            'close': price_base * (1 + price_trend + price_noise),
            'volume': np.random.randint(1000, 100000, n_samples),
            'symbol': np.random.choice(['BTC', 'ETH', 'ADA'], n_samples)
        }, index=dates)
        
        # Fix OHLC relationships
        synthetic_data['high'] = np.maximum(
            synthetic_data[['open', 'close']].max(axis=1), 
            synthetic_data['high']
        )
        synthetic_data['low'] = np.minimum(
            synthetic_data[['open', 'close']].min(axis=1), 
            synthetic_data['low']
        )
        
        # Prepare the synthetic data
        synthetic_data = self._prepare_training_data(synthetic_data)
        
        # Split for training/validation
        split_idx = int(len(synthetic_data) * 0.8)
        self.training_data = synthetic_data.iloc[:split_idx]
        self.validation_data = synthetic_data.iloc[split_idx:]
        
        print(f"✅ Synthetic data created: {len(self.training_data)} train, {len(self.validation_data)} validation")
    
    def _prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with features and targets for training."""
        
        # Basic features
        data['momentum_1'] = data['close'].pct_change(1) * 100
        data['momentum_4'] = data['close'].pct_change(4) * 100
        data['momentum_12'] = data['close'].pct_change(12) * 100
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        
        # Bollinger Bands
        bb_period = 20
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        bb_std = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume features
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Market structure
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['higher_low'] = (data['low'] > data['low'].shift(1)).astype(int)
        data['market_structure'] = data['higher_high'] + data['higher_low'] - 1
        
        # Volatility
        data['volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        data['atr'] = self._calculate_atr(data)
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        
        # Trend features
        sma_10 = data['close'].rolling(10).mean()
        sma_20 = data['close'].rolling(20).mean()
        data['trend_strength'] = np.abs(sma_10 - sma_20) / data['close']
        data['price_vs_sma10'] = (data['close'] - sma_10) / sma_10
        
        # Pattern features
        data['momentum_strength'] = data['momentum_1'] / 4  # Convert to hourly
        data['is_high_momentum'] = (data['momentum_strength'] > 1.780).astype(int)
        data['is_optimal_hour'] = (data['hour'] == 3).astype(int)
        
        data['success_score'] = (
            data['is_high_momentum'] * 3 +
            (data['market_structure'] >= 0).astype(int) * 2 +
            data['is_optimal_hour'] * 1 +
            (data['volume_ratio'] > 1.0).astype(int) * 1 +
            (data['rsi'] < 70).astype(int) * 1
        )
        
        data['success_probability'] = np.clip(data['success_score'] / 8.0 * 0.636, 0.0, 1.0)
        
        # Target variables
        for horizon in [1, 2, 4]:
            data[f'price_up_{horizon}h'] = (data['close'].shift(-horizon) > data['close']).astype(int)
            data[f'price_change_{horizon}h'] = (data['close'].shift(-horizon) / data['close'] - 1) * 100
            data[f'profitable_{horizon}h'] = (data[f'price_change_{horizon}h'] > 0.68).astype(int)
        
        # Position sizing targets
        data['position_size_signal'] = np.where(
            data['is_high_momentum'] == 1,
            0.800,
            np.where(data['momentum_strength'] > 0.5, 0.588, 0.464)
        )
        
        # Clean data
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return true_range.rolling(period).mean()
    
    def train_and_evaluate(self, config: Dict[str, Any], training_fraction: float = 1.0) -> Dict[str, Any]:
        """Train models with given configuration and evaluate performance."""
        
        try:
            # Sample training data based on fraction
            if training_fraction < 1.0:
                sample_size = int(len(self.training_data) * training_fraction)
                train_data = self.training_data.sample(n=sample_size, random_state=42)
            else:
                train_data = self.training_data
            
            # Train models
            models = self._train_models(train_data, config)
            
            # Evaluate on validation data
            backtest_results = self._backtest_models(models, self.validation_data, config)
            
            return backtest_results
            
        except Exception as e:
            print(f"❌ Training/evaluation error: {e}")
            return {
                'trades': [],
                'portfolio_values': [100000],
                'error': str(e)
            }
    
    def _train_models(self, train_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train Random Forest models with given configuration."""
        
        models = {}
        scalers = {}
        
        # Entry model
        if 'entry_model' in config:
            entry_features = [
                'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
                'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
                'hour', 'day_of_week', 'volatility', 'trend_strength'
            ]
            
            X_entry = train_data[entry_features].dropna()
            y_entry = train_data.loc[X_entry.index, 'price_up_2h']
            
            if len(X_entry) > 50:
                scaler = StandardScaler()
                X_entry_scaled = scaler.fit_transform(X_entry)
                
                entry_model = RandomForestClassifier(
                    random_state=42,
                    **config['entry_model']
                )
                entry_model.fit(X_entry_scaled, y_entry)
                
                models['entry'] = entry_model
                scalers['entry'] = scaler
        
        # Position model
        if 'position_model' in config:
            position_features = [
                'momentum_strength', 'is_high_momentum', 'volatility', 'atr',
                'rsi', 'market_structure', 'success_probability', 'volume_ratio'
            ]
            
            X_position = train_data[position_features].dropna()
            y_position = train_data.loc[X_position.index, 'position_size_signal']
            
            if len(X_position) > 50:
                scaler = StandardScaler()
                X_position_scaled = scaler.fit_transform(X_position)
                
                position_model = RandomForestRegressor(
                    random_state=42,
                    **config['position_model']
                )
                position_model.fit(X_position_scaled, y_position)
                
                models['position'] = position_model
                scalers['position'] = scaler
        
        # Exit model
        if 'exit_model' in config:
            exit_features = [
                'momentum_1', 'rsi', 'bb_position', 'market_structure',
                'volatility', 'volume_ratio', 'price_vs_sma10', 'macd'
            ]
            
            # Create exit signals (simplified)
            exit_signals = (
                (train_data['rsi'] > 70) |
                (train_data['momentum_strength'] < 0) |
                (train_data['bb_position'] > 0.8)
            ).astype(int)
            
            X_exit = train_data[exit_features].dropna()
            y_exit = exit_signals.loc[X_exit.index]
            
            if len(X_exit) > 50:
                scaler = StandardScaler()
                X_exit_scaled = scaler.fit_transform(X_exit)
                
                exit_model = RandomForestClassifier(
                    random_state=42,
                    **config['exit_model']
                )
                exit_model.fit(X_exit_scaled, y_exit)
                
                models['exit'] = exit_model
                scalers['exit'] = scaler
        
        # Profit model
        if 'profit_model' in config:
            profit_features = [
                'momentum_1', 'momentum_4', 'rsi', 'macd', 'bb_position',
                'market_structure', 'volume_ratio', 'volatility', 'success_probability'
            ]
            
            X_profit = train_data[profit_features].dropna()
            y_profit = train_data.loc[X_profit.index, 'profitable_2h']
            
            if len(X_profit) > 50:
                scaler = StandardScaler()
                X_profit_scaled = scaler.fit_transform(X_profit)
                
                profit_model = RandomForestClassifier(
                    random_state=42,
                    **config['profit_model']
                )
                profit_model.fit(X_profit_scaled, y_profit)
                
                models['profit'] = profit_model
                scalers['profit'] = scaler
        
        return {'models': models, 'scalers': scalers}
    
    def _backtest_models(self, trained_models: Dict[str, Any], test_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest trained models on validation data."""
        
        models = trained_models['models']
        scalers = trained_models['scalers']
        
        # Trading parameters
        trading_params = config.get('trading_params', {})
        momentum_threshold = trading_params.get('momentum_threshold', 1.78)
        position_range_min = trading_params.get('position_range_min', 0.464)
        position_range_max = trading_params.get('position_range_max', 0.800)
        confidence_threshold = trading_params.get('confidence_threshold', 0.6)
        exit_threshold = trading_params.get('exit_threshold', 0.5)
        
        # Initialize backtesting
        portfolio_values = [100000]
        trades = []
        cash = 100000
        btc_position = 0
        
        # Feature lists (same as training)
        entry_features = [
            'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
            'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
            'hour', 'day_of_week', 'volatility', 'trend_strength'
        ]
        
        position_features = [
            'momentum_strength', 'is_high_momentum', 'volatility', 'atr',
            'rsi', 'market_structure', 'success_probability', 'volume_ratio'
        ]
        
        exit_features = [
            'momentum_1', 'rsi', 'bb_position', 'market_structure',
            'volatility', 'volume_ratio', 'price_vs_sma10', 'macd'
        ]
        
        profit_features = [
            'momentum_1', 'momentum_4', 'rsi', 'macd', 'bb_position',
            'market_structure', 'volume_ratio', 'volatility', 'success_probability'
        ]
        
        # Backtest loop
        for idx, row in test_data.iterrows():
            current_price = row['close']
            
            # Calculate current portfolio value
            portfolio_value = cash + (btc_position * current_price)
            portfolio_values.append(portfolio_value)
            
            try:
                # Generate signals
                signals = {}
                
                # Entry signal
                if 'entry' in models:
                    entry_data = [[row[f] for f in entry_features]]
                    entry_scaled = scalers['entry'].transform(entry_data)
                    entry_prob = models['entry'].predict_proba(entry_scaled)[0]
                    signals['entry_prob'] = entry_prob[1] if len(entry_prob) > 1 else entry_prob[0]
                    signals['should_enter'] = signals['entry_prob'] > confidence_threshold
                else:
                    signals['should_enter'] = row['momentum_strength'] > momentum_threshold
                    signals['entry_prob'] = 0.7
                
                # Position sizing
                if 'position' in models:
                    position_data = [[row[f] for f in position_features]]
                    position_scaled = scalers['position'].transform(position_data)
                    position_size = models['position'].predict(position_scaled)[0]
                    signals['position_size'] = np.clip(position_size, position_range_min, position_range_max)
                else:
                    signals['position_size'] = 0.588  # Default
                
                # Exit signal
                if 'exit' in models:
                    exit_data = [[row[f] for f in exit_features]]
                    exit_scaled = scalers['exit'].transform(exit_data)
                    exit_prob = models['exit'].predict_proba(exit_scaled)[0]
                    signals['exit_prob'] = exit_prob[1] if len(exit_prob) > 1 else exit_prob[0]
                    signals['should_exit'] = signals['exit_prob'] > exit_threshold
                else:
                    signals['should_exit'] = (row['rsi'] > 70) or (row['bb_position'] > 0.8)
                    signals['exit_prob'] = 0.6
                
                # Execute trades
                if signals['should_enter'] and btc_position == 0 and cash > 1000:
                    # Buy
                    target_value = portfolio_value * signals['position_size']
                    btc_to_buy = min(target_value, cash * 0.99) / current_price
                    
                    if btc_to_buy > 0.001:
                        cost = btc_to_buy * current_price
                        cash -= cost
                        btc_position += btc_to_buy
                        
                        trades.append({
                            'timestamp': idx,
                            'action': 'BUY',
                            'quantity': btc_to_buy,
                            'price': current_price,
                            'value': cost,
                            'pnl': 0
                        })
                
                elif signals['should_exit'] and btc_position > 0.001:
                    # Sell
                    proceeds = btc_position * current_price
                    original_cost = sum(t['value'] for t in trades if t['action'] == 'BUY')
                    pnl = proceeds - original_cost
                    
                    cash += proceeds
                    
                    trades.append({
                        'timestamp': idx,
                        'action': 'SELL',
                        'quantity': btc_position,
                        'price': current_price,
                        'value': proceeds,
                        'pnl': pnl
                    })
                    
                    btc_position = 0
                    
            except Exception as e:
                # Skip this iteration on error
                continue
        
        # Final portfolio value
        final_portfolio_value = cash + (btc_position * test_data['close'].iloc[-1])
        portfolio_values.append(final_portfolio_value)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_value': final_portfolio_value,
            'total_return': (final_portfolio_value - 100000) / 100000 * 100,
            'num_trades': len(trades)
        }

def main():
    """Test the retrain worker."""
    
    print("🔄 Testing Retrain Worker")
    print("=" * 40)
    
    # Test configuration
    test_config = {
        'entry_model': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'position_model': {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'exit_model': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'profit_model': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'trading_params': {
            'momentum_threshold': 1.78,
            'position_range_min': 0.464,
            'position_range_max': 0.800,
            'confidence_threshold': 0.6,
            'exit_threshold': 0.5
        }
    }
    
    try:
        # Initialize worker
        worker = RetrainWorker()
        
        # Train and evaluate
        results = worker.train_and_evaluate(test_config)
        
        print(f"✅ Training complete!")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:+.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()