#!/usr/bin/env python3
"""
Simplified Enhanced Random Forest Trainer

Trains Random Forest models using available features from the 4-hour dataset
to achieve higher profit percentages than the 2.82% baseline.

Usage: python3 simplified_rf_trainer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimplifiedRFTrainer:
    """Simplified Random Forest trainer using available dataset features."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, dataset_path: str) -> pd.DataFrame:
        """Load and prepare the 4-hour cryptocurrency dataset."""
        print(f"📁 Loading data from {dataset_path}")
        
        data = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        
        # Remove any problematic columns
        if 'Datetime' in data.columns:
            data = data.drop('Datetime', axis=1)
        
        # Convert categorical movement columns to numeric
        movement_cols = [col for col in data.columns if 'movement_' in col]
        le = LabelEncoder()
        for col in movement_cols:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                data[col] = le.fit_transform(data[col].astype(str))
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create enhanced features based on successful trading patterns
        data = self._add_pattern_features(data)
        
        print(f"✅ Loaded {len(data)} records with {len(data.columns)} features")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
        
        return data
    
    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern features based on successful trading analysis."""
        
        # Momentum strength (based on 1.780%/hour threshold)
        data['momentum_strength'] = data['momentum_1'] / 4  # Convert to per-hour
        data['is_high_momentum'] = (data['momentum_strength'] > 1.780).astype(int)
        
        # Optimal timing (3 AM was best in analysis)
        data['is_optimal_hour'] = (data['hour'] == 3).astype(int)
        
        # Position sizing indicators (0.464-0.800 BTC range)
        data['position_size_signal'] = np.where(
            data['is_high_momentum'] == 1,
            0.800,  # Max position for high momentum
            np.where(
                data['momentum_strength'] > 0.5,
                0.588,  # Median position for medium momentum  
                0.464   # Min position for low momentum
            )
        )
        
        # Success probability based on multiple indicators
        data['success_score'] = (
            data['is_high_momentum'] * 3 +
            (data['market_structure'] >= 1).astype(int) * 2 +
            data['is_optimal_hour'] * 1 +
            (data['volume_ratio'] > 1.0).astype(int) * 1 +
            (data['rsi'] < 70).astype(int) * 1  # Not overbought
        )
        
        data['success_probability'] = np.clip(data['success_score'] / 8.0 * 0.636, 0.0, 1.0)
        
        # Entry signal (combination of favorable conditions)
        data['entry_signal'] = (
            (data['is_high_momentum'] == 1) &
            (data['market_structure'] >= 1) &
            (data['rsi'] < 70) &
            (data['volume_ratio'] > 1.0)
        ).astype(int)
        
        # Exit signal (based on optimal 0.6-1.2h cycle duration)
        data['exit_signal'] = (
            (data['rsi'] > 70) |  # Overbought
            (data['momentum_strength'] < 0) |  # Momentum turning negative
            (data['bb_position'] > 0.8)  # Near upper Bollinger Band
        ).astype(int)
        
        # Risk indicators
        data['high_volatility'] = (data['volatility'] > data['volatility'].rolling(20).quantile(0.8)).astype(int)
        data['risk_score'] = (
            data['high_volatility'] * 2 +
            (data['rsi'] > 80).astype(int) * 1 +
            (data['market_structure'] < 0).astype(int) * 1
        )
        
        return data
    
    def prepare_training_datasets(self, data: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare specialized datasets for different trading tasks."""
        print("🔧 Preparing training datasets...")
        
        datasets = {}
        
        # Entry Timing Model
        entry_features = [
            'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
            'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
            'hour', 'day_of_week', 'volatility', 'trend_strength'
        ]
        X_entry = data[entry_features].copy()
        y_entry = data['price_up_2h']  # Predict if price goes up in next 8 hours
        datasets['entry'] = (X_entry, y_entry)
        self.feature_names['entry'] = entry_features
        
        # Position Sizing Model (regression)
        position_features = [
            'momentum_strength', 'is_high_momentum', 'volatility', 'atr',
            'rsi', 'market_structure', 'success_probability', 'volume_ratio'
        ]
        X_position = data[position_features].copy()
        y_position = data['position_size_signal']
        datasets['position'] = (X_position, y_position)
        self.feature_names['position'] = position_features
        
        # Exit Timing Model
        exit_features = [
            'momentum_1', 'rsi', 'bb_position', 'market_structure',
            'volatility', 'volume_ratio', 'price_vs_sma10', 'macd'
        ]
        X_exit = data[exit_features].copy()
        y_exit = data['exit_signal']
        datasets['exit'] = (X_exit, y_exit)
        self.feature_names['exit'] = exit_features
        
        # Profit Prediction Model
        profit_features = [
            'momentum_1', 'momentum_4', 'rsi', 'macd', 'bb_position',
            'market_structure', 'volume_ratio', 'volatility', 'success_probability'
        ]
        X_profit = data[profit_features].copy()
        y_profit = (data['price_change_2h'] > 0.68).astype(int)  # Based on successful 0.68% average
        datasets['profit'] = (X_profit, y_profit)
        self.feature_names['profit'] = profit_features
        
        print(f"✅ Prepared {len(datasets)} specialized datasets")
        for name, (X, y) in datasets.items():
            print(f"   {name}: {len(X)} samples, {len(X.columns)} features")
        
        return datasets
    
    def train_models(self, datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Dict]:
        """Train all Random Forest models."""
        print("🚀 Training Enhanced Random Forest Models...")
        print("=" * 60)
        
        performance_metrics = {}
        
        for model_name, (X, y) in datasets.items():
            print(f"📊 Training {model_name.upper()} model...")
            
            # Remove rows with missing targets
            mask = ~y.isna()
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 100:
                print(f"   ⚠️ Insufficient data for {model_name}: {len(X_clean)} samples")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42, 
                stratify=y_clean if model_name != 'position' else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
            
            # Choose model type
            if model_name == 'position':
                # Regression for position sizing
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate regression
                score = model.score(X_test_scaled, y_test)
                y_pred = model.predict(X_test_scaled)
                mse = np.mean((y_test - y_pred) ** 2)
                
                performance_metrics[model_name] = {
                    'r2_score': score,
                    'mse': mse,
                    'samples': len(X_clean)
                }
                print(f"   ✅ {model_name}: R² = {score:.3f}, MSE = {mse:.4f}")
                
            else:
                # Classification for other models
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate classification
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                performance_metrics[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'samples': len(X_clean)
                }
                print(f"   ✅ {model_name}: Acc = {accuracy:.3f}, Prec = {precision:.3f}, Rec = {recall:.3f}")
            
            self.models[model_name] = model
        
        self.is_trained = True
        print("=" * 60)
        print("✅ All models trained successfully!")
        
        return performance_metrics
    
    def predict_trading_signals(self, current_data: pd.DataFrame) -> Dict:
        """Generate trading signals from the ensemble."""
        if not self.is_trained:
            raise ValueError("Models must be trained first")
        
        # Add pattern features to current data
        current_data_enhanced = self._add_pattern_features(current_data.copy())
        latest_row = current_data_enhanced.iloc[-1]
        
        signals = {}
        
        # Entry signal
        if 'entry' in self.models:
            entry_features = [latest_row[f] for f in self.feature_names['entry']]
            entry_scaled = self.scalers['entry'].transform([entry_features])
            entry_prob = self.models['entry'].predict_proba(entry_scaled)[0][1]
            signals['entry_probability'] = entry_prob
            signals['should_enter'] = entry_prob > 0.6
        
        # Position sizing
        if 'position' in self.models:
            position_features = [latest_row[f] for f in self.feature_names['position']]
            position_scaled = self.scalers['position'].transform([position_features])
            position_size = self.models['position'].predict(position_scaled)[0]
            signals['position_size'] = np.clip(position_size, 0.464, 0.800)
        
        # Exit signal
        if 'exit' in self.models:
            exit_features = [latest_row[f] for f in self.feature_names['exit']]
            exit_scaled = self.scalers['exit'].transform([exit_features])
            exit_prob = self.models['exit'].predict_proba(exit_scaled)[0][1]
            signals['exit_probability'] = exit_prob
            signals['should_exit'] = exit_prob > 0.5
        
        # Profit prediction
        if 'profit' in self.models:
            profit_features = [latest_row[f] for f in self.feature_names['profit']]
            profit_scaled = self.scalers['profit'].transform([profit_features])
            profit_prob = self.models['profit'].predict_proba(profit_scaled)[0][1]
            signals['profit_probability'] = profit_prob
        
        # Combined action
        action = 'HOLD'
        confidence = 0.0
        
        if signals.get('should_enter', False) and not signals.get('should_exit', False):
            action = 'BUY'
            confidence = signals.get('entry_probability', 0) * signals.get('profit_probability', 0.5)
        elif signals.get('should_exit', False):
            action = 'SELL'
            confidence = signals.get('exit_probability', 0)
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': signals.get('position_size', 0.588),
            'details': signals
        }
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance for all models."""
        importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                features = self.feature_names[model_name]
                importance = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                importance_data[model_name] = importance_df
        
        return importance_data
    
    def save_models(self, filepath: str = "models/enhanced_rf_models.pkl"):
        """Save all trained models."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
        
        return filepath

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
        
        print("🚀 Enhanced Random Forest Training for Higher Profits")
        print("=" * 70)
        print(f"📊 Dataset: {latest_file}")
        print(f"🎯 Target: 4-6% returns (vs 2.82% baseline)")
        print("=" * 70)
        
        # Initialize trainer
        trainer = SimplifiedRFTrainer()
        
        # Load and prepare data
        data = trainer.load_and_prepare_data(dataset_path)
        
        # Prepare training datasets
        datasets = trainer.prepare_training_datasets(data)
        
        # Train models
        performance = trainer.train_models(datasets)
        
        # Save models
        model_path = trainer.save_models()
        
        # Test prediction on recent data
        test_data = data.tail(50)
        signals = trainer.predict_trading_signals(test_data)
        
        # Get feature importance
        importance = trainer.get_feature_importance()
        
        # Results summary
        print("\n" + "=" * 70)
        print("✅ Enhanced Random Forest Training Complete!")
        print("\n📊 Model Performance:")
        for model_name, metrics in performance.items():
            print(f"   {model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"     {metric}: {value:.3f}")
                else:
                    print(f"     {metric}: {value}")
        
        print(f"\n🎯 Sample Trading Signal:")
        print(f"   Action: {signals['action']}")
        print(f"   Confidence: {signals['confidence']:.3f}")
        print(f"   Position Size: {signals['position_size']:.3f}")
        
        print(f"\n📈 Top 3 Important Features by Model:")
        for model_name, df in importance.items():
            print(f"   {model_name.upper()}: {', '.join(df.head(3)['feature'].tolist())}")
        
        print(f"\n💾 Models saved to: {model_path}")
        print(f"🚀 Ready for enhanced trading with 4-6% target returns!")
        
        # Save results summary
        results = {
            'performance': performance,
            'sample_signals': signals,
            'feature_importance': {k: v.to_dict('records') for k, v in importance.items()},
            'training_timestamp': datetime.now().isoformat(),
            'target_returns': '4-6%',
            'baseline_return': '2.82%'
        }
        
        with open('models/training_summary.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📋 Training summary saved to: models/training_summary.json")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()