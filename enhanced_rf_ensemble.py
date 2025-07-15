#!/usr/bin/env python3
"""
Enhanced Random Forest Ensemble System for High-Profit Trading

Multi-model ensemble system targeting 4-6% returns based on successful trading patterns:
- Entry Timing Model: Identifies optimal entry points (1.780%/hour momentum threshold)
- Position Sizing Model: Optimizes position sizes (0.464-0.800 BTC range)
- Exit Timing Model: Determines optimal exits (0.6-1.2h cycles)
- Risk Management Model: Manages downside risk and drawdowns

Usage: python3 enhanced_rf_ensemble.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Local imports
from enhanced_momentum_features import MomentumFeatureEngineer

class EnhancedRFEnsemble:
    """Multi-model Random Forest ensemble for high-performance trading."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_engineer = MomentumFeatureEngineer()
        self.feature_groups = self.feature_engineer.get_feature_importance_groups()
        self.is_trained = False
        
    def _get_default_config(self) -> Dict:
        """Default configuration optimized for crypto trading."""
        return {
            'entry_model': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'position_model': {
                'n_estimators': 150,
                'max_depth': 8,
                'min_samples_split': 15,
                'min_samples_leaf': 8,
                'random_state': 42
            },
            'exit_model': {
                'n_estimators': 180,
                'max_depth': 12,
                'min_samples_split': 25,
                'min_samples_leaf': 12,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'risk_model': {
                'n_estimators': 120,
                'max_depth': 6,
                'min_samples_split': 30,
                'min_samples_leaf': 15,
                'random_state': 42
            },
            'profit_targets': {
                'conservative': 0.68,  # Based on successful average
                'moderate': 1.50,
                'aggressive': 3.00
            },
            'risk_thresholds': {
                'max_position': 0.800,  # From pattern analysis
                'min_position': 0.464,
                'stop_loss': -2.0,
                'max_duration': 1.2  # Hours
            }
        }
    
    def prepare_training_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare specialized datasets for each model."""
        print("🔧 Preparing training data for ensemble models...")
        
        # Create enhanced features
        enhanced_data = self.feature_engineer.create_enhanced_features(data)
        enhanced_data = self.feature_engineer.create_target_variables(enhanced_data)
        
        # Remove rows with missing targets
        enhanced_data = enhanced_data.dropna()
        
        if len(enhanced_data) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")
        
        # Prepare entry timing dataset
        entry_features = self._get_entry_features()
        entry_data = enhanced_data[entry_features + ['optimal_cycle_start', 'profitable_2h']].copy()
        entry_data = entry_data.dropna()
        
        # Prepare position sizing dataset
        position_features = self._get_position_features()
        position_data = enhanced_data[position_features + ['risk_adjusted_position']].copy()
        position_data = position_data.dropna()
        
        # Prepare exit timing dataset
        exit_features = self._get_exit_features()
        exit_data = enhanced_data[exit_features + ['is_exit_window', 'high_profit_2h']].copy()
        exit_data = exit_data.dropna()
        
        # Prepare risk management dataset
        risk_features = self._get_risk_features()
        risk_data = enhanced_data[risk_features + ['good_risk_adj_2h']].copy()
        risk_data = risk_data.dropna()
        
        datasets = {
            'entry': entry_data,
            'position': position_data,
            'exit': exit_data,
            'risk': risk_data
        }
        
        print(f"✅ Training data prepared:")
        for name, df in datasets.items():
            print(f"   {name}: {len(df)} samples, {len(df.columns)-1} features")
        
        return datasets
    
    def _get_entry_features(self) -> List[str]:
        """Get features for entry timing model."""
        return [
            # Momentum features
            'momentum_2p', 'momentum_4p', 'momentum_strength_2p', 'is_high_momentum',
            'ema_momentum_diff', 'momentum_acceleration',
            
            # Volume features
            'volume_momentum', 'obv_momentum', 'is_volume_increasing',
            
            # Pattern features
            'bullish_engulfing', 'trend_continuation', 'pattern_score',
            
            # Structure features
            'market_structure_score', 'trend_alignment', 'distance_to_resistance',
            
            # Timing features
            'is_optimal_hour', 'is_asian_session',
            
            # Success indicators
            'success_probability', 'is_position_entry_optimal'
        ]
    
    def _get_position_features(self) -> List[str]:
        """Get features for position sizing model."""
        return [
            # Momentum strength
            'momentum_strength_2p', 'is_high_momentum', 'price_efficiency',
            
            # Volatility
            'atr_pct', 'volatility_momentum', 'is_vol_expanding',
            
            # Market conditions
            'trend_alignment', 'market_structure_score',
            
            # Risk indicators
            'distance_to_resistance', 'distance_to_support',
            
            # Success probability
            'success_probability', 'confidence_level'
        ]
    
    def _get_exit_features(self) -> List[str]:
        """Get features for exit timing model."""
        return [
            # Current momentum
            'momentum_2p', 'momentum_acceleration', 'price_velocity',
            
            # Market structure changes
            'market_structure_score', 'higher_high', 'lower_high',
            
            # Volatility expansion
            'volatility_momentum', 'is_vol_expanding',
            
            # Pattern completion
            'is_shooting_star', 'gap_up',
            
            # Time factors
            'cycles_since_entry', 'is_exit_window'
        ]
    
    def _get_risk_features(self) -> List[str]:
        """Get features for risk management model."""
        return [
            # Volatility risk
            'atr_pct', 'volatility_momentum', 'price_efficiency',
            
            # Market risk
            'market_structure_score', 'distance_to_support',
            
            # Momentum risk
            'momentum_acceleration', 'price_acceleration',
            
            # Pattern risk
            'is_doji', 'gap_down', 'bearish_structure'
        ]
    
    def train_ensemble(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Train all models in the ensemble."""
        print("🚀 Training Enhanced Random Forest Ensemble...")
        print("=" * 60)
        
        performance_metrics = {}
        
        # Train Entry Timing Model
        print("📈 Training Entry Timing Model...")
        entry_performance = self._train_entry_model(training_data['entry'])
        performance_metrics['entry'] = entry_performance
        
        # Train Position Sizing Model
        print("📊 Training Position Sizing Model...")
        position_performance = self._train_position_model(training_data['position'])
        performance_metrics['position'] = position_performance
        
        # Train Exit Timing Model
        print("📉 Training Exit Timing Model...")
        exit_performance = self._train_exit_model(training_data['exit'])
        performance_metrics['exit'] = exit_performance
        
        # Train Risk Management Model
        print("🛡️ Training Risk Management Model...")
        risk_performance = self._train_risk_model(training_data['risk'])
        performance_metrics['risk'] = risk_performance
        
        self.is_trained = True
        
        print("=" * 60)
        print("✅ Ensemble training complete!")
        
        return performance_metrics
    
    def _train_entry_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the entry timing model."""
        features = self._get_entry_features()
        X = data[features]
        y_cycle = data['optimal_cycle_start']
        y_profit = data['profitable_2h']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['entry'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_cycle, test_size=0.2, random_state=42, stratify=y_cycle
        )
        
        # Train model
        model = RandomForestClassifier(**self.config['entry_model'])
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        
        self.models['entry'] = model
        
        print(f"   ✅ Entry Model - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
        return {'accuracy': accuracy, 'precision': precision}
    
    def _train_position_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the position sizing model."""
        features = self._get_position_features()
        X = data[features]
        
        # Convert categorical to numeric if needed
        if 'confidence_level' in X.columns:
            le = LabelEncoder()
            X['confidence_level'] = le.fit_transform(X['confidence_level'].astype(str))
        
        y = data['risk_adjusted_position']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['position'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(**self.config['position_model'])
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        
        self.models['position'] = model
        
        print(f"   ✅ Position Model - R²: {score:.3f}, MSE: {mse:.4f}")
        return {'r2_score': score, 'mse': mse}
    
    def _train_exit_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the exit timing model."""
        features = self._get_exit_features()
        X = data[features]
        y_exit = data['is_exit_window']
        y_profit = data['high_profit_2h']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['exit'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_exit, test_size=0.2, random_state=42, stratify=y_exit
        )
        
        # Train model
        model = RandomForestClassifier(**self.config['exit_model'])
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        
        self.models['exit'] = model
        
        print(f"   ✅ Exit Model - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
        return {'accuracy': accuracy, 'precision': precision}
    
    def _train_risk_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the risk management model."""
        features = self._get_risk_features()
        X = data[features]
        y = data['good_risk_adj_2h']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['risk'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(**self.config['risk_model'])
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        
        self.models['risk'] = model
        
        print(f"   ✅ Risk Model - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
        return {'accuracy': accuracy, 'precision': precision}
    
    def predict_trading_signals(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trading signals from ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Create enhanced features
        enhanced_data = self.feature_engineer.create_enhanced_features(current_data)
        latest_data = enhanced_data.iloc[-1:]  # Get latest row
        
        # Entry signal
        entry_features = latest_data[self._get_entry_features()]
        entry_scaled = self.scalers['entry'].transform(entry_features)
        entry_prob = self.models['entry'].predict_proba(entry_scaled)[0]
        entry_signal = entry_prob[1] > 0.6  # 60% confidence threshold
        
        # Position sizing
        position_features = latest_data[self._get_position_features()]
        if 'confidence_level' in position_features.columns:
            position_features['confidence_level'] = 2  # Default to medium
        position_scaled = self.scalers['position'].transform(position_features)
        position_size = self.models['position'].predict(position_scaled)[0]
        
        # Exit signal
        exit_features = latest_data[self._get_exit_features()]
        exit_scaled = self.scalers['exit'].transform(exit_features)
        exit_prob = self.models['exit'].predict_proba(exit_scaled)[0]
        exit_signal = exit_prob[1] > 0.5  # 50% confidence threshold
        
        # Risk assessment
        risk_features = latest_data[self._get_risk_features()]
        risk_scaled = self.scalers['risk'].transform(risk_features)
        risk_prob = self.models['risk'].predict_proba(risk_scaled)[0]
        risk_acceptable = risk_prob[1] > 0.4  # 40% confidence threshold
        
        # Combined signal
        combined_signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'risk_level': 'HIGH',
            'details': {
                'entry_probability': entry_prob[1],
                'exit_probability': exit_prob[1],
                'risk_probability': risk_prob[1],
                'position_size_raw': position_size
            }
        }
        
        # Determine action
        if entry_signal and risk_acceptable and not exit_signal:
            combined_signal['action'] = 'BUY'
            combined_signal['position_size'] = np.clip(position_size, 0.464, 0.800)
            combined_signal['confidence'] = entry_prob[1]
            combined_signal['risk_level'] = 'ACCEPTABLE' if risk_prob[1] > 0.6 else 'MODERATE'
            
        elif exit_signal:
            combined_signal['action'] = 'SELL'
            combined_signal['confidence'] = exit_prob[1]
            combined_signal['risk_level'] = 'PROFIT_TAKING'
            
        elif not risk_acceptable:
            combined_signal['action'] = 'RISK_OFF'
            combined_signal['risk_level'] = 'HIGH'
        
        return combined_signal
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance for all models."""
        if not self.is_trained:
            raise ValueError("Models must be trained first")
        
        importance_data = {}
        
        for model_name in ['entry', 'exit', 'risk']:
            if model_name in self.models:
                features = getattr(self, f'_get_{model_name}_features')()
                importance = self.models[model_name].feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                importance_data[model_name] = importance_df
        
        return importance_data
    
    def save_ensemble(self, filepath: str = "models/enhanced_rf_ensemble.pkl"):
        """Save the trained ensemble."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        ensemble_data = {
            'models': self.models,
            'scalers': self.scalers,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        print(f"✅ Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str = "models/enhanced_rf_ensemble.pkl"):
        """Load a trained ensemble."""
        ensemble_data = joblib.load(filepath)
        
        self.models = ensemble_data['models']
        self.scalers = ensemble_data['scalers']
        self.config = ensemble_data['config']
        self.is_trained = ensemble_data['is_trained']
        
        print(f"✅ Ensemble loaded from {filepath}")

def main():
    """Test the enhanced Random Forest ensemble."""
    print("🚀 Testing Enhanced Random Forest Ensemble")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='4H')
    
    sample_data = pd.DataFrame({
        'open': 120000 + np.cumsum(np.random.randn(1000) * 100),
        'high': 120000 + np.cumsum(np.random.randn(1000) * 100) + 200,
        'low': 120000 + np.cumsum(np.random.randn(1000) * 100) - 200,
        'close': 120000 + np.cumsum(np.random.randn(1000) * 100),
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Fix OHLC logic
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    try:
        # Initialize ensemble
        ensemble = EnhancedRFEnsemble()
        
        # Prepare training data
        training_data = ensemble.prepare_training_data(sample_data)
        
        # Train ensemble
        performance = ensemble.train_ensemble(training_data)
        
        # Test prediction
        test_data = sample_data.tail(20)
        signals = ensemble.predict_trading_signals(test_data)
        
        print("=" * 60)
        print("🎯 Sample Trading Signal:")
        print(f"Action: {signals['action']}")
        print(f"Confidence: {signals['confidence']:.3f}")
        print(f"Position Size: {signals['position_size']:.3f}")
        print(f"Risk Level: {signals['risk_level']}")
        
        # Feature importance
        importance = ensemble.get_feature_importance()
        print("\n📊 Top 5 Features by Model:")
        for model_name, df in importance.items():
            print(f"\n{model_name.upper()} MODEL:")
            print(df.head().to_string(index=False))
        
        print("\n✅ Enhanced Random Forest Ensemble test complete!")
        
    except Exception as e:
        print(f"❌ Error in ensemble test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()