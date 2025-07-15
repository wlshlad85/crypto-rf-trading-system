"""Enhanced Random Forest model optimized for high-frequency minute-level cryptocurrency trading."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from utils.config import ModelConfig
from .random_forest_model import CryptoRandomForestModel


class MinuteRandomForestModel:
    """Enhanced Random Forest model optimized for minute-level high-frequency trading."""
    
    def __init__(self, config: ModelConfig = None, hf_config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.hf_config = hf_config or self._get_default_hf_config()
        self.logger = logging.getLogger(__name__)
        
        # Specialized models for different prediction horizons
        self.horizon_models = {}  # 1min, 5min, 15min, 1hour
        self.scalers = {}
        self.is_fitted = False
        
        # Online learning components
        self.online_buffer = {}
        self.model_performance_tracker = {}
        
        # Memory management
        self.max_memory_usage_gb = self.hf_config.get('max_memory_gb', 8)
        self.feature_selection_threshold = 0.001  # Remove features with importance < 0.1%
        
        # Specialized feature importance tracking
        self.feature_importance_history = []
        self.prediction_horizons = [1, 5, 15, 60]  # minutes
        
    def _get_default_config(self) -> ModelConfig:
        """Get default configuration optimized for minute data."""
        from types import SimpleNamespace
        config = SimpleNamespace()
        
        # High-frequency optimized parameters
        config.n_estimators = 150  # Reduced for speed
        config.max_depth = 12  # Moderate depth to prevent overfitting on noise
        config.min_samples_split = 10  # Higher to reduce overfitting
        config.min_samples_leaf = 5   # Higher for stability
        config.max_features = 'sqrt'  # Feature subsampling for variance
        config.bootstrap = True
        config.n_jobs = -1
        config.random_state = 42
        
        # High-frequency specific settings
        config.target_type = "returns"
        config.target_horizon = 5  # 5 minutes default
        config.scoring = 'neg_mean_squared_error'
        config.cv_folds = 3  # Reduced for speed
        config.walk_forward_splits = 10
        
        return config
        
    def _get_default_hf_config(self) -> Dict[str, Any]:
        """Get high-frequency specific configuration."""
        return {
            'enable_online_learning': True,
            'online_update_frequency': 60,  # Update every 60 minutes
            'model_ensemble_size': 3,       # Smaller ensemble for speed
            'feature_selection_method': 'importance',
            'max_memory_gb': 8,
            'parallel_training': True,
            'incremental_training': True,
            'prediction_caching': True,
            'model_rotation_hours': 24,     # Retrain models every 24 hours
            
            # Market regime adaptation
            'regime_detection': True,
            'regime_models': ['trend', 'volatile', 'sideways'],
            
            # Performance optimization
            'use_fast_scaler': True,        # RobustScaler instead of StandardScaler
            'feature_preprocessing': 'optimized',
            'prediction_batching': True,
            'memory_cleanup_frequency': 100,  # Clean memory every 100 predictions
        }
    
    def create_multi_horizon_models(self) -> Dict[int, RandomForestRegressor]:
        """Create specialized models for different prediction horizons."""
        models = {}
        
        for horizon in self.prediction_horizons:
            # Adjust model parameters based on prediction horizon
            params = self._get_horizon_specific_params(horizon)
            
            if self.hf_config['feature_selection_method'] == 'extra_trees':
                model = ExtraTreesRegressor(**params)
            else:
                model = RandomForestRegressor(**params)
            
            models[horizon] = model
            
        return models
    
    def _get_horizon_specific_params(self, horizon_minutes: int) -> Dict[str, Any]:
        """Get model parameters optimized for specific prediction horizon."""
        base_params = {
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'bootstrap': self.config.bootstrap
        }
        
        # Adjust parameters based on prediction horizon
        if horizon_minutes <= 5:  # Ultra-short term (1-5 minutes)
            params = {
                **base_params,
                'n_estimators': 100,        # Faster training
                'max_depth': 8,             # Shallower to avoid noise
                'min_samples_split': 15,    # Higher to reduce overfitting
                'min_samples_leaf': 7,
                'max_features': 'sqrt'
            }
        elif horizon_minutes <= 15:  # Short term (5-15 minutes)
            params = {
                **base_params,
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt'
            }
        else:  # Medium term (15+ minutes)
            params = {
                **base_params,
                'n_estimators': 200,        # More trees for longer horizons
                'max_depth': 15,            # Deeper trees
                'min_samples_split': 5,     # Allow more splitting
                'min_samples_leaf': 3,
                'max_features': 'log2'      # Different feature selection
            }
        
        return params
    
    def prepare_multi_horizon_targets(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Create targets for multiple prediction horizons."""
        all_targets = pd.DataFrame(index=data.index)
        
        for symbol in symbols:
            close_col = f"{symbol}_close"
            if close_col not in data.columns:
                continue
                
            close_prices = data[close_col]
            
            # Create targets for each horizon
            for horizon in self.prediction_horizons:
                if self.config.target_type == "returns":
                    target = close_prices.pct_change(horizon).shift(-horizon)
                elif self.config.target_type == "log_returns":
                    target = (np.log(close_prices) - np.log(close_prices.shift(horizon))).shift(-horizon)
                else:  # price_change
                    target = (close_prices.shift(-horizon) - close_prices) / close_prices
                
                target_col = f"{symbol}_{horizon}min_target"
                all_targets[target_col] = target
        
        return all_targets
    
    def prepare_optimized_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare and optimize features for high-frequency trading."""
        # Memory optimization
        if self.hf_config['feature_preprocessing'] == 'optimized':
            features = self._optimize_feature_memory(features)
        
        # Feature selection based on previous importance
        if hasattr(self, 'selected_features') and self.selected_features:
            available_features = [f for f in self.selected_features if f in features.columns]
            if available_features:
                features = features[available_features]
        
        # Handle missing values efficiently
        features = self._fast_fillna(features)
        
        return features
    
    def _optimize_feature_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage of features."""
        # Convert float64 to float32 where possible
        float_cols = df.select_dtypes(include=[np.float64]).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert int64 to smaller ints where possible  
        int_cols = df.select_dtypes(include=[np.int64]).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def _fast_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast NaN filling optimized for time series."""
        # Forward fill with limit to avoid lookahead bias
        df = df.fillna(method='ffill', limit=3)
        
        # Fill remaining with median (cached for speed)
        if not hasattr(self, '_feature_medians'):
            self._feature_medians = df.median()
        
        df = df.fillna(self._feature_medians)
        
        return df
    
    def train_multi_horizon_models(self, features: pd.DataFrame, targets: pd.DataFrame, 
                                 symbols: List[str]) -> Dict[str, Any]:
        """Train models for all prediction horizons."""
        self.logger.info("Training multi-horizon models for minute-level predictions")
        
        results = {}
        self.horizon_models = {}
        self.scalers = {}
        
        # Prepare optimized features
        features_optimized = self.prepare_optimized_features(features)
        
        for symbol in symbols:
            symbol_results = {}
            symbol_models = {}
            symbol_scalers = {}
            
            for horizon in self.prediction_horizons:
                target_col = f"{symbol}_{horizon}min_target"
                
                if target_col not in targets.columns:
                    continue
                
                self.logger.info(f"Training {symbol} {horizon}min model...")
                
                # Prepare data
                X, y = self._prepare_horizon_data(features_optimized, targets, target_col)
                
                if len(X) < 100:  # Minimum training samples
                    self.logger.warning(f"Insufficient data for {symbol} {horizon}min: {len(X)} samples")
                    continue
                
                # Create and configure scaler
                if self.hf_config['use_fast_scaler']:
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                # Scale features
                X_scaled = scaler.fit_transform(X)
                
                # Create model
                model_params = self._get_horizon_specific_params(horizon)
                model = RandomForestRegressor(**model_params)
                
                # Train with validation
                training_result = self._train_single_model(model, X_scaled, y, horizon)
                
                # Store model and scaler
                symbol_models[horizon] = model
                symbol_scalers[horizon] = scaler
                symbol_results[horizon] = training_result
                
                # Feature importance tracking
                self._update_feature_importance(model, X.columns, symbol, horizon)
            
            self.horizon_models[symbol] = symbol_models
            self.scalers[symbol] = symbol_scalers
            results[symbol] = symbol_results
        
        self.is_fitted = True
        
        # Memory cleanup
        if self.hf_config['memory_cleanup_frequency']:
            gc.collect()
        
        self.logger.info("Multi-horizon model training completed")
        return results
    
    def _prepare_horizon_data(self, features: pd.DataFrame, targets: pd.DataFrame, 
                            target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for a specific horizon."""
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        X = features.loc[common_index]
        y = targets.loc[common_index, target_col]
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Feature selection for this horizon
        if len(X.columns) > 200:  # Feature selection for large feature sets
            X = self._select_features_for_horizon(X, y)
        
        return X, y
    
    def _select_features_for_horizon(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most relevant features for specific horizon."""
        # Quick feature selection using correlation
        correlations = abs(X.corrwith(y))
        top_features = correlations.nlargest(min(100, len(X.columns))).index
        
        return X[top_features]
    
    def _train_single_model(self, model: RandomForestRegressor, X: np.ndarray, 
                          y: pd.Series, horizon: int) -> Dict[str, float]:
        """Train a single model with validation."""
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
        
        # Train on full data
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        
        return {
            'cv_score_mean': -cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'n_samples': len(y),
            'horizon_minutes': horizon
        }
    
    def _update_feature_importance(self, model: RandomForestRegressor, feature_names: pd.Index, 
                                 symbol: str, horizon: int):
        """Update feature importance tracking."""
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        self.feature_importance_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'horizon': horizon,
            'importance': importance_dict
        })
        
        # Keep only recent history
        if len(self.feature_importance_history) > 1000:
            self.feature_importance_history = self.feature_importance_history[-500:]
    
    def predict_multi_horizon(self, features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Make predictions for all horizons and symbols."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = pd.DataFrame(index=features.index)
        features_optimized = self.prepare_optimized_features(features)
        
        for symbol in symbols:
            if symbol not in self.horizon_models:
                continue
            
            for horizon in self.prediction_horizons:
                if horizon not in self.horizon_models[symbol]:
                    continue
                
                model = self.horizon_models[symbol][horizon]
                scaler = self.scalers[symbol][horizon]
                
                # Scale features
                X_scaled = scaler.transform(features_optimized)
                
                # Make prediction
                pred = model.predict(X_scaled)
                
                pred_col = f"{symbol}_{horizon}min_pred"
                predictions[pred_col] = pred
        
        return predictions
    
    def predict_with_confidence(self, features: pd.DataFrame, symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Make predictions with confidence intervals."""
        predictions = self.predict_multi_horizon(features, symbols)
        
        # Calculate confidence based on historical performance
        confidence = pd.DataFrame(index=features.index)
        
        for symbol in symbols:
            if symbol not in self.model_performance_tracker:
                continue
            
            for horizon in self.prediction_horizons:
                pred_col = f"{symbol}_{horizon}min_pred"
                conf_col = f"{symbol}_{horizon}min_confidence"
                
                if pred_col in predictions.columns:
                    # Use historical RMSE as confidence measure
                    perf_key = f"{symbol}_{horizon}"
                    if perf_key in self.model_performance_tracker:
                        rmse = self.model_performance_tracker[perf_key].get('rmse', 0.01)
                        confidence[conf_col] = 1.0 / (1.0 + rmse)  # Higher RMSE = lower confidence
                    else:
                        confidence[conf_col] = 0.5  # Default confidence
        
        return predictions, confidence
    
    def online_update(self, new_features: pd.DataFrame, new_targets: pd.DataFrame, 
                     symbols: List[str]) -> Dict[str, Any]:
        """Perform online learning update with new data."""
        if not self.hf_config['enable_online_learning']:
            return {'status': 'online_learning_disabled'}
        
        self.logger.info("Performing online model update")
        
        update_results = {}
        
        for symbol in symbols:
            if symbol not in self.horizon_models:
                continue
            
            symbol_results = {}
            
            for horizon in self.prediction_horizons:
                target_col = f"{symbol}_{horizon}min_target"
                
                if target_col not in new_targets.columns:
                    continue
                
                # Prepare new data
                X_new, y_new = self._prepare_horizon_data(new_features, new_targets, target_col)
                
                if len(X_new) < 10:  # Minimum samples for update
                    continue
                
                # Scale new features
                scaler = self.scalers[symbol][horizon]
                X_new_scaled = scaler.transform(X_new)
                
                # Partial fit (simulate online learning)
                model = self.horizon_models[symbol][horizon]
                
                # For RandomForest, we retrain with recent data
                if len(X_new) >= 50:  # Enough data for retraining
                    model.fit(X_new_scaled, y_new)
                    
                    # Update performance tracking
                    y_pred = model.predict(X_new_scaled)
                    perf_key = f"{symbol}_{horizon}"
                    self.model_performance_tracker[perf_key] = {
                        'rmse': np.sqrt(mean_squared_error(y_new, y_pred)),
                        'r2': r2_score(y_new, y_pred),
                        'last_update': datetime.now()
                    }
                    
                    symbol_results[horizon] = 'updated'
                else:
                    symbol_results[horizon] = 'insufficient_data'
            
            update_results[symbol] = symbol_results
        
        return update_results
    
    def get_feature_importance_summary(self, top_n: int = 20) -> pd.DataFrame:
        """Get aggregated feature importance across all models."""
        if not self.feature_importance_history:
            return pd.DataFrame()
        
        # Aggregate importance across all models and time
        all_importance = {}
        
        for record in self.feature_importance_history[-100:]:  # Recent history
            for feature, importance in record['importance'].items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(importance)
        
        # Calculate average importance
        avg_importance = {}
        for feature, importances in all_importance.items():
            avg_importance[feature] = {
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'count': len(importances)
            }
        
        # Create DataFrame
        importance_df = pd.DataFrame.from_dict(avg_importance, orient='index')
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary."""
        if not self.model_performance_tracker:
            return {'status': 'no_performance_data'}
        
        summary = {
            'total_models': len(self.model_performance_tracker),
            'model_performance': {},
            'average_metrics': {}
        }
        
        all_rmse = []
        all_r2 = []
        
        for model_key, metrics in self.model_performance_tracker.items():
            summary['model_performance'][model_key] = metrics
            all_rmse.append(metrics.get('rmse', 0))
            all_r2.append(metrics.get('r2', 0))
        
        if all_rmse:
            summary['average_metrics'] = {
                'avg_rmse': np.mean(all_rmse),
                'avg_r2': np.mean(all_r2),
                'best_rmse': min(all_rmse),
                'best_r2': max(all_r2)
            }
        
        return summary
    
    def save_models(self, base_filepath: str):
        """Save all trained models."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")
        
        model_data = {
            'horizon_models': self.horizon_models,
            'scalers': self.scalers,
            'config': self.config,
            'hf_config': self.hf_config,
            'feature_importance_history': self.feature_importance_history,
            'model_performance_tracker': self.model_performance_tracker,
            'prediction_horizons': self.prediction_horizons,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, f"{base_filepath}_minute_rf_models.pkl")
        self.logger.info(f"Multi-horizon models saved to {base_filepath}_minute_rf_models.pkl")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        model_data = joblib.load(filepath)
        
        self.horizon_models = model_data['horizon_models']
        self.scalers = model_data['scalers']
        self.config = model_data.get('config', self.config)
        self.hf_config = model_data.get('hf_config', self.hf_config)
        self.feature_importance_history = model_data.get('feature_importance_history', [])
        self.model_performance_tracker = model_data.get('model_performance_tracker', {})
        self.prediction_horizons = model_data.get('prediction_horizons', [1, 5, 15, 60])
        
        self.is_fitted = True
        self.logger.info(f"Multi-horizon models loaded from {filepath}")
    
    def get_models_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all models."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        summary = {
            'status': 'fitted',
            'model_type': 'MinuteRandomForest',
            'prediction_horizons': self.prediction_horizons,
            'symbols_trained': list(self.horizon_models.keys()),
            'total_models': sum(len(models) for models in self.horizon_models.values()),
            'hf_config': self.hf_config,
            'performance_summary': self.get_model_performance_summary(),
            'feature_importance_available': len(self.feature_importance_history) > 0
        }
        
        # Per-symbol summary
        symbol_summary = {}
        for symbol, models in self.horizon_models.items():
            symbol_summary[symbol] = {
                'horizons': list(models.keys()),
                'model_count': len(models)
            }
        
        summary['symbol_summary'] = symbol_summary
        
        return summary


class EnsembleMinuteRandomForest:
    """Ensemble of minute-level Random Forest models for improved robustness."""
    
    def __init__(self, config: ModelConfig = None, hf_config: Dict[str, Any] = None, n_models: int = 3):
        self.config = config
        self.hf_config = hf_config
        self.n_models = n_models
        self.models = []
        self.logger = logging.getLogger(__name__)
        
        # Create ensemble of models
        for i in range(n_models):
            model_config = config
            if model_config:
                model_config.random_state = (config.random_state or 42) + i
            
            model = MinuteRandomForestModel(model_config, hf_config)
            self.models.append(model)
    
    def train(self, features: pd.DataFrame, targets: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        self.logger.info(f"Training ensemble of {self.n_models} minute-level models")
        
        results = []
        
        for i, model in enumerate(self.models):
            self.logger.info(f"Training ensemble model {i+1}/{self.n_models}")
            
            # Use different data samples for each model
            sample_frac = 0.85 + (0.15 * np.random.random())  # 85-100% of data
            sample_size = int(len(features) * sample_frac)
            
            # Time-aware sampling (recent data has higher probability)
            weights = np.linspace(0.5, 1.0, len(features))
            sample_indices = np.random.choice(
                len(features), 
                size=sample_size, 
                replace=False,
                p=weights/weights.sum()
            )
            
            features_sample = features.iloc[sample_indices]
            targets_sample = targets.iloc[sample_indices]
            
            # Train model
            model_results = model.train_multi_horizon_models(features_sample, targets_sample, symbols)
            results.append(model_results)
        
        return {
            'ensemble_results': results,
            'n_models': self.n_models
        }
    
    def predict_multi_horizon(self, features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Make ensemble predictions across all horizons."""
        all_predictions = []
        
        for model in self.models:
            if model.is_fitted:
                pred = model.predict_multi_horizon(features, symbols)
                all_predictions.append(pred)
        
        if not all_predictions:
            return pd.DataFrame(index=features.index)
        
        # Average predictions
        ensemble_pred = pd.concat(all_predictions).groupby(level=0).mean()
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, features: pd.DataFrame, symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Make predictions with uncertainty estimation."""
        all_predictions = []
        
        for model in self.models:
            if model.is_fitted:
                pred = model.predict_multi_horizon(features, symbols)
                all_predictions.append(pred)
        
        if not all_predictions:
            empty_df = pd.DataFrame(index=features.index)
            return empty_df, empty_df
        
        # Stack predictions
        predictions_array = np.stack([pred.values for pred in all_predictions])
        
        # Calculate mean and standard deviation
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        # Create DataFrames
        columns = all_predictions[0].columns
        mean_df = pd.DataFrame(mean_pred, index=features.index, columns=columns)
        std_df = pd.DataFrame(std_pred, index=features.index, columns=[f"{col}_std" for col in columns])
        
        return mean_df, std_df
    
    @property
    def is_fitted(self) -> bool:
        """Check if ensemble is fitted."""
        return any(model.is_fitted for model in self.models)
    
    def online_update(self, new_features: pd.DataFrame, new_targets: pd.DataFrame, 
                     symbols: List[str]) -> Dict[str, Any]:
        """Perform online update for all models in ensemble."""
        results = []
        
        for i, model in enumerate(self.models):
            if model.is_fitted:
                result = model.online_update(new_features, new_targets, symbols)
                result['model_index'] = i
                results.append(result)
        
        return {'ensemble_updates': results}
    
    def save_models(self, base_filepath: str):
        """Save ensemble models."""
        for i, model in enumerate(self.models):
            if model.is_fitted:
                model.save_models(f"{base_filepath}_ensemble_{i}")
        
        ensemble_data = {
            'n_models': self.n_models,
            'config': self.config,
            'hf_config': self.hf_config,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, f"{base_filepath}_ensemble_meta.pkl")
        self.logger.info(f"Ensemble models saved with base path: {base_filepath}")
    
    def load_models(self, base_filepath: str):
        """Load ensemble models."""
        # Load meta information
        ensemble_data = joblib.load(f"{base_filepath}_ensemble_meta.pkl")
        self.n_models = ensemble_data['n_models']
        
        # Load individual models
        self.models = []
        for i in range(self.n_models):
            model = MinuteRandomForestModel(self.config, self.hf_config)
            try:
                model.load_models(f"{base_filepath}_ensemble_{i}_minute_rf_models.pkl")
                self.models.append(model)
            except FileNotFoundError:
                self.logger.warning(f"Could not load ensemble model {i}")
        
        self.logger.info(f"Loaded {len(self.models)} ensemble models")


# Utility functions

def create_minute_rf_model(symbols: List[str] = None, config: ModelConfig = None) -> MinuteRandomForestModel:
    """Create a configured minute Random Forest model."""
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    return MinuteRandomForestModel(config)


def create_ensemble_minute_rf(symbols: List[str] = None, n_models: int = 3) -> EnsembleMinuteRandomForest:
    """Create an ensemble of minute Random Forest models."""
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    return EnsembleMinuteRandomForest(n_models=n_models)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample minute data
    dates = pd.date_range('2024-01-01', '2024-01-03', freq='1T')
    np.random.seed(42)
    
    # Sample features (minute-level data)
    n_features = 50
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    sample_features = pd.DataFrame(
        np.random.randn(len(dates), n_features),
        index=dates,
        columns=feature_names
    )
    
    # Sample price data for targets
    symbols = ['BTC-USD', 'ETH-USD']
    sample_data = pd.DataFrame(index=dates)
    
    for symbol in symbols:
        price = 50000 + np.random.randn(len(dates)).cumsum() * 100
        sample_data[f'{symbol}_close'] = price
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample features shape: {sample_features.shape}")
    
    # Create and test model
    model = create_minute_rf_model(symbols)
    
    # Create targets
    targets = model.prepare_multi_horizon_targets(sample_data, symbols)
    print(f"Targets shape: {targets.shape}")
    print(f"Target columns: {list(targets.columns)}")
    
    # Train models
    print("\nTraining models...")
    results = model.train_multi_horizon_models(sample_features, targets, symbols)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict_multi_horizon(sample_features.tail(100), symbols)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction columns: {list(predictions.columns)}")
    
    # Model summary
    print("\nModel summary:")
    summary = model.get_models_summary()
    print(f"Status: {summary['status']}")
    print(f"Total models: {summary['total_models']}")
    print(f"Symbols: {summary['symbols_trained']}")
    print(f"Horizons: {summary['prediction_horizons']}")