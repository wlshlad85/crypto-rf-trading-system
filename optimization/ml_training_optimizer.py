"""Optimized ML training with parallel processing and efficient memory usage."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

from utils.config import ModelConfig


class OptimizedRandomForestModel:
    """Optimized Random Forest model with parallel training and efficient memory usage."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use all available cores
        self.n_jobs = mp.cpu_count() - 1 if config.n_jobs == -1 else config.n_jobs
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Performance tracking
        self.feature_importance = {}
        self.best_params = None
        
        # Memory optimization
        self.dtype_optimization = {
            'float64': 'float32',
            'int64': 'int32'
        }
    
    def prepare_data_optimized(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with memory optimization."""
        # Optimize data types to reduce memory usage
        df = self._optimize_dtypes(df)
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['float32', 'int32']]
        X = df[feature_cols]
        y = df[target_col].astype('float32')
        
        # Remove rows with NaN values efficiently
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.logger.info(f"Prepared data shape: X={X.shape}, y={len(y)}")
        self.logger.info(f"Memory usage: {X.memory_usage().sum() / 1024**2:.2f} MB")
        
        return X, y
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'float64':
                df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                # Check if we can use smaller int type
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        return df
    
    def train_optimized(self, X: pd.DataFrame, y: pd.Series, 
                       validation_split: float = 0.2) -> Dict[str, Any]:
        """Train model with optimizations."""
        self.logger.info("Starting optimized model training")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features - use float32 for efficiency
        X_train_scaled = self.scaler.fit_transform(X_train).astype('float32')
        X_val_scaled = self.scaler.transform(X_val).astype('float32')
        
        # Create optimized model
        self.model = self._create_optimized_model()
        
        # Train with early stopping callback
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train.values, y_train_pred)
        val_metrics = self._calculate_metrics(y_val.values, y_val_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        self.is_fitted = True
        
        self.logger.info(f"Training completed. Val R2: {val_metrics['r2']:.4f}")
        
        return {
            'train': train_metrics,
            'validation': val_metrics,
            'feature_importance': self.feature_importance
        }
    
    def _create_optimized_model(self) -> Any:
        """Create an optimized model based on configuration."""
        params = self.best_params or {}
        
        if self.config.model_type == 'lightgbm':
            # LightGBM is often faster than RandomForest
            default_params = {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_jobs': self.n_jobs,
                'random_state': self.config.random_state,
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            }
            default_params.update(params)
            return LGBMRegressor(**default_params)
        
        elif self.config.model_type == 'catboost':
            # CatBoost with GPU support if available
            default_params = {
                'iterations': 300,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'RMSE',
                'random_state': self.config.random_state,
                'thread_count': self.n_jobs,
                'verbose': False,
                'allow_writing_files': False
            }
            default_params.update(params)
            
            # Check for GPU availability
            try:
                import pycuda.driver as cuda
                cuda.init()
                if cuda.Device.count() > 0:
                    default_params['task_type'] = 'GPU'
                    self.logger.info("Using GPU for CatBoost training")
            except:
                pass
            
            return CatBoostRegressor(**default_params)
        
        else:
            # Optimized RandomForest
            default_params = {
                'n_estimators': params.get('n_estimators', 200),
                'max_depth': params.get('max_depth', 10),
                'min_samples_split': params.get('min_samples_split', 5),
                'min_samples_leaf': params.get('min_samples_leaf', 2),
                'max_features': params.get('max_features', 'sqrt'),
                'bootstrap': params.get('bootstrap', True),
                'n_jobs': self.n_jobs,
                'random_state': self.config.random_state,
                'max_samples': 0.8  # Subsample for faster training
            }
            return RandomForestRegressor(**default_params)
    
    def hyperparameter_tuning_optimized(self, X: pd.DataFrame, y: pd.Series, 
                                       n_trials: int = 100) -> Dict[str, Any]:
        """Optimized hyperparameter tuning with Optuna."""
        self.logger.info(f"Starting optimized hyperparameter tuning with {n_trials} trials")
        
        # Scale features once
        X_scaled = self.scaler.fit_transform(X).astype('float32')
        
        # Create objective function
        def objective(trial):
            # Model-specific parameter suggestions
            if self.config.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                model = LGBMRegressor(
                    **params,
                    n_jobs=self.n_jobs,
                    random_state=self.config.random_state,
                    verbose=-1
                )
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestRegressor(
                    **params,
                    n_jobs=self.n_jobs,
                    random_state=self.config.random_state
                )
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=20,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Predict and score
                y_pred = model.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study with optimizations
        sampler = TPESampler(seed=self.config.random_state, n_startup_trials=10)
        pruner = HyperbandPruner()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize with parallel trials
        study.optimize(
            objective, 
            n_trials=n_trials,
            n_jobs=min(4, self.n_jobs),  # Limit parallel trials
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        
        results = {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        self.logger.info(f"Optuna tuning completed. Best score: {study.best_value:.4f}")
        
        return results
    
    def walk_forward_validation_parallel(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Parallel walk-forward validation."""
        self.logger.info("Starting parallel walk-forward validation")
        
        n_splits = self.config.walk_forward_splits
        split_size = len(X) // n_splits
        
        # Prepare split indices
        split_indices = []
        for i in range(n_splits - 1):
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = min(test_start + split_size, len(X))
            
            if test_start < len(X):
                split_indices.append((
                    list(range(train_end)),
                    list(range(test_start, test_end))
                ))
        
        # Parallel validation
        with ProcessPoolExecutor(max_workers=min(n_splits, self.n_jobs)) as executor:
            futures = []
            
            for i, (train_idx, test_idx) in enumerate(split_indices):
                future = executor.submit(
                    self._validate_fold,
                    X.iloc[train_idx],
                    X.iloc[test_idx],
                    y.iloc[train_idx],
                    y.iloc[test_idx],
                    i
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
        
        # Aggregate results
        all_metrics = {
            'r2_scores': [r['r2'] for r in results],
            'rmse_scores': [r['rmse'] for r in results],
            'mae_scores': [r['mae'] for r in results]
        }
        
        avg_results = {
            'avg_r2': np.mean(all_metrics['r2_scores']),
            'avg_rmse': np.mean(all_metrics['rmse_scores']),
            'avg_mae': np.mean(all_metrics['mae_scores']),
            'std_r2': np.std(all_metrics['r2_scores']),
            'std_rmse': np.std(all_metrics['rmse_scores']),
            'std_mae': np.std(all_metrics['mae_scores'])
        }
        
        self.logger.info(f"Walk-forward validation completed. Avg R2: {avg_results['avg_r2']:.4f}")
        
        return {**all_metrics, **avg_results}
    
    def _validate_fold(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series, fold_id: int) -> Dict[str, float]:
        """Validate a single fold."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype('float32')
        X_test_scaled = scaler.transform(X_test).astype('float32')
        
        # Train model
        model = self._create_optimized_model()
        
        if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test.values, y_pred)
        metrics['fold_id'] = fold_id
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
    
    def predict_batch(self, X: pd.DataFrame, batch_size: int = 10000) -> np.ndarray:
        """Make predictions in batches for memory efficiency."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Predict in batches
        predictions = []
        n_samples = len(X)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X.iloc[start_idx:end_idx]
            
            # Scale batch
            batch_scaled = self.scaler.transform(batch).astype('float32')
            
            # Predict
            batch_pred = self.model.predict(batch_scaled)
            predictions.extend(batch_pred)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save model efficiently."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        # Use compression for smaller file size
        joblib.dump(model_data, filepath, compress=3)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model efficiently."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.best_params = model_data['best_params']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Model loaded from {filepath}")