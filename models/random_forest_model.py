"""Random Forest model implementation for cryptocurrency prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

from utils.config import ModelConfig


class CryptoRandomForestModel:
    """Random Forest model for cryptocurrency prediction."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Performance tracking
        self.training_history = []
        self.validation_scores = {}
        self.feature_importance = {}
        
        # Best parameters from hyperparameter tuning
        self.best_params = None
    
    def create_model(self, params: Dict[str, Any] = None) -> RandomForestRegressor:
        """Create Random Forest model with given parameters."""
        if params is None:
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'min_samples_split': self.config.min_samples_split,
                'min_samples_leaf': self.config.min_samples_leaf,
                'max_features': self.config.max_features,
                'bootstrap': self.config.bootstrap,
                'n_jobs': self.config.n_jobs,
                'random_state': self.config.random_state
            }
        
        return RandomForestRegressor(**params)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        # Remove target column from features
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        # Remove non-numeric columns (like 'symbol')
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.logger.info(f"Prepared data shape: X={X.shape}, y={len(y)}")
        
        return X, y
    
    def create_targets(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Create target variables for prediction."""
        targets = pd.DataFrame(index=data.index)
        
        for symbol in symbols:
            close_col = f"{symbol}_close"
            
            if close_col not in data.columns:
                continue
            
            close_prices = data[close_col]
            
            # Future returns
            if self.config.target_type == "returns":
                target = close_prices.pct_change(self.config.target_horizon).shift(-self.config.target_horizon)
            elif self.config.target_type == "log_returns":
                target = (np.log(close_prices) - np.log(close_prices.shift(self.config.target_horizon))).shift(-self.config.target_horizon)
            else:  # price_change
                target = (close_prices.shift(-self.config.target_horizon) - close_prices) / close_prices
            
            targets[f"{symbol}_target"] = target
        
        return targets
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the Random Forest model."""
        self.logger.info("Starting model training")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create and train model
        self.model = self.create_model(self.best_params)
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        # Store results
        self.validation_scores = {
            'train': train_metrics,
            'validation': val_metrics
        }
        
        # Feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        self.is_fitted = True
        
        self.logger.info(f"Training completed. Val R2: {val_metrics['r2']:.4f}")
        
        return self.validation_scores
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, method: str = 'optuna') -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        self.logger.info(f"Starting hyperparameter tuning with {method}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'grid_search':
            return self._grid_search_tuning(X_scaled, y)
        elif method == 'optuna':
            return self._optuna_tuning(X_scaled, y)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
    
    def _grid_search_tuning(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """Perform grid search hyperparameter tuning."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestRegressor(random_state=self.config.random_state, n_jobs=self.config.n_jobs)
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=tscv, 
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        
        results = {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Grid search completed. Best score: {grid_search.best_score_:.4f}")
        
        return results
    
    def _optuna_tuning(self, X: np.ndarray, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
        """Perform Optuna hyperparameter tuning."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs
            }
            
            rf = RandomForestRegressor(**params)
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            # Calculate cross-validation score
            scores = cross_val_score(rf, X, y, cv=tscv, scoring=self.config.scoring)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        
        results = {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
        
        self.logger.info(f"Optuna tuning completed. Best score: {study.best_value:.4f}")
        
        return results
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Perform walk-forward validation."""
        self.logger.info("Starting walk-forward validation")
        
        n_splits = self.config.walk_forward_splits
        split_size = len(X) // n_splits
        
        results = {
            'r2_scores': [],
            'rmse_scores': [],
            'mae_scores': []
        }
        
        for i in range(n_splits):
            # Define train and test indices
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = min(test_start + split_size, len(X))
            
            if test_start >= len(X):
                break
            
            # Split data
            X_train = X.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_train = y.iloc[:train_end]
            y_test = y.iloc[test_start:test_end]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.create_model(self.best_params)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            results['r2_scores'].append(metrics['r2'])
            results['rmse_scores'].append(metrics['rmse'])
            results['mae_scores'].append(metrics['mae'])
        
        # Calculate average scores
        avg_results = {
            'avg_r2': np.mean(results['r2_scores']),
            'avg_rmse': np.mean(results['rmse_scores']),
            'avg_mae': np.mean(results['mae_scores']),
            'std_r2': np.std(results['r2_scores']),
            'std_rmse': np.std(results['rmse_scores']),
            'std_mae': np.std(results['mae_scores'])
        }
        
        self.logger.info(f"Walk-forward validation completed. Avg R2: {avg_results['avg_r2']:.4f}")
        
        return {**results, **avg_results}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba_ranking(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict and rank assets by expected returns."""
        predictions = self.predict(X)
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame({
            'prediction': predictions,
            'rank': pd.Series(predictions).rank(ascending=False),
            'percentile': pd.Series(predictions).rank(pct=True)
        }, index=X.index)
        
        return ranking_df
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'validation_scores': self.validation_scores,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_params = model_data.get('best_params')
        self.feature_importance = model_data.get('feature_importance', {})
        self.validation_scores = model_data.get('validation_scores', {})
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "model_type": "RandomForest",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "validation_scores": self.validation_scores,
            "n_features": len(self.feature_importance),
            "best_params": self.best_params
        }


class EnsembleRandomForestModel:
    """Ensemble of Random Forest models for improved robustness."""
    
    def __init__(self, config: ModelConfig, n_models: int = 5):
        self.config = config
        self.n_models = n_models
        self.models = []
        self.logger = logging.getLogger(__name__)
        
        # Create multiple models with different random states
        for i in range(n_models):
            model_config = self.config
            model_config.random_state = self.config.random_state + i
            self.models.append(CryptoRandomForestModel(model_config))
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        self.logger.info(f"Training ensemble of {self.n_models} models")
        
        results = []
        
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i+1}/{self.n_models}")
            
            # Use different data splits for each model
            sample_frac = 0.8 + (0.2 * np.random.random())  # 80-100% of data
            sample_indices = np.random.choice(
                len(X), 
                size=int(len(X) * sample_frac), 
                replace=False
            )
            
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            
            # Train model
            model_results = model.train(X_sample, y_sample)
            results.append(model_results)
        
        return {
            'individual_results': results,
            'n_models': self.n_models
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def create_targets(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Create target variables for prediction using the first model."""
        # Delegate to the first model in the ensemble
        return self.models[0].create_targets(data, symbols)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training using the first model."""
        # Delegate to the first model in the ensemble
        return self.models[0].prepare_data(df, target_col)
    
    @property
    def is_fitted(self) -> bool:
        """Check if all models in the ensemble are fitted."""
        return all(model.is_fitted for model in self.models)
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, method: str = 'optuna') -> Dict[str, Any]:
        """Perform hyperparameter tuning for all models in the ensemble."""
        self.logger.info(f"Starting hyperparameter tuning for ensemble with {method}")
        
        # Tune the first model and apply the best parameters to all models
        tuning_results = self.models[0].hyperparameter_tuning(X, y, method)
        
        # Apply the best parameters to all models
        for model in self.models:
            model.best_params = tuning_results['best_params']
        
        self.logger.info("Hyperparameter tuning completed for ensemble")
        return tuning_results
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform walk-forward validation for the ensemble."""
        self.logger.info("Starting walk-forward validation for ensemble")
        
        # Delegate to the first model for validation logic
        validation_results = self.models[0].walk_forward_validation(X, y)
        
        # Calculate ensemble-specific metrics by running validation on all models
        individual_validations = []
        for model in self.models:
            individual_validations.append(model.walk_forward_validation(X, y))
        
        # Average the results across all models
        avg_scores = {}
        for key in validation_results.keys():
            if key.endswith('_scores'):
                scores = [val[key] for val in individual_validations]
                avg_scores[key] = [np.mean([score[i] for score in scores]) for i in range(len(scores[0]))]
        
        # Calculate overall averages
        avg_r2 = np.mean(avg_scores.get('r2_scores', [0]))
        
        return {
            'r2_scores': avg_scores.get('r2_scores', []),
            'rmse_scores': avg_scores.get('rmse_scores', []),
            'mae_scores': avg_scores.get('mae_scores', []),
            'avg_r2': avg_r2,
            'individual_validations': individual_validations
        }
    
    def save_model(self, filepath: str):
        """Save the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")
        
        ensemble_data = {
            'models': [model for model in self.models],
            'n_models': self.n_models,
            'config': self.config,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, filepath)
        self.logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load an ensemble model."""
        ensemble_data = joblib.load(filepath)
        
        self.models = ensemble_data['models']
        self.n_models = ensemble_data['n_models']
        self.config = ensemble_data['config']
        
        self.logger.info(f"Ensemble model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get ensemble model summary."""
        if not self.is_fitted:
            return {"status": "not_fitted", "n_models": self.n_models}
        
        individual_summaries = [model.get_model_summary() for model in self.models]
        
        return {
            "status": "fitted",
            "model_type": "EnsembleRandomForest",
            "n_models": self.n_models,
            "individual_models": individual_summaries
        }