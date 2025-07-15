"""
ULTRATHINK Hyperparameter Optimizer
Week 5 - DAY 31-32 Implementation

Advanced hyperparameter optimization pipeline with Bayesian optimization,
multi-objective optimization, and automated experiment tracking.
"""

import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Using grid search fallback.")

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_profiler import PerformanceProfiler, profile


@dataclass
class HyperparameterResult:
    """Single hyperparameter optimization result"""
    model_name: str
    optimization_method: str
    best_params: Dict[str, Any]
    best_score: float
    best_scores: Dict[str, float]  # Multi-objective scores
    optimization_time: float
    n_trials: int
    cv_scores: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "optimization_method": self.optimization_method,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_scores": self.best_scores,
            "optimization_time": self.optimization_time,
            "n_trials": self.n_trials,
            "cv_scores": self.cv_scores,
            "cv_mean": np.mean(self.cv_scores),
            "cv_std": np.std(self.cv_scores)
        }


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization system"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.results: List[HyperparameterResult] = []
        self.study_storage = {}
        
        # Optimization settings
        self.n_trials = 100
        self.cv_folds = 5
        self.random_state = 42
        self.timeout = 3600  # 1 hour max
        
        # Multi-objective weights
        self.objective_weights = {
            'accuracy': 0.4,
            'precision': 0.2,
            'recall': 0.2,
            'f1': 0.2
        }
        
        # Performance constraints
        self.max_training_time = 300  # 5 minutes max training time
        self.max_inference_time = 0.01  # 10ms max inference time
        
        # Parameter search spaces
        self.search_spaces = {
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]},
                'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']}
            },
            'logistic_regression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100.0, 'log': True},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']},
                'max_iter': {'type': 'int', 'low': 100, 'high': 1000}
            },
            'svm': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100.0, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear', 'poly']},
                'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
                'degree': {'type': 'int', 'low': 2, 'high': 5}
            }
        }
    
    def optimize_hyperparameters(self, model_name: str, model_class: type,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: Optional[np.ndarray] = None,
                                y_test: Optional[np.ndarray] = None,
                                method: str = 'bayesian') -> HyperparameterResult:
        """Optimize hyperparameters for a model"""
        
        print(f"\nOptimizing hyperparameters for: {model_name}")
        print(f"Method: {method}")
        print(f"Trials: {self.n_trials}")
        print("=" * 50)
        
        start_time = time.time()
        
        if method == 'bayesian' and OPTUNA_AVAILABLE:
            result = self._optimize_bayesian(
                model_name, model_class, X_train, y_train, X_test, y_test
            )
        elif method == 'grid':
            result = self._optimize_grid_search(
                model_name, model_class, X_train, y_train, X_test, y_test
            )
        elif method == 'random':
            result = self._optimize_random_search(
                model_name, model_class, X_train, y_train, X_test, y_test
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimization_time = time.time() - start_time
        
        # Add timing information
        result.optimization_time = optimization_time
        
        # Store result
        self.results.append(result)
        
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best score: {result.best_score:.4f}")
        print(f"Best parameters: {result.best_params}")
        
        return result
    
    def _optimize_bayesian(self, model_name: str, model_class: type,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: Optional[np.ndarray] = None,
                          y_test: Optional[np.ndarray] = None) -> HyperparameterResult:
        """Bayesian optimization using Optuna"""
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        # Define objective function
        def objective(trial):
            return self._evaluate_trial(
                trial, model_name, model_class, X_train, y_train, X_test, y_test
            )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        
        # Calculate final CV scores with best parameters
        cv_scores = self._calculate_cv_scores(
            model_class, best_params, X_train, y_train
        )
        
        # Multi-objective scores
        best_scores = self._calculate_multi_objective_scores(
            model_class, best_params, X_train, y_train, X_test, y_test
        )
        
        return HyperparameterResult(
            model_name=model_name,
            optimization_method='bayesian',
            best_params=best_params,
            best_score=best_score,
            best_scores=best_scores,
            optimization_time=0,  # Will be set by caller
            n_trials=len(study.trials),
            cv_scores=cv_scores
        )
    
    def _optimize_grid_search(self, model_name: str, model_class: type,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_test: Optional[np.ndarray] = None,
                             y_test: Optional[np.ndarray] = None) -> HyperparameterResult:
        """Grid search optimization"""
        
        from sklearn.model_selection import GridSearchCV
        
        # Create parameter grid
        param_grid = self._create_grid_search_params(model_name)
        
        # Create cross-validator
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            model_class(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Calculate final CV scores
        cv_scores = self._calculate_cv_scores(
            model_class, best_params, X_train, y_train
        )
        
        # Multi-objective scores
        best_scores = self._calculate_multi_objective_scores(
            model_class, best_params, X_train, y_train, X_test, y_test
        )
        
        return HyperparameterResult(
            model_name=model_name,
            optimization_method='grid',
            best_params=best_params,
            best_score=best_score,
            best_scores=best_scores,
            optimization_time=0,
            n_trials=len(grid_search.cv_results_['params']),
            cv_scores=cv_scores
        )
    
    def _optimize_random_search(self, model_name: str, model_class: type,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: Optional[np.ndarray] = None,
                               y_test: Optional[np.ndarray] = None) -> HyperparameterResult:
        """Random search optimization"""
        
        from sklearn.model_selection import RandomizedSearchCV
        
        # Create parameter distribution
        param_dist = self._create_random_search_params(model_name)
        
        # Create cross-validator
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Random search
        random_search = RandomizedSearchCV(
            model_class(),
            param_dist,
            n_iter=self.n_trials,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        # Get results
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        # Calculate final CV scores
        cv_scores = self._calculate_cv_scores(
            model_class, best_params, X_train, y_train
        )
        
        # Multi-objective scores
        best_scores = self._calculate_multi_objective_scores(
            model_class, best_params, X_train, y_train, X_test, y_test
        )
        
        return HyperparameterResult(
            model_name=model_name,
            optimization_method='random',
            best_params=best_params,
            best_score=best_score,
            best_scores=best_scores,
            optimization_time=0,
            n_trials=self.n_trials,
            cv_scores=cv_scores
        )
    
    def _evaluate_trial(self, trial, model_name: str, model_class: type,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: Optional[np.ndarray] = None,
                       y_test: Optional[np.ndarray] = None) -> float:
        """Evaluate a single trial for Bayesian optimization"""
        
        # Suggest parameters
        params = self._suggest_parameters(trial, model_name)
        
        # Create model
        model = model_class(**params)
        
        # Measure training time
        start_time = time.time()
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        try:
            # Fit model for timing
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Check training time constraint
            if training_time > self.max_training_time:
                trial.set_user_attr('training_time_exceeded', True)
                return 0.0
            
            # Measure inference time
            if X_test is not None:
                start_time = time.time()
                predictions = model.predict(X_test[:100])  # Sample for timing
                inference_time = (time.time() - start_time) / 100  # Per sample
                
                # Check inference time constraint
                if inference_time > self.max_inference_time:
                    trial.set_user_attr('inference_time_exceeded', True)
                    return 0.0
            
            # Calculate cross-validation score
            scores = cross_val_score(
                model_class(**params), X_train, y_train, 
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            
            mean_score = scores.mean()
            
            # Multi-objective optimization
            if X_test is not None and y_test is not None:
                # Calculate additional metrics
                test_predictions = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, test_predictions)
                precision = precision_score(y_test, test_predictions, average='weighted')
                recall = recall_score(y_test, test_predictions, average='weighted')
                f1 = f1_score(y_test, test_predictions, average='weighted')
                
                # Weighted multi-objective score
                multi_score = (
                    self.objective_weights['accuracy'] * accuracy +
                    self.objective_weights['precision'] * precision +
                    self.objective_weights['recall'] * recall +
                    self.objective_weights['f1'] * f1
                )
                
                # Store additional metrics
                trial.set_user_attr('test_accuracy', accuracy)
                trial.set_user_attr('test_precision', precision)
                trial.set_user_attr('test_recall', recall)
                trial.set_user_attr('test_f1', f1)
                trial.set_user_attr('multi_objective_score', multi_score)
                
                return multi_score
            
            return mean_score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    def _suggest_parameters(self, trial, model_name: str) -> Dict[str, Any]:
        """Suggest parameters for a trial"""
        
        if model_name not in self.search_spaces:
            raise ValueError(f"Unknown model: {model_name}")
        
        search_space = self.search_spaces[model_name]
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_loguniform(
                        param_name, param_config['low'], param_config['high']
                    )
                else:
                    params[param_name] = trial.suggest_uniform(
                        param_name, param_config['low'], param_config['high']
                    )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
        
        return params
    
    def _create_grid_search_params(self, model_name: str) -> Dict[str, List[Any]]:
        """Create parameter grid for grid search"""
        
        if model_name == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
        elif model_name == 'logistic_regression':
            return {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 500, 1000]
            }
        elif model_name == 'svm':
            return {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        else:
            return {}
    
    def _create_random_search_params(self, model_name: str) -> Dict[str, Any]:
        """Create parameter distribution for random search"""
        
        from scipy.stats import randint, uniform
        
        if model_name == 'random_forest':
            return {
                'n_estimators': randint(10, 200),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        elif model_name == 'logistic_regression':
            return {
                'C': uniform(0.01, 100.0),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': randint(100, 1000)
            }
        elif model_name == 'svm':
            return {
                'C': uniform(0.01, 100.0),
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        else:
            return {}
    
    def _calculate_cv_scores(self, model_class: type, params: Dict[str, Any],
                            X_train: np.ndarray, y_train: np.ndarray) -> List[float]:
        """Calculate cross-validation scores"""
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        model = model_class(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        return scores.tolist()
    
    def _calculate_multi_objective_scores(self, model_class: type, params: Dict[str, Any],
                                        X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: Optional[np.ndarray] = None,
                                        y_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate multi-objective scores"""
        
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        scores = {}
        
        # Training scores
        train_pred = model.predict(X_train)
        scores['train_accuracy'] = accuracy_score(y_train, train_pred)
        scores['train_precision'] = precision_score(y_train, train_pred, average='weighted')
        scores['train_recall'] = recall_score(y_train, train_pred, average='weighted')
        scores['train_f1'] = f1_score(y_train, train_pred, average='weighted')
        
        # Test scores (if available)
        if X_test is not None and y_test is not None:
            test_pred = model.predict(X_test)
            scores['test_accuracy'] = accuracy_score(y_test, test_pred)
            scores['test_precision'] = precision_score(y_test, test_pred, average='weighted')
            scores['test_recall'] = recall_score(y_test, test_pred, average='weighted')
            scores['test_f1'] = f1_score(y_test, test_pred, average='weighted')
        
        return scores
    
    def optimize_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: Optional[np.ndarray] = None,
                                y_test: Optional[np.ndarray] = None) -> Dict[str, HyperparameterResult]:
        """Optimize multiple models and compare results"""
        
        models = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }
        
        results = {}
        
        for model_name, model_class in models.items():
            try:
                print(f"\n{'='*60}")
                print(f"Optimizing {model_name}")
                print('='*60)
                
                result = self.optimize_hyperparameters(
                    model_name, model_class, X_train, y_train, X_test, y_test
                )
                results[model_name] = result
                
            except Exception as e:
                print(f"Error optimizing {model_name}: {e}")
                continue
        
        return results
    
    def get_best_model(self, results: Dict[str, HyperparameterResult]) -> Tuple[str, HyperparameterResult]:
        """Get the best model from optimization results"""
        
        best_model_name = None
        best_result = None
        best_score = 0
        
        for model_name, result in results.items():
            if result.best_score > best_score:
                best_score = result.best_score
                best_model_name = model_name
                best_result = result
        
        return best_model_name, best_result
    
    def export_optimization_report(self, output_path: str = "optimization/hyperparameter_report.json"):
        """Export optimization report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "optimization_settings": {
                "n_trials": self.n_trials,
                "cv_folds": self.cv_folds,
                "timeout": self.timeout,
                "objective_weights": self.objective_weights,
                "max_training_time": self.max_training_time,
                "max_inference_time": self.max_inference_time
            },
            "results": [result.to_dict() for result in self.results],
            "summary": {
                "total_optimizations": len(self.results),
                "best_overall_score": max([r.best_score for r in self.results]) if self.results else 0,
                "avg_optimization_time": np.mean([r.optimization_time for r in self.results]) if self.results else 0,
                "total_trials": sum([r.n_trials for r in self.results])
            }
        }
        
        # Find best model
        if self.results:
            best_result = max(self.results, key=lambda r: r.best_score)
            report["best_model"] = {
                "name": best_result.model_name,
                "score": best_result.best_score,
                "parameters": best_result.best_params,
                "method": best_result.optimization_method
            }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nHyperparameter optimization report saved to: {output_path}")
        return report
    
    def save_optimized_model(self, model_name: str, model_class: type, 
                           best_params: Dict[str, Any], X_train: np.ndarray, 
                           y_train: np.ndarray, output_path: str):
        """Save optimized model"""
        
        # Train final model
        model = model_class(**best_params)
        model.fit(X_train, y_train)
        
        # Save model
        Path(output_path).parent.mkdir(exist_ok=True)
        joblib.dump(model, output_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_class": str(model_class),
            "best_params": best_params,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = output_path.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {output_path}")
        print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    # Test the hyperparameter optimizer
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create test data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer()
    optimizer.n_trials = 20  # Reduce for demo
    
    # Optimize multiple models
    results = optimizer.optimize_multiple_models(
        X_train, y_train, X_test, y_test
    )
    
    # Find best model
    best_model_name, best_result = optimizer.get_best_model(results)
    print(f"\nBest model: {best_model_name}")
    print(f"Best score: {best_result.best_score:.4f}")
    print(f"Best parameters: {best_result.best_params}")
    
    # Export report
    report = optimizer.export_optimization_report()
    
    # Save best model
    if best_model_name == 'random_forest':
        model_class = RandomForestClassifier
    elif best_model_name == 'logistic_regression':
        model_class = LogisticRegression
    elif best_model_name == 'svm':
        model_class = SVC
    
    optimizer.save_optimized_model(
        best_model_name,
        model_class,
        best_result.best_params,
        X_train,
        y_train,
        f"optimization/best_model_{best_model_name}.joblib"
    )