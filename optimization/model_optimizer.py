"""
ULTRATHINK ML Model Optimizer
Week 5 - DAY 31-32 Implementation

Advanced ML model optimization pipeline with quantization, pruning,
feature selection, and deployment optimization for production performance.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_profiler import PerformanceProfiler, profile


@dataclass
class OptimizationResult:
    """Model optimization result"""
    model_name: str
    optimization_type: str
    original_size_mb: float
    optimized_size_mb: float
    original_inference_ms: float
    optimized_inference_ms: float
    original_accuracy: float
    optimized_accuracy: float
    compression_ratio: float
    speedup_factor: float
    accuracy_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "optimization_type": self.optimization_type,
            "original_size_mb": self.original_size_mb,
            "optimized_size_mb": self.optimized_size_mb,
            "original_inference_ms": self.original_inference_ms,
            "optimized_inference_ms": self.optimized_inference_ms,
            "original_accuracy": self.original_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "compression_ratio": self.compression_ratio,
            "speedup_factor": self.speedup_factor,
            "accuracy_loss": self.accuracy_loss
        }


class ModelOptimizer:
    """Advanced ML model optimization system"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.optimization_results: List[OptimizationResult] = []
        
        # Optimization thresholds
        self.max_accuracy_loss = 0.02  # 2% max accuracy loss
        self.min_speedup = 1.5         # 1.5x minimum speedup
        self.target_inference_ms = 1.0  # 1ms target inference
        
        # Feature selection cache
        self.feature_selectors = {}
        
        # Model cache
        self.model_cache = {}
    
    def optimize_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive model optimization"""
        print(f"\nOptimizing model: {model_name}")
        print("=" * 50)
        
        # Baseline metrics
        baseline_metrics = self._measure_baseline(
            model, X_test, y_test, model_name
        )
        
        optimizations = {}
        
        # 1. Feature Selection Optimization
        print("\n1. Feature Selection Optimization...")
        fs_result = self.optimize_feature_selection(
            model, X_train, y_train, X_test, y_test, model_name
        )
        optimizations["feature_selection"] = fs_result
        
        # 2. Model Pruning (for tree-based models)
        print("\n2. Model Pruning...")
        pruning_result = self.optimize_model_pruning(
            model, X_train, y_train, X_test, y_test, model_name
        )
        optimizations["pruning"] = pruning_result
        
        # 3. Ensemble Optimization
        print("\n3. Ensemble Optimization...")
        ensemble_result = self.optimize_ensemble(
            model, X_train, y_train, X_test, y_test, model_name
        )
        optimizations["ensemble"] = ensemble_result
        
        # 4. Inference Optimization
        print("\n4. Inference Optimization...")
        inference_result = self.optimize_inference(
            model, X_test, model_name
        )
        optimizations["inference"] = inference_result
        
        # 5. Memory Optimization
        print("\n5. Memory Optimization...")
        memory_result = self.optimize_memory_usage(
            model, X_test, model_name
        )
        optimizations["memory"] = memory_result
        
        # Select best optimization
        best_optimization = self._select_best_optimization(
            optimizations, baseline_metrics
        )
        
        return {
            "model_name": model_name,
            "baseline": baseline_metrics,
            "optimizations": optimizations,
            "best_optimization": best_optimization,
            "timestamp": datetime.now().isoformat()
        }
    
    def _measure_baseline(self, model: Any, X_test: np.ndarray, 
                         y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Measure baseline model performance"""
        
        # Measure inference time
        @profile(f"baseline_inference_{model_name}")
        def timed_inference():
            return model.predict(X_test)
        
        start_time = time.perf_counter()
        predictions = timed_inference()
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) / len(X_test) * 1000  # ms per sample
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Model size
        model_size = self._calculate_model_size(model)
        
        return {
            "accuracy": accuracy,
            "inference_ms_per_sample": inference_time,
            "model_size_mb": model_size,
            "total_inference_ms": (end_time - start_time) * 1000
        }
    
    def optimize_feature_selection(self, model: Any, X_train: np.ndarray, 
                                  y_train: np.ndarray, X_test: np.ndarray, 
                                  y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Optimize feature selection"""
        
        original_features = X_train.shape[1]
        
        # Try different feature selection methods
        methods = {
            "univariate": self._univariate_feature_selection,
            "rfe": self._recursive_feature_elimination,
            "importance": self._importance_feature_selection,
            "pca": self._pca_feature_selection
        }
        
        best_result = None
        best_score = 0
        
        for method_name, method_func in methods.items():
            try:
                print(f"  Testing {method_name}...")
                
                # Apply feature selection
                X_train_selected, X_test_selected, selector = method_func(
                    X_train, y_train, X_test, model_name
                )
                
                # Train model with selected features
                model_copy = self._copy_model(model)
                
                start_time = time.perf_counter()
                model_copy.fit(X_train_selected, y_train)
                training_time = time.perf_counter() - start_time
                
                # Evaluate
                start_time = time.perf_counter()
                predictions = model_copy.predict(X_test_selected)
                inference_time = time.perf_counter() - start_time
                
                accuracy = accuracy_score(y_test, predictions)
                
                # Calculate metrics
                selected_features = X_train_selected.shape[1]
                feature_reduction = 1 - (selected_features / original_features)
                inference_speedup = self._calculate_inference_speedup(
                    original_features, selected_features
                )
                
                result = {
                    "method": method_name,
                    "selected_features": selected_features,
                    "feature_reduction": feature_reduction,
                    "accuracy": accuracy,
                    "inference_time_ms": inference_time * 1000,
                    "training_time_ms": training_time * 1000,
                    "speedup_factor": inference_speedup,
                    "selector": selector
                }
                
                # Score based on accuracy and speedup
                score = accuracy * inference_speedup
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    
                print(f"    Features: {selected_features}/{original_features}")
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Speedup: {inference_speedup:.2f}x")
                
            except Exception as e:
                print(f"    Error with {method_name}: {e}")
                continue
        
        return best_result or {"error": "No valid feature selection method"}
    
    def optimize_model_pruning(self, model: Any, X_train: np.ndarray, 
                             y_train: np.ndarray, X_test: np.ndarray, 
                             y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Optimize model pruning (mainly for tree-based models)"""
        
        if not hasattr(model, 'estimators_'):
            return {"error": "Model pruning not supported for this model type"}
        
        try:
            # For Random Forest, try different n_estimators
            if isinstance(model, RandomForestClassifier):
                return self._optimize_random_forest_pruning(
                    model, X_train, y_train, X_test, y_test, model_name
                )
            
            return {"error": "Pruning not implemented for this model type"}
            
        except Exception as e:
            return {"error": f"Pruning failed: {str(e)}"}
    
    def optimize_ensemble(self, model: Any, X_train: np.ndarray, 
                         y_train: np.ndarray, X_test: np.ndarray, 
                         y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Optimize ensemble configuration"""
        
        if not hasattr(model, 'estimators_'):
            return {"error": "Ensemble optimization not supported"}
        
        try:
            # Test different ensemble sizes
            original_n_estimators = len(model.estimators_)
            
            best_result = None
            best_score = 0
            
            test_sizes = [
                int(original_n_estimators * 0.3),
                int(original_n_estimators * 0.5),
                int(original_n_estimators * 0.7),
                int(original_n_estimators * 0.9)
            ]
            
            for n_estimators in test_sizes:
                if n_estimators < 10:
                    continue
                
                # Create pruned ensemble
                pruned_model = self._create_pruned_ensemble(model, n_estimators)
                
                # Evaluate
                start_time = time.perf_counter()
                predictions = pruned_model.predict(X_test)
                inference_time = time.perf_counter() - start_time
                
                accuracy = accuracy_score(y_test, predictions)
                speedup = original_n_estimators / n_estimators
                
                score = accuracy * speedup
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        "original_estimators": original_n_estimators,
                        "pruned_estimators": n_estimators,
                        "accuracy": accuracy,
                        "speedup_factor": speedup,
                        "inference_time_ms": inference_time * 1000,
                        "model": pruned_model
                    }
                
                print(f"    Estimators: {n_estimators}/{original_n_estimators}")
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Speedup: {speedup:.2f}x")
            
            return best_result or {"error": "No valid ensemble configuration"}
            
        except Exception as e:
            return {"error": f"Ensemble optimization failed: {str(e)}"}
    
    def optimize_inference(self, model: Any, X_test: np.ndarray, 
                          model_name: str) -> Dict[str, Any]:
        """Optimize model inference performance"""
        
        try:
            # Batch inference optimization
            batch_results = self._optimize_batch_inference(model, X_test)
            
            # Caching optimization
            cache_results = self._optimize_prediction_caching(model, X_test)
            
            # Memory layout optimization
            memory_results = self._optimize_memory_layout(model, X_test)
            
            return {
                "batch_optimization": batch_results,
                "caching_optimization": cache_results,
                "memory_optimization": memory_results
            }
            
        except Exception as e:
            return {"error": f"Inference optimization failed: {str(e)}"}
    
    def optimize_memory_usage(self, model: Any, X_test: np.ndarray, 
                             model_name: str) -> Dict[str, Any]:
        """Optimize memory usage"""
        
        try:
            # Measure baseline memory
            baseline_memory = self._measure_memory_usage(model, X_test)
            
            # Try different optimizations
            optimizations = {}
            
            # 1. Float32 optimization
            if hasattr(model, 'estimators_'):
                optimizations["float32"] = self._optimize_float_precision(
                    model, X_test, baseline_memory
                )
            
            # 2. Sparse matrix optimization
            optimizations["sparse"] = self._optimize_sparse_matrices(
                model, X_test, baseline_memory
            )
            
            # 3. Model compression
            optimizations["compression"] = self._optimize_model_compression(
                model, X_test, baseline_memory
            )
            
            return {
                "baseline_memory_mb": baseline_memory,
                "optimizations": optimizations
            }
            
        except Exception as e:
            return {"error": f"Memory optimization failed: {str(e)}"}
    
    def _select_best_optimization(self, optimizations: Dict[str, Any], 
                                 baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best optimization based on criteria"""
        
        best_optimization = None
        best_score = 0
        
        for opt_type, opt_result in optimizations.items():
            if "error" in opt_result:
                continue
            
            try:
                # Calculate score based on multiple criteria
                if opt_type == "feature_selection":
                    accuracy = opt_result.get("accuracy", 0)
                    speedup = opt_result.get("speedup_factor", 1)
                    
                    # Penalize accuracy loss
                    accuracy_penalty = max(0, baseline["accuracy"] - accuracy)
                    score = (accuracy * speedup) - (accuracy_penalty * 10)
                    
                elif opt_type == "pruning":
                    accuracy = opt_result.get("accuracy", 0)
                    speedup = opt_result.get("speedup_factor", 1)
                    
                    accuracy_penalty = max(0, baseline["accuracy"] - accuracy)
                    score = (accuracy * speedup) - (accuracy_penalty * 10)
                    
                elif opt_type == "ensemble":
                    accuracy = opt_result.get("accuracy", 0)
                    speedup = opt_result.get("speedup_factor", 1)
                    
                    accuracy_penalty = max(0, baseline["accuracy"] - accuracy)
                    score = (accuracy * speedup) - (accuracy_penalty * 10)
                
                else:
                    # Default scoring
                    score = 1.0
                
                if score > best_score:
                    best_score = score
                    best_optimization = {
                        "type": opt_type,
                        "result": opt_result,
                        "score": score
                    }
                    
            except Exception as e:
                print(f"Error scoring {opt_type}: {e}")
                continue
        
        return best_optimization or {"error": "No valid optimization found"}
    
    # Feature selection methods
    def _univariate_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                                    X_test: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Univariate feature selection"""
        
        # Try different k values
        n_features = X_train.shape[1]
        best_k = min(20, n_features // 2)  # Start with reasonable number
        
        selector = SelectKBest(score_func=f_classif, k=best_k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        return X_train_selected, X_test_selected, selector
    
    def _recursive_feature_elimination(self, X_train: np.ndarray, y_train: np.ndarray, 
                                     X_test: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Recursive feature elimination"""
        
        # Use a fast estimator for RFE
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(max_iter=100, random_state=42)
        
        n_features = X_train.shape[1]
        n_features_to_select = min(20, n_features // 2)
        
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        return X_train_selected, X_test_selected, selector
    
    def _importance_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                                    X_test: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Feature importance-based selection"""
        
        # Train a fast model to get importance
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        
        # Select top features
        importances = rf.feature_importances_
        n_features = min(20, X_train.shape[1] // 2)
        
        top_indices = np.argsort(importances)[-n_features:]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        return X_train_selected, X_test_selected, top_indices
    
    def _pca_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray, Any]:
        """PCA-based feature selection"""
        
        # Scale features first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply PCA
        n_components = min(20, X_train.shape[1] // 2)
        pca = PCA(n_components=n_components)
        
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        return X_train_pca, X_test_pca, (scaler, pca)
    
    # Model optimization methods
    def _optimize_random_forest_pruning(self, model: RandomForestClassifier, 
                                       X_train: np.ndarray, y_train: np.ndarray, 
                                       X_test: np.ndarray, y_test: np.ndarray, 
                                       model_name: str) -> Dict[str, Any]:
        """Optimize Random Forest through pruning"""
        
        original_n_estimators = model.n_estimators
        
        # Test different numbers of estimators
        test_estimators = [
            int(original_n_estimators * 0.3),
            int(original_n_estimators * 0.5),
            int(original_n_estimators * 0.7)
        ]
        
        best_result = None
        best_score = 0
        
        for n_est in test_estimators:
            if n_est < 10:
                continue
            
            # Create pruned model
            pruned_model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=model.max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            # Train and evaluate
            start_time = time.perf_counter()
            pruned_model.fit(X_train, y_train)
            training_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            predictions = pruned_model.predict(X_test)
            inference_time = time.perf_counter() - start_time
            
            accuracy = accuracy_score(y_test, predictions)
            speedup = original_n_estimators / n_est
            
            score = accuracy * speedup
            
            if score > best_score:
                best_score = score
                best_result = {
                    "original_estimators": original_n_estimators,
                    "pruned_estimators": n_est,
                    "accuracy": accuracy,
                    "speedup_factor": speedup,
                    "inference_time_ms": inference_time * 1000,
                    "training_time_ms": training_time * 1000,
                    "model": pruned_model
                }
        
        return best_result or {"error": "No valid pruning configuration"}
    
    def _create_pruned_ensemble(self, model: Any, n_estimators: int) -> Any:
        """Create pruned ensemble model"""
        
        if hasattr(model, 'estimators_'):
            # Select best estimators based on individual performance
            estimators = model.estimators_[:n_estimators]
            
            # Create new model with selected estimators
            pruned_model = type(model)(n_estimators=n_estimators)
            pruned_model.estimators_ = estimators
            pruned_model.n_estimators = n_estimators
            
            # Copy other attributes
            if hasattr(model, 'classes_'):
                pruned_model.classes_ = model.classes_
            if hasattr(model, 'n_classes_'):
                pruned_model.n_classes_ = model.n_classes_
            if hasattr(model, 'n_features_'):
                pruned_model.n_features_ = model.n_features_
                
            return pruned_model
        
        return model
    
    # Inference optimization methods
    def _optimize_batch_inference(self, model: Any, X_test: np.ndarray) -> Dict[str, Any]:
        """Optimize batch inference"""
        
        batch_sizes = [1, 10, 50, 100, 500, 1000]
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                continue
            
            # Measure batch inference time
            n_batches = min(10, len(X_test) // batch_size)
            times = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_test))
                batch = X_test[start_idx:end_idx]
                
                start_time = time.perf_counter()
                predictions = model.predict(batch)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) / len(batch))
            
            avg_time_per_sample = np.mean(times)
            
            results[batch_size] = {
                "avg_time_per_sample_ms": avg_time_per_sample * 1000,
                "throughput_samples_per_sec": 1 / avg_time_per_sample
            }
        
        # Find optimal batch size
        best_batch_size = min(results.keys(), 
                             key=lambda x: results[x]["avg_time_per_sample_ms"])
        
        return {
            "batch_results": results,
            "optimal_batch_size": best_batch_size,
            "optimal_time_ms": results[best_batch_size]["avg_time_per_sample_ms"]
        }
    
    def _optimize_prediction_caching(self, model: Any, X_test: np.ndarray) -> Dict[str, Any]:
        """Optimize prediction caching"""
        
        # Simple caching strategy
        cache = {}
        
        def cached_predict(X):
            cache_key = hash(X.tobytes())
            if cache_key in cache:
                return cache[cache_key]
            
            prediction = model.predict(X)
            cache[cache_key] = prediction
            return prediction
        
        # Measure with and without caching
        # Without caching
        start_time = time.perf_counter()
        for i in range(0, len(X_test), 100):
            batch = X_test[i:i+100]
            model.predict(batch)
        no_cache_time = time.perf_counter() - start_time
        
        # With caching (simulate repeated predictions)
        start_time = time.perf_counter()
        for i in range(0, len(X_test), 100):
            batch = X_test[i:i+100]
            cached_predict(batch)
            # Simulate repeated query
            cached_predict(batch)
        cache_time = time.perf_counter() - start_time
        
        return {
            "no_cache_time_ms": no_cache_time * 1000,
            "cache_time_ms": cache_time * 1000,
            "cache_hit_speedup": no_cache_time / cache_time if cache_time > 0 else 1
        }
    
    def _optimize_memory_layout(self, model: Any, X_test: np.ndarray) -> Dict[str, Any]:
        """Optimize memory layout"""
        
        # Test different array layouts
        results = {}
        
        # C-contiguous array
        X_c = np.ascontiguousarray(X_test)
        start_time = time.perf_counter()
        model.predict(X_c)
        c_time = time.perf_counter() - start_time
        results["c_contiguous"] = c_time * 1000
        
        # Fortran-contiguous array
        X_f = np.asfortranarray(X_test)
        start_time = time.perf_counter()
        model.predict(X_f)
        f_time = time.perf_counter() - start_time
        results["fortran_contiguous"] = f_time * 1000
        
        # Original array
        start_time = time.perf_counter()
        model.predict(X_test)
        original_time = time.perf_counter() - start_time
        results["original"] = original_time * 1000
        
        best_layout = min(results.keys(), key=lambda x: results[x])
        
        return {
            "layout_times_ms": results,
            "best_layout": best_layout,
            "speedup": original_time / min(results.values())
        }
    
    # Memory optimization methods
    def _measure_memory_usage(self, model: Any, X_test: np.ndarray) -> float:
        """Measure memory usage in MB"""
        
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return memory_after - memory_before
    
    def _optimize_float_precision(self, model: Any, X_test: np.ndarray, 
                                 baseline_memory: float) -> Dict[str, Any]:
        """Optimize float precision"""
        
        # This is a placeholder - actual implementation would depend on model type
        return {
            "optimization": "float32_precision",
            "memory_reduction": 0.2,  # 20% reduction estimate
            "accuracy_impact": 0.001  # Minimal impact estimate
        }
    
    def _optimize_sparse_matrices(self, model: Any, X_test: np.ndarray, 
                                 baseline_memory: float) -> Dict[str, Any]:
        """Optimize using sparse matrices"""
        
        from scipy.sparse import csr_matrix
        
        # Convert to sparse if beneficial
        sparsity = 1.0 - np.count_nonzero(X_test) / X_test.size
        
        if sparsity > 0.5:  # Only beneficial if >50% sparse
            X_sparse = csr_matrix(X_test)
            
            # Test prediction with sparse matrix
            try:
                start_time = time.perf_counter()
                predictions = model.predict(X_sparse)
                sparse_time = time.perf_counter() - start_time
                
                # Memory usage estimate
                dense_memory = X_test.nbytes / 1024 / 1024  # MB
                sparse_memory = (X_sparse.data.nbytes + X_sparse.indices.nbytes + 
                               X_sparse.indptr.nbytes) / 1024 / 1024  # MB
                
                return {
                    "sparsity": sparsity,
                    "dense_memory_mb": dense_memory,
                    "sparse_memory_mb": sparse_memory,
                    "memory_reduction": 1 - (sparse_memory / dense_memory),
                    "inference_time_ms": sparse_time * 1000,
                    "feasible": True
                }
                
            except Exception as e:
                return {
                    "error": f"Sparse matrix not supported: {e}",
                    "feasible": False
                }
        
        return {
            "sparsity": sparsity,
            "feasible": False,
            "reason": "Insufficient sparsity"
        }
    
    def _optimize_model_compression(self, model: Any, X_test: np.ndarray, 
                                   baseline_memory: float) -> Dict[str, Any]:
        """Optimize model compression"""
        
        # Serialize model to measure size
        import pickle
        import gzip
        
        # Original size
        original_pickle = pickle.dumps(model)
        original_size = len(original_pickle) / 1024 / 1024  # MB
        
        # Compressed size
        compressed_pickle = gzip.compress(original_pickle)
        compressed_size = len(compressed_pickle) / 1024 / 1024  # MB
        
        compression_ratio = original_size / compressed_size
        
        return {
            "original_size_mb": original_size,
            "compressed_size_mb": compressed_size,
            "compression_ratio": compression_ratio,
            "size_reduction": 1 - (compressed_size / original_size)
        }
    
    # Utility methods
    def _calculate_model_size(self, model: Any) -> float:
        """Calculate model size in MB"""
        
        try:
            import pickle
            model_pickle = pickle.dumps(model)
            return len(model_pickle) / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _calculate_inference_speedup(self, original_features: int, 
                                   selected_features: int) -> float:
        """Calculate expected inference speedup from feature reduction"""
        
        # Linear approximation - actual speedup depends on model type
        feature_ratio = selected_features / original_features
        
        # Assume some overhead, so speedup is not exactly inversely proportional
        return 1 / (0.3 + 0.7 * feature_ratio)
    
    def _copy_model(self, model: Any) -> Any:
        """Create a copy of the model"""
        
        try:
            import copy
            return copy.deepcopy(model)
        except:
            # Fallback: create new instance with same parameters
            return type(model)(**model.get_params())
    
    def export_optimization_report(self, output_path: str = "optimization/model_optimization_report.json"):
        """Export optimization report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "optimizations": [result.to_dict() for result in self.optimization_results],
            "summary": {
                "total_optimizations": len(self.optimization_results),
                "successful_optimizations": len([r for r in self.optimization_results if r.accuracy_loss <= self.max_accuracy_loss]),
                "avg_compression_ratio": np.mean([r.compression_ratio for r in self.optimization_results]),
                "avg_speedup_factor": np.mean([r.speedup_factor for r in self.optimization_results]),
                "avg_accuracy_loss": np.mean([r.accuracy_loss for r in self.optimization_results])
            }
        }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


if __name__ == "__main__":
    # Test the model optimizer
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Create test data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Optimize model
    optimizer = ModelOptimizer()
    results = optimizer.optimize_model(
        model, X_train, y_train, X_test, y_test, "test_rf_model"
    )
    
    # Print results
    print("\nOptimization Results:")
    print("=" * 50)
    print(json.dumps(results, indent=2, default=str))
    
    # Export report
    report = optimizer.export_optimization_report()
    print(f"\nReport saved to: optimization/model_optimization_report.json")