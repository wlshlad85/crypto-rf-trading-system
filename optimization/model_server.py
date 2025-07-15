"""
ULTRATHINK Model Server
Week 5 - DAY 31-32 Implementation

Production-optimized model serving system with caching, batching,
versioning, and A/B testing capabilities for real-time trading.
"""

import asyncio
import time
import json
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import hashlib
import pickle
import gc
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import joblib
from sklearn.base import BaseEstimator

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_profiler import PerformanceProfiler, profile


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    predictions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    last_prediction: Optional[datetime] = None
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        return self.error_count / self.predictions if self.predictions > 0 else 0.0


@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    features: np.ndarray
    model_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResponse:
    """Prediction response"""
    request_id: str
    prediction: Any
    confidence: Optional[float]
    model_id: str
    model_version: str
    latency_ms: float
    from_cache: bool
    timestamp: datetime


class ModelCache:
    """Intelligent model prediction cache"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, features: np.ndarray, model_id: str) -> str:
        """Generate cache key from features and model ID"""
        # Hash features for cache key
        features_bytes = features.tobytes()
        feature_hash = hashlib.md5(features_bytes).hexdigest()
        return f"{model_id}_{feature_hash}"
    
    def get(self, features: np.ndarray, model_id: str) -> Optional[Any]:
        """Get prediction from cache"""
        key = self._generate_key(features, model_id)
        
        with self.lock:
            if key in self.cache:
                prediction, timestamp = self.cache[key]
                
                # Check TTL
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    self.access_times[key] = datetime.now()
                    return prediction
                else:
                    # Remove expired entry
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        
        return None
    
    def put(self, features: np.ndarray, model_id: str, prediction: Any):
        """Store prediction in cache"""
        key = self._generate_key(features, model_id)
        
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (prediction, datetime.now())
            self.access_times[key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "utilization": len(self.cache) / self.max_size
            }


class ModelServer:
    """Production model serving system"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.models: Dict[str, BaseEstimator] = {}
        self.model_versions: Dict[str, str] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Caching system
        self.cache = ModelCache()
        
        # Metrics tracking
        self.metrics: Dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self.latency_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Batch processing
        self.batch_queue: Dict[str, List[PredictionRequest]] = defaultdict(list)
        self.batch_size = 32
        self.batch_timeout = 0.01  # 10ms max batch wait
        
        # A/B testing
        self.ab_test_config: Dict[str, Dict[str, Any]] = {}
        self.ab_test_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Model warmup
        self.warmup_cache: Dict[str, np.ndarray] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'latency_p99_ms': 10.0,
            'error_rate': 0.05,
            'cache_hit_rate': 0.5
        }
    
    def load_model(self, model_id: str, model_path: str, version: str = "1.0.0"):
        """Load a model for serving"""
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Store model
            self.models[model_id] = model
            self.model_versions[model_id] = version
            
            # Load metadata if available
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_id] = json.load(f)
            
            # Initialize metrics
            self.metrics[model_id] = ModelMetrics(model_id=model_id)
            
            print(f"Loaded model {model_id} v{version} from {model_path}")
            
            # Warm up model
            self._warmup_model(model_id)
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise
    
    def _warmup_model(self, model_id: str):
        """Warm up model with dummy predictions"""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Create dummy features based on model requirements
        if hasattr(model, 'n_features_'):
            n_features = model.n_features_
        else:
            # Default to common feature size
            n_features = 50
        
        # Generate warmup data
        warmup_features = np.random.randn(10, n_features)
        self.warmup_cache[model_id] = warmup_features
        
        # Perform warmup predictions
        for i in range(10):
            try:
                features = warmup_features[i:i+1]
                model.predict(features)
            except Exception as e:
                print(f"Warmup prediction failed for {model_id}: {e}")
        
        print(f"Model {model_id} warmed up successfully")
    
    @profile("model_prediction")
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction with caching and monitoring"""
        start_time = time.perf_counter()
        
        model_id = request.model_id
        
        # Check if model exists
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Check cache first
        cached_prediction = self.cache.get(request.features, model_id)
        if cached_prediction is not None:
            # Cache hit
            self.metrics[model_id].cache_hits += 1
            self.metrics[model_id].predictions += 1
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return PredictionResponse(
                request_id=request.request_id,
                prediction=cached_prediction,
                confidence=None,
                model_id=model_id,
                model_version=self.model_versions[model_id],
                latency_ms=latency_ms,
                from_cache=True,
                timestamp=datetime.now()
            )
        
        # Cache miss - make prediction
        self.metrics[model_id].cache_misses += 1
        
        try:
            model = self.models[model_id]
            
            # Make prediction
            prediction = model.predict(request.features)
            
            # Calculate confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(request.features)
                    confidence = float(np.max(probabilities))
                except:
                    confidence = None
            
            # Store in cache
            self.cache.put(request.features, model_id, prediction)
            
            # Update metrics
            self.metrics[model_id].predictions += 1
            self.metrics[model_id].last_prediction = datetime.now()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_buffer[model_id].append(latency_ms)
            
            # Update latency metrics
            self._update_latency_metrics(model_id)
            
            return PredictionResponse(
                request_id=request.request_id,
                prediction=prediction,
                confidence=confidence,
                model_id=model_id,
                model_version=self.model_versions[model_id],
                latency_ms=latency_ms,
                from_cache=False,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            # Update error metrics
            self.metrics[model_id].error_count += 1
            self.metrics[model_id].predictions += 1
            
            print(f"Prediction error for model {model_id}: {e}")
            raise
    
    async def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Make batch predictions"""
        
        # Group requests by model
        model_groups = defaultdict(list)
        for request in requests:
            model_groups[request.model_id].append(request)
        
        # Process each model group
        responses = []
        
        for model_id, model_requests in model_groups.items():
            try:
                batch_responses = await self._process_batch(model_id, model_requests)
                responses.extend(batch_responses)
            except Exception as e:
                print(f"Batch prediction error for model {model_id}: {e}")
                # Create error responses
                for request in model_requests:
                    error_response = PredictionResponse(
                        request_id=request.request_id,
                        prediction=None,
                        confidence=None,
                        model_id=model_id,
                        model_version=self.model_versions.get(model_id, "unknown"),
                        latency_ms=0.0,
                        from_cache=False,
                        timestamp=datetime.now()
                    )
                    responses.append(error_response)
        
        return responses
    
    async def _process_batch(self, model_id: str, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Process batch of requests for a single model"""
        start_time = time.perf_counter()
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        responses = []
        
        # Separate cached and non-cached requests
        cached_requests = []
        non_cached_requests = []
        
        for request in requests:
            cached_pred = self.cache.get(request.features, model_id)
            if cached_pred is not None:
                cached_requests.append((request, cached_pred))
            else:
                non_cached_requests.append(request)
        
        # Process cached requests
        for request, cached_pred in cached_requests:
            self.metrics[model_id].cache_hits += 1
            self.metrics[model_id].predictions += 1
            
            response = PredictionResponse(
                request_id=request.request_id,
                prediction=cached_pred,
                confidence=None,
                model_id=model_id,
                model_version=self.model_versions[model_id],
                latency_ms=0.1,  # Minimal cache access time
                from_cache=True,
                timestamp=datetime.now()
            )
            responses.append(response)
        
        # Process non-cached requests in batch
        if non_cached_requests:
            # Stack features for batch prediction
            batch_features = np.vstack([req.features for req in non_cached_requests])
            
            try:
                # Make batch prediction
                batch_predictions = model.predict(batch_features)
                
                # Calculate confidence if available
                batch_confidence = None
                if hasattr(model, 'predict_proba'):
                    try:
                        batch_probabilities = model.predict_proba(batch_features)
                        batch_confidence = np.max(batch_probabilities, axis=1)
                    except:
                        batch_confidence = None
                
                # Create responses
                for i, request in enumerate(non_cached_requests):
                    prediction = batch_predictions[i]
                    confidence = batch_confidence[i] if batch_confidence is not None else None
                    
                    # Store in cache
                    self.cache.put(request.features, model_id, prediction)
                    
                    # Update metrics
                    self.metrics[model_id].cache_misses += 1
                    self.metrics[model_id].predictions += 1
                
                batch_latency = (time.perf_counter() - start_time) * 1000
                per_request_latency = batch_latency / len(non_cached_requests)
                
                # Update latency buffer
                self.latency_buffer[model_id].extend([per_request_latency] * len(non_cached_requests))
                
                # Create responses
                for i, request in enumerate(non_cached_requests):
                    prediction = batch_predictions[i]
                    confidence = batch_confidence[i] if batch_confidence is not None else None
                    
                    response = PredictionResponse(
                        request_id=request.request_id,
                        prediction=prediction,
                        confidence=confidence,
                        model_id=model_id,
                        model_version=self.model_versions[model_id],
                        latency_ms=per_request_latency,
                        from_cache=False,
                        timestamp=datetime.now()
                    )
                    responses.append(response)
                
            except Exception as e:
                # Update error metrics
                self.metrics[model_id].error_count += len(non_cached_requests)
                self.metrics[model_id].predictions += len(non_cached_requests)
                raise
        
        # Update latency metrics
        self._update_latency_metrics(model_id)
        
        return responses
    
    def _update_latency_metrics(self, model_id: str):
        """Update latency metrics"""
        if model_id not in self.latency_buffer:
            return
        
        latencies = list(self.latency_buffer[model_id])
        if not latencies:
            return
        
        self.metrics[model_id].avg_latency_ms = np.mean(latencies)
        self.metrics[model_id].p95_latency_ms = np.percentile(latencies, 95)
        self.metrics[model_id].p99_latency_ms = np.percentile(latencies, 99)
    
    def setup_ab_test(self, test_name: str, model_a: str, model_b: str,
                     traffic_split: float = 0.5, duration_hours: int = 24):
        """Setup A/B test between two models"""
        
        self.ab_test_config[test_name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'duration_hours': duration_hours,
            'active': True
        }
        
        # Initialize results tracking
        self.ab_test_results[test_name] = {
            'model_a': {'predictions': 0, 'latencies': [], 'errors': 0},
            'model_b': {'predictions': 0, 'latencies': [], 'errors': 0}
        }
        
        print(f"A/B test '{test_name}' started: {model_a} vs {model_b}")
    
    def get_ab_test_model(self, test_name: str, request_id: str) -> str:
        """Get model for A/B test based on request ID"""
        
        if test_name not in self.ab_test_config:
            raise ValueError(f"A/B test '{test_name}' not found")
        
        config = self.ab_test_config[test_name]
        
        # Check if test is still active
        if not config['active']:
            return config['model_a']  # Default to model A
        
        # Check if test duration exceeded
        elapsed = datetime.now() - config['start_time']
        if elapsed.total_seconds() > config['duration_hours'] * 3600:
            config['active'] = False
            return config['model_a']
        
        # Use hash of request ID for consistent assignment
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        
        if (hash_value % 100) < (config['traffic_split'] * 100):
            return config['model_a']
        else:
            return config['model_b']
    
    def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        if model_id not in self.metrics:
            return {}
        
        metrics = self.metrics[model_id]
        
        return {
            'model_id': model_id,
            'predictions': metrics.predictions,
            'cache_hit_rate': metrics.cache_hit_rate,
            'error_rate': metrics.error_rate,
            'avg_latency_ms': metrics.avg_latency_ms,
            'p95_latency_ms': metrics.p95_latency_ms,
            'p99_latency_ms': metrics.p99_latency_ms,
            'last_prediction': metrics.last_prediction.isoformat() if metrics.last_prediction else None
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all model metrics"""
        
        all_metrics = {}
        
        for model_id in self.models.keys():
            all_metrics[model_id] = self.get_model_metrics(model_id)
        
        # Add cache statistics
        all_metrics['cache_stats'] = self.cache.get_stats()
        
        # Add A/B test results
        all_metrics['ab_tests'] = self.ab_test_results
        
        return all_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        
        health_status = {
            'healthy': True,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'alerts': []
        }
        
        # Check each model
        for model_id in self.models.keys():
            model_metrics = self.get_model_metrics(model_id)
            
            model_health = {
                'healthy': True,
                'last_prediction': model_metrics.get('last_prediction'),
                'error_rate': model_metrics.get('error_rate', 0),
                'p99_latency_ms': model_metrics.get('p99_latency_ms', 0),
                'cache_hit_rate': model_metrics.get('cache_hit_rate', 0)
            }
            
            # Check thresholds
            if model_metrics.get('p99_latency_ms', 0) > self.alert_thresholds['latency_p99_ms']:
                model_health['healthy'] = False
                health_status['alerts'].append(f"Model {model_id}: P99 latency too high")
            
            if model_metrics.get('error_rate', 0) > self.alert_thresholds['error_rate']:
                model_health['healthy'] = False
                health_status['alerts'].append(f"Model {model_id}: Error rate too high")
            
            if model_metrics.get('cache_hit_rate', 0) < self.alert_thresholds['cache_hit_rate']:
                health_status['alerts'].append(f"Model {model_id}: Cache hit rate too low")
            
            health_status['models'][model_id] = model_health
            
            if not model_health['healthy']:
                health_status['healthy'] = False
        
        return health_status
    
    def export_metrics(self, output_path: str = "optimization/model_server_metrics.json"):
        """Export metrics to file"""
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.get_all_metrics(),
            'health_check': self.health_check()
        }
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Metrics exported to: {output_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        # Clear cache
        self.cache.clear()
        
        # Clear models
        self.models.clear()
        self.model_versions.clear()
        self.model_metadata.clear()
        
        # Clear metrics
        self.metrics.clear()
        self.latency_buffer.clear()
        
        # Force garbage collection
        gc.collect()
        
        print("Model server cleaned up")


if __name__ == "__main__":
    # Demo the model server
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import uuid
    
    # Create test data and model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = "optimization/test_model.joblib"
    Path(model_path).parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    # Create model server
    server = ModelServer()
    server.load_model("test_model", model_path, "1.0.0")
    
    async def demo():
        # Single prediction
        request = PredictionRequest(
            request_id=str(uuid.uuid4()),
            features=X_test[0:1],
            model_id="test_model",
            timestamp=datetime.now()
        )
        
        response = await server.predict(request)
        print(f"Single prediction: {response.prediction}")
        print(f"Latency: {response.latency_ms:.2f}ms")
        print(f"From cache: {response.from_cache}")
        
        # Batch predictions
        batch_requests = []
        for i in range(10):
            batch_requests.append(PredictionRequest(
                request_id=str(uuid.uuid4()),
                features=X_test[i:i+1],
                model_id="test_model",
                timestamp=datetime.now()
            ))
        
        batch_responses = await server.predict_batch(batch_requests)
        print(f"\nBatch predictions: {len(batch_responses)} responses")
        
        # Test cache hit
        cached_response = await server.predict(request)
        print(f"Cached prediction: {cached_response.from_cache}")
        
        # Get metrics
        metrics = server.get_model_metrics("test_model")
        print(f"\nModel metrics:")
        print(f"  Predictions: {metrics['predictions']}")
        print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        print(f"  P99 latency: {metrics['p99_latency_ms']:.2f}ms")
        
        # Health check
        health = server.health_check()
        print(f"\nHealth check: {'HEALTHY' if health['healthy'] else 'UNHEALTHY'}")
        
        # Export metrics
        server.export_metrics()
    
    # Run demo
    asyncio.run(demo())
    
    # Cleanup
    server.cleanup()
    os.remove(model_path)