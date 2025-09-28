"""Comprehensive result caching system for expensive computations."""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, Optional, Callable, Tuple
import logging
import os
from pathlib import Path
from functools import wraps
import gc

logger = logging.getLogger(__name__)


class ResultCache:
    """Advanced caching system for expensive computations."""

    def __init__(self, cache_dir: str = None, max_cache_size: int = 100,
                 ttl_seconds: int = 3600):
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'cache')
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache_metadata = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'saves': 0
        }

        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

    def _get_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate deterministic cache key."""
        # Create a string representation of the function call
        key_data = {
            'func_name': func_name,
            'args': args,
            'kwargs': kwargs
        }

        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"{func_name}_{key_hash}"

    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _get_metadata_path(self) -> str:
        """Get metadata file path."""
        return os.path.join(self.cache_dir, 'cache_metadata.json')

    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if os.path.exists(self._get_metadata_path()):
                with open(self._get_metadata_path(), 'r') as f:
                    self.cache_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
            self.cache_metadata = {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self._get_metadata_path(), 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache metadata: {e}")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache_metadata:
            return False

        metadata = self.cache_metadata[cache_key]
        cache_time = metadata.get('timestamp', 0)
        current_time = time.time()

        # Check TTL
        if current_time - cache_time > self.ttl_seconds:
            return False

        # Check if cache file exists
        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return False

        return True

    def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached result."""
        if self._is_cache_valid(cache_key):
            try:
                cache_path = self._get_cache_path(cache_key)
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)

                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return result

            except Exception as e:
                logger.warning(f"Error loading cache for key {cache_key}: {e}")
                self._evict_cache(cache_key)

        self.cache_stats['misses'] += 1
        return None

    def put(self, cache_key: str, result: Any):
        """Store result in cache."""
        try:
            # Check cache size limit
            if len(self.cache_metadata) >= self.max_cache_size:
                self._evict_lru()

            # Save result
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            # Update metadata
            self.cache_metadata[cache_key] = {
                'timestamp': time.time(),
                'size_bytes': os.path.getsize(cache_path)
            }

            self.cache_stats['saves'] += 1
            self._save_metadata()

        except Exception as e:
            logger.warning(f"Error saving cache for key {cache_key}: {e}")

    def _evict_cache(self, cache_key: str):
        """Remove cache entry."""
        try:
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)

            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]

            self.cache_stats['evictions'] += 1

        except Exception as e:
            logger.warning(f"Error evicting cache for key {cache_key}: {e}")

    def _evict_lru(self):
        """Evict least recently used cache entries."""
        if not self.cache_metadata:
            return

        # Sort by timestamp (oldest first)
        sorted_keys = sorted(
            self.cache_metadata.keys(),
            key=lambda k: self.cache_metadata[k]['timestamp']
        )

        # Evict oldest entries
        evict_count = max(1, len(sorted_keys) // 4)  # Evict 25% of entries
        for key in sorted_keys[:evict_count]:
            self._evict_cache(key)

    def clear(self):
        """Clear all cache entries."""
        try:
            for cache_key in list(self.cache_metadata.keys()):
                self._evict_cache(cache_key)

            self.cache_metadata.clear()
            self.cache_stats = {k: 0 for k in self.cache_stats}

        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests) if total_requests > 0 else 0

        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache_metadata),
            'cache_dir_size_mb': self._get_cache_dir_size() / (1024 * 1024)
        }

    def _get_cache_dir_size(self) -> int:
        """Get total size of cache directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Error calculating cache dir size: {e}")

        return total_size


def cache_result(cache_instance: ResultCache, ttl_seconds: int = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._get_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache_instance.put(cache_key, result)

            return result

        return wrapper
    return decorator


def cache_dataframe_computation(cache_dir: str = None):
    """Decorator specifically for DataFrame computations."""
    cache_instance = ResultCache(cache_dir=cache_dir)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a more specific cache key for DataFrame operations
            # Include DataFrame shapes and column info if available
            enhanced_kwargs = kwargs.copy()

            # Check if any args are DataFrames
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    enhanced_kwargs[f'df_{i}_shape'] = str(arg.shape)
                    enhanced_kwargs[f'df_{i}_columns'] = str(list(arg.columns))

            cache_key = cache_instance._get_cache_key(func.__name__, args, enhanced_kwargs)

            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                cache_instance.cache_stats['hits'] += 1
                return result

            cache_instance.cache_stats['misses'] += 1

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache_instance.put(cache_key, result)
            cache_instance.cache_stats['saves'] += 1

            return result

        return wrapper
    return decorator


# Global cache instance
_global_cache = ResultCache()


def get_global_cache() -> ResultCache:
    """Get the global cache instance."""
    return _global_cache


def clear_global_cache():
    """Clear the global cache."""
    _global_cache.clear()


def cache_with_ttl(ttl_seconds: int = 3600):
    """Decorator for caching with custom TTL."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = _global_cache._get_cache_key(func.__name__, args, kwargs)

            # Override TTL for this call
            original_ttl = _global_cache.ttl_seconds
            _global_cache.ttl_seconds = ttl_seconds

            try:
                # Try to get from cache
                result = _global_cache.get(cache_key)
                if result is not None:
                    return result

                # Compute result
                result = func(*args, **kwargs)

                # Cache result
                _global_cache.put(cache_key, result)

                return result

            finally:
                # Restore original TTL
                _global_cache.ttl_seconds = original_ttl

        return wrapper
    return decorator