#!/usr/bin/env python3
"""
ULTRATHINK Distributed Context Cache - Week 4 Day 25
Advanced distributed caching system for team environments
"""

import json
import time
import asyncio
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import logging
from collections import defaultdict, OrderedDict
import aioredis
import socket
import zlib
import uuid
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    content: str
    content_hash: str
    size: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: int
    priority: int
    node_id: str
    
@dataclass
class CacheNode:
    """Cache node information"""
    node_id: str
    host: str
    port: int
    capacity: int
    current_size: int
    status: str
    last_heartbeat: datetime
    
@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    network_latency: float
    storage_efficiency: float
    replication_factor: float
    
class DistributedContextCache:
    """
    Advanced distributed caching system for team environments
    Provides high-performance context caching with intelligent distribution
    """
    
    def __init__(self, system_root: Path, node_id: str = None):
        self.system_root = system_root
        self.node_id = node_id or str(uuid.uuid4())
        self.cache_dir = system_root / ".claude" / "caching" / self.node_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self.db_path = self.cache_dir / "cache.db"
        self.init_database()
        
        # Local cache (LRU)
        self.local_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_local_size = 1000  # Maximum local cache entries
        self.cache_lock = threading.RLock()
        
        # Distributed cache configuration
        self.redis_client = None
        self.cache_nodes: Dict[str, CacheNode] = {}
        self.replication_factor = 3
        
        # Performance tracking
        self.metrics = CacheMetrics(
            hit_rate=0.0,
            miss_rate=0.0,
            eviction_rate=0.0,
            network_latency=0.0,
            storage_efficiency=0.0,
            replication_factor=2.0
        )
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'network_requests': 0,
            'bytes_cached': 0,
            'compression_ratio': 0.0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.heartbeat_task = None
        self.sync_task = None
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
    def init_database(self):
        """Initialize SQLite database for persistent cache"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                content TEXT,
                content_hash TEXT,
                size INTEGER,
                created_at DATETIME,
                last_accessed DATETIME,
                access_count INTEGER,
                ttl INTEGER,
                priority INTEGER,
                node_id TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_nodes (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                capacity INTEGER,
                current_size INTEGER,
                status TEXT,
                last_heartbeat DATETIME
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_metrics (
                id INTEGER PRIMARY KEY,
                metric_name TEXT,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS replication_log (
                id INTEGER PRIMARY KEY,
                key TEXT,
                source_node TEXT,
                target_node TEXT,
                operation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def initialize_distributed_cache(self, redis_host: str = "localhost", 
                                         redis_port: int = 6379):
        """Initialize distributed cache with Redis backend"""
        try:
            # Connect to Redis
            self.redis_client = await aioredis.from_url(
                f"redis://{redis_host}:{redis_port}",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            
            # Register this node
            await self.register_cache_node()
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self.cleanup_expired_entries())
            self.heartbeat_task = asyncio.create_task(self.send_heartbeat())
            self.sync_task = asyncio.create_task(self.sync_with_cluster())
            
        except Exception as e:
            logger.warning(f"Failed to initialize distributed cache: {e}")
            logger.info("Falling back to local cache only")
            
    async def register_cache_node(self):
        """Register this node in the distributed cache cluster"""
        try:
            node_info = {
                'node_id': self.node_id,
                'host': socket.gethostname(),
                'port': 0,  # Will be set if we start a server
                'capacity': self.max_local_size,
                'current_size': len(self.local_cache),
                'status': 'active',
                'last_heartbeat': datetime.now().isoformat()
            }
            
            # Register in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    f"cache_nodes:{self.node_id}",
                    mapping=node_info
                )
                
            # Register in local database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_nodes 
                (node_id, host, port, capacity, current_size, status, last_heartbeat)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.node_id, node_info['host'], node_info['port'],
                  node_info['capacity'], node_info['current_size'],
                  node_info['status'], datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cache node registered: {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to register cache node: {e}")
            
    async def put(self, key: str, content: str, ttl: int = 3600, 
                 priority: int = 1) -> bool:
        """
        Put content into distributed cache
        Target: < 100ms for local cache, < 200ms for distributed
        """
        start_time = time.time()
        
        try:
            # Create cache entry
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            compressed_content = self._compress_content(content)
            
            entry = CacheEntry(
                key=key,
                content=compressed_content,
                content_hash=content_hash,
                size=len(compressed_content),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl=ttl,
                priority=priority,
                node_id=self.node_id
            )
            
            # Store in local cache
            with self.cache_lock:
                self.local_cache[key] = entry
                self._evict_if_necessary()
                
            # Store in distributed cache
            if self.redis_client:
                await self._put_distributed(key, entry)
                
            # Store in persistent cache
            await self._put_persistent(key, entry)
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['bytes_cached'] += entry.size
            
            operation_time = time.time() - start_time
            logger.debug(f"Cache put for {key} completed in {operation_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to put cache entry {key}: {e}")
            return False
            
    async def get(self, key: str) -> Optional[str]:
        """
        Get content from distributed cache
        Target: < 50ms for local cache, < 150ms for distributed
        """
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # Try local cache first
            with self.cache_lock:
                if key in self.local_cache:
                    entry = self.local_cache[key]
                    
                    # Check if expired
                    if self._is_expired(entry):
                        del self.local_cache[key]
                    else:
                        # Update access info
                        entry.last_accessed = datetime.now()
                        entry.access_count += 1
                        
                        # Move to end (LRU)
                        self.local_cache.move_to_end(key)
                        
                        self.stats['cache_hits'] += 1
                        self._update_hit_rate()
                        
                        decompressed = self._decompress_content(entry.content)
                        
                        operation_time = time.time() - start_time
                        logger.debug(f"Local cache hit for {key} in {operation_time:.3f}s")
                        
                        return decompressed
                        
            # Try distributed cache
            if self.redis_client:
                distributed_content = await self._get_distributed(key)
                if distributed_content:
                    # Store in local cache for future use
                    await self.put(key, distributed_content, ttl=3600)
                    
                    self.stats['cache_hits'] += 1
                    self._update_hit_rate()
                    
                    operation_time = time.time() - start_time
                    logger.debug(f"Distributed cache hit for {key} in {operation_time:.3f}s")
                    
                    return distributed_content
                    
            # Try persistent cache
            persistent_content = await self._get_persistent(key)
            if persistent_content:
                # Store in local cache for future use
                await self.put(key, persistent_content, ttl=3600)
                
                self.stats['cache_hits'] += 1
                self._update_hit_rate()
                
                operation_time = time.time() - start_time
                logger.debug(f"Persistent cache hit for {key} in {operation_time:.3f}s")
                
                return persistent_content
                
            # Cache miss
            self.stats['cache_misses'] += 1
            self._update_hit_rate()
            
            operation_time = time.time() - start_time
            logger.debug(f"Cache miss for {key} in {operation_time:.3f}s")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cache entry {key}: {e}")
            return None
            
    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across all nodes"""
        try:
            # Remove from local cache
            with self.cache_lock:
                if key in self.local_cache:
                    del self.local_cache[key]
                    
            # Remove from distributed cache
            if self.redis_client:
                await self._invalidate_distributed(key)
                
            # Remove from persistent cache
            await self._invalidate_persistent(key)
            
            logger.info(f"Cache entry invalidated: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache entry {key}: {e}")
            return False
            
    async def _put_distributed(self, key: str, entry: CacheEntry):
        """Store entry in distributed cache"""
        try:
            # Serialize entry
            entry_data = {
                'content': entry.content,
                'content_hash': entry.content_hash,
                'size': entry.size,
                'created_at': entry.created_at.isoformat(),
                'ttl': entry.ttl,
                'priority': entry.priority,
                'node_id': entry.node_id
            }
            
            # Store in Redis with TTL
            await self.redis_client.setex(
                f"cache:{key}",
                entry.ttl,
                json.dumps(entry_data)
            )
            
            # Replicate to other nodes
            await self._replicate_to_nodes(key, entry_data)
            
        except Exception as e:
            logger.error(f"Failed to put distributed cache entry: {e}")
            
    async def _get_distributed(self, key: str) -> Optional[str]:
        """Get entry from distributed cache"""
        try:
            # Get from Redis
            entry_json = await self.redis_client.get(f"cache:{key}")
            if entry_json:
                entry_data = json.loads(entry_json)
                content = entry_data['content']
                
                # Decompress if necessary
                return self._decompress_content(content)
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get distributed cache entry: {e}")
            return None
            
    async def _invalidate_distributed(self, key: str):
        """Invalidate entry in distributed cache"""
        try:
            # Remove from Redis
            await self.redis_client.delete(f"cache:{key}")
            
            # Notify other nodes
            await self._notify_nodes_invalidation(key)
            
        except Exception as e:
            logger.error(f"Failed to invalidate distributed cache entry: {e}")
            
    async def _put_persistent(self, key: str, entry: CacheEntry):
        """Store entry in persistent cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (key, content, content_hash, size, created_at, last_accessed, 
                 access_count, ttl, priority, node_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (key, entry.content, entry.content_hash, entry.size,
                  entry.created_at, entry.last_accessed, entry.access_count,
                  entry.ttl, entry.priority, entry.node_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to put persistent cache entry: {e}")
            
    async def _get_persistent(self, key: str) -> Optional[str]:
        """Get entry from persistent cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT content, created_at, ttl FROM cache_entries
                WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                content, created_at, ttl = result
                created_time = datetime.fromisoformat(created_at)
                
                # Check if expired
                if datetime.now() - created_time > timedelta(seconds=ttl):
                    await self._invalidate_persistent(key)
                    return None
                    
                return self._decompress_content(content)
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get persistent cache entry: {e}")
            return None
            
    async def _invalidate_persistent(self, key: str):
        """Invalidate entry in persistent cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to invalidate persistent cache entry: {e}")
            
    def _compress_content(self, content: str) -> str:
        """Compress content for storage"""
        try:
            compressed = zlib.compress(content.encode('utf-8'))
            compression_ratio = len(compressed) / len(content.encode('utf-8'))
            self.stats['compression_ratio'] = (
                self.stats['compression_ratio'] * 0.9 + compression_ratio * 0.1
            )
            return compressed.hex()
        except Exception as e:
            logger.error(f"Failed to compress content: {e}")
            return content
            
    def _decompress_content(self, compressed_content: str) -> str:
        """Decompress content from storage"""
        try:
            compressed_bytes = bytes.fromhex(compressed_content)
            decompressed = zlib.decompress(compressed_bytes)
            return decompressed.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decompress content: {e}")
            return compressed_content
            
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return (datetime.now() - entry.created_at).total_seconds() > entry.ttl
        
    def _evict_if_necessary(self):
        """Evict entries if cache is full (LRU)"""
        while len(self.local_cache) > self.max_local_size:
            # Remove least recently used item
            key, entry = self.local_cache.popitem(last=False)
            self.stats['evictions'] += 1
            logger.debug(f"Evicted cache entry: {key}")
            
    def _update_hit_rate(self):
        """Update cache hit rate metrics"""
        total_requests = self.stats['total_requests']
        if total_requests > 0:
            self.metrics.hit_rate = self.stats['cache_hits'] / total_requests
            self.metrics.miss_rate = self.stats['cache_misses'] / total_requests
            
    async def _replicate_to_nodes(self, key: str, entry_data: Dict):
        """Replicate entry to other nodes"""
        try:
            # This would implement replication logic
            # For now, we'll just log it
            logger.debug(f"Replicating {key} to {self.replication_factor} nodes")
            
        except Exception as e:
            logger.error(f"Failed to replicate entry: {e}")
            
    async def _notify_nodes_invalidation(self, key: str):
        """Notify other nodes about invalidation"""
        try:
            # This would implement invalidation notification
            logger.debug(f"Notifying nodes about invalidation of {key}")
            
        except Exception as e:
            logger.error(f"Failed to notify nodes about invalidation: {e}")
            
    async def cleanup_expired_entries(self):
        """Background task to cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.now()
                expired_keys = []
                
                with self.cache_lock:
                    for key, entry in self.local_cache.items():
                        if self._is_expired(entry):
                            expired_keys.append(key)
                            
                    for key in expired_keys:
                        del self.local_cache[key]
                        
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired entries")
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                
    async def send_heartbeat(self):
        """Send heartbeat to distributed cache cluster"""
        while True:
            try:
                await asyncio.sleep(30)  # Send every 30 seconds
                
                if self.redis_client:
                    heartbeat_data = {
                        'node_id': self.node_id,
                        'timestamp': datetime.now().isoformat(),
                        'cache_size': len(self.local_cache),
                        'stats': self.stats
                    }
                    
                    await self.redis_client.setex(
                        f"heartbeat:{self.node_id}",
                        60,  # 1 minute TTL
                        json.dumps(heartbeat_data)
                    )
                    
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                
    async def sync_with_cluster(self):
        """Sync with other nodes in the cluster"""
        while True:
            try:
                await asyncio.sleep(60)  # Sync every minute
                
                if self.redis_client:
                    # Get list of active nodes
                    node_keys = await self.redis_client.keys("heartbeat:*")
                    
                    active_nodes = []
                    for key in node_keys:
                        heartbeat_data = await self.redis_client.get(key)
                        if heartbeat_data:
                            node_info = json.loads(heartbeat_data)
                            active_nodes.append(node_info)
                            
                    logger.debug(f"Found {len(active_nodes)} active nodes")
                    
            except Exception as e:
                logger.error(f"Error syncing with cluster: {e}")
                
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'node_info': {
                'node_id': self.node_id,
                'local_cache_size': len(self.local_cache),
                'max_local_size': self.max_local_size
            },
            'performance_metrics': asdict(self.metrics),
            'statistics': self.stats,
            'cache_efficiency': {
                'hit_rate': f"{self.metrics.hit_rate:.2%}",
                'compression_ratio': f"{self.stats['compression_ratio']:.2f}",
                'storage_efficiency': f"{self.metrics.storage_efficiency:.2%}"
            }
        }
        
    async def shutdown(self):
        """Shutdown the cache system gracefully"""
        try:
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.sync_task:
                self.sync_task.cancel()
                
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
                
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("Cache system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Usage example
async def main():
    """Test the distributed context cache"""
    system_root = Path("/mnt/c/Users/RICHARD/OneDrive/Documents/crypto-rf-trading-system")
    cache = DistributedContextCache(system_root)
    
    # Initialize (will fall back to local cache if Redis not available)
    await cache.initialize_distributed_cache()
    
    # Test cache operations
    test_content = "This is a test context content for the Kelly criterion implementation"
    
    # Put content
    success = await cache.put("kelly_criterion", test_content, ttl=3600)
    print(f"Cache put success: {success}")
    
    # Get content
    retrieved = await cache.get("kelly_criterion")
    print(f"Cache get success: {retrieved is not None}")
    print(f"Content matches: {retrieved == test_content}")
    
    # Test cache miss
    missing = await cache.get("non_existent_key")
    print(f"Cache miss handled correctly: {missing is None}")
    
    # Get statistics
    stats = cache.get_cache_statistics()
    print("\nCache Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Cleanup
    await cache.shutdown()

if __name__ == "__main__":
    asyncio.run(main())