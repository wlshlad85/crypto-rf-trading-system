"""
ULTRATHINK Circuit Breaker System
Week 5 - DAY 33-34 Implementation

Advanced circuit breaker and fallback system for production trading system.
Implements fail-safe mechanisms, rate limiting, and automatic recovery.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from collections import deque, defaultdict
import logging
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.performance_profiler import PerformanceProfiler, profile


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, calls are blocked
    HALF_OPEN = "half_open" # Testing if service is recovered


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    name: str
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds to wait before trying recovery
    success_threshold: int = 3          # Successes needed to close circuit
    timeout_seconds: float = 30.0       # Operation timeout
    sliding_window_size: int = 100      # Window size for failure tracking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout_seconds,
            "sliding_window_size": self.sliding_window_size
        }


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        return self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "timeout_calls": self.timeout_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None
        }


class CircuitBreakerError(Exception):
    """Circuit breaker specific exception"""
    pass


class CircuitBreaker:
    """Individual circuit breaker implementation"""
    
    def __init__(self, config: CircuitConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.failure_window = deque(maxlen=config.sliding_window_size)
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"circuit_breaker.{config.name}")
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        
        def sync_wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        
        with self.lock:
            self.metrics.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit {self.config.name} transitioning to HALF_OPEN")
                else:
                    self.metrics.rejected_calls += 1
                    raise CircuitBreakerError(f"Circuit {self.config.name} is OPEN")
            
            # Execute function
            try:
                start_time = time.time()
                
                # Apply timeout
                result = self._execute_with_timeout(func, args, kwargs)
                
                execution_time = time.time() - start_time
                
                # Record success
                self._record_success()
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record failure
                self._record_failure(e)
                
                raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function through circuit breaker"""
        
        with self.lock:
            self.metrics.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit {self.config.name} transitioning to HALF_OPEN")
                else:
                    self.metrics.rejected_calls += 1
                    raise CircuitBreakerError(f"Circuit {self.config.name} is OPEN")
        
        # Execute function
        try:
            start_time = time.time()
            
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success()
            
            return result
            
        except asyncio.TimeoutError:
            self.metrics.timeout_calls += 1
            self._record_failure(TimeoutError("Operation timed out"))
            raise CircuitBreakerError(f"Operation timed out after {self.config.timeout_seconds}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure
            self._record_failure(e)
            
            raise
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with timeout for synchronous calls"""
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.timeout_seconds))
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _record_success(self):
        """Record successful execution"""
        
        with self.lock:
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = datetime.now()
            
            # Add success to sliding window
            self.failure_window.append(False)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                # Check if we should close the circuit
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    self.metrics.state_changes += 1
                    self.logger.info(f"Circuit {self.config.name} CLOSED after recovery")
    
    def _record_failure(self, exception: Exception):
        """Record failed execution"""
        
        with self.lock:
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = datetime.now()
            self.last_failure_time = time.time()
            
            # Add failure to sliding window
            self.failure_window.append(True)
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                failure_count = sum(self.failure_window)
                
                if failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.metrics.state_changes += 1
                    self.logger.warning(f"Circuit {self.config.name} OPENED due to failures")
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self.state = CircuitState.OPEN
                self.success_count = 0
                self.metrics.state_changes += 1
                self.logger.warning(f"Circuit {self.config.name} OPENED during recovery attempt")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    def reset(self):
        """Manually reset circuit breaker"""
        
        with self.lock:
            self.state = CircuitState.CLOSED
            self.success_count = 0
            self.failure_window.clear()
            self.last_failure_time = None
            self.metrics.state_changes += 1
            self.logger.info(f"Circuit {self.config.name} manually reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_metrics(self) -> CircuitMetrics:
        """Get circuit metrics"""
        return self.metrics


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_tokens: int, refill_rate: float, name: str = "rate_limiter"):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.name = name
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = threading.RLock()
        
        # Metrics
        self.total_requests = 0
        self.allowed_requests = 0
        self.rejected_requests = 0
        
        # Logging
        self.logger = logging.getLogger(f"rate_limiter.{name}")
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens"""
        
        with self.lock:
            self.total_requests += 1
            
            # Refill tokens
            self._refill_tokens()
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.allowed_requests += 1
                return True
            else:
                self.rejected_requests += 1
                self.logger.warning(f"Rate limit exceeded for {self.name}")
                return False
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        
        self.last_refill = now
    
    def get_available_tokens(self) -> int:
        """Get number of available tokens"""
        with self.lock:
            self._refill_tokens()
            return int(self.tokens)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics"""
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": self.rejected_requests / self.total_requests if self.total_requests > 0 else 0,
            "available_tokens": self.get_available_tokens(),
            "max_tokens": self.max_tokens,
            "refill_rate": self.refill_rate
        }


class FallbackSystem:
    """Fallback system for circuit breakers"""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.fallback_metrics: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger("fallback_system")
    
    def register_fallback(self, circuit_name: str, handler: Callable):
        """Register fallback handler for a circuit"""
        self.fallback_handlers[circuit_name] = handler
        self.logger.info(f"Registered fallback handler for {circuit_name}")
    
    def execute_fallback(self, circuit_name: str, *args, **kwargs) -> Any:
        """Execute fallback handler"""
        
        if circuit_name in self.fallback_handlers:
            self.fallback_metrics[circuit_name] += 1
            self.logger.info(f"Executing fallback for {circuit_name}")
            
            try:
                return self.fallback_handlers[circuit_name](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback failed for {circuit_name}: {e}")
                raise
        else:
            self.logger.error(f"No fallback handler registered for {circuit_name}")
            raise CircuitBreakerError(f"No fallback available for {circuit_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fallback metrics"""
        return {
            "registered_handlers": len(self.fallback_handlers),
            "fallback_executions": dict(self.fallback_metrics),
            "total_fallbacks": sum(self.fallback_metrics.values())
        }


class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.circuits: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.fallback_system = FallbackSystem()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("circuit_breaker_manager")
        
        # Trading specific configurations
        self._setup_trading_circuits()
    
    def _setup_trading_circuits(self):
        """Setup circuit breakers for trading system"""
        
        # Trading decision circuit
        self.create_circuit("trading_decision", CircuitConfig(
            name="trading_decision",
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2,
            timeout_seconds=5.0,
            sliding_window_size=50
        ))
        
        # Market data circuit
        self.create_circuit("market_data", CircuitConfig(
            name="market_data",
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=3,
            timeout_seconds=10.0,
            sliding_window_size=100
        ))
        
        # Order execution circuit
        self.create_circuit("order_execution", CircuitConfig(
            name="order_execution",
            failure_threshold=2,
            recovery_timeout=120,
            success_threshold=5,
            timeout_seconds=30.0,
            sliding_window_size=20
        ))
        
        # Risk management circuit
        self.create_circuit("risk_management", CircuitConfig(
            name="risk_management",
            failure_threshold=1,
            recovery_timeout=10,
            success_threshold=1,
            timeout_seconds=1.0,
            sliding_window_size=10
        ))
        
        # Setup rate limiters
        self.create_rate_limiter("trading_operations", max_tokens=100, refill_rate=10.0)
        self.create_rate_limiter("market_data_requests", max_tokens=1000, refill_rate=100.0)
        self.create_rate_limiter("order_submissions", max_tokens=50, refill_rate=5.0)
        
        # Setup fallback handlers
        self._setup_fallback_handlers()
    
    def _setup_fallback_handlers(self):
        """Setup fallback handlers for trading circuits"""
        
        # Trading decision fallback
        def trading_decision_fallback(*args, **kwargs):
            self.logger.info("Using conservative trading decision fallback")
            return {"action": "hold", "confidence": 0.0, "reason": "circuit_breaker_fallback"}
        
        # Market data fallback
        def market_data_fallback(*args, **kwargs):
            self.logger.info("Using cached market data fallback")
            return {"price": 50000, "volume": 0, "timestamp": time.time(), "source": "fallback"}
        
        # Order execution fallback
        def order_execution_fallback(*args, **kwargs):
            self.logger.info("Rejecting order due to circuit breaker")
            return {"status": "rejected", "reason": "circuit_breaker_protection"}
        
        # Risk management fallback
        def risk_management_fallback(*args, **kwargs):
            self.logger.info("Using conservative risk management fallback")
            return {"risk_ok": False, "reason": "circuit_breaker_protection"}
        
        # Register fallbacks
        self.fallback_system.register_fallback("trading_decision", trading_decision_fallback)
        self.fallback_system.register_fallback("market_data", market_data_fallback)
        self.fallback_system.register_fallback("order_execution", order_execution_fallback)
        self.fallback_system.register_fallback("risk_management", risk_management_fallback)
    
    def create_circuit(self, name: str, config: CircuitConfig) -> CircuitBreaker:
        """Create a new circuit breaker"""
        circuit = CircuitBreaker(config)
        self.circuits[name] = circuit
        self.logger.info(f"Created circuit breaker: {name}")
        return circuit
    
    def create_rate_limiter(self, name: str, max_tokens: int, refill_rate: float) -> RateLimiter:
        """Create a new rate limiter"""
        rate_limiter = RateLimiter(max_tokens, refill_rate, name)
        self.rate_limiters[name] = rate_limiter
        self.logger.info(f"Created rate limiter: {name}")
        return rate_limiter
    
    def get_circuit(self, name: str) -> CircuitBreaker:
        """Get circuit breaker by name"""
        if name not in self.circuits:
            raise ValueError(f"Circuit {name} not found")
        return self.circuits[name]
    
    def get_rate_limiter(self, name: str) -> RateLimiter:
        """Get rate limiter by name"""
        if name not in self.rate_limiters:
            raise ValueError(f"Rate limiter {name} not found")
        return self.rate_limiters[name]
    
    @profile("protected_trading_decision")
    def protected_trading_decision(self, func: Callable, *args, **kwargs) -> Any:
        """Execute trading decision with circuit breaker and rate limiting"""
        
        # Check rate limit
        rate_limiter = self.get_rate_limiter("trading_operations")
        if not rate_limiter.acquire():
            self.logger.warning("Trading operation rate limited")
            return self.fallback_system.execute_fallback("trading_decision", *args, **kwargs)
        
        # Execute through circuit breaker
        circuit = self.get_circuit("trading_decision")
        
        try:
            return circuit.call(func, *args, **kwargs)
        except CircuitBreakerError:
            return self.fallback_system.execute_fallback("trading_decision", *args, **kwargs)
    
    @profile("protected_order_execution")
    def protected_order_execution(self, func: Callable, *args, **kwargs) -> Any:
        """Execute order with circuit breaker and rate limiting"""
        
        # Check rate limit
        rate_limiter = self.get_rate_limiter("order_submissions")
        if not rate_limiter.acquire():
            self.logger.warning("Order submission rate limited")
            return self.fallback_system.execute_fallback("order_execution", *args, **kwargs)
        
        # Execute through circuit breaker
        circuit = self.get_circuit("order_execution")
        
        try:
            return circuit.call(func, *args, **kwargs)
        except CircuitBreakerError:
            return self.fallback_system.execute_fallback("order_execution", *args, **kwargs)
    
    def reset_all_circuits(self):
        """Reset all circuit breakers"""
        for circuit in self.circuits.values():
            circuit.reset()
        self.logger.info("All circuits reset")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "circuits": {},
            "rate_limiters": {},
            "fallback_system": self.fallback_system.get_metrics()
        }
        
        # Check circuit health
        for name, circuit in self.circuits.items():
            metrics = circuit.get_metrics()
            circuit_health = {
                "state": circuit.get_state().value,
                "healthy": circuit.get_state() != CircuitState.OPEN,
                "metrics": metrics.to_dict()
            }
            
            health["circuits"][name] = circuit_health
            
            if not circuit_health["healthy"]:
                health["overall_healthy"] = False
        
        # Check rate limiter health
        for name, rate_limiter in self.rate_limiters.items():
            metrics = rate_limiter.get_metrics()
            health["rate_limiters"][name] = metrics
        
        return health
    
    def export_metrics(self, output_path: str = "optimization/circuit_breaker_metrics.json"):
        """Export circuit breaker metrics"""
        
        health = self.get_system_health()
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(health, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")
        return health


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def circuit_breaker(name: str):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        circuit = circuit_manager.get_circuit(name)
        return circuit(func)
    return decorator


def rate_limited(name: str, tokens: int = 1):
    """Decorator for rate limiting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            rate_limiter = circuit_manager.get_rate_limiter(name)
            if rate_limiter.acquire(tokens):
                return func(*args, **kwargs)
            else:
                raise CircuitBreakerError(f"Rate limit exceeded for {name}")
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo the circuit breaker system
    import random
    
    # Create manager
    manager = CircuitBreakerManager()
    
    # Test functions
    @circuit_breaker("trading_decision")
    def make_trading_decision(symbol: str) -> Dict[str, Any]:
        """Mock trading decision function"""
        
        # Simulate occasional failures
        if random.random() < 0.3:
            raise Exception("Market data unavailable")
        
        return {
            "action": "buy",
            "symbol": symbol,
            "confidence": 0.8,
            "timestamp": time.time()
        }
    
    @rate_limited("trading_operations", tokens=2)
    def execute_trade(symbol: str, amount: float) -> Dict[str, Any]:
        """Mock trade execution function"""
        
        # Simulate execution
        time.sleep(0.1)
        
        return {
            "status": "executed",
            "symbol": symbol,
            "amount": amount,
            "timestamp": time.time()
        }
    
    # Test circuit breaker
    print("Testing Circuit Breaker System")
    print("=" * 50)
    
    # Test normal operations
    for i in range(10):
        try:
            decision = make_trading_decision("BTC/USD")
            print(f"Decision {i+1}: {decision['action']}")
        except Exception as e:
            print(f"Decision {i+1}: Failed - {e}")
    
    # Test rate limiting
    print("\nTesting Rate Limiting")
    print("=" * 30)
    
    for i in range(5):
        try:
            result = execute_trade("BTC/USD", 0.1)
            print(f"Trade {i+1}: {result['status']}")
        except Exception as e:
            print(f"Trade {i+1}: Failed - {e}")
    
    # Get system health
    health = manager.get_system_health()
    print(f"\nSystem Health: {'HEALTHY' if health['overall_healthy'] else 'UNHEALTHY'}")
    
    # Export metrics
    manager.export_metrics()
    
    print("\nCircuit Breaker Demo Complete")