"""
ULTRATHINK Fallback Systems
Week 5 - DAY 33-34 Implementation

Advanced fallback mechanisms for production trading system.
Provides degraded mode operation, backup data sources, emergency controls,
and manual override capabilities.
"""

import time
import asyncio
import json
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import logging
import sys
import os
import sqlite3
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_profiler import PerformanceProfiler, profile


class OperationMode(Enum):
    """System operation modes"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class FallbackTrigger(Enum):
    """Fallback trigger types"""
    CIRCUIT_BREAKER = "circuit_breaker"
    TIMEOUT = "timeout"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class FallbackEvent:
    """Fallback event record"""
    event_id: str
    trigger: FallbackTrigger
    component: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "trigger": self.trigger.value,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


class BackupDataSource:
    """Backup data source manager"""
    
    def __init__(self, name: str, cache_size: int = 1000):
        self.name = name
        self.cache_size = cache_size
        self.data_cache = deque(maxlen=cache_size)
        self.last_update = None
        self.is_stale = False
        self.staleness_threshold = 60  # seconds
        
    def update_cache(self, data: Any):
        """Update data cache"""
        self.data_cache.append({
            'data': data,
            'timestamp': time.time()
        })
        self.last_update = time.time()
        self.is_stale = False
    
    def get_latest_data(self) -> Optional[Any]:
        """Get latest cached data"""
        if not self.data_cache:
            return None
        
        # Check staleness
        if self.last_update:
            age = time.time() - self.last_update
            self.is_stale = age > self.staleness_threshold
        
        return self.data_cache[-1]['data']
    
    def get_historical_data(self, seconds_back: int = 300) -> List[Any]:
        """Get historical data from cache"""
        cutoff_time = time.time() - seconds_back
        
        historical = []
        for item in self.data_cache:
            if item['timestamp'] >= cutoff_time:
                historical.append(item['data'])
        
        return historical


class EmergencyControls:
    """Emergency trading controls"""
    
    def __init__(self):
        self.emergency_active = False
        self.kill_switch_active = False
        self.position_limits = {
            'max_position_size': 10000,
            'max_total_exposure': 50000,
            'max_drawdown': 0.15
        }
        self.emergency_actions = []
        self.logger = logging.getLogger("emergency_controls")
        
    def activate_emergency_mode(self, reason: str):
        """Activate emergency mode"""
        self.emergency_active = True
        self.emergency_actions.append({
            'action': 'emergency_activated',
            'reason': reason,
            'timestamp': datetime.now()
        })
        self.logger.critical(f"Emergency mode activated: {reason}")
    
    def activate_kill_switch(self, reason: str):
        """Activate kill switch - stop all trading"""
        self.kill_switch_active = True
        self.emergency_active = True
        self.emergency_actions.append({
            'action': 'kill_switch_activated',
            'reason': reason,
            'timestamp': datetime.now()
        })
        self.logger.critical(f"Kill switch activated: {reason}")
    
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Emergency position closure"""
        if not self.emergency_active:
            return []
        
        # Mock position closure
        closed_positions = [
            {
                'symbol': 'BTC/USD',
                'size': 0.5,
                'close_price': 50000,
                'timestamp': datetime.now()
            },
            {
                'symbol': 'ETH/USD',
                'size': 10,
                'close_price': 3000,
                'timestamp': datetime.now()
            }
        ]
        
        self.emergency_actions.append({
            'action': 'positions_closed',
            'positions': closed_positions,
            'timestamp': datetime.now()
        })
        
        self.logger.warning(f"Emergency position closure: {len(closed_positions)} positions")
        return closed_positions
    
    def reduce_position_limits(self, reduction_factor: float = 0.5):
        """Reduce position limits during emergency"""
        for limit_type, limit_value in self.position_limits.items():
            if isinstance(limit_value, (int, float)):
                self.position_limits[limit_type] = limit_value * reduction_factor
        
        self.emergency_actions.append({
            'action': 'limits_reduced',
            'reduction_factor': reduction_factor,
            'new_limits': self.position_limits.copy(),
            'timestamp': datetime.now()
        })
        
        self.logger.warning(f"Position limits reduced by {reduction_factor}")
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed"""
        return not self.kill_switch_active
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """Get emergency status"""
        return {
            'emergency_active': self.emergency_active,
            'kill_switch_active': self.kill_switch_active,
            'position_limits': self.position_limits.copy(),
            'recent_actions': self.emergency_actions[-10:],  # Last 10 actions
            'trading_allowed': self.is_trading_allowed()
        }


class DegradedModeController:
    """Controls degraded mode operations"""
    
    def __init__(self):
        self.degraded_mode_active = False
        self.degraded_features = set()
        self.performance_limits = {
            'max_operations_per_second': 100,
            'max_concurrent_operations': 10,
            'simplified_risk_checks': True,
            'disable_complex_strategies': True
        }
        self.logger = logging.getLogger("degraded_mode")
        
    def activate_degraded_mode(self, features_to_disable: List[str]):
        """Activate degraded mode"""
        self.degraded_mode_active = True
        self.degraded_features.update(features_to_disable)
        self.logger.warning(f"Degraded mode activated, disabled: {features_to_disable}")
    
    def deactivate_degraded_mode(self):
        """Deactivate degraded mode"""
        self.degraded_mode_active = False
        self.degraded_features.clear()
        self.logger.info("Degraded mode deactivated")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled"""
        return feature not in self.degraded_features
    
    def get_performance_limit(self, limit_type: str) -> Any:
        """Get performance limit"""
        return self.performance_limits.get(limit_type)
    
    def simple_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified trading decision for degraded mode"""
        # Very basic decision logic
        price = market_data.get('price', 0)
        
        if price > 50000:
            return {
                'action': 'hold',
                'reason': 'degraded_mode_conservative',
                'confidence': 0.5
            }
        else:
            return {
                'action': 'buy',
                'reason': 'degraded_mode_basic',
                'confidence': 0.6
            }


class ManualOverrides:
    """Manual override system"""
    
    def __init__(self):
        self.active_overrides = {}
        self.override_history = []
        self.authorized_users = {'admin', 'operator', 'risk_manager'}
        self.logger = logging.getLogger("manual_overrides")
        
    def set_override(self, user: str, component: str, action: str, 
                    duration_minutes: int = 60, reason: str = ""):
        """Set manual override"""
        
        if user not in self.authorized_users:
            raise ValueError(f"User {user} not authorized for overrides")
        
        override_id = f"{component}_{action}_{int(time.time())}"
        expiry_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        override = {
            'id': override_id,
            'user': user,
            'component': component,
            'action': action,
            'reason': reason,
            'created': datetime.now(),
            'expires': expiry_time,
            'active': True
        }
        
        self.active_overrides[override_id] = override
        self.override_history.append(override.copy())
        
        self.logger.warning(f"Manual override set by {user}: {component}.{action}")
        return override_id
    
    def remove_override(self, override_id: str, user: str):
        """Remove manual override"""
        
        if override_id in self.active_overrides:
            override = self.active_overrides[override_id]
            override['active'] = False
            override['removed_by'] = user
            override['removed_at'] = datetime.now()
            
            del self.active_overrides[override_id]
            self.logger.info(f"Manual override removed by {user}: {override_id}")
    
    def is_override_active(self, component: str, action: str) -> bool:
        """Check if override is active"""
        
        # Clean up expired overrides
        self._cleanup_expired_overrides()
        
        for override in self.active_overrides.values():
            if override['component'] == component and override['action'] == action:
                return True
        
        return False
    
    def _cleanup_expired_overrides(self):
        """Remove expired overrides"""
        now = datetime.now()
        expired_ids = []
        
        for override_id, override in self.active_overrides.items():
            if now > override['expires']:
                expired_ids.append(override_id)
        
        for override_id in expired_ids:
            override = self.active_overrides[override_id]
            override['active'] = False
            override['expired'] = True
            del self.active_overrides[override_id]
    
    def get_active_overrides(self) -> List[Dict[str, Any]]:
        """Get active overrides"""
        self._cleanup_expired_overrides()
        return list(self.active_overrides.values())


class FallbackSystemManager:
    """Main fallback system manager"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.operation_mode = OperationMode.NORMAL
        
        # Core components
        self.emergency_controls = EmergencyControls()
        self.degraded_mode = DegradedModeController()
        self.manual_overrides = ManualOverrides()
        
        # Backup data sources
        self.backup_sources = {}
        self._setup_backup_sources()
        
        # Fallback events
        self.fallback_events = []
        self.event_counter = 0
        
        # Monitoring
        self.health_checks = {}
        self.monitoring_active = False
        
        # Database for persistence
        self.db_path = "optimization/fallback_system.db"
        self._init_database()
        
        # Logger
        self.logger = logging.getLogger("fallback_system_manager")
        
    def _setup_backup_sources(self):
        """Setup backup data sources"""
        self.backup_sources = {
            'market_data': BackupDataSource('market_data', 1000),
            'price_feeds': BackupDataSource('price_feeds', 500),
            'portfolio_state': BackupDataSource('portfolio_state', 100),
            'risk_metrics': BackupDataSource('risk_metrics', 200)
        }
    
    def _init_database(self):
        """Initialize fallback events database"""
        Path(self.db_path).parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fallback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    trigger TEXT,
                    component TEXT,
                    timestamp TEXT,
                    details TEXT,
                    resolved BOOLEAN,
                    resolution_time TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    operation_mode TEXT,
                    emergency_active BOOLEAN,
                    degraded_active BOOLEAN,
                    details TEXT
                )
            """)
    
    def trigger_fallback(self, trigger: FallbackTrigger, component: str, 
                        details: Dict[str, Any] = None) -> str:
        """Trigger fallback mechanism"""
        
        event_id = f"fallback_{self.event_counter}_{int(time.time())}"
        self.event_counter += 1
        
        event = FallbackEvent(
            event_id=event_id,
            trigger=trigger,
            component=component,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self.fallback_events.append(event)
        
        # Store in database
        self._store_fallback_event(event)
        
        # Execute fallback logic
        self._execute_fallback_logic(event)
        
        self.logger.warning(f"Fallback triggered: {component} due to {trigger.value}")
        return event_id
    
    def _execute_fallback_logic(self, event: FallbackEvent):
        """Execute fallback logic based on trigger and component"""
        
        if event.trigger == FallbackTrigger.CIRCUIT_BREAKER:
            self._handle_circuit_breaker_fallback(event)
        elif event.trigger == FallbackTrigger.TIMEOUT:
            self._handle_timeout_fallback(event)
        elif event.trigger == FallbackTrigger.ERROR_RATE:
            self._handle_error_rate_fallback(event)
        elif event.trigger == FallbackTrigger.RESOURCE_EXHAUSTION:
            self._handle_resource_exhaustion_fallback(event)
        elif event.trigger == FallbackTrigger.MANUAL:
            self._handle_manual_fallback(event)
    
    def _handle_circuit_breaker_fallback(self, event: FallbackEvent):
        """Handle circuit breaker fallback"""
        
        component = event.component
        
        if component == "trading_decision":
            # Switch to degraded mode with simplified decisions
            self.degraded_mode.activate_degraded_mode(['complex_strategies', 'ml_models'])
        
        elif component == "market_data":
            # Use backup data source
            backup_data = self.backup_sources['market_data'].get_latest_data()
            if backup_data:
                self.logger.info("Using backup market data")
            else:
                self.emergency_controls.activate_emergency_mode("No backup market data available")
        
        elif component == "order_execution":
            # Reduce position limits and activate emergency controls
            self.emergency_controls.reduce_position_limits(0.3)
            self.emergency_controls.activate_emergency_mode("Order execution circuit breaker")
        
        elif component == "risk_management":
            # Activate kill switch - too dangerous to continue
            self.emergency_controls.activate_kill_switch("Risk management system failure")
    
    def _handle_timeout_fallback(self, event: FallbackEvent):
        """Handle timeout fallback"""
        
        # Switch to degraded mode to reduce system load
        self.degraded_mode.activate_degraded_mode(['real_time_processing', 'complex_calculations'])
        
        # Reduce performance limits
        self.degraded_mode.performance_limits['max_operations_per_second'] = 50
        self.degraded_mode.performance_limits['max_concurrent_operations'] = 5
    
    def _handle_error_rate_fallback(self, event: FallbackEvent):
        """Handle high error rate fallback"""
        
        error_rate = event.details.get('error_rate', 0)
        
        if error_rate > 0.1:  # 10% error rate
            self.emergency_controls.activate_emergency_mode("High error rate detected")
            self.degraded_mode.activate_degraded_mode(['automated_trading', 'position_scaling'])
        
        elif error_rate > 0.05:  # 5% error rate
            self.degraded_mode.activate_degraded_mode(['complex_strategies'])
    
    def _handle_resource_exhaustion_fallback(self, event: FallbackEvent):
        """Handle resource exhaustion fallback"""
        
        resource_type = event.details.get('resource_type', 'unknown')
        
        if resource_type == 'memory':
            # Clear caches and reduce operations
            for backup_source in self.backup_sources.values():
                backup_source.data_cache.clear()
            
            self.degraded_mode.activate_degraded_mode(['data_caching', 'historical_analysis'])
        
        elif resource_type == 'cpu':
            # Reduce computational complexity
            self.degraded_mode.activate_degraded_mode(['complex_calculations', 'ml_inference'])
        
        elif resource_type == 'network':
            # Use backup data sources
            self.degraded_mode.activate_degraded_mode(['real_time_data', 'external_apis'])
    
    def _handle_manual_fallback(self, event: FallbackEvent):
        """Handle manual fallback"""
        
        action = event.details.get('action', 'unknown')
        
        if action == 'emergency_stop':
            self.emergency_controls.activate_kill_switch("Manual emergency stop")
        elif action == 'degraded_mode':
            features = event.details.get('features', [])
            self.degraded_mode.activate_degraded_mode(features)
        elif action == 'close_positions':
            self.emergency_controls.close_all_positions()
    
    def _store_fallback_event(self, event: FallbackEvent):
        """Store fallback event in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fallback_events 
                (event_id, trigger, component, timestamp, details, resolved, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.trigger.value,
                event.component,
                event.timestamp.isoformat(),
                json.dumps(event.details),
                event.resolved,
                event.resolution_time.isoformat() if event.resolution_time else None
            ))
    
    def resolve_fallback_event(self, event_id: str):
        """Resolve fallback event"""
        
        for event in self.fallback_events:
            if event.event_id == event_id:
                event.resolved = True
                event.resolution_time = datetime.now()
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE fallback_events 
                        SET resolved = ?, resolution_time = ?
                        WHERE event_id = ?
                    """, (True, event.resolution_time.isoformat(), event_id))
                
                self.logger.info(f"Fallback event resolved: {event_id}")
                return True
        
        return False
    
    @profile("fallback_trading_decision")
    def get_fallback_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback trading decision"""
        
        # Check if manual override is active
        if self.manual_overrides.is_override_active('trading', 'disable'):
            return {
                'action': 'hold',
                'reason': 'manual_override',
                'confidence': 0.0
            }
        
        # Check emergency controls
        if not self.emergency_controls.is_trading_allowed():
            return {
                'action': 'hold',
                'reason': 'emergency_controls',
                'confidence': 0.0
            }
        
        # Use degraded mode decision if active
        if self.degraded_mode.degraded_mode_active:
            return self.degraded_mode.simple_trading_decision(market_data)
        
        # Normal fallback decision
        return {
            'action': 'hold',
            'reason': 'fallback_conservative',
            'confidence': 0.3
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'operation_mode': self.operation_mode.value,
            'emergency_controls': self.emergency_controls.get_emergency_status(),
            'degraded_mode': {
                'active': self.degraded_mode.degraded_mode_active,
                'disabled_features': list(self.degraded_mode.degraded_features),
                'performance_limits': self.degraded_mode.performance_limits
            },
            'manual_overrides': {
                'active_count': len(self.manual_overrides.get_active_overrides()),
                'active_overrides': self.manual_overrides.get_active_overrides()
            },
            'recent_events': [
                event.to_dict() for event in self.fallback_events[-10:]
            ],
            'backup_sources': {
                name: {
                    'cache_size': len(source.data_cache),
                    'is_stale': source.is_stale,
                    'last_update': source.last_update
                }
                for name, source in self.backup_sources.items()
            }
        }
    
    def export_status_report(self, output_path: str = "optimization/fallback_system_report.json"):
        """Export system status report"""
        
        status = self.get_system_status()
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        self.logger.info(f"Status report exported to {output_path}")
        return status


# Global fallback system manager
fallback_manager = FallbackSystemManager()


if __name__ == "__main__":
    # Demo the fallback system
    manager = FallbackSystemManager()
    
    print("ULTRATHINK Fallback System Demo")
    print("=" * 50)
    
    # Simulate market data
    market_data = {
        'price': 48000,
        'volume': 1000,
        'timestamp': time.time()
    }
    
    # Update backup data source
    manager.backup_sources['market_data'].update_cache(market_data)
    
    # Test normal operation
    print("\n1. Normal Operation:")
    decision = manager.get_fallback_trading_decision(market_data)
    print(f"   Decision: {decision}")
    
    # Test circuit breaker fallback
    print("\n2. Circuit Breaker Fallback:")
    event_id = manager.trigger_fallback(
        FallbackTrigger.CIRCUIT_BREAKER,
        "trading_decision",
        {'error_count': 5}
    )
    decision = manager.get_fallback_trading_decision(market_data)
    print(f"   Decision: {decision}")
    
    # Test manual override
    print("\n3. Manual Override:")
    manager.manual_overrides.set_override(
        'admin', 'trading', 'disable', 30, 'Testing override'
    )
    decision = manager.get_fallback_trading_decision(market_data)
    print(f"   Decision: {decision}")
    
    # Test emergency controls
    print("\n4. Emergency Controls:")
    manager.emergency_controls.activate_kill_switch("Demo emergency")
    decision = manager.get_fallback_trading_decision(market_data)
    print(f"   Decision: {decision}")
    
    # Get system status
    print("\n5. System Status:")
    status = manager.get_system_status()
    print(f"   Mode: {status['operation_mode']}")
    print(f"   Emergency: {status['emergency_controls']['emergency_active']}")
    print(f"   Degraded: {status['degraded_mode']['active']}")
    print(f"   Events: {len(status['recent_events'])}")
    
    # Export report
    manager.export_status_report()
    
    print("\nFallback System Demo Complete")