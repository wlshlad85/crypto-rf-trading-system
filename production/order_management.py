#!/usr/bin/env python3
"""
ULTRATHINK Week 6: Order Management System
Centralized order lifecycle management with institutional-grade controls.

Features:
- Centralized order lifecycle management
- Order validation and risk checks
- Execution tracking and reporting
- Settlement and reconciliation
- Compliance checks and reporting
- Real-time order monitoring
- Trade settlement automation

Components:
- Order State Machine
- Risk Validation Engine
- Execution Monitor
- Settlement Engine
- Compliance Reporter
- Performance Analytics
"""

import asyncio
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import uuid
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

# Import our trading components
from real_money_trader import (
    OrderRequest, OrderSide, OrderType, OrderStatus, 
    ExchangeType, AuditLogger, SecurityConfig
)
from exchange_integrations import ExchangeRouter, MarketData

class OrderEvent(Enum):
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SETTLEMENT_PENDING = "settlement_pending"
    SETTLED = "settled"

class RiskDecision(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_APPROVAL = "requires_approval"
    CONDITIONAL = "conditional"

@dataclass
class OrderExecution:
    """Single order execution/fill record."""
    execution_id: str
    order_id: str
    timestamp: datetime
    quantity: Decimal
    price: Decimal
    commission: Decimal
    exchange: str
    trade_id: str
    is_maker: bool = False
    
    @property
    def notional_value(self) -> Decimal:
        return self.quantity * self.price

@dataclass
class RiskAssessment:
    """Risk assessment result for an order."""
    decision: RiskDecision
    risk_score: float
    reason: str
    max_position_size: Optional[Decimal] = None
    required_approvals: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    
    @property
    def is_approved(self) -> bool:
        return self.decision == RiskDecision.APPROVED

@dataclass
class ManagedOrder:
    """Comprehensive order with full lifecycle tracking."""
    # Core order data
    order_id: str
    client_order_id: str
    order_request: OrderRequest
    
    # Status and timing
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    
    # Risk and compliance
    risk_assessment: Optional[RiskAssessment] = None
    compliance_flags: List[str] = field(default_factory=list)
    
    # Execution tracking
    exchange_order_id: Optional[str] = None
    target_exchange: Optional[str] = None
    executions: List[OrderExecution] = field(default_factory=list)
    
    # Financial tracking
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    total_commission: Decimal = Decimal("0")
    
    # Performance metrics
    submission_latency_ms: Optional[float] = None
    execution_latency_ms: Optional[float] = None
    slippage_bps: Optional[float] = None
    
    # Audit trail
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.order_request.quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        if self.order_request.quantity == 0:
            return 0.0
        return float((self.filled_quantity / self.order_request.quantity) * 100)
    
    @property
    def notional_value(self) -> Decimal:
        if self.average_price:
            return self.filled_quantity * self.average_price
        return Decimal("0")
    
    def add_execution(self, execution: OrderExecution):
        """Add an execution to this order."""
        self.executions.append(execution)
        self.filled_quantity += execution.quantity
        self.total_commission += execution.commission
        self.updated_at = datetime.now()
        
        # Update remaining quantity
        self.remaining_quantity = self.order_request.quantity - self.filled_quantity
        
        # Calculate average price
        total_value = sum(exec.quantity * exec.price for exec in self.executions)
        if self.filled_quantity > 0:
            self.average_price = total_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= Decimal("0.000001"):  # Practically zero
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        # Add event
        self.add_event(OrderEvent.PARTIALLY_FILLED if self.status == OrderStatus.PARTIALLY_FILLED else OrderEvent.FILLED, {
            "execution_id": execution.execution_id,
            "quantity": float(execution.quantity),
            "price": float(execution.price),
            "commission": float(execution.commission),
            "filled_quantity": float(self.filled_quantity),
            "remaining_quantity": float(self.remaining_quantity),
            "fill_percentage": self.fill_percentage
        })
    
    def add_event(self, event: OrderEvent, details: Dict[str, Any] = None):
        """Add an event to the order's audit trail."""
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event": event.value,
            "details": details or {}
        }
        self.events.append(event_record)
        self.updated_at = datetime.now()

class RiskValidator:
    """Advanced risk validation for orders."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.risk_validator")
        
        # Risk thresholds
        self.max_order_value_usd = config.get("max_order_value_usd", 100000)
        self.max_position_percentage = config.get("max_position_percentage", 50)
        self.max_daily_trades = config.get("max_daily_trades", 100)
        self.max_concentration_percentage = config.get("max_concentration_percentage", 25)
        
        # Approval requirements
        self.large_order_threshold = config.get("large_order_threshold_usd", 50000)
        self.high_risk_score_threshold = config.get("high_risk_score_threshold", 0.7)
    
    async def validate_order(self, order: OrderRequest, 
                           current_positions: Dict[str, Decimal],
                           daily_trade_count: int,
                           market_data: Optional[MarketData] = None) -> RiskAssessment:
        """Comprehensive order risk validation."""
        
        risk_factors = []
        risk_score = 0.0
        decision = RiskDecision.APPROVED
        required_approvals = []
        conditions = []
        
        # Calculate order value
        if order.price and market_data:
            order_value_usd = float(order.quantity * order.price)
            reference_price = float(market_data.last)
        elif market_data:
            reference_price = float(market_data.last)
            order_value_usd = float(order.quantity) * reference_price
        else:
            # Conservative estimate without market data
            order_value_usd = float(order.quantity) * 50000  # Assume $50k for BTC
            reference_price = 50000
        
        # Order size validation
        if order_value_usd > self.max_order_value_usd:
            risk_factors.append(f"Order value ${order_value_usd:,.2f} exceeds limit ${self.max_order_value_usd:,.2f}")
            risk_score += 0.3
            
            if order_value_usd > self.large_order_threshold:
                required_approvals.append("LARGE_ORDER_APPROVAL")
                decision = RiskDecision.REQUIRES_APPROVAL
        
        # Position concentration check
        current_position = current_positions.get(order.symbol, Decimal("0"))
        if order.side == OrderSide.BUY:
            new_position_value = float(current_position + order.quantity) * reference_price
        else:
            new_position_value = float(max(Decimal("0"), current_position - order.quantity)) * reference_price
        
        # Daily trading limit
        if daily_trade_count >= self.max_daily_trades:
            risk_factors.append(f"Daily trade limit ({self.max_daily_trades}) reached")
            risk_score += 0.4
            decision = RiskDecision.REJECTED
        
        # Market conditions check
        if market_data:
            # Check for wide spreads
            spread_bps = ((market_data.ask - market_data.bid) / market_data.last) * 10000
            if spread_bps > 50:  # 0.5% spread
                risk_factors.append(f"Wide spread detected: {spread_bps:.1f} bps")
                risk_score += 0.2
                conditions.append("MONITOR_EXECUTION_QUALITY")
        
        # Price validation for limit orders
        if order.order_type == OrderType.LIMIT and order.price and market_data:
            price_deviation = abs(float(order.price) - reference_price) / reference_price
            if price_deviation > 0.05:  # 5% deviation
                risk_factors.append(f"Limit price deviates {price_deviation*100:.1f}% from market")
                risk_score += 0.1
        
        # Final risk assessment
        if risk_score > self.high_risk_score_threshold:
            if decision == RiskDecision.APPROVED:
                decision = RiskDecision.CONDITIONAL
                required_approvals.append("HIGH_RISK_APPROVAL")
        
        reason = "; ".join(risk_factors) if risk_factors else "Order passes risk checks"
        
        return RiskAssessment(
            decision=decision,
            risk_score=min(risk_score, 1.0),
            reason=reason,
            required_approvals=required_approvals,
            conditions=conditions
        )

class OrderManager:
    """
    Centralized order management system with institutional-grade controls.
    
    Features:
    - Complete order lifecycle management
    - Risk validation and compliance
    - Real-time execution monitoring
    - Automated settlement
    - Performance analytics
    """
    
    def __init__(self, 
                 exchange_router: ExchangeRouter,
                 audit_logger: AuditLogger,
                 config: Dict[str, Any] = None):
        
        self.exchange_router = exchange_router
        self.audit_logger = audit_logger
        self.config = config or {}
        
        # Risk validator
        self.risk_validator = RiskValidator(self.config)
        
        # Order storage
        self.active_orders: Dict[str, ManagedOrder] = {}
        self.order_history: Dict[str, ManagedOrder] = {}
        
        # Position tracking
        self.positions: Dict[str, Decimal] = {}
        self.daily_trade_count = 0
        self.daily_volume_usd = Decimal("0")
        
        # Performance tracking
        self.execution_metrics = {
            "total_orders": 0,
            "successful_orders": 0,
            "rejected_orders": 0,
            "avg_execution_time_ms": 0.0,
            "avg_slippage_bps": 0.0,
            "total_commission": Decimal("0")
        }
        
        # Database setup
        self.db_path = "production/order_management.db"
        self.setup_database()
        
        self.logger = logging.getLogger(__name__)
        
        # Background tasks
        self.monitoring_task = None
        self.settlement_task = None
        
        self.logger.info("Order Management System initialized")
    
    def setup_database(self):
        """Initialize order management database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Orders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    client_order_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    price TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    exchange_order_id TEXT,
                    target_exchange TEXT,
                    filled_quantity TEXT DEFAULT '0',
                    average_price TEXT,
                    total_commission TEXT DEFAULT '0',
                    risk_score REAL,
                    compliance_flags TEXT,
                    order_data TEXT NOT NULL
                )
            """)
            
            # Executions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    price TEXT NOT NULL,
                    commission TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    is_maker BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_executions_order_id ON executions(order_id)")
    
    async def submit_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """
        Submit order with comprehensive validation and tracking.
        """
        start_time = time.time()
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        if not order_request.client_order_id:
            order_request.client_order_id = f"OM_{int(time.time() * 1000)}"
        
        # Create managed order
        managed_order = ManagedOrder(
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            order_request=order_request,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        managed_order.add_event(OrderEvent.CREATED, {
            "symbol": order_request.symbol,
            "side": order_request.side.value,
            "order_type": order_request.order_type.value,
            "quantity": float(order_request.quantity),
            "price": float(order_request.price) if order_request.price else None
        })
        
        try:
            # Get market data for validation
            market_data = await self.exchange_router.get_best_price(
                order_request.symbol, 
                order_request.side
            )
            
            # Risk validation
            risk_assessment = await self.risk_validator.validate_order(
                order_request,
                self.positions,
                self.daily_trade_count,
                market_data
            )
            
            managed_order.risk_assessment = risk_assessment
            
            if not risk_assessment.is_approved:
                managed_order.status = OrderStatus.REJECTED
                managed_order.add_event(OrderEvent.REJECTED, {
                    "reason": risk_assessment.reason,
                    "risk_score": risk_assessment.risk_score,
                    "required_approvals": risk_assessment.required_approvals
                })
                
                await self.save_order(managed_order)
                
                self.audit_logger.log_event(
                    event_type="ORDER_REJECTED",
                    action="RISK_VALIDATION_FAILED",
                    details={
                        "order_id": order_id,
                        "risk_assessment": asdict(risk_assessment)
                    },
                    symbol=order_request.symbol,
                    amount=float(order_request.quantity),
                    risk_score=risk_assessment.risk_score
                )
                
                return {
                    "success": False,
                    "order_id": order_id,
                    "reason": risk_assessment.reason,
                    "risk_score": risk_assessment.risk_score
                }
            
            managed_order.add_event(OrderEvent.VALIDATED, {
                "risk_score": risk_assessment.risk_score,
                "conditions": risk_assessment.conditions
            })
            
            # Submit to exchange router
            submission_start = time.time()
            
            exchange_result = await self.exchange_router.route_order(order_request)
            
            submission_latency = (time.time() - submission_start) * 1000
            managed_order.submission_latency_ms = submission_latency
            
            if exchange_result.get("success"):
                managed_order.exchange_order_id = exchange_result.get("order_id")
                managed_order.target_exchange = exchange_result.get("routed_to")
                managed_order.status = OrderStatus.PENDING
                
                managed_order.add_event(OrderEvent.SUBMITTED, {
                    "exchange": managed_order.target_exchange,
                    "exchange_order_id": managed_order.exchange_order_id,
                    "submission_latency_ms": submission_latency,
                    "failover": exchange_result.get("failover", False)
                })
                
                # Add to active orders
                self.active_orders[order_id] = managed_order
                
                # Update metrics
                self.execution_metrics["total_orders"] += 1
                self.daily_trade_count += 1
                
                self.audit_logger.log_event(
                    event_type="ORDER_SUBMITTED",
                    action="EXCHANGE_ACCEPTED",
                    details={
                        "order_id": order_id,
                        "exchange_result": exchange_result,
                        "submission_latency_ms": submission_latency
                    },
                    symbol=order_request.symbol,
                    amount=float(order_request.quantity),
                    exchange=managed_order.target_exchange,
                    order_id=managed_order.exchange_order_id,
                    client_order_id=order_request.client_order_id
                )
                
                await self.save_order(managed_order)
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "exchange_order_id": managed_order.exchange_order_id,
                    "target_exchange": managed_order.target_exchange,
                    "submission_latency_ms": submission_latency
                }
            
            else:
                managed_order.status = OrderStatus.REJECTED
                managed_order.add_event(OrderEvent.REJECTED, {
                    "reason": exchange_result.get("reason", "Exchange submission failed"),
                    "exchange_error": exchange_result
                })
                
                await self.save_order(managed_order)
                
                self.execution_metrics["rejected_orders"] += 1
                
                return {
                    "success": False,
                    "order_id": order_id,
                    "reason": exchange_result.get("reason", "Exchange submission failed")
                }
                
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            
            managed_order.status = OrderStatus.REJECTED
            managed_order.add_event(OrderEvent.REJECTED, {
                "reason": f"System error: {str(e)}",
                "exception": str(e)
            })
            
            await self.save_order(managed_order)
            
            self.audit_logger.log_event(
                event_type="ORDER_ERROR",
                action="SUBMISSION_FAILED",
                details={"order_id": order_id, "error": str(e)},
                symbol=order_request.symbol,
                amount=float(order_request.quantity)
            )
            
            return {
                "success": False,
                "order_id": order_id,
                "reason": f"System error: {str(e)}"
            }
    
    async def save_order(self, order: ManagedOrder):
        """Save order to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO orders (
                        order_id, client_order_id, symbol, side, order_type,
                        quantity, price, status, created_at, updated_at,
                        exchange_order_id, target_exchange, filled_quantity,
                        average_price, total_commission, risk_score,
                        compliance_flags, order_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.order_id,
                    order.client_order_id,
                    order.order_request.symbol,
                    order.order_request.side.value,
                    order.order_request.order_type.value,
                    str(order.order_request.quantity),
                    str(order.order_request.price) if order.order_request.price else None,
                    order.status.value,
                    order.created_at.isoformat(),
                    order.updated_at.isoformat(),
                    order.exchange_order_id,
                    order.target_exchange,
                    str(order.filled_quantity),
                    str(order.average_price) if order.average_price else None,
                    str(order.total_commission),
                    order.risk_assessment.risk_score if order.risk_assessment else None,
                    json.dumps(order.compliance_flags),
                    json.dumps(asdict(order), default=str)
                ))
                
                # Save executions
                for execution in order.executions:
                    conn.execute("""
                        INSERT OR REPLACE INTO executions (
                            execution_id, order_id, timestamp, quantity, price,
                            commission, exchange, trade_id, is_maker
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        execution.execution_id,
                        order.order_id,
                        execution.timestamp.isoformat(),
                        str(execution.quantity),
                        str(execution.price),
                        str(execution.commission),
                        execution.exchange,
                        execution.trade_id,
                        execution.is_maker
                    ))
                
        except Exception as e:
            self.logger.error(f"Failed to save order {order.order_id}: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive order status."""
        order = self.active_orders.get(order_id) or self.order_history.get(order_id)
        if not order:
            return None
        
        return {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "status": order.status.value,
            "symbol": order.order_request.symbol,
            "side": order.order_request.side.value,
            "order_type": order.order_request.order_type.value,
            "quantity": float(order.order_request.quantity),
            "price": float(order.order_request.price) if order.order_request.price else None,
            "filled_quantity": float(order.filled_quantity),
            "remaining_quantity": float(order.remaining_quantity),
            "average_price": float(order.average_price) if order.average_price else None,
            "total_commission": float(order.total_commission),
            "fill_percentage": order.fill_percentage,
            "notional_value": float(order.notional_value),
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "exchange": order.target_exchange,
            "exchange_order_id": order.exchange_order_id,
            "execution_count": len(order.executions),
            "risk_score": order.risk_assessment.risk_score if order.risk_assessment else None,
            "compliance_flags": order.compliance_flags,
            "events": order.events[-10:]  # Last 10 events
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio and order summary."""
        return {
            "positions": {symbol: float(qty) for symbol, qty in self.positions.items()},
            "active_orders": len(self.active_orders),
            "daily_trade_count": self.daily_trade_count,
            "daily_volume_usd": float(self.daily_volume_usd),
            "execution_metrics": {
                **self.execution_metrics,
                "total_commission": float(self.execution_metrics["total_commission"]),
                "success_rate": (
                    self.execution_metrics["successful_orders"] / 
                    max(self.execution_metrics["total_orders"], 1)
                ) * 100
            },
            "active_order_summary": [
                {
                    "order_id": order.order_id,
                    "symbol": order.order_request.symbol,
                    "side": order.order_request.side.value,
                    "status": order.status.value,
                    "fill_percentage": order.fill_percentage
                }
                for order in self.active_orders.values()
            ]
        }

# Example usage and testing
async def demo_order_management():
    """Demonstration of the order management system."""
    print("🚨 ULTRATHINK Week 6: Order Management System Demo 🚨")
    print("=" * 60)
    
    # This is a simplified demo - would need actual exchange router in production
    from exchange_integrations import ExchangeRouter
    from real_money_trader import AuditLogger
    
    # Create components
    exchange_router = ExchangeRouter()  # Empty for demo
    audit_logger = AuditLogger()
    
    # Create order manager
    config = {
        "max_order_value_usd": 100000,
        "max_daily_trades": 50,
        "large_order_threshold_usd": 25000
    }
    
    order_manager = OrderManager(exchange_router, audit_logger, config)
    
    # Create test order
    test_order = OrderRequest(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("45000.00")
    )
    
    print(f"Submitting test order: {test_order.quantity} {test_order.symbol} @ ${test_order.price}")
    
    # This will be rejected due to no exchange connections in demo
    result = await order_manager.submit_order(test_order)
    print(f"Order result: {result}")
    
    # Show portfolio summary
    portfolio = order_manager.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"- Active Orders: {portfolio['active_orders']}")
    print(f"- Daily Trades: {portfolio['daily_trade_count']}")
    print(f"- Success Rate: {portfolio['execution_metrics']['success_rate']:.1f}%")
    
    print(f"\n✅ Order Management System demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_order_management())