#!/usr/bin/env python3
"""
Test utilities for ULTRATHINK compliance framework testing.
Provides mock implementations and test data generators.
"""

import sys
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random
import logging

# Add parent directory for imports
sys.path.append('/home/richardw/crypto_rf_trading_system')

# Import actual types we need to mock
from compliance.compliance_engine import (
    ComplianceRule, ComplianceViolation, ViolationType, 
    RegulatoryFramework, ComplianceLevel
)
from compliance.audit_trail import AuditLevel, AuditCategory, AuditContext
from production.portfolio_manager import Position, AssetInfo, AssetClass, PositionType
from production.order_management import OrderRequest, OrderSide, OrderType

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class MockComplianceEngine:
    """Mock compliance engine for testing."""
    
    def __init__(self):
        self.rules = {}
        self.violations = []
        self.logger = logging.getLogger("MockComplianceEngine")
        
    def add_rule(self, rule: ComplianceRule):
        """Add compliance rule."""
        self.rules[rule.rule_id] = rule
        
    def check_compliance(self, action: str, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Mock compliance check - returns empty list (no violations)."""
        return []
        
    def log_violation(self, violation: ComplianceViolation):
        """Log compliance violation."""
        self.violations.append(violation)


class MockAuditLogger:
    """Mock audit logger for testing."""
    
    def __init__(self):
        self.events = []
        self.logger = logging.getLogger("MockAuditLogger")
        
    def log_event(self, level: AuditLevel, category: AuditCategory, 
                  event_type: str, description: str, context: AuditContext, **kwargs):
        """Log audit event."""
        event = {
            'timestamp': datetime.now(),
            'level': level,
            'category': category,
            'event_type': event_type,
            'description': description,
            'context': context,
            'details': kwargs
        }
        self.events.append(event)
        return str(uuid.uuid4())
        
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all logged events."""
        return self.events


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.assets: Dict[str, AssetInfo] = {}
        self.total_value = Decimal("100000.00")  # $100k starting capital
        self.logger = logging.getLogger("MockPortfolioManager")
        
        # Initialize with some test positions
        self._init_test_positions()
        
    def _init_test_positions(self):
        """Initialize with test positions."""
        # BTC position
        self.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            exchange="test_exchange",
            quantity=Decimal("1.5"),
            average_price=Decimal("45000.00"),
            current_price=Decimal("46000.00"),
            unrealized_pnl=Decimal("1500.00"),
            realized_pnl=Decimal("0.00"),
            total_commission=Decimal("100.00"),
            position_type=PositionType.LONG,
            opened_at=datetime.now() - timedelta(days=7),
            updated_at=datetime.now(),
            var_1d=Decimal("2000.00"),
            var_10d=Decimal("6000.00")
        )
        
        # ETH position
        self.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            exchange="test_exchange",
            quantity=Decimal("10.0"),
            average_price=Decimal("3000.00"),
            current_price=Decimal("3100.00"),
            unrealized_pnl=Decimal("1000.00"),
            realized_pnl=Decimal("0.00"),
            total_commission=Decimal("50.00"),
            position_type=PositionType.LONG,
            opened_at=datetime.now() - timedelta(days=3),
            updated_at=datetime.now(),
            var_1d=Decimal("1500.00"),
            var_10d=Decimal("4500.00")
        )
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
        
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
        
    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        position_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.total_value + position_value
        
    def update_position(self, symbol: str, quantity: Decimal, price: Decimal):
        """Update position (mock implementation)."""
        if symbol in self.positions:
            self.positions[symbol].quantity = quantity
            self.positions[symbol].current_price = price
            self.positions[symbol].updated_at = datetime.now()


class MockRiskEngine:
    """Mock risk engine for testing."""
    
    def __init__(self, compliance_engine=None, audit_logger=None, portfolio_manager=None):
        self.compliance_engine = compliance_engine or MockComplianceEngine()
        self.audit_logger = audit_logger or MockAuditLogger()
        self.portfolio_manager = portfolio_manager or MockPortfolioManager()
        self.limits = {}
        self.logger = logging.getLogger("MockRiskEngine")
        
    def validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Mock order validation - always approves for testing."""
        return {
            "approved": True,
            "risk_score": 0.3,
            "reason": "Mock approval for testing",
            "checks_performed": ["position_limit", "concentration", "leverage"]
        }
        
    def check_limits(self, limit_type: str, value: Decimal) -> bool:
        """Mock limit check - always passes."""
        return True
        
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get mock risk metrics."""
        return {
            "portfolio_var": Decimal("5000.00"),
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "current_leverage": 1.2
        }


class TestDataGenerator:
    """Generate test data for compliance testing."""
    
    @staticmethod
    def create_test_order() -> OrderRequest:
        """Create a test order request."""
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "MATIC-USD"]
        return OrderRequest(
            symbol=random.choice(symbols),
            side=random.choice([OrderSide.BUY, OrderSide.SELL]),
            order_type=OrderType.LIMIT,
            quantity=Decimal(str(round(random.uniform(0.1, 2.0), 2))),
            price=Decimal(str(round(random.uniform(30000, 50000), 2)))
        )
        
    @staticmethod
    def create_test_trade() -> Dict[str, Any]:
        """Create a test trade."""
        return {
            "trade_id": f"TRADE_{uuid.uuid4().hex[:8]}",
            "symbol": random.choice(["BTC-USD", "ETH-USD"]),
            "side": random.choice(["BUY", "SELL"]),
            "quantity": round(random.uniform(0.1, 2.0), 4),
            "price": round(random.uniform(30000, 50000), 2),
            "timestamp": datetime.now(),
            "exchange": "test_exchange",
            "fee": round(random.uniform(10, 100), 2)
        }
        
    @staticmethod
    def create_test_context() -> AuditContext:
        """Create a test audit context."""
        return AuditContext(
            session_id=f"TEST_SESSION_{uuid.uuid4().hex[:8]}",
            user_id=f"TEST_USER_{random.randint(1, 100)}",
            client_id=f"CLIENT_{random.randint(1000, 9999)}",
            ip_address="192.168.1.100"
        )
        
    @staticmethod
    def create_compliance_violation() -> ComplianceViolation:
        """Create a test compliance violation."""
        return ComplianceViolation(
            violation_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            rule_id=f"RULE_{random.randint(100, 999)}",
            rule_name="Test Compliance Rule",
            violation_type=random.choice(list(ViolationType)),
            severity="high",
            description="Test violation for compliance testing",
            context={
                "symbol": "BTC-USD",
                "value": random.uniform(50000, 100000)
            }
        )


def create_test_environment() -> Dict[str, Any]:
    """
    Create a complete test environment with all mocked components.
    Returns a dictionary with all initialized components.
    """
    # Create mock components
    compliance_engine = MockComplianceEngine()
    audit_logger = MockAuditLogger()
    portfolio_manager = MockPortfolioManager()
    risk_engine = MockRiskEngine(compliance_engine, audit_logger, portfolio_manager)
    
    # Add some test rules to compliance engine
    test_rule = ComplianceRule(
        rule_id="TEST_POSITION_LIMIT",
        name="Test Position Limit",
        framework=RegulatoryFramework.MIFID_II,
        description="Test position size limit",
        severity=ComplianceLevel.YELLOW,
        threshold_value=100000.0,
        parameters={"max_position_usd": 100000},
        active=True
    )
    compliance_engine.add_rule(test_rule)
    
    return {
        "compliance_engine": compliance_engine,
        "audit_logger": audit_logger,
        "portfolio_manager": portfolio_manager,
        "risk_engine": risk_engine,
        "data_generator": TestDataGenerator()
    }


def run_component_test(component_name: str, test_func: callable) -> bool:
    """
    Run a test for a specific component and return success status.
    """
    try:
        print(f"\n🧪 Testing {component_name}...")
        test_func()
        print(f"✅ {component_name} test passed!")
        return True
    except Exception as e:
        print(f"❌ {component_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the test utilities themselves
    print("Testing compliance test utilities...")
    
    # Create test environment
    env = create_test_environment()
    print(f"✅ Created test environment with {len(env)} components")
    
    # Test data generator
    test_order = TestDataGenerator.create_test_order()
    print(f"✅ Generated test order: {test_order.symbol} {test_order.side.value} "
          f"{test_order.quantity} @ ${test_order.price}")
    
    # Test mock compliance engine
    violations = env["compliance_engine"].check_compliance("test_action", {})
    print(f"✅ Mock compliance check returned {len(violations)} violations")
    
    # Test mock risk engine
    risk_result = env["risk_engine"].validate_order({"symbol": "BTC-USD", "value_usd": 50000})
    print(f"✅ Mock risk validation: {risk_result}")
    
    # Test mock portfolio manager
    portfolio_value = env["portfolio_manager"].get_portfolio_value()
    print(f"✅ Mock portfolio value: ${portfolio_value:,.2f}")
    
    print("\n✅ All test utilities working correctly!")