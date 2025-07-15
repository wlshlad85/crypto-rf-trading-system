#!/usr/bin/env python3
"""
ULTRATHINK Week 6 DAY 39: Compliance Risk Management Controls
Advanced risk controls with regulatory compliance integration.

Features:
- Pre-trade risk checks with regulatory validation
- Position limit enforcement with real-time monitoring
- Concentration risk monitoring across multiple dimensions
- Counterparty risk assessment and limits
- Stress testing integration for compliance scenarios
- Regulatory capital adequacy monitoring
- Liquidity risk management
- Operational risk controls
- Market risk limits and alerts
- Credit risk assessment

Regulatory Compliance:
- Basel III: Capital adequacy and liquidity requirements
- MiFID II: Position limits and concentration risk
- CFTC: Position limits and large trader reporting
- SEC: Net capital requirements and liquidity
- EMIR: Risk mitigation techniques
- Dodd-Frank: Volcker Rule compliance
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import existing components
import sys
sys.path.append('/home/richardw/crypto_rf_trading_system')
from production.order_management import OrderRequest, ManagedOrder, RiskAssessment, RiskDecision
from production.portfolio_manager import Position, PortfolioManager
from compliance.compliance_engine import ComplianceEngine, ComplianceViolation, ViolationType
from compliance.audit_trail import EnhancedAuditLogger, AuditLevel, AuditCategory, AuditContext

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class LimitType(Enum):
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    COUNTERPARTY_LIMIT = "counterparty_limit"
    SECTOR_LIMIT = "sector_limit"
    CURRENCY_LIMIT = "currency_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    VAR_LIMIT = "var_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    NOTIONAL_LIMIT = "notional_limit"

class RiskAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    REDUCE = "reduce"
    REJECT = "reject"
    ESCALATE = "escalate"
    HALT = "halt"

class MonitoringFrequency(Enum):
    REAL_TIME = "real_time"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class RiskLimit:
    """Risk limit definition with regulatory compliance."""
    limit_id: str
    name: str
    limit_type: LimitType
    limit_value: Decimal
    warning_threshold: Decimal
    
    # Scope
    scope: str = "global"  # global, client, symbol, sector, etc.
    scope_value: Optional[str] = None
    
    # Regulatory context
    regulatory_requirement: Optional[str] = None
    regulatory_framework: Optional[str] = None
    
    # Monitoring
    monitoring_frequency: MonitoringFrequency = MonitoringFrequency.REAL_TIME
    
    # Actions
    warning_action: RiskAction = RiskAction.WARN
    breach_action: RiskAction = RiskAction.REJECT
    
    # Metadata
    description: str = ""
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Breach tracking
    breach_count: int = 0
    last_breach: Optional[datetime] = None
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate current utilization percentage."""
        # This would be calculated based on current positions/exposure
        return 0.0
    
    @property
    def is_breached(self) -> bool:
        """Check if limit is currently breached."""
        return self.utilization_percentage > 100.0
    
    @property
    def is_warning(self) -> bool:
        """Check if limit is at warning level."""
        warning_pct = float(self.warning_threshold / self.limit_value * 100)
        return self.utilization_percentage >= warning_pct

@dataclass
class RiskBreach:
    """Risk limit breach record."""
    breach_id: str
    limit_id: str
    timestamp: datetime
    
    # Breach details
    limit_value: Decimal
    current_value: Decimal
    breach_amount: Decimal
    breach_percentage: float
    
    # Context
    triggered_by: str  # order_id, position_id, etc.
    client_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # Resolution
    action_taken: RiskAction = RiskAction.WARN
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: str = ""
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    escalated: bool = False
    
    @property
    def age_minutes(self) -> float:
        """Get breach age in minutes."""
        return (datetime.now() - self.timestamp).total_seconds() / 60
    
    @property
    def is_critical(self) -> bool:
        """Check if breach is critical."""
        return self.risk_level == RiskLevel.CRITICAL

@dataclass
class CounterpartyRisk:
    """Counterparty risk assessment."""
    counterparty_id: str
    name: str
    
    # Risk metrics
    credit_rating: str = "BBB"
    probability_of_default: float = 0.01  # 1%
    loss_given_default: float = 0.60  # 60%
    expected_loss: float = 0.006  # 0.6%
    
    # Exposure
    current_exposure: Decimal = Decimal("0")
    maximum_exposure: Decimal = Decimal("0")
    potential_future_exposure: Decimal = Decimal("0")
    
    # Limits
    exposure_limit: Decimal = Decimal("1000000")  # $1M
    
    # Monitoring
    last_assessment: datetime = field(default_factory=datetime.now)
    next_review: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate exposure utilization."""
        if self.exposure_limit == 0:
            return 0.0
        return float(self.current_exposure / self.exposure_limit * 100)
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score."""
        # Simplified risk scoring
        base_score = self.probability_of_default * self.loss_given_default
        exposure_factor = min(self.utilization_percentage / 100, 1.0)
        return base_score * exposure_factor * 100

@dataclass
class StressTesting:
    """Stress testing configuration and results."""
    test_id: str
    name: str
    description: str
    
    # Scenarios
    market_shock: float = -0.20  # -20% market shock
    volatility_shock: float = 2.0  # 2x volatility
    liquidity_shock: float = 0.50  # 50% liquidity reduction
    correlation_shock: float = 0.30  # 30% correlation increase
    
    # Results
    portfolio_impact: Optional[Decimal] = None
    var_impact: Optional[Decimal] = None
    liquidity_impact: Optional[float] = None
    
    # Compliance
    regulatory_requirement: Optional[str] = None
    pass_threshold: Decimal = Decimal("0.15")  # 15% max loss
    
    # Metadata
    last_run: Optional[datetime] = None
    passed: Optional[bool] = None
    
    def run_stress_test(self, portfolio_value: Decimal, positions: Dict[str, Position]) -> Dict[str, Any]:
        """Run stress test scenario."""
        results = {
            'test_id': self.test_id,
            'test_name': self.name,
            'portfolio_value': portfolio_value,
            'scenario_impact': {},
            'total_impact': Decimal('0'),
            'percentage_impact': 0.0,
            'passed': False,
            'timestamp': datetime.now()
        }
        
        # Apply market shock
        market_impact = Decimal('0')
        for symbol, position in positions.items():
            position_impact = position.market_value * Decimal(str(self.market_shock))
            market_impact += position_impact
        
        results['scenario_impact']['market_shock'] = market_impact
        
        # Apply volatility shock (simplified)
        volatility_impact = market_impact * Decimal(str(self.volatility_shock - 1))
        results['scenario_impact']['volatility_shock'] = volatility_impact
        
        # Apply liquidity shock
        liquidity_impact = market_impact * Decimal(str(self.liquidity_shock))
        results['scenario_impact']['liquidity_shock'] = liquidity_impact
        
        # Calculate total impact
        total_impact = market_impact + volatility_impact + liquidity_impact
        results['total_impact'] = total_impact
        
        # Calculate percentage impact
        if portfolio_value > 0:
            percentage_impact = float(abs(total_impact) / portfolio_value * 100)
            results['percentage_impact'] = percentage_impact
        
        # Check if passed
        results['passed'] = abs(total_impact) <= self.pass_threshold * portfolio_value
        
        # Update test record
        self.last_run = datetime.now()
        self.passed = results['passed']
        self.portfolio_impact = total_impact
        
        return results

class RiskEngine:
    """Advanced risk engine with regulatory compliance."""
    
    def __init__(self, 
                 compliance_engine: ComplianceEngine,
                 audit_logger: EnhancedAuditLogger,
                 portfolio_manager: PortfolioManager):
        
        self.compliance_engine = compliance_engine
        self.audit_logger = audit_logger
        self.portfolio_manager = portfolio_manager
        
        # Risk limits
        self.limits: Dict[str, RiskLimit] = {}
        self.breaches: Dict[str, RiskBreach] = {}
        
        # Counterparty risk
        self.counterparties: Dict[str, CounterpartyRisk] = {}
        
        # Stress testing
        self.stress_tests: Dict[str, StressTesting] = {}
        
        # Monitoring
        self.monitoring_enabled = True
        self.alert_handlers: Dict[str, Callable] = {}
        
        # Database
        self.db_path = "compliance/risk_controls.db"
        self.setup_database()
        
        # Load default limits
        self.load_default_limits()
        
        # Load default stress tests
        self.load_default_stress_tests()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Risk Engine initialized with compliance controls")
    
    def setup_database(self):
        """Initialize risk controls database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Risk limits table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_limits (
                    limit_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    limit_type TEXT NOT NULL,
                    limit_value TEXT NOT NULL,
                    warning_threshold TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_value TEXT,
                    regulatory_requirement TEXT,
                    regulatory_framework TEXT,
                    monitoring_frequency TEXT NOT NULL,
                    warning_action TEXT NOT NULL,
                    breach_action TEXT NOT NULL,
                    active BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    breach_count INTEGER DEFAULT 0,
                    last_breach TEXT,
                    limit_data TEXT NOT NULL
                )
            """)
            
            # Risk breaches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_breaches (
                    breach_id TEXT PRIMARY KEY,
                    limit_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    limit_value TEXT NOT NULL,
                    current_value TEXT NOT NULL,
                    breach_amount TEXT NOT NULL,
                    breach_percentage REAL NOT NULL,
                    triggered_by TEXT NOT NULL,
                    client_id TEXT,
                    symbol TEXT,
                    action_taken TEXT NOT NULL,
                    resolved BOOLEAN NOT NULL,
                    resolution_time TEXT,
                    resolution_notes TEXT,
                    risk_level TEXT NOT NULL,
                    escalated BOOLEAN NOT NULL,
                    breach_data TEXT NOT NULL,
                    FOREIGN KEY (limit_id) REFERENCES risk_limits (limit_id)
                )
            """)
            
            # Counterparty risk table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS counterparty_risk (
                    counterparty_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    credit_rating TEXT NOT NULL,
                    probability_of_default REAL NOT NULL,
                    loss_given_default REAL NOT NULL,
                    expected_loss REAL NOT NULL,
                    current_exposure TEXT NOT NULL,
                    maximum_exposure TEXT NOT NULL,
                    potential_future_exposure TEXT NOT NULL,
                    exposure_limit TEXT NOT NULL,
                    last_assessment TEXT NOT NULL,
                    next_review TEXT NOT NULL,
                    counterparty_data TEXT NOT NULL
                )
            """)
            
            # Stress testing table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_tests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    market_shock REAL NOT NULL,
                    volatility_shock REAL NOT NULL,
                    liquidity_shock REAL NOT NULL,
                    correlation_shock REAL NOT NULL,
                    portfolio_impact TEXT,
                    var_impact TEXT,
                    liquidity_impact REAL,
                    regulatory_requirement TEXT,
                    pass_threshold TEXT NOT NULL,
                    last_run TEXT,
                    passed BOOLEAN,
                    test_data TEXT NOT NULL
                )
            """)
            
            # Risk monitoring metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    limit_value REAL,
                    utilization_percentage REAL,
                    risk_level TEXT NOT NULL,
                    client_id TEXT,
                    symbol TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_breaches_timestamp ON risk_breaches(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_breaches_limit_id ON risk_breaches(limit_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_breaches_resolved ON risk_breaches(resolved)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON risk_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON risk_metrics(metric_type)")
    
    def load_default_limits(self):
        """Load default risk limits."""
        
        # Position limits
        self.add_limit(RiskLimit(
            limit_id="POSITION_LIMIT_GLOBAL",
            name="Global Position Limit",
            limit_type=LimitType.POSITION_LIMIT,
            limit_value=Decimal("250000"),  # $250K max position
            warning_threshold=Decimal("200000"),  # $200K warning
            scope="global",
            regulatory_requirement="MiFID II Position Limits",
            regulatory_framework="MiFID II",
            monitoring_frequency=MonitoringFrequency.REAL_TIME,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.REJECT,
            description="Maximum position size per symbol"
        ))
        
        # Concentration limits
        self.add_limit(RiskLimit(
            limit_id="CONCENTRATION_LIMIT_PORTFOLIO",
            name="Portfolio Concentration Limit",
            limit_type=LimitType.CONCENTRATION_LIMIT,
            limit_value=Decimal("0.25"),  # 25% max concentration
            warning_threshold=Decimal("0.20"),  # 20% warning
            scope="portfolio",
            regulatory_requirement="Risk Concentration Limits",
            regulatory_framework="Basel III",
            monitoring_frequency=MonitoringFrequency.REAL_TIME,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.REDUCE,
            description="Maximum concentration per asset"
        ))
        
        # Leverage limits
        self.add_limit(RiskLimit(
            limit_id="LEVERAGE_LIMIT_GLOBAL",
            name="Global Leverage Limit",
            limit_type=LimitType.LEVERAGE_LIMIT,
            limit_value=Decimal("3.0"),  # 3:1 max leverage
            warning_threshold=Decimal("2.5"),  # 2.5:1 warning
            scope="global",
            regulatory_requirement="Leverage Ratio",
            regulatory_framework="Basel III",
            monitoring_frequency=MonitoringFrequency.REAL_TIME,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.HALT,
            description="Maximum portfolio leverage"
        ))
        
        # VaR limits
        self.add_limit(RiskLimit(
            limit_id="VAR_LIMIT_DAILY",
            name="Daily VaR Limit",
            limit_type=LimitType.VAR_LIMIT,
            limit_value=Decimal("50000"),  # $50K daily VaR
            warning_threshold=Decimal("40000"),  # $40K warning
            scope="portfolio",
            regulatory_requirement="Market Risk Limits",
            regulatory_framework="Basel III",
            monitoring_frequency=MonitoringFrequency.DAILY,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.REDUCE,
            description="Maximum daily Value at Risk"
        ))
        
        # Drawdown limits
        self.add_limit(RiskLimit(
            limit_id="DRAWDOWN_LIMIT_PORTFOLIO",
            name="Portfolio Drawdown Limit",
            limit_type=LimitType.DRAWDOWN_LIMIT,
            limit_value=Decimal("0.15"),  # 15% max drawdown
            warning_threshold=Decimal("0.10"),  # 10% warning
            scope="portfolio",
            regulatory_requirement="Risk Management",
            regulatory_framework="MiFID II",
            monitoring_frequency=MonitoringFrequency.REAL_TIME,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.HALT,
            description="Maximum portfolio drawdown"
        ))
        
        # Liquidity limits
        self.add_limit(RiskLimit(
            limit_id="LIQUIDITY_LIMIT_PORTFOLIO",
            name="Portfolio Liquidity Limit",
            limit_type=LimitType.LIQUIDITY_LIMIT,
            limit_value=Decimal("0.20"),  # 20% max illiquid assets
            warning_threshold=Decimal("0.15"),  # 15% warning
            scope="portfolio",
            regulatory_requirement="Liquidity Risk Management",
            regulatory_framework="Basel III",
            monitoring_frequency=MonitoringFrequency.DAILY,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.REDUCE,
            description="Maximum illiquid asset concentration"
        ))
        
        # Counterparty limits
        self.add_limit(RiskLimit(
            limit_id="COUNTERPARTY_LIMIT_EXCHANGE",
            name="Exchange Counterparty Limit",
            limit_type=LimitType.COUNTERPARTY_LIMIT,
            limit_value=Decimal("500000"),  # $500K per exchange
            warning_threshold=Decimal("400000"),  # $400K warning
            scope="counterparty",
            regulatory_requirement="Counterparty Risk Limits",
            regulatory_framework="EMIR",
            monitoring_frequency=MonitoringFrequency.REAL_TIME,
            warning_action=RiskAction.WARN,
            breach_action=RiskAction.REJECT,
            description="Maximum exposure per exchange"
        ))
        
        self.logger.info(f"Loaded {len(self.limits)} default risk limits")
    
    def load_default_stress_tests(self):
        """Load default stress testing scenarios."""
        
        # Market crash scenario
        self.add_stress_test(StressTesting(
            test_id="MARKET_CRASH_20",
            name="Market Crash 20%",
            description="20% market crash scenario",
            market_shock=-0.20,
            volatility_shock=2.0,
            liquidity_shock=0.50,
            correlation_shock=0.30,
            regulatory_requirement="CFTC Stress Testing",
            pass_threshold=Decimal("0.15")
        ))
        
        # Volatility spike scenario
        self.add_stress_test(StressTesting(
            test_id="VOLATILITY_SPIKE",
            name="Volatility Spike",
            description="3x volatility spike scenario",
            market_shock=-0.10,
            volatility_shock=3.0,
            liquidity_shock=0.30,
            correlation_shock=0.50,
            regulatory_requirement="Basel III Stress Testing",
            pass_threshold=Decimal("0.20")
        ))
        
        # Liquidity crisis scenario
        self.add_stress_test(StressTesting(
            test_id="LIQUIDITY_CRISIS",
            name="Liquidity Crisis",
            description="Severe liquidity crisis scenario",
            market_shock=-0.15,
            volatility_shock=2.5,
            liquidity_shock=0.80,
            correlation_shock=0.40,
            regulatory_requirement="MiFID II Stress Testing",
            pass_threshold=Decimal("0.25")
        ))
        
        self.logger.info(f"Loaded {len(self.stress_tests)} default stress tests")
    
    def add_limit(self, limit: RiskLimit):
        """Add risk limit to the engine."""
        self.limits[limit.limit_id] = limit
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO risk_limits (
                    limit_id, name, limit_type, limit_value, warning_threshold,
                    scope, scope_value, regulatory_requirement, regulatory_framework,
                    monitoring_frequency, warning_action, breach_action, active,
                    created_at, updated_at, breach_count, last_breach, limit_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                limit.limit_id,
                limit.name,
                limit.limit_type.value,
                str(limit.limit_value),
                str(limit.warning_threshold),
                limit.scope,
                limit.scope_value,
                limit.regulatory_requirement,
                limit.regulatory_framework,
                limit.monitoring_frequency.value,
                limit.warning_action.value,
                limit.breach_action.value,
                limit.active,
                limit.created_at.isoformat(),
                limit.updated_at.isoformat(),
                limit.breach_count,
                limit.last_breach.isoformat() if limit.last_breach else None,
                json.dumps(asdict(limit), default=str)
            ))
    
    def add_stress_test(self, stress_test: StressTesting):
        """Add stress test to the engine."""
        self.stress_tests[stress_test.test_id] = stress_test
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stress_tests (
                    test_id, name, description, market_shock, volatility_shock,
                    liquidity_shock, correlation_shock, portfolio_impact,
                    var_impact, liquidity_impact, regulatory_requirement,
                    pass_threshold, last_run, passed, test_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stress_test.test_id,
                stress_test.name,
                stress_test.description,
                stress_test.market_shock,
                stress_test.volatility_shock,
                stress_test.liquidity_shock,
                stress_test.correlation_shock,
                str(stress_test.portfolio_impact) if stress_test.portfolio_impact else None,
                str(stress_test.var_impact) if stress_test.var_impact else None,
                stress_test.liquidity_impact,
                stress_test.regulatory_requirement,
                str(stress_test.pass_threshold),
                stress_test.last_run.isoformat() if stress_test.last_run else None,
                stress_test.passed,
                json.dumps(asdict(stress_test), default=str)
            ))
    
    def check_pre_trade_risk(self, order: OrderRequest) -> RiskAssessment:
        """Comprehensive pre-trade risk check."""
        risk_factors = []
        risk_score = 0.0
        decision = RiskDecision.APPROVED
        
        # Check all applicable limits
        for limit_id, limit in self.limits.items():
            if not limit.active:
                continue
            
            breach = self.check_limit_breach(limit, order)
            if breach:
                risk_factors.append(f"Limit breach: {limit.name}")
                risk_score += 0.3
                
                # Record breach
                self.record_breach(breach)
                
                # Determine action
                if breach.risk_level == RiskLevel.CRITICAL:
                    decision = RiskDecision.REJECTED
                elif breach.risk_level == RiskLevel.HIGH:
                    decision = RiskDecision.CONDITIONAL
                elif decision == RiskDecision.APPROVED:
                    decision = RiskDecision.REQUIRES_APPROVAL
        
        # Check counterparty risk
        counterparty_risk = self.check_counterparty_risk(order)
        if counterparty_risk > 0.5:
            risk_factors.append(f"High counterparty risk: {counterparty_risk:.1%}")
            risk_score += 0.2
            if decision == RiskDecision.APPROVED:
                decision = RiskDecision.CONDITIONAL
        
        # Check position concentration
        concentration_risk = self.check_concentration_risk(order)
        if concentration_risk > 0.25:  # 25% concentration
            risk_factors.append(f"High concentration risk: {concentration_risk:.1%}")
            risk_score += 0.2
            if decision == RiskDecision.APPROVED:
                decision = RiskDecision.CONDITIONAL
        
        # Check liquidity risk
        liquidity_risk = self.check_liquidity_risk(order)
        if liquidity_risk > 0.3:  # 30% liquidity risk
            risk_factors.append(f"High liquidity risk: {liquidity_risk:.1%}")
            risk_score += 0.1
        
        # Create assessment
        assessment = RiskAssessment(
            decision=decision,
            risk_score=min(risk_score, 1.0),
            reason="; ".join(risk_factors) if risk_factors else "Risk checks passed",
            conditions=[]
        )
        
        # Add conditions based on risk level
        if decision == RiskDecision.CONDITIONAL:
            assessment.conditions.append("Enhanced monitoring required")
            
        if risk_score > 0.7:
            assessment.conditions.append("Manual review required")
        
        # Log risk assessment
        self.audit_logger.log_event(
            level=AuditLevel.INFO if decision == RiskDecision.APPROVED else AuditLevel.WARNING,
            category=AuditCategory.RISK_MANAGEMENT,
            event_type="PRE_TRADE_RISK_CHECK",
            description=f"Pre-trade risk check: {decision.value}",
            context=AuditContext(
                session_id="system",
                user_id="risk_engine",
                component="risk_controls"
            ),
            symbol=order.symbol,
            amount=order.quantity,
            price=order.price,
            risk_score=risk_score,
            compliance_flags=risk_factors
        )
        
        return assessment
    
    def check_limit_breach(self, limit: RiskLimit, order: OrderRequest) -> Optional[RiskBreach]:
        """Check if order would breach risk limit."""
        current_value = self.calculate_current_exposure(limit, order)
        
        # Check if breach would occur
        if current_value > limit.limit_value:
            breach_amount = current_value - limit.limit_value
            breach_percentage = float(breach_amount / limit.limit_value * 100)
            
            # Determine risk level
            if breach_percentage > 50:
                risk_level = RiskLevel.CRITICAL
            elif breach_percentage > 25:
                risk_level = RiskLevel.HIGH
            elif breach_percentage > 10:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            return RiskBreach(
                breach_id=str(uuid.uuid4()),
                limit_id=limit.limit_id,
                timestamp=datetime.now(),
                limit_value=limit.limit_value,
                current_value=current_value,
                breach_amount=breach_amount,
                breach_percentage=breach_percentage,
                triggered_by=order.client_order_id or "unknown",
                client_id=getattr(order, 'client_id', None),
                symbol=order.symbol,
                action_taken=limit.breach_action,
                risk_level=risk_level
            )
        
        return None
    
    def calculate_current_exposure(self, limit: RiskLimit, order: OrderRequest) -> Decimal:
        """Calculate current exposure for limit check."""
        # This is a simplified calculation - real implementation would be more complex
        
        if limit.limit_type == LimitType.POSITION_LIMIT:
            # Current position + new order
            current_position = self.portfolio_manager.positions.get(order.symbol, None)
            if current_position:
                return current_position.market_value + (order.quantity * (order.price or Decimal('0')))
            else:
                return order.quantity * (order.price or Decimal('0'))
        
        elif limit.limit_type == LimitType.CONCENTRATION_LIMIT:
            # Position concentration
            total_value = self.portfolio_manager.get_total_portfolio_value()
            if total_value == 0:
                return Decimal('0')
            
            symbol_value = Decimal('0')
            if order.symbol in self.portfolio_manager.positions:
                symbol_value = self.portfolio_manager.positions[order.symbol].market_value
            
            new_order_value = order.quantity * (order.price or Decimal('0'))
            return (symbol_value + new_order_value) / total_value
        
        elif limit.limit_type == LimitType.LEVERAGE_LIMIT:
            # Portfolio leverage
            total_value = self.portfolio_manager.get_total_portfolio_value()
            total_notional = sum(pos.market_value for pos in self.portfolio_manager.positions.values())
            new_notional = order.quantity * (order.price or Decimal('0'))
            
            if total_value == 0:
                return Decimal('0')
            
            return (total_notional + new_notional) / total_value
        
        # Default to zero exposure
        return Decimal('0')
    
    def check_counterparty_risk(self, order: OrderRequest) -> float:
        """Check counterparty risk for order."""
        # Simplified counterparty risk check
        # In practice, this would involve complex credit risk models
        
        # For demo purposes, return risk based on order size
        order_value = float(order.quantity * (order.price or Decimal('0')))
        
        if order_value > 100000:  # $100K
            return 0.3  # 30% risk
        elif order_value > 50000:  # $50K
            return 0.2  # 20% risk
        else:
            return 0.1  # 10% risk
    
    def check_concentration_risk(self, order: OrderRequest) -> float:
        """Check concentration risk for order."""
        total_value = self.portfolio_manager.get_total_portfolio_value()
        if total_value == 0:
            return 0.0
        
        # Current symbol concentration
        current_position = self.portfolio_manager.positions.get(order.symbol, None)
        current_value = current_position.market_value if current_position else Decimal('0')
        
        # Add new order value
        new_order_value = order.quantity * (order.price or Decimal('0'))
        total_symbol_value = current_value + new_order_value
        
        return float(total_symbol_value / total_value)
    
    def check_liquidity_risk(self, order: OrderRequest) -> float:
        """Check liquidity risk for order."""
        # Simplified liquidity risk assessment
        # In practice, this would use order book data and market impact models
        
        # For demo purposes, return risk based on order size and symbol
        order_value = float(order.quantity * (order.price or Decimal('0')))
        
        # Assume BTC has lower liquidity risk
        if order.symbol == "BTC-USD":
            if order_value > 1000000:  # $1M
                return 0.4  # 40% risk
            elif order_value > 500000:  # $500K
                return 0.2  # 20% risk
            else:
                return 0.1  # 10% risk
        else:
            # Other symbols have higher liquidity risk
            return min(0.5, order_value / 1000000)  # Max 50% risk
    
    def record_breach(self, breach: RiskBreach):
        """Record risk limit breach."""
        self.breaches[breach.breach_id] = breach
        
        # Update limit breach count
        limit = self.limits[breach.limit_id]
        limit.breach_count += 1
        limit.last_breach = breach.timestamp
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_breaches (
                    breach_id, limit_id, timestamp, limit_value, current_value,
                    breach_amount, breach_percentage, triggered_by, client_id,
                    symbol, action_taken, resolved, resolution_time,
                    resolution_notes, risk_level, escalated, breach_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                breach.breach_id,
                breach.limit_id,
                breach.timestamp.isoformat(),
                str(breach.limit_value),
                str(breach.current_value),
                str(breach.breach_amount),
                breach.breach_percentage,
                breach.triggered_by,
                breach.client_id,
                breach.symbol,
                breach.action_taken.value,
                breach.resolved,
                breach.resolution_time.isoformat() if breach.resolution_time else None,
                breach.resolution_notes,
                breach.risk_level.value,
                breach.escalated,
                json.dumps(asdict(breach), default=str)
            ))
        
        # Log breach
        self.audit_logger.log_event(
            level=AuditLevel.CRITICAL if breach.is_critical else AuditLevel.WARNING,
            category=AuditCategory.RISK_MANAGEMENT,
            event_type="RISK_LIMIT_BREACH",
            description=f"Risk limit breach: {limit.name}",
            context=AuditContext(
                session_id="system",
                user_id="risk_engine",
                component="risk_controls"
            ),
            symbol=breach.symbol,
            client_id=breach.client_id,
            risk_score=breach.breach_percentage / 100,
            compliance_flags=[breach.risk_level.value]
        )
        
        # Handle breach
        self.handle_breach(breach)
    
    def handle_breach(self, breach: RiskBreach):
        """Handle risk limit breach."""
        limit = self.limits[breach.limit_id]
        
        if breach.action_taken == RiskAction.HALT:
            self.logger.critical(f"TRADING HALT: {limit.name} breached by {breach.breach_percentage:.1f}%")
            # Would integrate with trading engine to halt trading
            
        elif breach.action_taken == RiskAction.ESCALATE:
            self.logger.warning(f"ESCALATION: {limit.name} breach requires attention")
            breach.escalated = True
            
        elif breach.action_taken == RiskAction.REDUCE:
            self.logger.warning(f"POSITION REDUCTION: {limit.name} breach requires position reduction")
            # Would integrate with portfolio manager to reduce positions
            
        elif breach.action_taken == RiskAction.REJECT:
            self.logger.warning(f"ORDER REJECTION: {limit.name} breach blocks new orders")
            # Order would be rejected
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run all stress tests."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_manager.get_total_portfolio_value(),
            'test_results': {},
            'overall_passed': True,
            'risk_assessment': 'LOW'
        }
        
        portfolio_value = self.portfolio_manager.get_total_portfolio_value()
        positions = self.portfolio_manager.positions
        
        for test_id, stress_test in self.stress_tests.items():
            test_results = stress_test.run_stress_test(portfolio_value, positions)
            results['test_results'][test_id] = test_results
            
            if not test_results['passed']:
                results['overall_passed'] = False
        
        # Determine overall risk assessment
        if not results['overall_passed']:
            failed_tests = [
                test_id for test_id, result in results['test_results'].items()
                if not result['passed']
            ]
            
            if len(failed_tests) >= 2:
                results['risk_assessment'] = 'HIGH'
            else:
                results['risk_assessment'] = 'MEDIUM'
        
        # Log stress test results
        self.audit_logger.log_event(
            level=AuditLevel.WARNING if not results['overall_passed'] else AuditLevel.INFO,
            category=AuditCategory.RISK_MANAGEMENT,
            event_type="STRESS_TEST_EXECUTION",
            description=f"Stress tests executed: {results['risk_assessment']} risk",
            context=AuditContext(
                session_id="system",
                user_id="risk_engine",
                component="stress_testing"
            ),
            risk_score=0.8 if results['risk_assessment'] == 'HIGH' else 0.5,
            compliance_flags=[results['risk_assessment']]
        )
        
        return results
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard."""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'ACTIVE' if self.monitoring_enabled else 'INACTIVE',
            'limits': {
                'total_limits': len(self.limits),
                'active_limits': len([l for l in self.limits.values() if l.active]),
                'breached_limits': len([l for l in self.limits.values() if l.is_breached]),
                'warning_limits': len([l for l in self.limits.values() if l.is_warning])
            },
            'breaches': {
                'total_breaches': len(self.breaches),
                'critical_breaches': len([b for b in self.breaches.values() if b.is_critical]),
                'unresolved_breaches': len([b for b in self.breaches.values() if not b.resolved]),
                'recent_breaches': len([
                    b for b in self.breaches.values()
                    if b.age_minutes < 60
                ])
            },
            'stress_tests': {
                'total_tests': len(self.stress_tests),
                'last_run': max([
                    st.last_run for st in self.stress_tests.values()
                    if st.last_run
                ], default=None),
                'passed_tests': len([
                    st for st in self.stress_tests.values()
                    if st.passed
                ]),
                'failed_tests': len([
                    st for st in self.stress_tests.values()
                    if st.passed is False
                ])
            },
            'counterparties': {
                'total_counterparties': len(self.counterparties),
                'high_risk_counterparties': len([
                    cp for cp in self.counterparties.values()
                    if cp.risk_score > 50
                ])
            }
        }
        
        return dashboard
    
    def generate_risk_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter recent breaches
        recent_breaches = [
            b for b in self.breaches.values()
            if b.timestamp > cutoff_date
        ]
        
        # Calculate metrics
        breach_rate = len(recent_breaches) / len(self.limits) if self.limits else 0
        resolution_rate = len([b for b in recent_breaches if b.resolved]) / len(recent_breaches) if recent_breaches else 1
        
        # Generate report
        report = {
            'report_period': f"{period_days} days",
            'report_date': datetime.now().isoformat(),
            'executive_summary': {
                'total_breaches': len(recent_breaches),
                'breach_rate': breach_rate,
                'resolution_rate': resolution_rate,
                'risk_level': 'HIGH' if breach_rate > 0.5 else 'MEDIUM' if breach_rate > 0.2 else 'LOW'
            },
            'limit_analysis': {
                'most_breached_limits': self.get_most_breached_limits(recent_breaches),
                'limit_effectiveness': self.calculate_limit_effectiveness(recent_breaches)
            },
            'stress_test_summary': {
                'tests_run': len([st for st in self.stress_tests.values() if st.last_run]),
                'pass_rate': len([st for st in self.stress_tests.values() if st.passed]) / len(self.stress_tests) if self.stress_tests else 1
            },
            'recommendations': self.generate_risk_recommendations(recent_breaches)
        }
        
        return report
    
    def get_most_breached_limits(self, breaches: List[RiskBreach]) -> List[Dict[str, Any]]:
        """Get most frequently breached limits."""
        breach_counts = defaultdict(int)
        for breach in breaches:
            breach_counts[breach.limit_id] += 1
        
        most_breached = []
        for limit_id, count in sorted(breach_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            limit = self.limits[limit_id]
            most_breached.append({
                'limit_id': limit_id,
                'limit_name': limit.name,
                'breach_count': count,
                'limit_type': limit.limit_type.value
            })
        
        return most_breached
    
    def calculate_limit_effectiveness(self, breaches: List[RiskBreach]) -> Dict[str, float]:
        """Calculate limit effectiveness metrics."""
        if not breaches:
            return {'overall_effectiveness': 1.0}
        
        # Calculate metrics
        resolution_time = np.mean([b.age_minutes for b in breaches if b.resolved])
        escalation_rate = len([b for b in breaches if b.escalated]) / len(breaches)
        repeat_breach_rate = len([b for b in breaches if self.limits[b.limit_id].breach_count > 1]) / len(breaches)
        
        # Overall effectiveness score
        effectiveness = 1.0 - (escalation_rate * 0.3 + repeat_breach_rate * 0.4 + min(resolution_time / 1440, 1.0) * 0.3)
        
        return {
            'overall_effectiveness': max(0.0, effectiveness),
            'avg_resolution_time_hours': resolution_time / 60 if resolution_time else 0,
            'escalation_rate': escalation_rate,
            'repeat_breach_rate': repeat_breach_rate
        }
    
    def generate_risk_recommendations(self, breaches: List[RiskBreach]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if not breaches:
            recommendations.append("Risk controls are operating effectively")
            return recommendations
        
        # Analyze breach patterns
        breach_types = defaultdict(int)
        for breach in breaches:
            limit = self.limits[breach.limit_id]
            breach_types[limit.limit_type.value] += 1
        
        # Generate specific recommendations
        if breach_types.get('position_limit', 0) > 5:
            recommendations.append("Review and tighten position limit thresholds")
        
        if breach_types.get('concentration_limit', 0) > 3:
            recommendations.append("Implement additional diversification controls")
        
        if breach_types.get('leverage_limit', 0) > 2:
            recommendations.append("Reduce maximum leverage ratios")
        
        # Check resolution efficiency
        unresolved_breaches = [b for b in breaches if not b.resolved]
        if len(unresolved_breaches) > 5:
            recommendations.append("Improve breach resolution processes")
        
        # Check escalation patterns
        escalated_breaches = [b for b in breaches if b.escalated]
        if len(escalated_breaches) > 3:
            recommendations.append("Review escalation procedures and thresholds")
        
        return recommendations if recommendations else ["No specific recommendations at this time"]

# Example usage and testing
async def demo_risk_controls():
    """Demonstration of compliance risk controls."""
    print("🚨 ULTRATHINK Week 6 DAY 39: Risk Management Controls Demo 🚨")
    print("=" * 60)
    
    # Create mock components
    from compliance.compliance_engine import ComplianceEngine
    from compliance.audit_trail import EnhancedAuditLogger
    from production.portfolio_manager import PortfolioManager
    from production.exchange_integrations import ExchangeRouter
    from production.order_management import OrderManager
    from production.real_money_trader import AuditLogger
    
    # Initialize components
    basic_audit_logger = AuditLogger()
    enhanced_audit_logger = EnhancedAuditLogger()
    compliance_engine = ComplianceEngine(basic_audit_logger)
    exchange_router = ExchangeRouter()
    order_manager = OrderManager(exchange_router, basic_audit_logger)
    portfolio_manager = PortfolioManager(exchange_router, order_manager, basic_audit_logger)
    
    # Create risk engine
    risk_engine = RiskEngine(compliance_engine, enhanced_audit_logger, portfolio_manager)
    
    print(f"✅ Risk Engine initialized with {len(risk_engine.limits)} limits")
    
    # Test pre-trade risk check
    test_order = OrderRequest(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("5.0"),  # Large order to trigger limits
        price=Decimal("45000")
    )
    
    print(f"\n🔍 Testing pre-trade risk check:")
    print(f"Order: {test_order.quantity} {test_order.symbol} @ ${test_order.price}")
    
    risk_assessment = risk_engine.check_pre_trade_risk(test_order)
    print(f"Risk Decision: {risk_assessment.decision.value}")
    print(f"Risk Score: {risk_assessment.risk_score:.2f}")
    print(f"Reason: {risk_assessment.reason}")
    
    # Run stress tests
    print(f"\n🧪 Running stress tests:")
    stress_results = risk_engine.run_stress_tests()
    print(f"Overall Result: {'PASSED' if stress_results['overall_passed'] else 'FAILED'}")
    print(f"Risk Assessment: {stress_results['risk_assessment']}")
    
    for test_id, result in stress_results['test_results'].items():
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"- {result['test_name']}: {status} ({result['percentage_impact']:.1f}% impact)")
    
    # Get risk dashboard
    dashboard = risk_engine.get_risk_dashboard()
    print(f"\n📊 Risk Dashboard:")
    print(f"- Monitoring Status: {dashboard['monitoring_status']}")
    print(f"- Active Limits: {dashboard['limits']['active_limits']}")
    print(f"- Total Breaches: {dashboard['breaches']['total_breaches']}")
    print(f"- Critical Breaches: {dashboard['breaches']['critical_breaches']}")
    print(f"- Stress Tests: {dashboard['stress_tests']['total_tests']}")
    
    # Generate risk report
    report = risk_engine.generate_risk_report(7)  # 7 days
    print(f"\n📋 Risk Report (7 days):")
    print(f"- Total Breaches: {report['executive_summary']['total_breaches']}")
    print(f"- Breach Rate: {report['executive_summary']['breach_rate']:.2f}")
    print(f"- Resolution Rate: {report['executive_summary']['resolution_rate']:.2f}")
    print(f"- Risk Level: {report['executive_summary']['risk_level']}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print(f"\n✅ Risk Management Controls demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_risk_controls())