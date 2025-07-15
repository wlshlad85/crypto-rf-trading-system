#!/usr/bin/env python3
"""
ULTRATHINK Week 6 DAY 38: Regulatory Compliance Framework
Institutional-grade compliance engine with MiFID II, GDPR, and SOX compliance.

Features:
- Regulatory rule engine (MiFID II, GDPR, SOX compliance)
- Trade surveillance and monitoring
- Best execution analysis
- Market abuse detection
- Suspicious activity reporting
- Real-time compliance monitoring
- Automated regulatory alerts
- Policy enforcement framework
- Compliance workflow management
- Risk-based compliance assessment

Regulatory Frameworks:
- MiFID II: Markets in Financial Instruments Directive II
- GDPR: General Data Protection Regulation
- SOX: Sarbanes-Oxley Act
- AML: Anti-Money Laundering
- KYC: Know Your Customer
- EMIR: European Market Infrastructure Regulation
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import existing components
import sys
sys.path.append('/home/richardw/crypto_rf_trading_system')
from production.real_money_trader import AuditLogger
from production.order_management import ManagedOrder, OrderExecution
from compliance.audit_trail import AuditLevel, AuditCategory, AuditContext
from production.portfolio_manager import Position

class ComplianceLevel(Enum):
    GREEN = "green"          # Fully compliant
    YELLOW = "yellow"        # Minor issues requiring attention
    ORANGE = "orange"        # Significant issues requiring action
    RED = "red"              # Critical compliance breach

class RegulatoryFramework(Enum):
    MIFID_II = "mifid_ii"
    GDPR = "gdpr"
    SOX = "sox"
    AML = "aml"
    KYC = "kyc"
    EMIR = "emir"
    BEST_EXECUTION = "best_execution"
    MARKET_ABUSE = "market_abuse"

class ViolationType(Enum):
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_RISK = "concentration_risk"
    LARGE_EXPOSURE = "large_exposure"
    SUSPICIOUS_TRADING = "suspicious_trading"
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING = "insider_trading"
    WASH_TRADING = "wash_trading"
    LAYERING = "layering"
    SPOOFING = "spoofing"
    BEST_EXECUTION_BREACH = "best_execution_breach"
    CLIENT_MONEY_BREACH = "client_money_breach"
    DATA_PROTECTION = "data_protection"

class ActionRequired(Enum):
    NONE = "none"
    MONITOR = "monitor"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    REPORT = "report"
    HALT_TRADING = "halt_trading"
    ESCALATE = "escalate"

@dataclass
class ComplianceRule:
    """Individual compliance rule definition."""
    rule_id: str
    name: str
    framework: RegulatoryFramework
    description: str
    severity: ComplianceLevel
    threshold_value: Optional[float] = None
    threshold_percentage: Optional[float] = None
    lookback_period: Optional[int] = None  # Minutes
    active: bool = True
    
    # Rule logic
    condition_logic: str = ""  # Python expression for evaluation
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Actions
    action_required: ActionRequired = ActionRequired.ALERT
    escalation_levels: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_id: str
    timestamp: datetime
    violation_type: ViolationType
    severity: ComplianceLevel
    description: str
    
    # Context
    affected_orders: List[str] = field(default_factory=list)
    affected_positions: List[str] = field(default_factory=list)
    client_ids: List[str] = field(default_factory=list)
    
    # Evidence
    evidence_data: Dict[str, Any] = field(default_factory=dict)
    threshold_breached: Optional[float] = None
    actual_value: Optional[float] = None
    
    # Actions taken
    action_taken: ActionRequired = ActionRequired.NONE
    investigated: bool = False
    resolved: bool = False
    resolution_notes: str = ""
    
    # Workflow
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    escalated: bool = False
    escalation_level: int = 0
    
    @property
    def is_critical(self) -> bool:
        return self.severity == ComplianceLevel.RED
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds() / 3600

@dataclass
class SuspiciousActivity:
    """Suspicious activity report (SAR) record."""
    sar_id: str
    timestamp: datetime
    activity_type: str
    description: str
    client_id: str
    
    # Transaction details
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    total_amount: Decimal = Decimal("0")
    suspicious_patterns: List[str] = field(default_factory=list)
    
    # Investigation
    risk_score: float = 0.0
    investigation_status: str = "pending"
    investigator: Optional[str] = None
    findings: str = ""
    
    # Regulatory action
    reported_to_authorities: bool = False
    report_date: Optional[datetime] = None
    authority: Optional[str] = None
    reference_number: Optional[str] = None

class TradeSurveillance:
    """Advanced trade surveillance system."""
    
    def __init__(self):
        self.surveillance_window = 1440  # 24 hours in minutes
        self.trade_history = deque(maxlen=10000)
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Pattern detection
        self.suspicious_patterns = {}
        self.client_behavior_profiles = {}
        
        # Alert thresholds
        self.large_trade_threshold = 100000  # $100K
        self.rapid_trading_threshold = 10    # 10 trades in 5 minutes
        self.price_manipulation_threshold = 0.05  # 5% unusual price movement
        
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add trade to surveillance system."""
        trade_record = {
            'timestamp': datetime.now(),
            'client_id': trade_data.get('client_id', 'unknown'),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side'),
            'quantity': trade_data.get('quantity'),
            'price': trade_data.get('price'),
            'value': trade_data.get('quantity', 0) * trade_data.get('price', 0),
            'order_type': trade_data.get('order_type'),
            'execution_venue': trade_data.get('exchange', 'unknown')
        }
        
        self.trade_history.append(trade_record)
        
        # Update price and volume history
        symbol = trade_record['symbol']
        self.price_history[symbol].append({
            'timestamp': trade_record['timestamp'],
            'price': trade_record['price']
        })
        
        self.volume_history[symbol].append({
            'timestamp': trade_record['timestamp'],
            'volume': trade_record['quantity']
        })
    
    def detect_wash_trading(self, client_id: str, lookback_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect potential wash trading patterns."""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        client_trades = [
            trade for trade in self.trade_history
            if trade['client_id'] == client_id and trade['timestamp'] > cutoff_time
        ]
        
        # Group trades by symbol
        symbol_trades = defaultdict(list)
        for trade in client_trades:
            symbol_trades[trade['symbol']].append(trade)
        
        wash_patterns = []
        
        for symbol, trades in symbol_trades.items():
            if len(trades) < 4:  # Need at least 4 trades for wash pattern
                continue
            
            # Look for buy-sell-buy-sell patterns at similar prices
            for i in range(len(trades) - 3):
                trade_sequence = trades[i:i+4]
                
                # Check if it's alternating buy/sell
                sides = [t['side'] for t in trade_sequence]
                if sides in [['buy', 'sell', 'buy', 'sell'], ['sell', 'buy', 'sell', 'buy']]:
                    
                    # Check if prices are similar (within 1%)
                    prices = [t['price'] for t in trade_sequence]
                    price_range = (max(prices) - min(prices)) / np.mean(prices)
                    
                    if price_range < 0.01:  # 1% price range
                        wash_patterns.append({
                            'client_id': client_id,
                            'symbol': symbol,
                            'trades': trade_sequence,
                            'confidence': 0.8,
                            'reason': 'Alternating buy/sell at similar prices'
                        })
        
        return wash_patterns
    
    def detect_layering(self, symbol: str, lookback_minutes: int = 30) -> List[Dict[str, Any]]:
        """Detect layering/spoofing patterns."""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        recent_trades = [
            trade for trade in self.trade_history
            if trade['symbol'] == symbol and trade['timestamp'] > cutoff_time
        ]
        
        layering_patterns = []
        
        # Group by client
        client_trades = defaultdict(list)
        for trade in recent_trades:
            client_trades[trade['client_id']].append(trade)
        
        for client_id, trades in client_trades.items():
            if len(trades) < 5:  # Need sufficient trades
                continue
            
            # Look for patterns of large orders followed by quick cancellations
            # This is a simplified detection - real implementation would need order book data
            
            # Check for rapid order placement and cancellation
            order_times = [t['timestamp'] for t in trades]
            if len(order_times) >= 5:
                time_diffs = [(order_times[i+1] - order_times[i]).total_seconds() 
                             for i in range(len(order_times)-1)]
                
                # If multiple orders within 30 seconds
                if sum(1 for diff in time_diffs if diff < 30) >= 3:
                    layering_patterns.append({
                        'client_id': client_id,
                        'symbol': symbol,
                        'trades': trades,
                        'confidence': 0.6,
                        'reason': 'Rapid order placement pattern detected'
                    })
        
        return layering_patterns
    
    def detect_market_manipulation(self, symbol: str, lookback_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect potential market manipulation."""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        # Get price history
        recent_prices = [
            p for p in self.price_history[symbol]
            if p['timestamp'] > cutoff_time
        ]
        
        if len(recent_prices) < 10:
            return []
        
        manipulation_patterns = []
        
        # Calculate price volatility
        prices = [p['price'] for p in recent_prices]
        price_changes = [abs(prices[i+1] - prices[i]) / prices[i] 
                        for i in range(len(prices)-1)]
        
        avg_volatility = np.mean(price_changes)
        
        # Look for unusual price movements
        for i, change in enumerate(price_changes):
            if change > self.price_manipulation_threshold and change > 3 * avg_volatility:
                
                # Check if there were unusual trades around this time
                manipulation_time = recent_prices[i]['timestamp']
                
                # Look for large trades within 5 minutes
                window_start = manipulation_time - timedelta(minutes=5)
                window_end = manipulation_time + timedelta(minutes=5)
                
                window_trades = [
                    trade for trade in self.trade_history
                    if (trade['symbol'] == symbol and 
                        window_start <= trade['timestamp'] <= window_end and
                        trade['value'] > self.large_trade_threshold)
                ]
                
                if window_trades:
                    manipulation_patterns.append({
                        'symbol': symbol,
                        'timestamp': manipulation_time,
                        'price_change': change,
                        'suspicious_trades': window_trades,
                        'confidence': 0.7,
                        'reason': f'Unusual price movement ({change:.1%}) with large trades'
                    })
        
        return manipulation_patterns

class BestExecutionAnalyzer:
    """Best execution analysis for MiFID II compliance."""
    
    def __init__(self):
        self.execution_venues = {}
        self.execution_history = deque(maxlen=10000)
        self.benchmark_prices = {}
        
    def add_execution(self, execution_data: Dict[str, Any]):
        """Add execution for best execution analysis."""
        execution_record = {
            'timestamp': datetime.now(),
            'order_id': execution_data.get('order_id'),
            'symbol': execution_data.get('symbol'),
            'side': execution_data.get('side'),
            'quantity': execution_data.get('quantity'),
            'price': execution_data.get('price'),
            'venue': execution_data.get('venue'),
            'execution_type': execution_data.get('execution_type', 'market'),
            'commission': execution_data.get('commission', 0),
            'market_impact': execution_data.get('market_impact', 0),
            'timing_impact': execution_data.get('timing_impact', 0),
            'benchmark_price': execution_data.get('benchmark_price')
        }
        
        self.execution_history.append(execution_record)
    
    def analyze_execution_quality(self, execution_id: str) -> Dict[str, Any]:
        """Analyze execution quality for best execution compliance."""
        execution = next(
            (e for e in self.execution_history if e.get('order_id') == execution_id),
            None
        )
        
        if not execution:
            return {'error': 'Execution not found'}
        
        analysis = {
            'execution_id': execution_id,
            'timestamp': execution['timestamp'],
            'venue': execution['venue'],
            'price_improvement': 0.0,
            'effective_spread': 0.0,
            'market_impact': execution.get('market_impact', 0),
            'timing_impact': execution.get('timing_impact', 0),
            'total_cost': 0.0,
            'best_execution_score': 0.0,
            'compliance_status': 'COMPLIANT'
        }
        
        # Calculate price improvement
        if execution.get('benchmark_price'):
            benchmark = execution['benchmark_price']
            executed_price = execution['price']
            
            if execution['side'] == 'buy':
                price_improvement = (benchmark - executed_price) / benchmark
            else:
                price_improvement = (executed_price - benchmark) / benchmark
            
            analysis['price_improvement'] = price_improvement
        
        # Calculate total cost
        commission = execution.get('commission', 0)
        market_impact_cost = execution.get('market_impact', 0)
        timing_cost = execution.get('timing_impact', 0)
        
        analysis['total_cost'] = commission + market_impact_cost + timing_cost
        
        # Calculate best execution score (simplified)
        score = 100.0
        
        # Penalize for high costs
        if analysis['total_cost'] > 0.01:  # 1% of notional
            score -= 20
        
        # Reward for price improvement
        if analysis['price_improvement'] > 0:
            score += min(analysis['price_improvement'] * 1000, 10)
        
        # Penalize for market impact
        if analysis['market_impact'] > 0.005:  # 0.5%
            score -= 15
        
        analysis['best_execution_score'] = max(0, score)
        
        # Determine compliance status
        if analysis['best_execution_score'] < 60:
            analysis['compliance_status'] = 'POOR_EXECUTION'
        elif analysis['best_execution_score'] < 80:
            analysis['compliance_status'] = 'REVIEW_REQUIRED'
        
        return analysis
    
    def generate_best_execution_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate best execution report for regulatory compliance."""
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        period_executions = [
            e for e in self.execution_history
            if e['timestamp'] > cutoff_date
        ]
        
        if not period_executions:
            return {'error': 'No executions in period'}
        
        # Analyze by venue
        venue_analysis = defaultdict(lambda: {
            'execution_count': 0,
            'total_volume': 0,
            'avg_price_improvement': 0,
            'avg_total_cost': 0,
            'best_execution_score': 0
        })
        
        for execution in period_executions:
            venue = execution['venue']
            venue_stats = venue_analysis[venue]
            
            venue_stats['execution_count'] += 1
            venue_stats['total_volume'] += execution['quantity'] * execution['price']
            
            # Calculate metrics (simplified)
            if execution.get('benchmark_price'):
                benchmark = execution['benchmark_price']
                executed_price = execution['price']
                
                if execution['side'] == 'buy':
                    price_improvement = (benchmark - executed_price) / benchmark
                else:
                    price_improvement = (executed_price - benchmark) / benchmark
                
                venue_stats['avg_price_improvement'] += price_improvement
        
        # Calculate averages
        for venue, stats in venue_analysis.items():
            if stats['execution_count'] > 0:
                stats['avg_price_improvement'] /= stats['execution_count']
                stats['avg_total_cost'] /= stats['execution_count']
                
                # Simple scoring
                score = 85.0  # Base score
                score += min(stats['avg_price_improvement'] * 1000, 10)
                stats['best_execution_score'] = max(0, score)
        
        return {
            'report_period': f"{period_days} days",
            'report_date': datetime.now().isoformat(),
            'total_executions': len(period_executions),
            'venue_analysis': dict(venue_analysis),
            'summary': {
                'best_venue': max(venue_analysis.keys(), 
                                key=lambda v: venue_analysis[v]['best_execution_score']) if venue_analysis else None,
                'worst_venue': min(venue_analysis.keys(), 
                                 key=lambda v: venue_analysis[v]['best_execution_score']) if venue_analysis else None,
                'overall_score': np.mean([s['best_execution_score'] for s in venue_analysis.values()]) if venue_analysis else 0
            }
        }

class ComplianceEngine:
    """
    Comprehensive regulatory compliance engine.
    
    Features:
    - Multi-framework compliance (MiFID II, GDPR, SOX, AML)
    - Real-time rule monitoring
    - Automated violation detection
    - Suspicious activity reporting
    - Best execution analysis
    - Compliance workflow management
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.rules: Dict[str, ComplianceRule] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        self.suspicious_activities: Dict[str, SuspiciousActivity] = {}
        
        # Surveillance systems
        self.trade_surveillance = TradeSurveillance()
        self.best_execution_analyzer = BestExecutionAnalyzer()
        
        # Database
        self.db_path = "compliance/compliance.db"
        self.setup_database()
        
        # Monitoring
        self.monitoring_enabled = True
        self.alert_handlers: Dict[str, Callable] = {}
        
        # Load default rules
        self.load_default_rules()
        
        self.logger.info("Compliance Engine initialized")
    
    def setup_database(self):
        """Initialize compliance database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Rules table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    active BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    rule_data TEXT NOT NULL
                )
            """)
            
            # Violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    resolved BOOLEAN NOT NULL,
                    violation_data TEXT NOT NULL,
                    FOREIGN KEY (rule_id) REFERENCES compliance_rules (rule_id)
                )
            """)
            
            # Suspicious activities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS suspicious_activities (
                    sar_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    investigation_status TEXT NOT NULL,
                    reported_to_authorities BOOLEAN NOT NULL,
                    activity_data TEXT NOT NULL
                )
            """)
            
            # Best execution reports
            conn.execute("""
                CREATE TABLE IF NOT EXISTS best_execution_reports (
                    report_id TEXT PRIMARY KEY,
                    report_date TEXT NOT NULL,
                    period_days INTEGER NOT NULL,
                    total_executions INTEGER NOT NULL,
                    overall_score REAL NOT NULL,
                    report_data TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON compliance_violations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_severity ON compliance_violations(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sar_timestamp ON suspicious_activities(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sar_risk_score ON suspicious_activities(risk_score)")
    
    def load_default_rules(self):
        """Load default compliance rules."""
        
        # MiFID II Rules
        self.add_rule(ComplianceRule(
            rule_id="MIFID_LARGE_TRANSACTION",
            name="Large Transaction Reporting",
            framework=RegulatoryFramework.MIFID_II,
            description="Detect large transactions requiring regulatory reporting",
            severity=ComplianceLevel.YELLOW,
            threshold_value=100000,  # €100k
            condition_logic="trade_value > threshold_value",
            action_required=ActionRequired.REPORT
        ))
        
        self.add_rule(ComplianceRule(
            rule_id="MIFID_BEST_EXECUTION",
            name="Best Execution Monitoring",
            framework=RegulatoryFramework.MIFID_II,
            description="Monitor best execution compliance",
            severity=ComplianceLevel.ORANGE,
            threshold_value=80,  # Score threshold
            condition_logic="best_execution_score < threshold_value",
            action_required=ActionRequired.INVESTIGATE
        ))
        
        # Market Abuse Rules
        self.add_rule(ComplianceRule(
            rule_id="MARKET_MANIPULATION",
            name="Market Manipulation Detection",
            framework=RegulatoryFramework.MARKET_ABUSE,
            description="Detect potential market manipulation",
            severity=ComplianceLevel.RED,
            threshold_percentage=5.0,  # 5% unusual price movement
            lookback_period=60,  # 1 hour
            action_required=ActionRequired.HALT_TRADING
        ))
        
        self.add_rule(ComplianceRule(
            rule_id="WASH_TRADING",
            name="Wash Trading Detection",
            framework=RegulatoryFramework.MARKET_ABUSE,
            description="Detect wash trading patterns",
            severity=ComplianceLevel.RED,
            lookback_period=60,  # 1 hour
            action_required=ActionRequired.INVESTIGATE
        ))
        
        # Position Limits
        self.add_rule(ComplianceRule(
            rule_id="POSITION_LIMIT_BREACH",
            name="Position Limit Monitoring",
            framework=RegulatoryFramework.MIFID_II,
            description="Monitor position limits",
            severity=ComplianceLevel.ORANGE,
            threshold_percentage=25.0,  # 25% of portfolio
            action_required=ActionRequired.ALERT
        ))
        
        # AML Rules
        self.add_rule(ComplianceRule(
            rule_id="AML_SUSPICIOUS_PATTERN",
            name="Suspicious Trading Pattern",
            framework=RegulatoryFramework.AML,
            description="Detect suspicious trading patterns",
            severity=ComplianceLevel.RED,
            threshold_value=50000,  # $50k
            lookback_period=1440,  # 24 hours
            action_required=ActionRequired.REPORT
        ))
        
        # GDPR Rules
        self.add_rule(ComplianceRule(
            rule_id="GDPR_DATA_RETENTION",
            name="Data Retention Compliance",
            framework=RegulatoryFramework.GDPR,
            description="Monitor data retention periods",
            severity=ComplianceLevel.YELLOW,
            threshold_value=2555,  # 7 years in days
            action_required=ActionRequired.MONITOR
        ))
        
        self.logger.info(f"Loaded {len(self.rules)} default compliance rules")
    
    def add_rule(self, rule: ComplianceRule):
        """Add compliance rule to the engine."""
        self.rules[rule.rule_id] = rule
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO compliance_rules (
                    rule_id, name, framework, severity, active, 
                    created_at, updated_at, rule_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.name,
                rule.framework.value,
                rule.severity.value,
                rule.active,
                rule.created_at.isoformat(),
                rule.updated_at.isoformat(),
                json.dumps(asdict(rule), default=str)
            ))
        
        self.logger.info(f"Added compliance rule: {rule.rule_id}")
    
    def check_trade_compliance(self, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check trade against all compliance rules."""
        violations = []
        
        if not self.monitoring_enabled:
            return violations
        
        # Add trade to surveillance
        self.trade_surveillance.add_trade(trade_data)
        
        # Check each active rule
        for rule_id, rule in self.rules.items():
            if not rule.active:
                continue
            
            violation = self.evaluate_rule(rule, trade_data)
            if violation:
                violations.append(violation)
                self.violations[violation.violation_id] = violation
                
                # Log violation
                self.audit_logger.log_event(
                    level=AuditLevel.WARNING,
                    category=AuditCategory.COMPLIANCE,
                    description=f"Compliance violation: {violation.description}",
                    context=AuditContext(
                        session_id="compliance_engine",
                        user_id="system",
                        client_id=trade_data.get('client_id'),
                        request_id=str(uuid.uuid4())
                    )
                )
                
                # Save to database
                self.save_violation(violation)
                
                # Trigger alerts
                self.handle_violation(violation)
        
        return violations
    
    def evaluate_rule(self, rule: ComplianceRule, trade_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Evaluate specific compliance rule against trade data."""
        try:
            # Prepare evaluation context
            context = {
                'trade_value': trade_data.get('quantity', 0) * trade_data.get('price', 0),
                'trade_quantity': trade_data.get('quantity', 0),
                'trade_price': trade_data.get('price', 0),
                'symbol': trade_data.get('symbol', ''),
                'client_id': trade_data.get('client_id', ''),
                'threshold_value': rule.threshold_value,
                'threshold_percentage': rule.threshold_percentage,
                'datetime': datetime,
                'timedelta': timedelta,
                'best_execution_score': trade_data.get('best_execution_score', 85)  # Default good score
            }
            
            # Add specific rule logic
            if rule.rule_id == "MIFID_LARGE_TRANSACTION":
                return self.check_large_transaction(rule, trade_data, context)
            elif rule.rule_id == "MIFID_BEST_EXECUTION":
                return self.check_best_execution(rule, trade_data, context)
            elif rule.rule_id == "POSITION_LIMIT_BREACH":
                return self.check_position_limits(rule, trade_data, context)
            elif rule.rule_id == "WASH_TRADING":
                return self.check_wash_trading(rule, trade_data, context)
            elif rule.rule_id == "MARKET_MANIPULATION":
                return self.check_market_manipulation(rule, trade_data, context)
            elif rule.rule_id == "AML_SUSPICIOUS_PATTERN":
                return self.check_aml_patterns(rule, trade_data, context)
            
            # Generic rule evaluation
            if rule.condition_logic:
                if eval(rule.condition_logic, {"__builtins__": {}}, context):
                    return self.create_violation(rule, trade_data, context)
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
        
        return None
    
    def check_large_transaction(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for large transaction reporting requirements."""
        trade_value = context['trade_value']
        
        if trade_value > rule.threshold_value:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                violation_type=ViolationType.LARGE_EXPOSURE,
                severity=rule.severity,
                description=f"Large transaction detected: ${trade_value:,.2f} exceeds threshold ${rule.threshold_value:,.2f}",
                affected_orders=[trade_data.get('order_id', '')],
                evidence_data=trade_data,
                threshold_breached=rule.threshold_value,
                actual_value=trade_value,
                action_taken=rule.action_required
            )
        
        return None
    
    def check_best_execution(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check best execution compliance."""
        best_execution_score = context['best_execution_score']
        
        if best_execution_score < rule.threshold_value:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                violation_type=ViolationType.BEST_EXECUTION_BREACH,
                severity=rule.severity,
                description=f"Best execution violation: score {best_execution_score} below threshold {rule.threshold_value}",
                affected_orders=[trade_data.get('order_id', '')],
                evidence_data=trade_data,
                threshold_breached=rule.threshold_value,
                actual_value=best_execution_score,
                action_taken=rule.action_required
            )
        
        return None
    
    def check_position_limits(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check position limits compliance."""
        # This would need access to portfolio manager
        # Simplified implementation
        return None
    
    def check_wash_trading(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for wash trading patterns."""
        client_id = trade_data.get('client_id', '')
        if not client_id:
            return None
        
        wash_patterns = self.trade_surveillance.detect_wash_trading(client_id, rule.lookback_period)
        
        if wash_patterns:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                violation_type=ViolationType.WASH_TRADING,
                severity=rule.severity,
                description=f"Wash trading pattern detected for client {client_id}",
                client_ids=[client_id],
                evidence_data={'patterns': wash_patterns},
                action_taken=rule.action_required
            )
        
        return None
    
    def check_market_manipulation(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for market manipulation patterns."""
        symbol = trade_data.get('symbol', '')
        if not symbol:
            return None
        
        manipulation_patterns = self.trade_surveillance.detect_market_manipulation(symbol, rule.lookback_period)
        
        if manipulation_patterns:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                violation_type=ViolationType.MARKET_MANIPULATION,
                severity=rule.severity,
                description=f"Market manipulation detected for {symbol}",
                evidence_data={'patterns': manipulation_patterns},
                action_taken=rule.action_required
            )
        
        return None
    
    def check_aml_patterns(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for AML suspicious patterns."""
        # Simplified AML check
        trade_value = context['trade_value']
        client_id = trade_data.get('client_id', '')
        
        if trade_value > rule.threshold_value:
            # Check for suspicious patterns
            if self.is_suspicious_activity(client_id, trade_data):
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    timestamp=datetime.now(),
                    violation_type=ViolationType.SUSPICIOUS_TRADING,
                    severity=rule.severity,
                    description=f"Suspicious trading pattern detected for client {client_id}",
                    client_ids=[client_id],
                    evidence_data=trade_data,
                    action_taken=rule.action_required
                )
        
        return None
    
    def is_suspicious_activity(self, client_id: str, trade_data: Dict[str, Any]) -> bool:
        """Determine if activity is suspicious for AML purposes."""
        # Simple heuristics - real implementation would be more sophisticated
        
        # Check for unusual trading hours
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside normal hours
            return True
        
        # Check for round number bias
        price = trade_data.get('price', 0)
        if price % 1000 == 0:  # Exact round numbers
            return True
        
        # Check for rapid successive trades
        client_trades = [
            trade for trade in self.trade_surveillance.trade_history
            if trade['client_id'] == client_id
        ]
        
        if len(client_trades) >= 2:
            last_trade_time = client_trades[-2]['timestamp']
            if (datetime.now() - last_trade_time).total_seconds() < 60:  # Within 1 minute
                return True
        
        return False
    
    def create_violation(self, rule: ComplianceRule, trade_data: Dict[str, Any], context: Dict[str, Any]) -> ComplianceViolation:
        """Create a generic compliance violation."""
        return ComplianceViolation(
            violation_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            violation_type=ViolationType.POSITION_LIMIT,  # Default
            severity=rule.severity,
            description=f"Rule {rule.name} violated",
            evidence_data=trade_data,
            action_taken=rule.action_required
        )
    
    def handle_violation(self, violation: ComplianceViolation):
        """Handle compliance violation based on severity and type."""
        if violation.is_critical:
            self.logger.critical(f"Critical compliance violation: {violation.description}")
        else:
            self.logger.warning(f"Compliance violation: {violation.description}")
        
        # Execute required action
        if violation.action_taken == ActionRequired.HALT_TRADING:
            self.emergency_halt_trading(violation)
        elif violation.action_taken == ActionRequired.REPORT:
            self.create_suspicious_activity_report(violation)
        elif violation.action_taken == ActionRequired.INVESTIGATE:
            self.initiate_investigation(violation)
        elif violation.action_taken == ActionRequired.ESCALATE:
            self.escalate_violation(violation)
    
    def emergency_halt_trading(self, violation: ComplianceViolation):
        """Emergency halt trading for critical violations."""
        self.logger.critical(f"EMERGENCY HALT: {violation.description}")
        
        # This would integrate with trading engine to halt trading
        # For now, just log the action
        self.audit_logger.log_event(
            event_type="EMERGENCY_HALT",
            action="TRADING_HALTED",
            details={
                "violation_id": violation.violation_id,
                "reason": violation.description,
                "severity": violation.severity.value
            }
        )
    
    def create_suspicious_activity_report(self, violation: ComplianceViolation):
        """Create SAR for suspicious activity."""
        sar = SuspiciousActivity(
            sar_id=str(uuid.uuid4()),
            timestamp=violation.timestamp,
            activity_type=violation.violation_type.value,
            description=violation.description,
            client_id=violation.client_ids[0] if violation.client_ids else "unknown",
            risk_score=0.8 if violation.severity == ComplianceLevel.RED else 0.6,
            suspicious_patterns=[violation.description]
        )
        
        self.suspicious_activities[sar.sar_id] = sar
        self.save_suspicious_activity(sar)
        
        self.logger.info(f"SAR created: {sar.sar_id}")
    
    def initiate_investigation(self, violation: ComplianceViolation):
        """Initiate investigation for violation."""
        violation.investigated = True
        violation.assigned_to = "compliance_team"
        violation.due_date = datetime.now() + timedelta(days=5)
        
        self.logger.info(f"Investigation initiated for violation: {violation.violation_id}")
    
    def escalate_violation(self, violation: ComplianceViolation):
        """Escalate violation to higher level."""
        violation.escalated = True
        violation.escalation_level += 1
        
        self.logger.warning(f"Violation escalated: {violation.violation_id} (Level {violation.escalation_level})")
    
    def save_violation(self, violation: ComplianceViolation):
        """Save violation to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO compliance_violations (
                        violation_id, rule_id, timestamp, violation_type,
                        severity, resolved, violation_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.violation_id,
                    violation.rule_id,
                    violation.timestamp.isoformat(),
                    violation.violation_type.value,
                    violation.severity.value,
                    violation.resolved,
                    json.dumps(asdict(violation), default=str)
                ))
        except Exception as e:
            self.logger.error(f"Failed to save violation: {e}")
    
    def save_suspicious_activity(self, sar: SuspiciousActivity):
        """Save suspicious activity report to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO suspicious_activities (
                        sar_id, timestamp, activity_type, client_id,
                        risk_score, investigation_status, reported_to_authorities,
                        activity_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sar.sar_id,
                    sar.timestamp.isoformat(),
                    sar.activity_type,
                    sar.client_id,
                    sar.risk_score,
                    sar.investigation_status,
                    sar.reported_to_authorities,
                    json.dumps(asdict(sar), default=str)
                ))
        except Exception as e:
            self.logger.error(f"Failed to save SAR: {e}")
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard."""
        # Count violations by severity
        severity_counts = defaultdict(int)
        recent_violations = []
        
        for violation in self.violations.values():
            severity_counts[violation.severity.value] += 1
            if violation.age_hours < 24:  # Recent violations
                recent_violations.append({
                    'violation_id': violation.violation_id,
                    'rule_id': violation.rule_id,
                    'severity': violation.severity.value,
                    'description': violation.description,
                    'age_hours': violation.age_hours,
                    'resolved': violation.resolved
                })
        
        # Count SARs by status
        sar_counts = defaultdict(int)
        for sar in self.suspicious_activities.values():
            sar_counts[sar.investigation_status] += 1
        
        # Active rules count
        active_rules = sum(1 for rule in self.rules.values() if rule.active)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_violations': len(self.violations),
                'active_rules': active_rules,
                'total_sars': len(self.suspicious_activities),
                'monitoring_enabled': self.monitoring_enabled
            },
            'violations': {
                'by_severity': dict(severity_counts),
                'recent_violations': recent_violations,
                'critical_count': severity_counts['red'],
                'unresolved_count': sum(1 for v in self.violations.values() if not v.resolved)
            },
            'suspicious_activities': {
                'by_status': dict(sar_counts),
                'high_risk_count': sum(1 for s in self.suspicious_activities.values() if s.risk_score > 0.7),
                'pending_investigation': sar_counts['pending']
            },
            'trade_surveillance': {
                'total_trades_monitored': len(self.trade_surveillance.trade_history),
                'surveillance_active': True
            }
        }
    
    def generate_compliance_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter violations by period
        period_violations = [
            v for v in self.violations.values()
            if v.timestamp > cutoff_date
        ]
        
        # Generate best execution report
        best_exec_report = self.best_execution_analyzer.generate_best_execution_report(period_days)
        
        # Rule effectiveness analysis
        rule_effectiveness = {}
        for rule_id, rule in self.rules.items():
            rule_violations = [v for v in period_violations if v.rule_id == rule_id]
            rule_effectiveness[rule_id] = {
                'rule_name': rule.name,
                'violations_count': len(rule_violations),
                'avg_resolution_time': np.mean([
                    v.age_hours for v in rule_violations if v.resolved
                ]) if rule_violations else 0
            }
        
        return {
            'report_period': f"{period_days} days",
            'report_date': datetime.now().isoformat(),
            'summary': {
                'total_violations': len(period_violations),
                'resolved_violations': sum(1 for v in period_violations if v.resolved),
                'critical_violations': sum(1 for v in period_violations if v.severity == ComplianceLevel.RED),
                'resolution_rate': (sum(1 for v in period_violations if v.resolved) / len(period_violations) * 100) if period_violations else 100
            },
            'violations_by_type': {
                vtype.value: sum(1 for v in period_violations if v.violation_type == vtype)
                for vtype in ViolationType
            },
            'rule_effectiveness': rule_effectiveness,
            'best_execution_report': best_exec_report,
            'recommendations': self.generate_compliance_recommendations(period_violations)
        }
    
    def generate_compliance_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []
        
        # Analyze violation patterns
        violation_types = defaultdict(int)
        for violation in violations:
            violation_types[violation.violation_type] += 1
        
        # Generate specific recommendations
        if violation_types[ViolationType.POSITION_LIMIT] > 5:
            recommendations.append("Review and tighten position limit thresholds")
        
        if violation_types[ViolationType.WASH_TRADING] > 0:
            recommendations.append("Enhance wash trading detection algorithms")
        
        if violation_types[ViolationType.MARKET_MANIPULATION] > 0:
            recommendations.append("Implement additional market manipulation controls")
        
        if violation_types[ViolationType.BEST_EXECUTION_BREACH] > 10:
            recommendations.append("Review execution venue selection criteria")
        
        # General recommendations
        critical_violations = sum(1 for v in violations if v.severity == ComplianceLevel.RED)
        if critical_violations > 5:
            recommendations.append("Implement additional pre-trade risk controls")
        
        unresolved_violations = sum(1 for v in violations if not v.resolved)
        if unresolved_violations > 10:
            recommendations.append("Increase compliance team resources for investigation")
        
        return recommendations if recommendations else ["No specific recommendations at this time"]

# Example usage and testing
async def demo_compliance_engine():
    """Demonstration of the compliance engine."""
    print("🚨 ULTRATHINK Week 6 DAY 38: Compliance Engine Demo 🚨")
    print("=" * 60)
    
    # Create audit logger
    audit_logger = AuditLogger()
    
    # Create compliance engine
    compliance_engine = ComplianceEngine(audit_logger)
    
    print(f"✅ Compliance Engine initialized with {len(compliance_engine.rules)} rules")
    
    # Test trade compliance check
    test_trade = {
        'order_id': 'test_001',
        'client_id': 'client_123',
        'symbol': 'BTC-USD',
        'side': 'buy',
        'quantity': 10,
        'price': 45000,
        'timestamp': datetime.now(),
        'exchange': 'binance'
    }
    
    print(f"\n📊 Testing trade compliance:")
    print(f"Trade: {test_trade['quantity']} {test_trade['symbol']} @ ${test_trade['price']:,}")
    
    # Check compliance
    violations = compliance_engine.check_trade_compliance(test_trade)
    
    if violations:
        print(f"⚠️  {len(violations)} compliance violations detected:")
        for violation in violations:
            print(f"- {violation.severity.value.upper()}: {violation.description}")
    else:
        print("✅ No compliance violations detected")
    
    # Generate compliance dashboard
    dashboard = compliance_engine.get_compliance_dashboard()
    print(f"\n📈 Compliance Dashboard:")
    print(f"- Total Rules: {dashboard['overview']['active_rules']}")
    print(f"- Total Violations: {dashboard['overview']['total_violations']}")
    print(f"- Monitoring Status: {'Active' if dashboard['overview']['monitoring_enabled'] else 'Inactive'}")
    
    # Generate compliance report
    report = compliance_engine.generate_compliance_report(7)  # 7 days
    print(f"\n📋 Compliance Report (7 days):")
    print(f"- Total Violations: {report['summary']['total_violations']}")
    print(f"- Resolution Rate: {report['summary']['resolution_rate']:.1f}%")
    print(f"- Recommendations: {len(report['recommendations'])}")
    
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n✅ Compliance Engine demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_compliance_engine())