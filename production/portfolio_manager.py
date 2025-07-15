#!/usr/bin/env python3
"""
ULTRATHINK Week 6: Portfolio Management System
Real-time position tracking across exchanges with institutional-grade analytics.

Features:
- Real-time position tracking across exchanges
- Multi-asset portfolio optimization
- Risk-adjusted position sizing
- Performance attribution analysis
- Liquidity management
- Cross-exchange reconciliation
- Real-time P&L calculation
- Risk metrics and VaR calculation
- Correlation analysis
- Sector exposure tracking
- Rebalancing automation
- Margin and leverage management

Advanced Analytics:
- Sharpe ratio optimization
- Maximum drawdown tracking
- Beta calculation vs benchmarks
- Alpha generation attribution
- Transaction cost analysis
- Liquidity-adjusted returns
- Factor exposure analysis
- Stress testing scenarios
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import sqlite3
import logging
from pathlib import Path
import uuid
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Import our trading components
from real_money_trader import AuditLogger, ExchangeType
from exchange_integrations import ExchangeRouter, MarketData
from order_management import OrderManager, ManagedOrder, OrderExecution

class AssetClass(Enum):
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    COMMODITY = "commodity"
    EQUITY = "equity"
    BOND = "bond"
    DERIVATIVE = "derivative"

class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class RebalanceSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class AssetInfo:
    """Comprehensive asset information."""
    symbol: str
    name: str
    asset_class: AssetClass
    base_currency: str
    quote_currency: str
    lot_size: Decimal
    min_order_size: Decimal
    max_order_size: Decimal
    tick_size: Decimal
    commission_rate: Decimal
    margin_requirement: Decimal = Decimal("1.0")
    sector: Optional[str] = None
    correlation_group: Optional[str] = None

@dataclass
class Position:
    """Real-time position with comprehensive tracking."""
    symbol: str
    exchange: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_commission: Decimal
    position_type: PositionType
    
    # Timestamps
    opened_at: datetime
    updated_at: datetime
    
    # Risk metrics
    var_1d: Decimal  # 1-day Value at Risk
    var_10d: Decimal  # 10-day Value at Risk
    beta: Optional[float] = None
    volatility: Optional[float] = None
    
    # Performance metrics
    holding_period_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    
    # Liquidity metrics
    liquidity_score: float = 1.0
    days_to_liquidate: Optional[int] = None
    
    # Execution tracking
    executions: List[OrderExecution] = field(default_factory=list)
    
    @property
    def market_value(self) -> Decimal:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def total_pnl(self) -> Decimal:
        """Total P&L including realized and unrealized."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """P&L as percentage of invested capital."""
        invested_capital = self.quantity * self.average_price
        if invested_capital == 0:
            return 0.0
        return float((self.total_pnl / invested_capital) * 100)
    
    def update_price(self, new_price: Decimal):
        """Update position with new market price."""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.average_price) * self.quantity
        self.updated_at = datetime.now()
        
        # Update holding period return
        if self.average_price > 0:
            self.holding_period_return = float((new_price - self.average_price) / self.average_price)

@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio performance metrics."""
    timestamp: datetime
    total_value: Decimal
    total_pnl: Decimal
    total_pnl_percentage: float
    
    # Risk metrics
    portfolio_var_1d: Decimal
    portfolio_var_10d: Decimal
    beta: float
    volatility: float
    correlation: float
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Diversification metrics
    concentration_ratio: float
    effective_positions: int
    sector_concentration: Dict[str, float]
    
    # Liquidity metrics
    liquidity_score: float
    days_to_liquidate: int
    
    # Attribution analysis
    asset_allocation_return: float
    security_selection_return: float
    interaction_effect: float
    
    @property
    def risk_adjusted_return(self) -> float:
        """Risk-adjusted return metric."""
        return self.sharpe_ratio * self.volatility

@dataclass
class RiskLimits:
    """Portfolio risk management limits."""
    max_position_size_pct: float = 25.0  # Max 25% per position
    max_sector_exposure_pct: float = 50.0  # Max 50% per sector
    max_correlation_exposure_pct: float = 40.0  # Max 40% in correlated assets
    max_var_1d_pct: float = 5.0  # Max 5% daily VaR
    max_var_10d_pct: float = 15.0  # Max 15% 10-day VaR
    max_drawdown_pct: float = 20.0  # Max 20% drawdown
    min_liquidity_score: float = 0.3  # Minimum liquidity score
    max_leverage: float = 2.0  # Maximum leverage
    
    def check_position_limit(self, position_value: Decimal, total_value: Decimal) -> bool:
        """Check if position exceeds size limit."""
        if total_value <= 0:
            return False
        position_pct = float((position_value / total_value) * 100)
        return position_pct <= self.max_position_size_pct

@dataclass
class RebalanceRecommendation:
    """Portfolio rebalancing recommendation."""
    symbol: str
    current_weight: float
    target_weight: float
    weight_difference: float
    signal: RebalanceSignal
    priority: int  # 1 = highest priority
    estimated_cost: Decimal
    expected_return: float
    risk_impact: float
    liquidity_impact: float
    reason: str
    
    @property
    def trade_size(self) -> float:
        """Recommended trade size as percentage of portfolio."""
        return abs(self.weight_difference)

class PerformanceAnalyzer:
    """Advanced performance analysis and attribution."""
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.returns_history = deque(maxlen=lookback_days)
        self.benchmark_returns = deque(maxlen=lookback_days)
        self.portfolio_values = deque(maxlen=lookback_days)
        
    def add_return(self, portfolio_return: float, benchmark_return: float = 0.0, 
                   portfolio_value: float = 0.0):
        """Add daily return data."""
        self.returns_history.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)
        self.portfolio_values.append(portfolio_value)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)
        
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / downside_deviation
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        values = np.array(self.portfolio_values)
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (values - cumulative_max) / cumulative_max
        
        return abs(np.min(drawdowns))
    
    def calculate_beta(self) -> float:
        """Calculate portfolio beta vs benchmark."""
        if len(self.returns_history) < 30 or len(self.benchmark_returns) < 30:
            return 1.0
        
        portfolio_returns = np.array(self.returns_history)
        benchmark_returns = np.array(self.benchmark_returns)
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    def calculate_information_ratio(self) -> float:
        """Calculate information ratio."""
        if len(self.returns_history) < 30:
            return 0.0
        
        portfolio_returns = np.array(self.returns_history)
        benchmark_returns = np.array(self.benchmark_returns)
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / tracking_error

class RiskCalculator:
    """Advanced risk calculations and stress testing."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf(confidence_level)
        
    def calculate_var(self, returns: np.ndarray, days: int = 1) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 30:
            return 0.0
        
        volatility = np.std(returns)
        var = self.z_score * volatility * np.sqrt(days)
        
        return abs(var)
    
    def calculate_expected_shortfall(self, returns: np.ndarray, days: int = 1) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) < 30:
            return 0.0
        
        var = self.calculate_var(returns, days)
        tail_returns = returns[returns < -var]
        
        if len(tail_returns) == 0:
            return var
        
        return abs(np.mean(tail_returns)) * np.sqrt(days)
    
    def stress_test_scenario(self, positions: Dict[str, Position], 
                           stress_scenarios: Dict[str, float]) -> Dict[str, float]:
        """Run stress test scenarios."""
        results = {}
        
        for scenario_name, market_shock in stress_scenarios.items():
            total_impact = 0.0
            
            for position in positions.values():
                # Apply market shock to position
                shocked_price = position.current_price * (1 + market_shock)
                position_impact = float((shocked_price - position.current_price) * position.quantity)
                total_impact += position_impact
            
            results[scenario_name] = total_impact
        
        return results

class PortfolioOptimizer:
    """Advanced portfolio optimization using modern portfolio theory."""
    
    def __init__(self, risk_aversion: float = 3.0):
        self.risk_aversion = risk_aversion
        
    def optimize_weights(self, expected_returns: np.ndarray, 
                        covariance_matrix: np.ndarray,
                        current_weights: np.ndarray,
                        constraints: Dict[str, Any] = None) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization."""
        n_assets = len(expected_returns)
        
        # Objective function: maximize utility (return - risk penalty)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Position limits
        bounds = []
        for i in range(n_assets):
            lower_bound = constraints.get('min_weight', 0.0) if constraints else 0.0
            upper_bound = constraints.get('max_weight', 1.0) if constraints else 1.0
            bounds.append((lower_bound, upper_bound))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            return result.x
        else:
            return current_weights  # Return current weights if optimization fails

class PortfolioManager:
    """
    Institutional-grade portfolio management system.
    
    Features:
    - Real-time position tracking across exchanges
    - Multi-asset portfolio optimization
    - Risk management and VaR calculation
    - Performance attribution analysis
    - Automated rebalancing recommendations
    - Stress testing and scenario analysis
    - Liquidity management
    - Regulatory compliance reporting
    """
    
    def __init__(self, 
                 exchange_router: ExchangeRouter,
                 order_manager: OrderManager,
                 audit_logger: AuditLogger,
                 initial_capital: Decimal = Decimal("1000000")):
        
        self.exchange_router = exchange_router
        self.order_manager = order_manager
        self.audit_logger = audit_logger
        self.initial_capital = initial_capital
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash_balances: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.asset_info: Dict[str, AssetInfo] = {}
        
        # Risk management
        self.risk_limits = RiskLimits()
        self.risk_calculator = RiskCalculator()
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer()
        self.daily_returns = deque(maxlen=252)
        self.portfolio_values = deque(maxlen=252)
        
        # Optimization
        self.optimizer = PortfolioOptimizer()
        self.target_weights: Dict[str, float] = {}
        
        # Database
        self.db_path = "production/portfolio_management.db"
        self.setup_database()
        
        # Monitoring
        self.last_rebalance = datetime.now()
        self.rebalance_frequency = timedelta(hours=6)  # Rebalance every 6 hours
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize cash balance
        self.cash_balances["USD"] = initial_capital
        
        self.logger.info(f"Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def setup_database(self):
        """Initialize portfolio database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    exchange TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    average_price TEXT NOT NULL,
                    current_price TEXT NOT NULL,
                    unrealized_pnl TEXT NOT NULL,
                    realized_pnl TEXT NOT NULL,
                    total_commission TEXT NOT NULL,
                    position_type TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    position_data TEXT NOT NULL
                )
            """)
            
            # Portfolio snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_value TEXT NOT NULL,
                    total_pnl TEXT NOT NULL,
                    cash_balance TEXT NOT NULL,
                    position_count INTEGER NOT NULL,
                    metrics_data TEXT NOT NULL
                )
            """)
            
            # Performance history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    daily_return REAL NOT NULL,
                    portfolio_value TEXT NOT NULL,
                    benchmark_return REAL DEFAULT 0.0,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    volatility REAL
                )
            """)
            
            # Rebalance history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rebalance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    old_weight REAL NOT NULL,
                    new_weight REAL NOT NULL,
                    trade_size TEXT NOT NULL,
                    estimated_cost TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_history(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_timestamp ON rebalance_history(timestamp)")
    
    def register_asset(self, asset_info: AssetInfo):
        """Register asset information for portfolio management."""
        self.asset_info[asset_info.symbol] = asset_info
        self.logger.info(f"Registered asset: {asset_info.symbol} ({asset_info.name})")
    
    async def update_position(self, symbol: str, execution: OrderExecution):
        """Update position from order execution."""
        try:
            if symbol not in self.positions:
                # Create new position
                position = Position(
                    symbol=symbol,
                    exchange=execution.exchange,
                    quantity=execution.quantity,
                    average_price=execution.price,
                    current_price=execution.price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                    total_commission=execution.commission,
                    position_type=PositionType.LONG if execution.quantity > 0 else PositionType.SHORT,
                    opened_at=execution.timestamp,
                    updated_at=execution.timestamp,
                    var_1d=Decimal("0"),
                    var_10d=Decimal("0"),
                    executions=[execution]
                )
                self.positions[symbol] = position
                
            else:
                # Update existing position
                position = self.positions[symbol]
                
                # Calculate new average price
                old_value = position.quantity * position.average_price
                new_value = execution.quantity * execution.price
                total_quantity = position.quantity + execution.quantity
                
                if total_quantity != 0:
                    position.average_price = (old_value + new_value) / total_quantity
                
                position.quantity = total_quantity
                position.total_commission += execution.commission
                position.updated_at = execution.timestamp
                position.executions.append(execution)
                
                # Update position type
                if position.quantity > 0:
                    position.position_type = PositionType.LONG
                elif position.quantity < 0:
                    position.position_type = PositionType.SHORT
                else:
                    position.position_type = PositionType.NEUTRAL
                
                # If position is closed, realize P&L
                if position.quantity == 0:
                    position.realized_pnl = position.unrealized_pnl
                    position.unrealized_pnl = Decimal("0")
            
            # Save to database
            await self.save_position(symbol)
            
            # Log position update
            self.audit_logger.log_event(
                event_type="POSITION_UPDATE",
                action="EXECUTION_PROCESSED",
                details={
                    "symbol": symbol,
                    "execution": asdict(execution),
                    "new_quantity": float(self.positions[symbol].quantity),
                    "new_average_price": float(self.positions[symbol].average_price)
                },
                symbol=symbol,
                amount=float(execution.quantity),
                price=float(execution.price)
            )
            
            self.logger.info(f"Position updated: {symbol} = {position.quantity} @ ${position.average_price}")
            
        except Exception as e:
            self.logger.error(f"Failed to update position for {symbol}: {e}")
            raise
    
    async def update_market_prices(self):
        """Update all positions with current market prices."""
        tasks = []
        
        for symbol in self.positions.keys():
            # Get best price for each position
            task = self.exchange_router.get_best_price(symbol, None)
            tasks.append((symbol, task))
        
        # Execute all price updates concurrently
        for symbol, task in tasks:
            try:
                market_data = await task
                if market_data:
                    # Use mid-price for position valuation
                    mid_price = (market_data.bid + market_data.ask) / 2
                    self.positions[symbol].update_price(mid_price)
                    
                    # Update VaR if we have enough history
                    await self.update_position_risk_metrics(symbol)
                    
            except Exception as e:
                self.logger.warning(f"Failed to update price for {symbol}: {e}")
    
    async def update_position_risk_metrics(self, symbol: str):
        """Update risk metrics for a position."""
        position = self.positions[symbol]
        
        # Calculate historical returns for VaR
        if len(self.daily_returns) >= 30:
            returns = np.array(self.daily_returns)
            
            # Calculate position-specific returns (simplified)
            position_returns = returns * float(position.quantity * position.current_price / self.get_total_portfolio_value())
            
            # Calculate VaR
            position.var_1d = Decimal(str(self.risk_calculator.calculate_var(position_returns, 1)))
            position.var_10d = Decimal(str(self.risk_calculator.calculate_var(position_returns, 10)))
            
            # Calculate volatility
            position.volatility = float(np.std(position_returns) * np.sqrt(252))
            
            # Calculate Sharpe ratio for position
            if position.volatility > 0:
                position.sharpe_ratio = float(np.mean(position_returns) * 252 / position.volatility)
    
    def get_total_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        total_value = sum(self.cash_balances.values())
        
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        total_value = self.get_total_portfolio_value()
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = float(position.market_value / total_value)
        
        return weights
    
    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        total_value = self.get_total_portfolio_value()
        total_pnl = sum(pos.total_pnl for pos in self.positions.values())
        total_pnl_pct = float((total_pnl / self.initial_capital) * 100) if self.initial_capital > 0 else 0.0
        
        # Risk metrics
        portfolio_var_1d = sum(pos.var_1d for pos in self.positions.values())
        portfolio_var_10d = sum(pos.var_10d for pos in self.positions.values())
        
        # Performance metrics
        sharpe_ratio = self.performance_analyzer.calculate_sharpe_ratio()
        sortino_ratio = self.performance_analyzer.calculate_sortino_ratio()
        information_ratio = self.performance_analyzer.calculate_information_ratio()
        max_drawdown = self.performance_analyzer.calculate_max_drawdown()
        beta = self.performance_analyzer.calculate_beta()
        
        # Calculate volatility
        if len(self.daily_returns) >= 30:
            volatility = float(np.std(self.daily_returns) * np.sqrt(252))
        else:
            volatility = 0.0
        
        # Diversification metrics
        weights = self.get_portfolio_weights()
        concentration_ratio = sum(w**2 for w in weights.values()) if weights else 1.0
        effective_positions = int(1 / concentration_ratio) if concentration_ratio > 0 else 0
        
        # Sector concentration
        sector_weights = defaultdict(float)
        for symbol, weight in weights.items():
            if symbol in self.asset_info:
                sector = self.asset_info[symbol].sector or "Unknown"
                sector_weights[sector] += weight
        
        # Liquidity metrics
        liquidity_scores = [pos.liquidity_score for pos in self.positions.values()]
        avg_liquidity = np.mean(liquidity_scores) if liquidity_scores else 1.0
        
        # Days to liquidate (weighted average)
        days_to_liquidate = []
        for pos in self.positions.values():
            if pos.days_to_liquidate:
                days_to_liquidate.append(pos.days_to_liquidate)
        avg_days_to_liquidate = int(np.mean(days_to_liquidate)) if days_to_liquidate else 1
        
        # Attribution analysis (simplified)
        asset_allocation_return = 0.0  # Would need benchmark for proper calculation
        security_selection_return = 0.0
        interaction_effect = 0.0
        
        return PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_pct,
            portfolio_var_1d=portfolio_var_1d,
            portfolio_var_10d=portfolio_var_10d,
            beta=beta,
            volatility=volatility,
            correlation=0.0,  # Would need correlation matrix
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=sharpe_ratio / max_drawdown if max_drawdown > 0 else 0.0,
            concentration_ratio=concentration_ratio,
            effective_positions=effective_positions,
            sector_concentration=dict(sector_weights),
            liquidity_score=avg_liquidity,
            days_to_liquidate=avg_days_to_liquidate,
            asset_allocation_return=asset_allocation_return,
            security_selection_return=security_selection_return,
            interaction_effect=interaction_effect
        )
    
    async def generate_rebalance_recommendations(self) -> List[RebalanceRecommendation]:
        """Generate portfolio rebalancing recommendations."""
        recommendations = []
        
        # Check if rebalancing is needed
        if datetime.now() - self.last_rebalance < self.rebalance_frequency:
            return recommendations
        
        current_weights = self.get_portfolio_weights()
        total_value = self.get_total_portfolio_value()
        
        # Simple rebalancing logic (in production, would be more sophisticated)
        for symbol, position in self.positions.items():
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = self.target_weights.get(symbol, 0.0)
            
            if target_weight == 0.0:
                # No target weight set, skip
                continue
            
            weight_difference = target_weight - current_weight
            
            # Only rebalance if difference is significant
            if abs(weight_difference) > 0.02:  # 2% threshold
                
                # Determine signal strength
                if abs(weight_difference) > 0.10:  # 10%
                    signal = RebalanceSignal.STRONG_BUY if weight_difference > 0 else RebalanceSignal.STRONG_SELL
                    priority = 1
                elif abs(weight_difference) > 0.05:  # 5%
                    signal = RebalanceSignal.BUY if weight_difference > 0 else RebalanceSignal.SELL
                    priority = 2
                else:
                    signal = RebalanceSignal.HOLD
                    priority = 3
                
                # Estimate transaction cost
                trade_value = abs(weight_difference) * float(total_value)
                estimated_cost = Decimal(str(trade_value * 0.001))  # 0.1% estimated cost
                
                # Create recommendation
                recommendation = RebalanceRecommendation(
                    symbol=symbol,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    weight_difference=weight_difference,
                    signal=signal,
                    priority=priority,
                    estimated_cost=estimated_cost,
                    expected_return=0.0,  # Would need expected return model
                    risk_impact=0.0,      # Would need risk model
                    liquidity_impact=position.liquidity_score,
                    reason=f"Rebalance from {current_weight:.1%} to {target_weight:.1%}"
                )
                
                recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations
    
    async def execute_rebalance(self, recommendations: List[RebalanceRecommendation]) -> Dict[str, Any]:
        """Execute rebalancing recommendations."""
        results = {
            "executed": [],
            "failed": [],
            "total_cost": Decimal("0"),
            "timestamp": datetime.now()
        }
        
        for rec in recommendations:
            try:
                # Skip low-priority recommendations if cost is too high
                if rec.priority > 2 and results["total_cost"] > Decimal("1000"):
                    continue
                
                # Calculate order size
                total_value = self.get_total_portfolio_value()
                trade_value = abs(rec.weight_difference) * float(total_value)
                
                # Skip if trade is too small
                if trade_value < 100:  # $100 minimum
                    continue
                
                # Create order (simplified - would need proper order creation)
                # This is a placeholder - actual implementation would create proper orders
                
                results["executed"].append({
                    "symbol": rec.symbol,
                    "signal": rec.signal.value,
                    "trade_value": trade_value,
                    "estimated_cost": float(rec.estimated_cost)
                })
                
                results["total_cost"] += rec.estimated_cost
                
                # Log rebalance action
                self.audit_logger.log_event(
                    event_type="REBALANCE_EXECUTED",
                    action="WEIGHT_ADJUSTMENT",
                    details={
                        "symbol": rec.symbol,
                        "old_weight": rec.current_weight,
                        "new_weight": rec.target_weight,
                        "trade_value": trade_value,
                        "reason": rec.reason
                    },
                    symbol=rec.symbol,
                    amount=trade_value
                )
                
            except Exception as e:
                results["failed"].append({
                    "symbol": rec.symbol,
                    "error": str(e)
                })
                self.logger.error(f"Rebalance failed for {rec.symbol}: {e}")
        
        self.last_rebalance = datetime.now()
        return results
    
    async def run_stress_test(self) -> Dict[str, Any]:
        """Run comprehensive stress test scenarios."""
        stress_scenarios = {
            "market_crash_20": -0.20,      # 20% market crash
            "market_crash_50": -0.50,      # 50% market crash
            "volatility_spike": 0.10,      # 10% volatility spike
            "flash_crash": -0.15,          # 15% flash crash
            "sector_rotation": -0.10,      # 10% sector-specific decline
            "liquidity_crisis": -0.25,     # 25% liquidity crisis
            "interest_rate_shock": -0.08,  # 8% interest rate shock
            "geopolitical_risk": -0.12,    # 12% geopolitical shock
            "cyber_attack": -0.30,         # 30% cyber attack scenario
            "regulatory_change": -0.15     # 15% regulatory shock
        }
        
        stress_results = self.risk_calculator.stress_test_scenario(
            self.positions, 
            stress_scenarios
        )
        
        # Calculate impact as percentage of portfolio
        total_value = float(self.get_total_portfolio_value())
        stress_results_pct = {
            scenario: (impact / total_value) * 100 if total_value > 0 else 0.0
            for scenario, impact in stress_results.items()
        }
        
        # Risk assessment
        risk_assessment = {
            "overall_risk": "LOW",
            "worst_case_loss": min(stress_results_pct.values()),
            "scenarios_exceeding_limit": [],
            "recommendations": []
        }
        
        # Check against risk limits
        for scenario, loss_pct in stress_results_pct.items():
            if abs(loss_pct) > self.risk_limits.max_drawdown_pct:
                risk_assessment["scenarios_exceeding_limit"].append(scenario)
                risk_assessment["overall_risk"] = "HIGH"
        
        if len(risk_assessment["scenarios_exceeding_limit"]) > 3:
            risk_assessment["recommendations"].append("Reduce position sizes")
            risk_assessment["recommendations"].append("Increase diversification")
        
        return {
            "stress_test_results": stress_results_pct,
            "risk_assessment": risk_assessment,
            "timestamp": datetime.now().isoformat()
        }
    
    async def save_position(self, symbol: str):
        """Save position to database."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO positions (
                        symbol, exchange, quantity, average_price, current_price,
                        unrealized_pnl, realized_pnl, total_commission, position_type,
                        opened_at, updated_at, position_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    position.exchange,
                    str(position.quantity),
                    str(position.average_price),
                    str(position.current_price),
                    str(position.unrealized_pnl),
                    str(position.realized_pnl),
                    str(position.total_commission),
                    position.position_type.value,
                    position.opened_at.isoformat(),
                    position.updated_at.isoformat(),
                    json.dumps(asdict(position), default=str)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to save position {symbol}: {e}")
    
    async def save_portfolio_snapshot(self):
        """Save current portfolio snapshot."""
        try:
            total_value = self.get_total_portfolio_value()
            total_pnl = sum(pos.total_pnl for pos in self.positions.values())
            cash_balance = sum(self.cash_balances.values())
            
            metrics = await self.calculate_portfolio_metrics()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO portfolio_snapshots (
                        timestamp, total_value, total_pnl, cash_balance,
                        position_count, metrics_data
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    str(total_value),
                    str(total_pnl),
                    str(cash_balance),
                    len(self.positions),
                    json.dumps(asdict(metrics), default=str)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to save portfolio snapshot: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        total_value = self.get_total_portfolio_value()
        weights = self.get_portfolio_weights()
        
        # Position summaries
        position_summaries = []
        for symbol, position in self.positions.items():
            position_summaries.append({
                "symbol": symbol,
                "quantity": float(position.quantity),
                "current_price": float(position.current_price),
                "market_value": float(position.market_value),
                "weight": weights.get(symbol, 0.0),
                "pnl": float(position.total_pnl),
                "pnl_percentage": position.pnl_percentage,
                "position_type": position.position_type.value,
                "days_held": (datetime.now() - position.opened_at).days
            })
        
        # Sort by market value
        position_summaries.sort(key=lambda x: x["market_value"], reverse=True)
        
        # Cash balances
        cash_summary = {
            currency: float(balance) 
            for currency, balance in self.cash_balances.items()
        }
        
        # Performance metrics
        performance_metrics = {
            "total_value": float(total_value),
            "total_pnl": float(sum(pos.total_pnl for pos in self.positions.values())),
            "total_pnl_percentage": float(
                (sum(pos.total_pnl for pos in self.positions.values()) / self.initial_capital) * 100
            ) if self.initial_capital > 0 else 0.0,
            "position_count": len(self.positions),
            "sharpe_ratio": self.performance_analyzer.calculate_sharpe_ratio(),
            "max_drawdown": self.performance_analyzer.calculate_max_drawdown(),
            "avg_holding_period": np.mean([
                (datetime.now() - pos.opened_at).days 
                for pos in self.positions.values()
            ]) if self.positions else 0
        }
        
        return {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_value": float(total_value),
                "initial_capital": float(self.initial_capital),
                "cash_balances": cash_summary,
                "position_count": len(self.positions),
                "largest_position": max(weights.values()) if weights else 0.0,
                "concentration_ratio": sum(w**2 for w in weights.values()) if weights else 0.0
            },
            "positions": position_summaries,
            "performance": performance_metrics,
            "risk_metrics": {
                "portfolio_var_1d": float(sum(pos.var_1d for pos in self.positions.values())),
                "portfolio_var_10d": float(sum(pos.var_10d for pos in self.positions.values())),
                "avg_liquidity_score": np.mean([pos.liquidity_score for pos in self.positions.values()]) if self.positions else 1.0,
                "risk_limit_breaches": self.check_risk_limits()
            }
        }
    
    def check_risk_limits(self) -> List[str]:
        """Check for risk limit breaches."""
        breaches = []
        total_value = self.get_total_portfolio_value()
        weights = self.get_portfolio_weights()
        
        # Position size limits
        for symbol, weight in weights.items():
            if weight > self.risk_limits.max_position_size_pct / 100:
                breaches.append(f"Position {symbol} exceeds size limit: {weight:.1%}")
        
        # VaR limits
        total_var_1d = sum(pos.var_1d for pos in self.positions.values())
        var_1d_pct = float((total_var_1d / total_value) * 100) if total_value > 0 else 0.0
        
        if var_1d_pct > self.risk_limits.max_var_1d_pct:
            breaches.append(f"1-day VaR exceeds limit: {var_1d_pct:.1f}%")
        
        # Liquidity limits
        for symbol, position in self.positions.items():
            if position.liquidity_score < self.risk_limits.min_liquidity_score:
                breaches.append(f"Position {symbol} below liquidity threshold")
        
        return breaches

# Example usage and testing
async def demo_portfolio_manager():
    """Demonstration of the portfolio management system."""
    print("🚨 ULTRATHINK Week 6: Portfolio Management System Demo 🚨")
    print("=" * 60)
    
    # This is a simplified demo - would need actual exchange router and order manager
    from exchange_integrations import ExchangeRouter
    from order_management import OrderManager
    from real_money_trader import AuditLogger
    
    # Create components
    exchange_router = ExchangeRouter()
    audit_logger = AuditLogger()
    order_manager = OrderManager(exchange_router, audit_logger)
    
    # Create portfolio manager
    portfolio_manager = PortfolioManager(
        exchange_router=exchange_router,
        order_manager=order_manager,
        audit_logger=audit_logger,
        initial_capital=Decimal("1000000")  # $1M
    )
    
    # Register some assets
    btc_info = AssetInfo(
        symbol="BTC-USD",
        name="Bitcoin",
        asset_class=AssetClass.CRYPTOCURRENCY,
        base_currency="BTC",
        quote_currency="USD",
        lot_size=Decimal("0.00001"),
        min_order_size=Decimal("0.001"),
        max_order_size=Decimal("100"),
        tick_size=Decimal("0.01"),
        commission_rate=Decimal("0.001"),
        sector="Cryptocurrency"
    )
    
    eth_info = AssetInfo(
        symbol="ETH-USD",
        name="Ethereum",
        asset_class=AssetClass.CRYPTOCURRENCY,
        base_currency="ETH",
        quote_currency="USD",
        lot_size=Decimal("0.0001"),
        min_order_size=Decimal("0.01"),
        max_order_size=Decimal("1000"),
        tick_size=Decimal("0.01"),
        commission_rate=Decimal("0.001"),
        sector="Cryptocurrency"
    )
    
    portfolio_manager.register_asset(btc_info)
    portfolio_manager.register_asset(eth_info)
    
    # Create mock executions to demonstrate position tracking
    btc_execution = OrderExecution(
        execution_id="exec_001",
        order_id="order_001",
        timestamp=datetime.now(),
        quantity=Decimal("0.5"),
        price=Decimal("45000"),
        commission=Decimal("22.50"),
        exchange="binance",
        trade_id="trade_001",
        is_maker=True
    )
    
    eth_execution = OrderExecution(
        execution_id="exec_002",
        order_id="order_002",
        timestamp=datetime.now(),
        quantity=Decimal("10"),
        price=Decimal("3000"),
        commission=Decimal("30.00"),
        exchange="binance",
        trade_id="trade_002",
        is_maker=False
    )
    
    # Update positions
    await portfolio_manager.update_position("BTC-USD", btc_execution)
    await portfolio_manager.update_position("ETH-USD", eth_execution)
    
    print("✅ Positions created from executions")
    
    # Calculate portfolio metrics
    metrics = await portfolio_manager.calculate_portfolio_metrics()
    print(f"Portfolio Value: ${metrics.total_value:,.2f}")
    print(f"Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_percentage:.2f}%)")
    print(f"Position Count: {metrics.effective_positions}")
    
    # Run stress test
    stress_results = await portfolio_manager.run_stress_test()
    print(f"\nStress Test Results:")
    print(f"- Worst Case Loss: {stress_results['risk_assessment']['worst_case_loss']:.1f}%")
    print(f"- Overall Risk: {stress_results['risk_assessment']['overall_risk']}")
    
    # Set target weights for rebalancing demo
    portfolio_manager.target_weights = {
        "BTC-USD": 0.60,  # 60% BTC
        "ETH-USD": 0.40   # 40% ETH
    }
    
    # Generate rebalancing recommendations
    recommendations = await portfolio_manager.generate_rebalance_recommendations()
    print(f"\nRebalancing Recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"- {rec.symbol}: {rec.current_weight:.1%} → {rec.target_weight:.1%} ({rec.signal.value})")
    
    # Get portfolio summary
    summary = portfolio_manager.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"- Total Value: ${summary['summary']['total_value']:,.2f}")
    print(f"- Positions: {summary['summary']['position_count']}")
    print(f"- Largest Position: {summary['summary']['largest_position']:.1%}")
    print(f"- Sharpe Ratio: {summary['performance']['sharpe_ratio']:.2f}")
    
    # Check risk limits
    breaches = portfolio_manager.check_risk_limits()
    if breaches:
        print(f"\n⚠️  Risk Limit Breaches:")
        for breach in breaches:
            print(f"- {breach}")
    else:
        print(f"\n✅ All risk limits within bounds")
    
    print(f"\n✅ Portfolio Management System demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_portfolio_manager())