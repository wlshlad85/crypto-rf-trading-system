"""UltraThink Decision Framework: Real-time decision making with advanced reasoning chains."""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from .reasoning_engine import UltraThinkReasoningEngine, ReasoningChain, ReasoningLevel
from .market_analyzer import UltraThinkMarketAnalyzer, MarketState
from .strategy_selector import UltraThinkStrategySelector, StrategyRecommendation


class DecisionType(Enum):
    """Types of trading decisions."""
    ENTRY = "entry"
    EXIT = "exit"
    POSITION_SIZE = "position_size"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    EMERGENCY_STOP = "emergency_stop"


class DecisionPriority(Enum):
    """Priority levels for decisions."""
    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class DecisionStatus(Enum):
    """Status of a decision."""
    PENDING = "pending"
    PROCESSING = "processing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TradingDecision:
    """Represents a trading decision with full reasoning."""
    decision_id: str
    decision_type: DecisionType
    priority: DecisionPriority
    symbol: str
    action: str  # 'buy', 'sell', 'hold', 'reduce', 'increase'
    quantity: float
    price: Optional[float]
    reasoning_chain: ReasoningChain
    confidence: float
    risk_score: float
    expected_outcome: Dict[str, Any]
    implementation_plan: List[str]
    risk_mitigation: List[str]
    created_at: datetime
    status: DecisionStatus = DecisionStatus.PENDING
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    post_decision_analysis: Optional[Dict[str, Any]] = None


@dataclass
class MarketContext:
    """Current market context for decision making."""
    timestamp: datetime
    market_state: MarketState
    active_positions: Dict[str, Any]
    portfolio_value: float
    available_capital: float
    current_risk: float
    recent_decisions: List[TradingDecision]
    market_regime: str
    volatility_level: str


@dataclass
class DecisionMetrics:
    """Metrics for decision quality tracking."""
    total_decisions: int = 0
    successful_decisions: int = 0
    failed_decisions: int = 0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    risk_adjusted_return: float = 0.0
    decision_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


class UltraThinkDecisionFramework:
    """Advanced real-time decision framework with reasoning chains."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.reasoning_engine = UltraThinkReasoningEngine()
        self.market_analyzer = UltraThinkMarketAnalyzer()
        self.strategy_selector = UltraThinkStrategySelector()
        
        # Decision management
        self.pending_decisions = {}
        self.decision_history = []
        self.decision_queue = asyncio.Queue()
        self.metrics = DecisionMetrics()
        
        # Real-time processing
        self.is_running = False
        self.decision_processor = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Risk management
        self.emergency_stops = {}
        self.risk_limits = self.config.get('risk_limits', {})
        
        # Decision callbacks
        self.execution_callbacks = []
        self.monitoring_callbacks = []
        
        # Performance tracking
        self.decision_performance = {}
        self.learning_data = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'decision_timeout': 30.0,  # seconds
            'max_concurrent_decisions': 5,
            'enable_real_time_processing': True,
            'confidence_threshold': 0.6,
            'risk_threshold': 0.8,
            'emergency_stop_enabled': True,
            'decision_validation': True,
            'parallel_reasoning': True,
            'learning_enabled': True,
            'risk_limits': {
                'max_position_size': 0.2,
                'max_portfolio_risk': 0.15,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.20
            },
            'decision_cooling_period': 300,  # seconds between similar decisions
            'auto_execution': False,  # Manual approval required by default
            'backup_decision_maker': True
        }
    
    async def start_real_time_processing(self):
        """Start real-time decision processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.decision_processor = asyncio.create_task(self._process_decisions_loop())
        self.logger.info("UltraThink Decision Framework started")
    
    async def stop_real_time_processing(self):
        """Stop real-time decision processing."""
        self.is_running = False
        if self.decision_processor:
            self.decision_processor.cancel()
            try:
                await self.decision_processor
            except asyncio.CancelledError:
                pass
        self.logger.info("UltraThink Decision Framework stopped")
    
    async def _process_decisions_loop(self):
        """Main loop for processing decisions."""
        while self.is_running:
            try:
                # Get decision from queue with timeout
                decision = await asyncio.wait_for(
                    self.decision_queue.get(),
                    timeout=1.0
                )
                
                # Process decision
                await self._process_decision(decision)
                
            except asyncio.TimeoutError:
                # No decisions to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in decision processing loop: {e}")
                await asyncio.sleep(1)
    
    async def make_decision(self, market_data: pd.DataFrame, symbol: str,
                          decision_type: DecisionType, context: MarketContext) -> TradingDecision:
        """Make a trading decision using ultrathink reasoning."""
        
        decision_id = f"{decision_type.value}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"Making decision: {decision_id}")
        
        start_time = time.time()
        
        try:
            # Analyze current market state
            market_state = await self._analyze_market_state(market_data, symbol, context)
            
            # Generate reasoning chain for decision
            reasoning_chain = await self._generate_decision_reasoning(
                market_data, symbol, decision_type, context, market_state
            )
            
            # Select optimal strategy
            strategy_recommendation = await self._select_strategy(
                market_data, symbol, context, market_state
            )
            
            # Generate specific decision
            decision_details = await self._generate_decision_details(
                decision_type, symbol, market_state, strategy_recommendation, context
            )
            
            # Create decision object
            decision = TradingDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                priority=self._determine_priority(decision_type, market_state, context),
                symbol=symbol,
                action=decision_details['action'],
                quantity=decision_details['quantity'],
                price=decision_details.get('price'),
                reasoning_chain=reasoning_chain,
                confidence=decision_details['confidence'],
                risk_score=decision_details['risk_score'],
                expected_outcome=decision_details['expected_outcome'],
                implementation_plan=decision_details['implementation_plan'],
                risk_mitigation=decision_details['risk_mitigation'],
                created_at=datetime.now()
            )
            
            # Validate decision
            if self.config['decision_validation']:
                validation_result = await self._validate_decision(decision, context)
                if not validation_result['valid']:
                    decision.status = DecisionStatus.FAILED
                    decision.execution_result = {'error': validation_result['reason']}
                    return decision
            
            # Add to processing queue
            await self.decision_queue.put(decision)
            self.pending_decisions[decision_id] = decision
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_decision_metrics(decision, processing_time)
            
            self.logger.info(f"Decision created: {decision_id} - {decision.action} "
                           f"{decision.quantity} {symbol} (confidence: {decision.confidence:.2%})")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making decision {decision_id}: {e}")
            # Create failed decision
            failed_decision = TradingDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                priority=DecisionPriority.MEDIUM,
                symbol=symbol,
                action='hold',
                quantity=0.0,
                price=None,
                reasoning_chain=ReasoningChain(
                    chain_id=f"failed_{decision_id}",
                    nodes=[],
                    final_conclusion=f"Decision failed: {str(e)}",
                    overall_confidence=0.0,
                    created_at=datetime.now(),
                    market_context={}
                ),
                confidence=0.0,
                risk_score=1.0,
                expected_outcome={'error': str(e)},
                implementation_plan=[],
                risk_mitigation=[],
                created_at=datetime.now(),
                status=DecisionStatus.FAILED
            )
            return failed_decision
    
    async def _analyze_market_state(self, data: pd.DataFrame, symbol: str,
                                  context: MarketContext) -> MarketState:
        """Analyze current market state."""
        market_context = {
            'portfolio_value': context.portfolio_value,
            'available_capital': context.available_capital,
            'current_risk': context.current_risk,
            'active_positions': len(context.active_positions),
            'recent_decisions': len(context.recent_decisions)
        }
        
        return self.market_analyzer.analyze_market(data, symbol, market_context)
    
    async def _generate_decision_reasoning(self, data: pd.DataFrame, symbol: str,
                                         decision_type: DecisionType, context: MarketContext,
                                         market_state: MarketState) -> ReasoningChain:
        """Generate reasoning chain for the decision."""
        
        reasoning_context = {
            'decision_type': decision_type.value,
            'symbol': symbol,
            'market_sentiment': market_state.overall_sentiment,
            'risk_level': market_state.risk_level,
            'portfolio_context': {
                'value': context.portfolio_value,
                'risk': context.current_risk,
                'positions': len(context.active_positions)
            },
            'decision_making': True
        }
        
        return self.reasoning_engine.reason_about_market(data, reasoning_context)
    
    async def _select_strategy(self, data: pd.DataFrame, symbol: str,
                             context: MarketContext, market_state: MarketState) -> StrategyRecommendation:
        """Select optimal strategy for the decision."""
        
        strategy_context = {
            'portfolio_value': context.portfolio_value,
            'available_capital': context.available_capital,
            'risk_tolerance': 1.0 - context.current_risk,
            'market_regime': context.market_regime
        }
        
        return self.strategy_selector.select_optimal_strategy(data, symbol, strategy_context)
    
    async def _generate_decision_details(self, decision_type: DecisionType, symbol: str,
                                       market_state: MarketState, strategy: StrategyRecommendation,
                                       context: MarketContext) -> Dict[str, Any]:
        """Generate specific decision details."""
        
        if decision_type == DecisionType.ENTRY:
            return await self._generate_entry_decision(symbol, market_state, strategy, context)
        elif decision_type == DecisionType.EXIT:
            return await self._generate_exit_decision(symbol, market_state, strategy, context)
        elif decision_type == DecisionType.POSITION_SIZE:
            return await self._generate_position_size_decision(symbol, market_state, strategy, context)
        elif decision_type == DecisionType.RISK_MANAGEMENT:
            return await self._generate_risk_management_decision(symbol, market_state, strategy, context)
        elif decision_type == DecisionType.PORTFOLIO_REBALANCE:
            return await self._generate_rebalance_decision(symbol, market_state, strategy, context)
        else:
            return {
                'action': 'hold',
                'quantity': 0.0,
                'confidence': 0.5,
                'risk_score': 0.5,
                'expected_outcome': {},
                'implementation_plan': [],
                'risk_mitigation': []
            }
    
    async def _generate_entry_decision(self, symbol: str, market_state: MarketState,
                                     strategy: StrategyRecommendation, context: MarketContext) -> Dict[str, Any]:
        """Generate entry decision details."""
        
        # Determine action based on market sentiment and strategy
        if market_state.overall_sentiment == 'bullish' and strategy.confidence > 0.7:
            action = 'buy'
        elif market_state.overall_sentiment == 'bearish' and strategy.confidence > 0.7:
            action = 'sell'
        else:
            action = 'hold'
        
        # Calculate position size
        max_position = context.available_capital * strategy.strategy_config.max_position_size
        risk_adjusted_size = max_position * (1 - market_state.risk_level_numeric())
        
        # Adjust for confidence
        confidence_adjusted_size = risk_adjusted_size * strategy.confidence
        
        quantity = confidence_adjusted_size / market_state.price if action != 'hold' else 0.0
        
        # Calculate expected outcome
        expected_return = strategy.expected_return
        expected_risk = strategy.risk_score
        
        expected_outcome = {
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'probability_of_profit': strategy.confidence,
            'max_loss': quantity * market_state.price * strategy.strategy_config.stop_loss,
            'max_gain': quantity * market_state.price * strategy.strategy_config.take_profit
        }
        
        # Implementation plan
        implementation_plan = [
            f"Enter {action} position of {quantity:.4f} {symbol}",
            f"Set stop loss at {strategy.strategy_config.stop_loss:.1%}",
            f"Set take profit at {strategy.strategy_config.take_profit:.1%}",
            "Monitor position for 1 hour after entry"
        ]
        
        # Risk mitigation
        risk_mitigation = [
            f"Position size limited to {strategy.strategy_config.max_position_size:.1%} of capital",
            "Automatic stop loss will limit downside",
            "Position will be reviewed every 4 hours",
            "Emergency stop available if market conditions deteriorate"
        ]
        
        return {
            'action': action,
            'quantity': quantity,
            'price': market_state.price,
            'confidence': strategy.confidence,
            'risk_score': expected_risk,
            'expected_outcome': expected_outcome,
            'implementation_plan': implementation_plan,
            'risk_mitigation': risk_mitigation
        }
    
    async def _generate_exit_decision(self, symbol: str, market_state: MarketState,
                                    strategy: StrategyRecommendation, context: MarketContext) -> Dict[str, Any]:
        """Generate exit decision details."""
        
        current_position = context.active_positions.get(symbol, {})
        position_size = current_position.get('quantity', 0)
        
        if position_size == 0:
            return {
                'action': 'hold',
                'quantity': 0.0,
                'confidence': 1.0,
                'risk_score': 0.0,
                'expected_outcome': {'reason': 'No position to exit'},
                'implementation_plan': [],
                'risk_mitigation': []
            }
        
        # Determine exit action
        position_pnl = current_position.get('unrealized_pnl', 0)
        position_age = datetime.now() - current_position.get('entry_time', datetime.now())
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Take profit
        if position_pnl > abs(position_size) * market_state.price * strategy.strategy_config.take_profit:
            should_exit = True
            exit_reason = "Take profit target reached"
        
        # Stop loss
        elif position_pnl < -abs(position_size) * market_state.price * strategy.strategy_config.stop_loss:
            should_exit = True
            exit_reason = "Stop loss triggered"
        
        # Strategy change
        elif strategy.market_fit_score < 0.3:
            should_exit = True
            exit_reason = "Strategy no longer fits market conditions"
        
        # Risk management
        elif market_state.risk_level == 'very_high':
            should_exit = True
            exit_reason = "Market risk too high"
        
        if should_exit:
            action = 'sell' if position_size > 0 else 'buy'  # Close position
            quantity = abs(position_size)
            confidence = 0.8
        else:
            action = 'hold'
            quantity = 0.0
            confidence = 0.6
        
        expected_outcome = {
            'exit_reason': exit_reason,
            'expected_pnl': position_pnl,
            'position_duration': position_age.total_seconds() / 3600  # hours
        }
        
        implementation_plan = [
            f"Exit {abs(position_size):.4f} {symbol} position",
            "Execute market order for immediate fill",
            "Update portfolio risk calculations",
            "Record trade performance metrics"
        ] if should_exit else ["Monitor position for exit signals"]
        
        risk_mitigation = [
            "Market order ensures immediate execution",
            "Position size known exactly",
            "No additional market exposure after exit"
        ] if should_exit else ["Continued monitoring of stop loss levels"]
        
        return {
            'action': action,
            'quantity': quantity,
            'confidence': confidence,
            'risk_score': 0.2 if should_exit else 0.6,
            'expected_outcome': expected_outcome,
            'implementation_plan': implementation_plan,
            'risk_mitigation': risk_mitigation
        }
    
    async def _generate_position_size_decision(self, symbol: str, market_state: MarketState,
                                             strategy: StrategyRecommendation, context: MarketContext) -> Dict[str, Any]:
        """Generate position sizing decision."""
        
        current_position = context.active_positions.get(symbol, {})
        current_size = current_position.get('quantity', 0)
        
        # Calculate optimal position size
        optimal_size = self._calculate_optimal_position_size(
            market_state, strategy, context
        )
        
        size_diff = optimal_size - abs(current_size)
        
        if abs(size_diff) < 0.01:  # Minimal change
            action = 'hold'
            quantity = 0.0
        elif size_diff > 0:
            action = 'increase'
            quantity = size_diff
        else:
            action = 'reduce'
            quantity = abs(size_diff)
        
        confidence = strategy.confidence * 0.8  # Slightly lower for sizing decisions
        
        expected_outcome = {
            'optimal_size': optimal_size,
            'current_size': current_size,
            'adjustment': size_diff,
            'new_portfolio_weight': optimal_size * market_state.price / context.portfolio_value
        }
        
        return {
            'action': action,
            'quantity': quantity,
            'confidence': confidence,
            'risk_score': strategy.risk_score,
            'expected_outcome': expected_outcome,
            'implementation_plan': [f"Adjust position size by {quantity:.4f} {symbol}"],
            'risk_mitigation': ["Gradual position adjustment to minimize market impact"]
        }
    
    async def _generate_risk_management_decision(self, symbol: str, market_state: MarketState,
                                               strategy: StrategyRecommendation, context: MarketContext) -> Dict[str, Any]:
        """Generate risk management decision."""
        
        # Assess current risk levels
        portfolio_risk = context.current_risk
        position_risk = self._calculate_position_risk(symbol, context, market_state)
        
        # Determine risk action
        if portfolio_risk > self.risk_limits['max_portfolio_risk']:
            action = 'reduce_risk'
            quantity = 0.3  # Reduce by 30%
            reason = "Portfolio risk exceeds limits"
        elif position_risk > 0.1:  # 10% position risk
            action = 'hedge_position'
            quantity = 0.5  # Partial hedge
            reason = "Individual position risk too high"
        elif market_state.risk_level == 'very_high':
            action = 'defensive'
            quantity = 0.2  # Reduce overall exposure
            reason = "Market risk elevated"
        else:
            action = 'monitor'
            quantity = 0.0
            reason = "Risk levels acceptable"
        
        expected_outcome = {
            'risk_reduction': f"Expected {quantity*100:.0f}% risk reduction",
            'reason': reason,
            'new_portfolio_risk': portfolio_risk * (1 - quantity)
        }
        
        return {
            'action': action,
            'quantity': quantity,
            'confidence': 0.9,  # High confidence in risk management
            'risk_score': 0.1,  # Low risk for risk management actions
            'expected_outcome': expected_outcome,
            'implementation_plan': [f"Implement {action} for {symbol}"],
            'risk_mitigation': ["Risk management action reduces overall portfolio risk"]
        }
    
    async def _generate_rebalance_decision(self, symbol: str, market_state: MarketState,
                                         strategy: StrategyRecommendation, context: MarketContext) -> Dict[str, Any]:
        """Generate portfolio rebalancing decision."""
        
        # Calculate target allocation
        target_allocation = self._calculate_target_allocation(symbol, market_state, strategy, context)
        current_allocation = self._calculate_current_allocation(symbol, context)
        
        allocation_diff = target_allocation - current_allocation
        
        if abs(allocation_diff) < 0.02:  # 2% threshold
            action = 'hold'
            quantity = 0.0
        elif allocation_diff > 0:
            action = 'increase_allocation'
            quantity = allocation_diff * context.portfolio_value / market_state.price
        else:
            action = 'decrease_allocation'
            quantity = abs(allocation_diff) * context.portfolio_value / market_state.price
        
        expected_outcome = {
            'target_allocation': target_allocation,
            'current_allocation': current_allocation,
            'rebalance_amount': allocation_diff,
            'portfolio_improvement': f"Better risk-return profile expected"
        }
        
        return {
            'action': action,
            'quantity': quantity,
            'confidence': 0.7,
            'risk_score': 0.3,
            'expected_outcome': expected_outcome,
            'implementation_plan': [f"Rebalance {symbol} allocation by {allocation_diff:.1%}"],
            'risk_mitigation': ["Rebalancing maintains target risk profile"]
        }
    
    def _calculate_optimal_position_size(self, market_state: MarketState,
                                       strategy: StrategyRecommendation,
                                       context: MarketContext) -> float:
        """Calculate optimal position size."""
        
        # Kelly criterion approach
        win_prob = strategy.confidence
        avg_win = strategy.strategy_config.take_profit
        avg_loss = strategy.strategy_config.stop_loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, strategy.strategy_config.max_position_size))
        
        # Adjust for market conditions
        market_adjustment = 1.0
        if market_state.risk_level == 'very_high':
            market_adjustment = 0.5
        elif market_state.risk_level == 'high':
            market_adjustment = 0.7
        elif market_state.risk_level == 'low':
            market_adjustment = 1.2
        
        optimal_fraction = kelly_fraction * market_adjustment
        optimal_size = optimal_fraction * context.available_capital / market_state.price
        
        return optimal_size
    
    def _calculate_position_risk(self, symbol: str, context: MarketContext,
                               market_state: MarketState) -> float:
        """Calculate risk for a specific position."""
        
        position = context.active_positions.get(symbol, {})
        if not position:
            return 0.0
        
        position_value = abs(position.get('quantity', 0)) * market_state.price
        position_risk = position_value / context.portfolio_value
        
        # Adjust for volatility
        volatility_multiplier = 1.0
        if 'volatility' in market_state.dimensions:
            vol_regime = market_state.dimensions['volatility'].analysis_result.get('regime', 'normal')
            if vol_regime == 'high':
                volatility_multiplier = 1.5
            elif vol_regime == 'extreme':
                volatility_multiplier = 2.0
        
        return position_risk * volatility_multiplier
    
    def _calculate_target_allocation(self, symbol: str, market_state: MarketState,
                                   strategy: StrategyRecommendation, context: MarketContext) -> float:
        """Calculate target allocation for symbol."""
        
        base_allocation = strategy.strategy_config.max_position_size
        
        # Adjust for market opportunity
        opportunity_adjustment = market_state.opportunity_score
        
        # Adjust for strategy confidence
        confidence_adjustment = strategy.confidence
        
        target = base_allocation * opportunity_adjustment * confidence_adjustment
        return min(target, 0.3)  # Cap at 30%
    
    def _calculate_current_allocation(self, symbol: str, context: MarketContext) -> float:
        """Calculate current allocation for symbol."""
        
        position = context.active_positions.get(symbol, {})
        if not position:
            return 0.0
        
        position_value = abs(position.get('quantity', 0)) * position.get('current_price', 0)
        return position_value / context.portfolio_value
    
    def _determine_priority(self, decision_type: DecisionType, market_state: MarketState,
                          context: MarketContext) -> DecisionPriority:
        """Determine decision priority."""
        
        if decision_type == DecisionType.EMERGENCY_STOP:
            return DecisionPriority.EMERGENCY
        
        if decision_type == DecisionType.RISK_MANAGEMENT:
            return DecisionPriority.HIGH
        
        if market_state.risk_level == 'very_high':
            return DecisionPriority.HIGH
        
        if decision_type in [DecisionType.ENTRY, DecisionType.EXIT]:
            return DecisionPriority.MEDIUM
        
        return DecisionPriority.LOW
    
    async def _validate_decision(self, decision: TradingDecision,
                               context: MarketContext) -> Dict[str, Any]:
        """Validate a decision before execution."""
        
        validation_checks = []
        
        # Risk limit checks
        if decision.risk_score > self.config['risk_threshold']:
            return {'valid': False, 'reason': 'Risk score too high'}
        
        # Confidence threshold
        if decision.confidence < self.config['confidence_threshold']:
            return {'valid': False, 'reason': 'Confidence too low'}
        
        # Portfolio risk check
        if context.current_risk > self.risk_limits['max_portfolio_risk']:
            if decision.action in ['buy', 'increase']:
                return {'valid': False, 'reason': 'Portfolio risk limit exceeded'}
        
        # Position size check
        position_value = decision.quantity * (decision.price or 0)
        position_fraction = position_value / context.portfolio_value
        
        if position_fraction > self.risk_limits['max_position_size']:
            return {'valid': False, 'reason': 'Position size limit exceeded'}
        
        # Cooling period check
        recent_similar = [
            d for d in context.recent_decisions
            if d.symbol == decision.symbol and d.decision_type == decision.decision_type
            and (datetime.now() - d.created_at).total_seconds() < self.config['decision_cooling_period']
        ]
        
        if recent_similar:
            return {'valid': False, 'reason': 'Decision cooling period active'}
        
        return {'valid': True, 'reason': 'All validation checks passed'}
    
    async def _process_decision(self, decision: TradingDecision):
        """Process a decision for execution."""
        
        decision.status = DecisionStatus.PROCESSING
        
        try:
            # Execute decision (if auto-execution enabled)
            if self.config['auto_execution']:
                execution_result = await self._execute_decision(decision)
                decision.execution_result = execution_result
                decision.executed_at = datetime.now()
                decision.status = DecisionStatus.EXECUTED
            else:
                # Wait for manual approval
                decision.status = DecisionStatus.PENDING
            
            # Move to history
            self.decision_history.append(decision)
            if decision.decision_id in self.pending_decisions:
                del self.pending_decisions[decision.decision_id]
            
            # Notify callbacks
            for callback in self.execution_callbacks:
                try:
                    await callback(decision)
                except Exception as e:
                    self.logger.error(f"Error in execution callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing decision {decision.decision_id}: {e}")
            decision.status = DecisionStatus.FAILED
            decision.execution_result = {'error': str(e)}
    
    async def _execute_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """Execute a trading decision."""
        
        # This would interface with actual trading API
        # For now, return simulated execution
        
        execution_result = {
            'executed': True,
            'execution_price': decision.price,
            'executed_quantity': decision.quantity,
            'execution_time': datetime.now(),
            'transaction_cost': decision.quantity * decision.price * 0.001,  # 0.1% fee
            'slippage': 0.0001  # 0.01% slippage
        }
        
        self.logger.info(f"Decision executed: {decision.decision_id}")
        
        return execution_result
    
    def _update_decision_metrics(self, decision: TradingDecision, processing_time: float):
        """Update decision metrics."""
        
        self.metrics.total_decisions += 1
        self.metrics.average_confidence = (
            (self.metrics.average_confidence * (self.metrics.total_decisions - 1) + decision.confidence) /
            self.metrics.total_decisions
        )
        self.metrics.average_execution_time = (
            (self.metrics.average_execution_time * (self.metrics.total_decisions - 1) + processing_time) /
            self.metrics.total_decisions
        )
    
    def add_execution_callback(self, callback: Callable):
        """Add callback for decision execution."""
        self.execution_callbacks.append(callback)
    
    def add_monitoring_callback(self, callback: Callable):
        """Add callback for decision monitoring."""
        self.monitoring_callbacks.append(callback)
    
    def get_decision_status(self, decision_id: str) -> Optional[TradingDecision]:
        """Get status of a specific decision."""
        
        # Check pending decisions
        if decision_id in self.pending_decisions:
            return self.pending_decisions[decision_id]
        
        # Check history
        for decision in reversed(self.decision_history):
            if decision.decision_id == decision_id:
                return decision
        
        return None
    
    def get_decision_metrics(self) -> DecisionMetrics:
        """Get current decision metrics."""
        return self.metrics
    
    def get_recent_decisions(self, hours: int = 24) -> List[TradingDecision]:
        """Get recent decisions within specified hours."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent = [
            d for d in self.decision_history
            if d.created_at >= cutoff_time
        ]
        
        return sorted(recent, key=lambda x: x.created_at, reverse=True)
    
    async def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop for all trading."""
        
        self.logger.warning(f"Emergency stop triggered: {reason}")
        
        # Create emergency stop decision for all active positions
        for symbol in self.emergency_stops.keys():
            emergency_decision = TradingDecision(
                decision_id=f"emergency_stop_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                decision_type=DecisionType.EMERGENCY_STOP,
                priority=DecisionPriority.EMERGENCY,
                symbol=symbol,
                action='close_all',
                quantity=0.0,  # Will be determined by position size
                price=None,
                reasoning_chain=ReasoningChain(
                    chain_id=f"emergency_{symbol}",
                    nodes=[],
                    final_conclusion=f"Emergency stop: {reason}",
                    overall_confidence=1.0,
                    created_at=datetime.now(),
                    market_context={'emergency': True}
                ),
                confidence=1.0,
                risk_score=0.0,
                expected_outcome={'reason': reason},
                implementation_plan=[f"Close all positions in {symbol}"],
                risk_mitigation=["Emergency liquidation"],
                created_at=datetime.now()
            )
            
            await self.decision_queue.put(emergency_decision)


# Helper extensions to MarketState for risk level conversion
def market_state_risk_level_numeric(self) -> float:
    """Convert risk level to numeric value."""
    risk_mapping = {
        'low': 0.2,
        'medium': 0.5,
        'high': 0.7,
        'very_high': 0.9
    }
    return risk_mapping.get(self.risk_level, 0.5)

# Add method to MarketState class
MarketState.risk_level_numeric = market_state_risk_level_numeric