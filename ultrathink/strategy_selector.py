"""UltraThink Strategy Selector: Adaptive strategy selection with advanced reasoning."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from abc import ABC, abstractmethod

from .reasoning_engine import UltraThinkReasoningEngine, ReasoningChain
from .market_analyzer import UltraThinkMarketAnalyzer, MarketState


class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    avg_trade_duration: float = 0.0
    trades_count: int = 0
    recent_performance: float = 0.0
    volatility: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyConfiguration:
    """Configuration for a trading strategy."""
    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any]
    market_regimes: List[MarketRegime]
    min_confidence: float = 0.5
    max_position_size: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.10
    enabled: bool = True
    priority: int = 1


@dataclass
class StrategyRecommendation:
    """A strategy recommendation with reasoning."""
    strategy_config: StrategyConfiguration
    confidence: float
    expected_return: float
    risk_score: float
    reasoning_chain: ReasoningChain
    market_fit_score: float
    implementation_notes: List[str]
    risk_factors: List[str]


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: StrategyConfiguration):
        self.config = config
        self.metrics = StrategyMetrics()
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    @abstractmethod
    def generate_signals(self, market_state: MarketState, 
                        data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on market state."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], 
                              portfolio_value: float) -> float:
        """Calculate position size for a signal."""
        pass
    
    @abstractmethod
    def get_market_fit_score(self, market_state: MarketState) -> float:
        """Calculate how well this strategy fits current market conditions."""
        pass
    
    def update_metrics(self, new_metrics: StrategyMetrics):
        """Update strategy performance metrics."""
        self.metrics = new_metrics
        self.metrics.last_updated = datetime.now()


class UltraThinkStrategySelector:
    """Advanced strategy selector with reasoning-based adaptation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.reasoning_engine = UltraThinkReasoningEngine()
        self.market_analyzer = UltraThinkMarketAnalyzer()
        
        # Strategy management
        self.available_strategies = {}
        self.strategy_history = {}
        self.regime_strategy_mapping = {}
        
        # Performance tracking
        self.strategy_metrics = {}
        self.adaptation_history = []
        
        # Initialize strategies
        self._initialize_strategies()
        self._build_regime_mapping()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'strategy_selection_method': 'reasoning_based',  # 'reasoning_based', 'performance_based', 'ensemble'
            'adaptation_frequency': 'daily',  # 'real_time', 'hourly', 'daily', 'weekly'
            'min_strategy_confidence': 0.6,
            'max_strategies_active': 3,
            'performance_window': 30,  # days
            'regime_detection_enabled': True,
            'ensemble_weighting': 'dynamic',  # 'equal', 'performance', 'dynamic'
            'risk_adjustment': True,
            'strategy_rotation_enabled': True,
            'emergency_stop_enabled': True,
            'emergency_drawdown_threshold': 0.15
        }
    
    def _initialize_strategies(self):
        """Initialize available trading strategies."""
        
        # Momentum Strategy
        momentum_config = StrategyConfiguration(
            strategy_type=StrategyType.MOMENTUM,
            name="UltraThink Momentum",
            description="Adaptive momentum strategy with regime detection",
            parameters={
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'volume_confirmation': True,
                'regime_adaptive': True
            },
            market_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT],
            min_confidence=0.7,
            max_position_size=0.3,
            stop_loss=0.04,
            take_profit=0.12
        )
        self.available_strategies['momentum'] = MomentumStrategy(momentum_config)
        
        # Mean Reversion Strategy
        mean_reversion_config = StrategyConfiguration(
            strategy_type=StrategyType.MEAN_REVERSION,
            name="UltraThink Mean Reversion",
            description="Advanced mean reversion with volatility clustering",
            parameters={
                'reversion_period': 50,
                'deviation_threshold': 2.0,
                'volatility_filter': True,
                'regime_adaptive': True
            },
            market_regimes=[MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY],
            min_confidence=0.6,
            max_position_size=0.25,
            stop_loss=0.03,
            take_profit=0.08
        )
        self.available_strategies['mean_reversion'] = MeanReversionStrategy(mean_reversion_config)
        
        # Breakout Strategy
        breakout_config = StrategyConfiguration(
            strategy_type=StrategyType.BREAKOUT,
            name="UltraThink Breakout",
            description="Multi-timeframe breakout with volume confirmation",
            parameters={
                'consolidation_period': 15,
                'breakout_threshold': 0.025,
                'volume_multiple': 1.5,
                'false_breakout_filter': True
            },
            market_regimes=[MarketRegime.LOW_VOLATILITY, MarketRegime.BREAKOUT],
            min_confidence=0.75,
            max_position_size=0.4,
            stop_loss=0.05,
            take_profit=0.15
        )
        self.available_strategies['breakout'] = BreakoutStrategy(breakout_config)
        
        # Trend Following Strategy
        trend_following_config = StrategyConfiguration(
            strategy_type=StrategyType.TREND_FOLLOWING,
            name="UltraThink Trend Following",
            description="Adaptive trend following with regime confirmation",
            parameters={
                'fast_ma': 10,
                'slow_ma': 30,
                'trend_strength_filter': True,
                'regime_confirmation': True
            },
            market_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            min_confidence=0.65,
            max_position_size=0.35,
            stop_loss=0.06,
            take_profit=0.18
        )
        self.available_strategies['trend_following'] = TrendFollowingStrategy(trend_following_config)
        
        # Volatility Strategy
        volatility_config = StrategyConfiguration(
            strategy_type=StrategyType.SCALPING,
            name="UltraThink Volatility",
            description="Volatility-based scalping strategy",
            parameters={
                'volatility_window': 20,
                'entry_threshold': 1.5,
                'quick_exit': True,
                'regime_adaptive': True
            },
            market_regimes=[MarketRegime.VOLATILE],
            min_confidence=0.8,
            max_position_size=0.2,
            stop_loss=0.02,
            take_profit=0.04
        )
        self.available_strategies['volatility'] = VolatilityStrategy(volatility_config)
    
    def _build_regime_mapping(self):
        """Build mapping between market regimes and optimal strategies."""
        self.regime_strategy_mapping = {
            MarketRegime.TRENDING_UP: ['momentum', 'trend_following'],
            MarketRegime.TRENDING_DOWN: ['momentum', 'trend_following'],
            MarketRegime.SIDEWAYS: ['mean_reversion', 'volatility'],
            MarketRegime.VOLATILE: ['volatility', 'breakout'],
            MarketRegime.LOW_VOLATILITY: ['mean_reversion', 'breakout'],
            MarketRegime.BREAKOUT: ['breakout', 'momentum'],
            MarketRegime.REVERSAL: ['mean_reversion']
        }
    
    def select_optimal_strategy(self, market_data: pd.DataFrame, 
                              symbol: str, context: Dict[str, Any] = None) -> StrategyRecommendation:
        """Select the optimal strategy using ultrathink reasoning."""
        
        self.logger.info(f"Selecting optimal strategy for {symbol}")
        
        # Analyze market state
        market_state = self.market_analyzer.analyze_market(market_data, symbol, context)
        
        # Detect market regime
        market_regime = self._detect_market_regime(market_state, market_data)
        
        # Generate reasoning about strategy selection
        strategy_reasoning = self._reason_about_strategy_selection(
            market_state, market_regime, market_data
        )
        
        # Score all strategies
        strategy_scores = {}
        for name, strategy in self.available_strategies.items():
            if not strategy.config.enabled:
                continue
                
            score = self._score_strategy(strategy, market_state, market_regime, market_data)
            strategy_scores[name] = score
        
        # Select best strategy
        if not strategy_scores:
            raise ValueError("No enabled strategies available")
        
        best_strategy_name = max(strategy_scores.keys(), key=lambda k: strategy_scores[k]['total_score'])
        best_strategy = self.available_strategies[best_strategy_name]
        best_score = strategy_scores[best_strategy_name]
        
        # Create recommendation
        recommendation = StrategyRecommendation(
            strategy_config=best_strategy.config,
            confidence=best_score['confidence'],
            expected_return=best_score['expected_return'],
            risk_score=best_score['risk_score'],
            reasoning_chain=strategy_reasoning,
            market_fit_score=best_score['market_fit'],
            implementation_notes=self._generate_implementation_notes(
                best_strategy, market_state, market_regime
            ),
            risk_factors=self._identify_risk_factors(
                best_strategy, market_state, market_regime
            )
        )
        
        # Log selection
        self.logger.info(f"Selected strategy: {best_strategy_name} "
                        f"(confidence: {recommendation.confidence:.2%}, "
                        f"fit: {recommendation.market_fit_score:.2%})")
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'selected_strategy': best_strategy_name,
            'market_regime': market_regime.value,
            'confidence': recommendation.confidence,
            'market_state': market_state.overall_sentiment,
            'reasoning': strategy_reasoning.final_conclusion
        })
        
        return recommendation
    
    def _detect_market_regime(self, market_state: MarketState, 
                            data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        
        # Analyze price action
        returns = data['close'].pct_change().dropna()
        
        # Trend detection
        if len(returns) >= 20:
            short_trend = returns.tail(10).mean()
            medium_trend = returns.tail(20).mean()
            
            # Volatility analysis
            current_vol = returns.tail(10).std()
            historical_vol = returns.tail(50).std() if len(returns) >= 50 else current_vol
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            
            # Regime classification logic
            if abs(short_trend) > 0.002 and abs(medium_trend) > 0.001:  # Strong trend
                if short_trend > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            elif vol_ratio > 1.5:  # High volatility
                return MarketRegime.VOLATILE
            elif vol_ratio < 0.7:  # Low volatility
                # Check for potential breakout setup
                price_range = (data['high'].tail(20).max() - data['low'].tail(20).min()) / data['close'].iloc[-1]
                if price_range < 0.1:  # Tight range
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.LOW_VOLATILITY
            else:  # Sideways market
                return MarketRegime.SIDEWAYS
        
        return MarketRegime.SIDEWAYS  # Default
    
    def _reason_about_strategy_selection(self, market_state: MarketState, 
                                       regime: MarketRegime, 
                                       data: pd.DataFrame) -> ReasoningChain:
        """Use reasoning engine to analyze strategy selection."""
        
        context = {
            'market_regime': regime.value,
            'market_sentiment': market_state.overall_sentiment,
            'risk_level': market_state.risk_level,
            'opportunity_score': market_state.opportunity_score,
            'strategy_selection': True
        }
        
        return self.reasoning_engine.reason_about_market(data, context)
    
    def _score_strategy(self, strategy: BaseStrategy, market_state: MarketState,
                       regime: MarketRegime, data: pd.DataFrame) -> Dict[str, float]:
        """Comprehensive strategy scoring."""
        
        # Market fit score
        market_fit = strategy.get_market_fit_score(market_state)
        
        # Regime compatibility
        regime_fit = 1.0 if regime in strategy.config.market_regimes else 0.3
        
        # Historical performance score
        performance_score = self._calculate_performance_score(strategy)
        
        # Risk-adjusted score
        risk_score = self._calculate_risk_score(strategy, market_state)
        
        # Confidence from market analysis
        market_confidence = market_state.confidence_score
        
        # Weighted total score
        weights = {
            'market_fit': 0.3,
            'regime_fit': 0.25,
            'performance': 0.2,
            'risk_adjusted': 0.15,
            'market_confidence': 0.1
        }
        
        total_score = (
            weights['market_fit'] * market_fit +
            weights['regime_fit'] * regime_fit +
            weights['performance'] * performance_score +
            weights['risk_adjusted'] * (1 - risk_score) +  # Lower risk is better
            weights['market_confidence'] * market_confidence
        )
        
        # Expected return estimation
        base_return = strategy.metrics.recent_performance
        market_multiplier = 1 + (market_state.opportunity_score - 0.5)
        expected_return = base_return * market_multiplier
        
        return {
            'total_score': total_score,
            'market_fit': market_fit,
            'regime_fit': regime_fit,
            'performance': performance_score,
            'risk_score': risk_score,
            'confidence': min(0.95, max(0.1, total_score)),
            'expected_return': expected_return
        }
    
    def _calculate_performance_score(self, strategy: BaseStrategy) -> float:
        """Calculate performance score for strategy."""
        metrics = strategy.metrics
        
        # Combine multiple performance metrics
        sharpe_score = min(1.0, max(0.0, (metrics.sharpe_ratio + 1) / 3))  # Scale Sharpe
        return_score = min(1.0, max(0.0, metrics.total_return + 0.5))  # Bias toward positive
        drawdown_score = max(0.0, 1 - abs(metrics.max_drawdown))
        win_rate_score = metrics.win_rate
        
        # Weighted combination
        performance_score = (
            0.3 * sharpe_score +
            0.25 * return_score +
            0.25 * drawdown_score +
            0.2 * win_rate_score
        )
        
        return performance_score
    
    def _calculate_risk_score(self, strategy: BaseStrategy, 
                            market_state: MarketState) -> float:
        """Calculate risk score for strategy in current market."""
        
        base_risk = 0.5  # Default risk level
        
        # Adjust for market risk
        market_risk_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3,
            'very_high': 1.6
        }
        
        market_risk = base_risk * market_risk_multiplier.get(market_state.risk_level, 1.0)
        
        # Adjust for strategy-specific risk
        strategy_vol = strategy.metrics.volatility
        volatility_risk = min(1.0, strategy_vol * 2)  # Scale volatility
        
        # Combine risks
        total_risk = min(1.0, (market_risk + volatility_risk) / 2)
        
        return total_risk
    
    def _generate_implementation_notes(self, strategy: BaseStrategy,
                                     market_state: MarketState,
                                     regime: MarketRegime) -> List[str]:
        """Generate implementation notes for the selected strategy."""
        
        notes = []
        
        # Market condition specific notes
        if market_state.risk_level in ['high', 'very_high']:
            notes.append("Reduce position sizes due to elevated market risk")
            notes.append("Consider tighter stop losses")
        
        if market_state.overall_sentiment == 'neutral':
            notes.append("Market shows mixed signals - be prepared for false signals")
        
        # Strategy specific notes
        if strategy.config.strategy_type == StrategyType.BREAKOUT:
            notes.append("Wait for volume confirmation before entering positions")
            notes.append("Be cautious of false breakouts in current regime")
        
        elif strategy.config.strategy_type == StrategyType.MOMENTUM:
            notes.append("Monitor for momentum exhaustion signals")
            notes.append("Consider scaling out at resistance levels")
        
        elif strategy.config.strategy_type == StrategyType.MEAN_REVERSION:
            notes.append("Ensure sufficient mean reversion setup before entry")
            notes.append("Be patient - wait for clear oversold/overbought conditions")
        
        # Regime specific notes
        if regime == MarketRegime.VOLATILE:
            notes.append("Expect increased noise - use larger position sizing buffers")
        
        return notes
    
    def _identify_risk_factors(self, strategy: BaseStrategy,
                             market_state: MarketState,
                             regime: MarketRegime) -> List[str]:
        """Identify risk factors for the selected strategy."""
        
        risk_factors = []
        
        # Market risk factors
        for dimension in market_state.dimensions.values():
            risk_factor = dimension.analysis_result.get('risk_factor')
            if risk_factor:
                risk_factors.append(f"Market risk: {risk_factor}")
        
        # Strategy-regime mismatch
        if regime not in strategy.config.market_regimes:
            risk_factors.append(f"Strategy not optimized for {regime.value} regime")
        
        # Performance concerns
        if strategy.metrics.recent_performance < -0.05:
            risk_factors.append("Strategy showing recent underperformance")
        
        if strategy.metrics.max_drawdown > 0.15:
            risk_factors.append("Strategy has history of large drawdowns")
        
        # Market structure risks
        if market_state.risk_level == 'very_high':
            risk_factors.append("Extreme market volatility may cause unexpected behavior")
        
        return risk_factors
    
    def get_strategy_ensemble(self, market_data: pd.DataFrame, symbol: str,
                            context: Dict[str, Any] = None) -> List[StrategyRecommendation]:
        """Get ensemble of top strategies with weights."""
        
        # Get primary recommendation
        primary_rec = self.select_optimal_strategy(market_data, symbol, context)
        
        if self.config['max_strategies_active'] == 1:
            return [primary_rec]
        
        # Get additional strategies
        market_state = self.market_analyzer.analyze_market(market_data, symbol, context)
        regime = self._detect_market_regime(market_state, market_data)
        
        all_recommendations = []
        strategy_scores = {}
        
        for name, strategy in self.available_strategies.items():
            if not strategy.config.enabled:
                continue
                
            score = self._score_strategy(strategy, market_state, regime, market_data)
            if score['confidence'] >= self.config['min_strategy_confidence']:
                reasoning = self._reason_about_strategy_selection(market_state, regime, market_data)
                
                rec = StrategyRecommendation(
                    strategy_config=strategy.config,
                    confidence=score['confidence'],
                    expected_return=score['expected_return'],
                    risk_score=score['risk_score'],
                    reasoning_chain=reasoning,
                    market_fit_score=score['market_fit'],
                    implementation_notes=self._generate_implementation_notes(strategy, market_state, regime),
                    risk_factors=self._identify_risk_factors(strategy, market_state, regime)
                )
                
                all_recommendations.append(rec)
                strategy_scores[name] = score['total_score']
        
        # Sort by score and take top N
        all_recommendations.sort(key=lambda x: x.confidence, reverse=True)
        top_strategies = all_recommendations[:self.config['max_strategies_active']]
        
        return top_strategies
    
    def update_strategy_performance(self, strategy_name: str, metrics: StrategyMetrics):
        """Update performance metrics for a strategy."""
        if strategy_name in self.available_strategies:
            self.available_strategies[strategy_name].update_metrics(metrics)
            self.strategy_metrics[strategy_name] = metrics
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of strategy adaptations."""
        return self.adaptation_history.copy()
    
    def emergency_stop_check(self, portfolio_metrics: Dict[str, float]) -> bool:
        """Check if emergency stop should be triggered."""
        if not self.config['emergency_stop_enabled']:
            return False
        
        current_drawdown = portfolio_metrics.get('drawdown', 0)
        return current_drawdown > self.config['emergency_drawdown_threshold']


# Strategy implementations

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""
    
    def generate_signals(self, market_state: MarketState, data: pd.DataFrame) -> Dict[str, Any]:
        # Implement momentum signal generation
        returns = data['close'].pct_change().dropna()
        lookback = self.config.parameters['lookback_period']
        
        if len(returns) >= lookback:
            momentum = returns.tail(lookback).mean()
            threshold = self.config.parameters['momentum_threshold']
            
            if momentum > threshold:
                return {'signal': 'buy', 'strength': min(1.0, momentum / threshold)}
            elif momentum < -threshold:
                return {'signal': 'sell', 'strength': min(1.0, abs(momentum) / threshold)}
        
        return {'signal': 'hold', 'strength': 0.0}
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> float:
        base_size = self.config.max_position_size * portfolio_value
        strength = signal.get('strength', 0.5)
        return base_size * strength
    
    def get_market_fit_score(self, market_state: MarketState) -> float:
        # Check for momentum-favorable conditions
        momentum_score = 0.5
        
        if 'momentum' in market_state.dimensions:
            momentum_dim = market_state.dimensions['momentum']
            if momentum_dim.analysis_result.get('price_momentum') in ['strong_positive', 'strong_negative']:
                momentum_score += 0.3
        
        if market_state.overall_sentiment in ['bullish', 'bearish']:
            momentum_score += 0.2
        
        return min(1.0, momentum_score)


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy."""
    
    def generate_signals(self, market_state: MarketState, data: pd.DataFrame) -> Dict[str, Any]:
        period = self.config.parameters['reversion_period']
        
        if len(data) >= period:
            mean_price = data['close'].tail(period).mean()
            current_price = data['close'].iloc[-1]
            std_dev = data['close'].tail(period).std()
            
            threshold = self.config.parameters['deviation_threshold']
            
            z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
            
            if z_score > threshold:
                return {'signal': 'sell', 'strength': min(1.0, abs(z_score) / threshold)}
            elif z_score < -threshold:
                return {'signal': 'buy', 'strength': min(1.0, abs(z_score) / threshold)}
        
        return {'signal': 'hold', 'strength': 0.0}
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> float:
        base_size = self.config.max_position_size * portfolio_value
        strength = signal.get('strength', 0.5)
        return base_size * strength
    
    def get_market_fit_score(self, market_state: MarketState) -> float:
        # Check for mean reversion conditions
        reversion_score = 0.4
        
        if market_state.overall_sentiment == 'neutral':
            reversion_score += 0.3
        
        if 'volatility' in market_state.dimensions:
            vol_dim = market_state.dimensions['volatility']
            if vol_dim.analysis_result.get('regime') in ['normal', 'low']:
                reversion_score += 0.3
        
        return min(1.0, reversion_score)


class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy."""
    
    def generate_signals(self, market_state: MarketState, data: pd.DataFrame) -> Dict[str, Any]:
        period = self.config.parameters['consolidation_period']
        
        if len(data) >= period:
            recent_high = data['high'].tail(period).max()
            recent_low = data['low'].tail(period).min()
            current_price = data['close'].iloc[-1]
            
            breakout_threshold = self.config.parameters['breakout_threshold']
            range_size = (recent_high - recent_low) / recent_low
            
            if current_price > recent_high * (1 + breakout_threshold) and range_size < 0.1:
                return {'signal': 'buy', 'strength': 0.8}
            elif current_price < recent_low * (1 - breakout_threshold) and range_size < 0.1:
                return {'signal': 'sell', 'strength': 0.8}
        
        return {'signal': 'hold', 'strength': 0.0}
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> float:
        return self.config.max_position_size * portfolio_value
    
    def get_market_fit_score(self, market_state: MarketState) -> float:
        # Check for breakout setup conditions
        breakout_score = 0.3
        
        if 'volatility' in market_state.dimensions:
            vol_dim = market_state.dimensions['volatility']
            if vol_dim.analysis_result.get('regime') == 'low':
                breakout_score += 0.4
        
        if market_state.risk_level == 'low':
            breakout_score += 0.3
        
        return min(1.0, breakout_score)


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy."""
    
    def generate_signals(self, market_state: MarketState, data: pd.DataFrame) -> Dict[str, Any]:
        fast_ma = self.config.parameters['fast_ma']
        slow_ma = self.config.parameters['slow_ma']
        
        if len(data) >= slow_ma:
            fast_ma_val = data['close'].tail(fast_ma).mean()
            slow_ma_val = data['close'].tail(slow_ma).mean()
            
            if fast_ma_val > slow_ma_val * 1.005:  # 0.5% threshold
                return {'signal': 'buy', 'strength': 0.7}
            elif fast_ma_val < slow_ma_val * 0.995:
                return {'signal': 'sell', 'strength': 0.7}
        
        return {'signal': 'hold', 'strength': 0.0}
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> float:
        return self.config.max_position_size * portfolio_value
    
    def get_market_fit_score(self, market_state: MarketState) -> float:
        # Check for trending conditions
        trend_score = 0.3
        
        if 'technical' in market_state.dimensions:
            tech_dim = market_state.dimensions['technical']
            if 'trend' in tech_dim.analysis_result.get('ma_trend', ''):
                trend_score += 0.4
        
        if market_state.overall_sentiment in ['bullish', 'bearish']:
            trend_score += 0.3
        
        return min(1.0, trend_score)


class VolatilityStrategy(BaseStrategy):
    """Volatility-based strategy."""
    
    def generate_signals(self, market_state: MarketState, data: pd.DataFrame) -> Dict[str, Any]:
        # Implement volatility-based signals
        returns = data['close'].pct_change().dropna()
        window = self.config.parameters['volatility_window']
        
        if len(returns) >= window:
            current_vol = returns.tail(5).std()
            historical_vol = returns.tail(window).std()
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            threshold = self.config.parameters['entry_threshold']
            
            if vol_ratio > threshold:
                # High volatility - look for quick trades
                recent_return = returns.iloc[-1]
                if recent_return > 0:
                    return {'signal': 'buy', 'strength': 0.6}
                else:
                    return {'signal': 'sell', 'strength': 0.6}
        
        return {'signal': 'hold', 'strength': 0.0}
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> float:
        # Smaller positions for volatility strategy
        return self.config.max_position_size * portfolio_value * 0.5
    
    def get_market_fit_score(self, market_state: MarketState) -> float:
        # Check for high volatility conditions
        vol_score = 0.2
        
        if 'volatility' in market_state.dimensions:
            vol_dim = market_state.dimensions['volatility']
            if vol_dim.analysis_result.get('regime') == 'high':
                vol_score += 0.6
        
        if market_state.risk_level in ['high', 'very_high']:
            vol_score += 0.2
        
        return min(1.0, vol_score)