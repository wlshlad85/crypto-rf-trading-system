"""UltraThink Reasoning Engine: Advanced multi-layered reasoning for trading decisions."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import threading
from collections import deque


class ReasoningLevel(Enum):
    """Different levels of reasoning depth."""
    SURFACE = 1      # Basic technical analysis
    TACTICAL = 2     # Short-term patterns and signals
    STRATEGIC = 3    # Medium-term market dynamics
    META = 4         # Long-term market regime analysis
    PHILOSOPHICAL = 5 # Fundamental market understanding


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning conclusions."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class ReasoningNode:
    """A single reasoning step or conclusion."""
    level: ReasoningLevel
    premise: str
    conclusion: str
    evidence: Dict[str, Any]
    confidence: float
    timestamp: datetime
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ReasoningChain:
    """A chain of connected reasoning nodes."""
    chain_id: str
    nodes: List[ReasoningNode]
    final_conclusion: str
    overall_confidence: float
    created_at: datetime
    market_context: Dict[str, Any]


class UltraThinkReasoningEngine:
    """Advanced reasoning engine that thinks through trading decisions at multiple levels."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Reasoning chains storage
        self.active_chains = {}
        self.completed_chains = deque(maxlen=1000)
        self.reasoning_history = deque(maxlen=10000)
        
        # Knowledge base
        self.market_patterns = {}
        self.learned_relationships = {}
        self.confidence_adjustments = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize reasoning modules
        self._initialize_reasoning_modules()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for reasoning engine."""
        return {
            "max_reasoning_depth": 5,
            "min_confidence_threshold": 0.3,
            "enable_meta_reasoning": True,
            "enable_philosophical_reasoning": True,
            "pattern_memory_size": 5000,
            "reasoning_timeout": 30.0,
            "parallel_reasoning": True,
            "confidence_decay_rate": 0.95,
            "evidence_weight_threshold": 0.1
        }
    
    def _initialize_reasoning_modules(self):
        """Initialize specialized reasoning modules."""
        self.surface_reasoner = SurfaceReasoner()
        self.tactical_reasoner = TacticalReasoner()
        self.strategic_reasoner = StrategicReasoner()
        self.meta_reasoner = MetaReasoner()
        self.philosophical_reasoner = PhilosophicalReasoner()
    
    def reason_about_market(self, market_data: pd.DataFrame, 
                          context: Dict[str, Any]) -> ReasoningChain:
        """Main entry point for reasoning about market conditions."""
        chain_id = f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"Starting ultrathink reasoning chain: {chain_id}")
        
        with self.lock:
            # Create new reasoning chain
            chain = ReasoningChain(
                chain_id=chain_id,
                nodes=[],
                final_conclusion="",
                overall_confidence=0.0,
                created_at=datetime.now(),
                market_context=context.copy()
            )
            
            self.active_chains[chain_id] = chain
        
        try:
            # Level 1: Surface reasoning (technical indicators)
            surface_nodes = self._reason_at_surface_level(market_data, context)
            chain.nodes.extend(surface_nodes)
            
            # Level 2: Tactical reasoning (short-term patterns)
            tactical_nodes = self._reason_at_tactical_level(market_data, context, surface_nodes)
            chain.nodes.extend(tactical_nodes)
            
            # Level 3: Strategic reasoning (medium-term dynamics)
            strategic_nodes = self._reason_at_strategic_level(market_data, context, tactical_nodes)
            chain.nodes.extend(strategic_nodes)
            
            # Level 4: Meta reasoning (market regime analysis)
            if self.config["enable_meta_reasoning"]:
                meta_nodes = self._reason_at_meta_level(market_data, context, strategic_nodes)
                chain.nodes.extend(meta_nodes)
            
            # Level 5: Philosophical reasoning (fundamental understanding)
            if self.config["enable_philosophical_reasoning"]:
                philosophical_nodes = self._reason_at_philosophical_level(market_data, context, chain.nodes)
                chain.nodes.extend(philosophical_nodes)
            
            # Synthesize final conclusion
            chain.final_conclusion, chain.overall_confidence = self._synthesize_conclusions(chain.nodes)
            
            # Store completed chain
            with self.lock:
                self.completed_chains.append(chain)
                if chain_id in self.active_chains:
                    del self.active_chains[chain_id]
            
            self.logger.info(f"Reasoning chain completed: {chain_id} with confidence {chain.overall_confidence:.3f}")
            
            return chain
            
        except Exception as e:
            self.logger.error(f"Error in reasoning chain {chain_id}: {e}")
            # Clean up
            with self.lock:
                if chain_id in self.active_chains:
                    del self.active_chains[chain_id]
            raise
    
    def _reason_at_surface_level(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[ReasoningNode]:
        """Level 1: Surface reasoning - basic technical analysis."""
        nodes = []
        
        # RSI Analysis
        if 'rsi' in data.columns:
            rsi_value = data['rsi'].iloc[-1]
            if rsi_value > 70:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.SURFACE,
                    premise=f"RSI is {rsi_value:.2f}, above 70",
                    conclusion="Market appears overbought from RSI perspective",
                    evidence={"rsi_value": rsi_value, "rsi_threshold": 70},
                    confidence=0.6,
                    timestamp=datetime.now()
                ))
            elif rsi_value < 30:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.SURFACE,
                    premise=f"RSI is {rsi_value:.2f}, below 30",
                    conclusion="Market appears oversold from RSI perspective",
                    evidence={"rsi_value": rsi_value, "rsi_threshold": 30},
                    confidence=0.6,
                    timestamp=datetime.now()
                ))
        
        # MACD Analysis
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            macd = data['macd'].iloc[-1]
            signal = data['macd_signal'].iloc[-1]
            if macd > signal and data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2]:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.SURFACE,
                    premise=f"MACD ({macd:.4f}) crossed above signal ({signal:.4f})",
                    conclusion="Bullish momentum signal detected",
                    evidence={"macd": macd, "signal": signal, "crossover": True},
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
        
        # Volume Analysis
        if 'volume' in data.columns:
            recent_volume = data['volume'].tail(5).mean()
            historical_volume = data['volume'].tail(50).mean()
            if recent_volume > historical_volume * 1.5:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.SURFACE,
                    premise=f"Recent volume ({recent_volume:.0f}) is 50% above historical average ({historical_volume:.0f})",
                    conclusion="Significant increase in trading activity",
                    evidence={"recent_volume": recent_volume, "historical_volume": historical_volume},
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
        
        return nodes
    
    def _reason_at_tactical_level(self, data: pd.DataFrame, context: Dict[str, Any], 
                                surface_nodes: List[ReasoningNode]) -> List[ReasoningNode]:
        """Level 2: Tactical reasoning - short-term patterns and signals."""
        nodes = []
        
        # Pattern Recognition
        if len(data) >= 20:
            # Support/Resistance Analysis
            highs = data['high'].tail(20)
            lows = data['low'].tail(20)
            current_price = data['close'].iloc[-1]
            
            resistance_level = highs.quantile(0.9)
            support_level = lows.quantile(0.1)
            
            if current_price > resistance_level * 0.98:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.TACTICAL,
                    premise=f"Price ({current_price:.2f}) is near resistance level ({resistance_level:.2f})",
                    conclusion="Price may face resistance, potential reversal zone",
                    evidence={"current_price": current_price, "resistance": resistance_level},
                    confidence=0.65,
                    timestamp=datetime.now()
                ))
            elif current_price < support_level * 1.02:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.TACTICAL,
                    premise=f"Price ({current_price:.2f}) is near support level ({support_level:.2f})",
                    conclusion="Price may find support, potential bounce zone",
                    evidence={"current_price": current_price, "support": support_level},
                    confidence=0.65,
                    timestamp=datetime.now()
                ))
        
        # Momentum Convergence Analysis
        bullish_signals = sum(1 for node in surface_nodes if "bullish" in node.conclusion.lower())
        bearish_signals = sum(1 for node in surface_nodes if "bearish" in node.conclusion.lower())
        
        if bullish_signals >= 2 and bearish_signals == 0:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.TACTICAL,
                premise=f"Multiple bullish signals ({bullish_signals}) with no bearish signals",
                conclusion="Strong bullish momentum convergence",
                evidence={"bullish_count": bullish_signals, "bearish_count": bearish_signals},
                confidence=0.8,
                timestamp=datetime.now(),
                dependencies=[node.premise for node in surface_nodes if "bullish" in node.conclusion.lower()]
            ))
        elif bearish_signals >= 2 and bullish_signals == 0:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.TACTICAL,
                premise=f"Multiple bearish signals ({bearish_signals}) with no bullish signals",
                conclusion="Strong bearish momentum convergence",
                evidence={"bullish_count": bullish_signals, "bearish_count": bearish_signals},
                confidence=0.8,
                timestamp=datetime.now(),
                dependencies=[node.premise for node in surface_nodes if "bearish" in node.conclusion.lower()]
            ))
        
        return nodes
    
    def _reason_at_strategic_level(self, data: pd.DataFrame, context: Dict[str, Any],
                                 tactical_nodes: List[ReasoningNode]) -> List[ReasoningNode]:
        """Level 3: Strategic reasoning - medium-term market dynamics."""
        nodes = []
        
        # Trend Analysis
        if len(data) >= 50:
            short_ma = data['close'].tail(10).mean()
            medium_ma = data['close'].tail(30).mean()
            long_ma = data['close'].tail(50).mean()
            
            if short_ma > medium_ma > long_ma:
                trend_strength = (short_ma - long_ma) / long_ma
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.STRATEGIC,
                    premise=f"Short MA ({short_ma:.2f}) > Medium MA ({medium_ma:.2f}) > Long MA ({long_ma:.2f})",
                    conclusion=f"Strong uptrend confirmed with {trend_strength:.2%} strength",
                    evidence={"short_ma": short_ma, "medium_ma": medium_ma, "long_ma": long_ma, "trend_strength": trend_strength},
                    confidence=0.75,
                    timestamp=datetime.now()
                ))
            elif short_ma < medium_ma < long_ma:
                trend_strength = (long_ma - short_ma) / long_ma
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.STRATEGIC,
                    premise=f"Short MA ({short_ma:.2f}) < Medium MA ({medium_ma:.2f}) < Long MA ({long_ma:.2f})",
                    conclusion=f"Strong downtrend confirmed with {trend_strength:.2%} strength",
                    evidence={"short_ma": short_ma, "medium_ma": medium_ma, "long_ma": long_ma, "trend_strength": trend_strength},
                    confidence=0.75,
                    timestamp=datetime.now()
                ))
        
        # Volatility Regime Analysis
        if 'volatility' in data.columns or len(data) >= 30:
            if 'volatility' in data.columns:
                current_vol = data['volatility'].iloc[-1]
                historical_vol = data['volatility'].tail(100).mean()
            else:
                returns = data['close'].pct_change().dropna()
                current_vol = returns.tail(10).std() * np.sqrt(365)
                historical_vol = returns.tail(100).std() * np.sqrt(365)
            
            vol_ratio = current_vol / historical_vol
            
            if vol_ratio > 1.5:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.STRATEGIC,
                    premise=f"Current volatility ({current_vol:.2%}) is {vol_ratio:.1f}x historical average",
                    conclusion="High volatility regime - expect larger price swings and uncertainty",
                    evidence={"current_vol": current_vol, "historical_vol": historical_vol, "vol_ratio": vol_ratio},
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
            elif vol_ratio < 0.7:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.STRATEGIC,
                    premise=f"Current volatility ({current_vol:.2%}) is {vol_ratio:.1f}x historical average",
                    conclusion="Low volatility regime - expect compressed price action and potential breakout",
                    evidence={"current_vol": current_vol, "historical_vol": historical_vol, "vol_ratio": vol_ratio},
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
        
        # Market Cycle Analysis
        if len(data) >= 100:
            recent_performance = (data['close'].iloc[-1] / data['close'].iloc[-30] - 1)
            medium_performance = (data['close'].iloc[-1] / data['close'].iloc[-60] - 1)
            long_performance = (data['close'].iloc[-1] / data['close'].iloc[-100] - 1)
            
            if recent_performance > 0.1 and medium_performance > 0.2:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.STRATEGIC,
                    premise=f"Strong recent ({recent_performance:.1%}) and medium-term ({medium_performance:.1%}) performance",
                    conclusion="Market in strong bull cycle phase",
                    evidence={"recent_perf": recent_performance, "medium_perf": medium_performance, "long_perf": long_performance},
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
            elif recent_performance < -0.1 and medium_performance < -0.2:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.STRATEGIC,
                    premise=f"Weak recent ({recent_performance:.1%}) and medium-term ({medium_performance:.1%}) performance",
                    conclusion="Market in bear cycle phase",
                    evidence={"recent_perf": recent_performance, "medium_perf": medium_performance, "long_perf": long_performance},
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
        
        return nodes
    
    def _reason_at_meta_level(self, data: pd.DataFrame, context: Dict[str, Any],
                            strategic_nodes: List[ReasoningNode]) -> List[ReasoningNode]:
        """Level 4: Meta reasoning - market regime and structural analysis."""
        nodes = []
        
        # Market Regime Classification
        regime_indicators = {
            "trend_strength": 0,
            "volatility_level": 0,
            "momentum_consistency": 0,
            "volume_pattern": 0
        }
        
        # Analyze previous reasoning nodes for regime clues
        for node in strategic_nodes:
            if "trend" in node.conclusion.lower():
                if "strong" in node.conclusion.lower():
                    regime_indicators["trend_strength"] += 0.8 if "up" in node.conclusion.lower() else -0.8
                else:
                    regime_indicators["trend_strength"] += 0.4 if "up" in node.conclusion.lower() else -0.4
            
            if "volatility" in node.conclusion.lower():
                if "high" in node.conclusion.lower():
                    regime_indicators["volatility_level"] = 0.8
                elif "low" in node.conclusion.lower():
                    regime_indicators["volatility_level"] = -0.8
        
        # Classify overall market regime
        regime_score = sum(regime_indicators.values()) / len(regime_indicators)
        
        if regime_score > 0.5:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.META,
                premise=f"Regime analysis score: {regime_score:.2f} based on trend, volatility, and momentum factors",
                conclusion="Market is in a STRONG BULL REGIME - expect continued upward bias with periodic corrections",
                evidence={"regime_score": regime_score, "indicators": regime_indicators},
                confidence=0.75,
                timestamp=datetime.now()
            ))
        elif regime_score < -0.5:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.META,
                premise=f"Regime analysis score: {regime_score:.2f} based on trend, volatility, and momentum factors",
                conclusion="Market is in a STRONG BEAR REGIME - expect continued downward pressure with relief rallies",
                evidence={"regime_score": regime_score, "indicators": regime_indicators},
                confidence=0.75,
                timestamp=datetime.now()
            ))
        else:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.META,
                premise=f"Regime analysis score: {regime_score:.2f} shows mixed signals",
                conclusion="Market is in TRANSITION REGIME - expect choppy, range-bound action with false breakouts",
                evidence={"regime_score": regime_score, "indicators": regime_indicators},
                confidence=0.6,
                timestamp=datetime.now()
            ))
        
        # Structural Market Analysis
        if context.get('market_cap') and context.get('trading_volume'):
            market_cap = context['market_cap']
            volume = context['trading_volume']
            liquidity_ratio = volume / market_cap if market_cap > 0 else 0
            
            if liquidity_ratio > 0.1:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.META,
                    premise=f"High liquidity ratio ({liquidity_ratio:.3f}) indicates active market participation",
                    conclusion="Market structure supports efficient price discovery and lower slippage",
                    evidence={"liquidity_ratio": liquidity_ratio, "market_cap": market_cap, "volume": volume},
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
            elif liquidity_ratio < 0.01:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.META,
                    premise=f"Low liquidity ratio ({liquidity_ratio:.3f}) indicates limited market participation",
                    conclusion="Market structure may lead to inefficient pricing and higher volatility",
                    evidence={"liquidity_ratio": liquidity_ratio, "market_cap": market_cap, "volume": volume},
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
        
        return nodes
    
    def _reason_at_philosophical_level(self, data: pd.DataFrame, context: Dict[str, Any],
                                     all_nodes: List[ReasoningNode]) -> List[ReasoningNode]:
        """Level 5: Philosophical reasoning - fundamental market understanding."""
        nodes = []
        
        # Market Efficiency Analysis
        price_movements = data['close'].pct_change().dropna()
        autocorrelation = price_movements.autocorr()
        
        if abs(autocorrelation) < 0.1:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.PHILOSOPHICAL,
                premise=f"Price autocorrelation is {autocorrelation:.3f}, close to zero",
                conclusion="Market exhibits semi-strong form efficiency - technical analysis has limited predictive power",
                evidence={"autocorrelation": autocorrelation, "efficiency_indication": "semi-strong"},
                confidence=0.6,
                timestamp=datetime.now()
            ))
        else:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.PHILOSOPHICAL,
                premise=f"Price autocorrelation is {autocorrelation:.3f}, significantly different from zero",
                conclusion="Market shows inefficiency patterns - technical analysis may have predictive value",
                evidence={"autocorrelation": autocorrelation, "efficiency_indication": "inefficient"},
                confidence=0.7,
                timestamp=datetime.now()
            ))
        
        # Risk-Return Philosophy
        if len(price_movements) >= 50:
            returns = price_movements.mean() * 365
            volatility = price_movements.std() * np.sqrt(365)
            sharpe_ratio = returns / volatility if volatility > 0 else 0
            
            if sharpe_ratio > 1.0:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.PHILOSOPHICAL,
                    premise=f"Sharpe ratio is {sharpe_ratio:.2f}, indicating strong risk-adjusted returns",
                    conclusion="Market offers favorable risk-return profile - consider higher allocation",
                    evidence={"sharpe_ratio": sharpe_ratio, "returns": returns, "volatility": volatility},
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
            elif sharpe_ratio < 0:
                nodes.append(ReasoningNode(
                    level=ReasoningLevel.PHILOSOPHICAL,
                    premise=f"Sharpe ratio is {sharpe_ratio:.2f}, indicating poor risk-adjusted returns",
                    conclusion="Market offers unfavorable risk-return profile - consider reduced exposure",
                    evidence={"sharpe_ratio": sharpe_ratio, "returns": returns, "volatility": volatility},
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
        
        # Complexity and Emergence Analysis
        reasoning_complexity = len(all_nodes)
        conflicting_conclusions = self._count_conflicting_conclusions(all_nodes)
        
        if conflicting_conclusions > reasoning_complexity * 0.3:
            nodes.append(ReasoningNode(
                level=ReasoningLevel.PHILOSOPHICAL,
                premise=f"High proportion of conflicting signals ({conflicting_conclusions}/{reasoning_complexity})",
                conclusion="Market exhibits complex emergent behavior - simple strategies likely insufficient",
                evidence={"complexity_ratio": conflicting_conclusions/reasoning_complexity, "total_signals": reasoning_complexity},
                confidence=0.7,
                timestamp=datetime.now()
            ))
        
        return nodes
    
    def _count_conflicting_conclusions(self, nodes: List[ReasoningNode]) -> int:
        """Count nodes with conflicting conclusions."""
        bullish_count = sum(1 for node in nodes if any(word in node.conclusion.lower() 
                           for word in ['bullish', 'bull', 'up', 'positive', 'strong', 'buy']))
        bearish_count = sum(1 for node in nodes if any(word in node.conclusion.lower() 
                           for word in ['bearish', 'bear', 'down', 'negative', 'weak', 'sell']))
        
        return min(bullish_count, bearish_count)
    
    def _synthesize_conclusions(self, nodes: List[ReasoningNode]) -> Tuple[str, float]:
        """Synthesize all reasoning nodes into a final conclusion."""
        if not nodes:
            return "Insufficient data for reasoning", 0.1
        
        # Weight nodes by level and confidence
        level_weights = {
            ReasoningLevel.SURFACE: 0.1,
            ReasoningLevel.TACTICAL: 0.2,
            ReasoningLevel.STRATEGIC: 0.3,
            ReasoningLevel.META: 0.3,
            ReasoningLevel.PHILOSOPHICAL: 0.1
        }
        
        weighted_conclusions = []
        total_weight = 0
        
        for node in nodes:
            weight = level_weights[node.level] * node.confidence
            weighted_conclusions.append({
                'conclusion': node.conclusion,
                'weight': weight,
                'level': node.level.name
            })
            total_weight += weight
        
        # Analyze sentiment
        bullish_weight = sum(w['weight'] for w in weighted_conclusions 
                           if any(word in w['conclusion'].lower() 
                                 for word in ['bullish', 'bull', 'up', 'positive', 'strong']))
        
        bearish_weight = sum(w['weight'] for w in weighted_conclusions 
                           if any(word in w['conclusion'].lower() 
                                 for word in ['bearish', 'bear', 'down', 'negative', 'weak']))
        
        neutral_weight = total_weight - bullish_weight - bearish_weight
        
        # Determine overall sentiment
        if bullish_weight > bearish_weight * 1.2:
            sentiment = "BULLISH"
            confidence = min(0.9, bullish_weight / total_weight + 0.1)
        elif bearish_weight > bullish_weight * 1.2:
            sentiment = "BEARISH"
            confidence = min(0.9, bearish_weight / total_weight + 0.1)
        else:
            sentiment = "NEUTRAL"
            confidence = max(0.3, neutral_weight / total_weight)
        
        # Create comprehensive conclusion
        conclusion = f"UltraThink Analysis: {sentiment} sentiment with {confidence:.1%} confidence. "
        conclusion += f"Based on {len(nodes)} reasoning nodes across {len(set(node.level for node in nodes))} levels of analysis. "
        
        # Add key insights
        high_confidence_insights = [node.conclusion for node in nodes if node.confidence > 0.7]
        if high_confidence_insights:
            conclusion += f"Key insights: {'; '.join(high_confidence_insights[:3])}"
        
        return conclusion, confidence
    
    def get_reasoning_summary(self, chain_id: str = None) -> Dict[str, Any]:
        """Get a summary of reasoning for a specific chain or the latest."""
        if chain_id:
            chain = next((c for c in self.completed_chains if c.chain_id == chain_id), None)
        else:
            chain = self.completed_chains[-1] if self.completed_chains else None
        
        if not chain:
            return {"error": "No reasoning chain found"}
        
        return {
            "chain_id": chain.chain_id,
            "timestamp": chain.created_at.isoformat(),
            "final_conclusion": chain.final_conclusion,
            "overall_confidence": chain.overall_confidence,
            "reasoning_levels": len(set(node.level for node in chain.nodes)),
            "total_nodes": len(chain.nodes),
            "nodes_by_level": {
                level.name: len([n for n in chain.nodes if n.level == level])
                for level in ReasoningLevel
            },
            "key_insights": [
                {
                    "level": node.level.name,
                    "premise": node.premise,
                    "conclusion": node.conclusion,
                    "confidence": node.confidence
                }
                for node in chain.nodes if node.confidence > 0.7
            ]
        }


class SurfaceReasoner:
    """Handles surface-level technical analysis reasoning."""
    pass

class TacticalReasoner:
    """Handles tactical-level pattern recognition reasoning."""
    pass

class StrategicReasoner:
    """Handles strategic-level market dynamics reasoning."""
    pass

class MetaReasoner:
    """Handles meta-level market regime reasoning."""
    pass

class PhilosophicalReasoner:
    """Handles philosophical-level market understanding reasoning."""
    pass