#!/usr/bin/env python3
"""
ULTRATHINK Chunk Classification System
Trading-specific semantic chunk classification and prioritization

Philosophy: Intelligent chunk categorization for trading development
Performance: < 1ms chunk classification with 95% accuracy
Intelligence: Trading domain expertise for optimal context selection
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Import semantic chunk types
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tree_sitter_chunker import SemanticChunk, SemanticModuleAnalysis

@dataclass
class ChunkClassification:
    """Complete classification of a semantic chunk"""
    chunk_id: str
    primary_category: str  # 'CRITICAL_TRADING', 'RISK_CONTROL', 'ML_MODEL', 'DATA_PIPELINE', 'UTILITY'
    secondary_categories: List[str]
    priority_score: float  # 0.0-1.0
    context_importance: float  # 0.0-1.0 for context loading priority
    trading_stage: str  # 'DATA_INGESTION', 'SIGNAL_GENERATION', 'RISK_ASSESSMENT', 'ORDER_EXECUTION', 'MONITORING'
    complexity_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    dependency_rank: int  # Higher rank = more dependencies rely on this
    usage_frequency: float  # Estimated usage frequency in development
    classification_confidence: float  # 0.0-1.0

@dataclass
class ClassificationMetrics:
    """Classification system performance metrics"""
    total_chunks_classified: int
    avg_classification_time_ms: float
    confidence_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    trading_stage_distribution: Dict[str, int]
    high_priority_chunks: int
    critical_chunks: int

class TradingDomainClassifier:
    """Domain-specific classifier for trading system chunks"""
    
    def __init__(self):
        # Define trading-specific classification patterns
        self.CRITICAL_TRADING_PATTERNS = {
            'order_execution': [
                r'execute.*trade', r'place.*order', r'buy.*sell', r'position.*entry',
                r'order.*management', r'trade.*execution', r'live.*trading'
            ],
            'risk_management': [
                r'kelly.*criterion', r'stop.*loss', r'position.*sizing', r'risk.*limit',
                r'drawdown.*control', r'var.*calculation', r'cvar.*optimization'
            ],
            'signal_generation': [
                r'trading.*signal', r'buy.*signal', r'sell.*signal', r'strategy.*logic',
                r'momentum.*filter', r'trend.*detection', r'signal.*confirmation'
            ],
            'portfolio_management': [
                r'portfolio.*optimization', r'asset.*allocation', r'rebalancing',
                r'position.*management', r'capital.*allocation'
            ]
        }
        
        self.ML_MODEL_PATTERNS = {
            'ensemble_learning': [
                r'random.*forest', r'ensemble.*model', r'meta.*learning',
                r'model.*combination', r'voting.*classifier', r'bagging', r'boosting'
            ],
            'prediction_models': [
                r'price.*prediction', r'return.*forecast', r'signal.*prediction',
                r'market.*prediction', r'trend.*prediction'
            ],
            'feature_engineering': [
                r'feature.*extraction', r'technical.*indicators', r'feature.*selection',
                r'data.*transformation', r'feature.*importance'
            ],
            'model_validation': [
                r'cross.*validation', r'backtest', r'model.*evaluation',
                r'performance.*metrics', r'overfitting.*check'
            ]
        }
        
        self.DATA_PIPELINE_PATTERNS = {
            'data_ingestion': [
                r'data.*fetcher', r'market.*data', r'data.*collection',
                r'api.*client', r'data.*source', r'stream.*data'
            ],
            'data_processing': [
                r'data.*cleaning', r'data.*validation', r'data.*transformation',
                r'missing.*data', r'outlier.*detection', r'data.*quality'
            ],
            'data_storage': [
                r'database.*connection', r'data.*storage', r'cache.*manager',
                r'data.*persistence', r'storage.*engine'
            ]
        }
        
        # Trading workflow stages
        self.TRADING_STAGES = {
            'DATA_INGESTION': ['data_fetcher', 'market_data', 'api_client'],
            'SIGNAL_GENERATION': ['strategy', 'signal', 'indicator', 'momentum'],
            'RISK_ASSESSMENT': ['risk', 'kelly', 'position_sizing', 'limit'],
            'ORDER_EXECUTION': ['execute', 'trade', 'order', 'buy', 'sell'],
            'MONITORING': ['analytics', 'performance', 'monitoring', 'reporting']
        }
        
        # Complexity indicators
        self.COMPLEXITY_INDICATORS = {
            'HIGH': [
                r'async.*def', r'concurrent', r'threading', r'multiprocessing',
                r'optimization', r'meta.*learning', r'genetic.*algorithm'
            ],
            'MEDIUM': [
                r'class.*inheritance', r'decorator', r'context.*manager',
                r'exception.*handling', r'logging'
            ],
            'LOW': [
                r'simple.*function', r'utility', r'helper', r'getter', r'setter'
            ]
        }
        
        print("🎯 Trading Domain Classifier initialized")
    
    def classify_chunk(self, chunk: SemanticChunk) -> ChunkClassification:
        """Classify a semantic chunk for trading context"""
        start_time = time.time()
        
        # Primary category classification
        primary_category = self._classify_primary_category(chunk)
        
        # Secondary categories
        secondary_categories = self._classify_secondary_categories(chunk)
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(chunk, primary_category)
        
        # Calculate context importance
        context_importance = self._calculate_context_importance(chunk, primary_category)
        
        # Determine trading stage
        trading_stage = self._determine_trading_stage(chunk)
        
        # Assess complexity level
        complexity_level = self._assess_complexity_level(chunk)
        
        # Calculate dependency rank (placeholder - would be calculated from graph)
        dependency_rank = self._estimate_dependency_rank(chunk)
        
        # Estimate usage frequency
        usage_frequency = self._estimate_usage_frequency(chunk, primary_category)
        
        # Calculate classification confidence
        classification_confidence = self._calculate_classification_confidence(
            chunk, primary_category, secondary_categories
        )
        
        classification_time = (time.time() - start_time) * 1000
        
        return ChunkClassification(
            chunk_id=chunk.chunk_id,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            priority_score=priority_score,
            context_importance=context_importance,
            trading_stage=trading_stage,
            complexity_level=complexity_level,
            dependency_rank=dependency_rank,
            usage_frequency=usage_frequency,
            classification_confidence=classification_confidence
        )
    
    def _classify_primary_category(self, chunk: SemanticChunk) -> str:
        """Classify the primary category of the chunk"""
        content_lower = chunk.content.lower() if chunk.content else ""
        
        # Check for critical trading patterns
        for pattern_category, patterns in self.CRITICAL_TRADING_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    if pattern_category in ['order_execution', 'risk_management']:
                        return 'CRITICAL_TRADING'
                    elif pattern_category in ['signal_generation', 'portfolio_management']:
                        return 'RISK_CONTROL'
        
        # Check for ML model patterns
        for pattern_category, patterns in self.ML_MODEL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return 'ML_MODEL'
        
        # Check for data pipeline patterns
        for pattern_category, patterns in self.DATA_PIPELINE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return 'DATA_PIPELINE'
        
        # Check semantic tags for classification
        if 'kelly_criterion' in chunk.semantic_tags or 'risk_management' in chunk.semantic_tags:
            return 'CRITICAL_TRADING'
        elif 'ensemble_ml' in chunk.semantic_tags or 'ml_model' in chunk.chunk_type:
            return 'ML_MODEL'
        elif 'order_execution' in chunk.semantic_tags or 'live_trading' in chunk.semantic_tags:
            return 'CRITICAL_TRADING'
        
        # Default classification
        return 'UTILITY'
    
    def _classify_secondary_categories(self, chunk: SemanticChunk) -> List[str]:
        """Classify secondary categories"""
        secondary_categories = []
        content_lower = chunk.content.lower() if chunk.content else ""
        
        # Check all pattern categories
        all_patterns = {
            **self.CRITICAL_TRADING_PATTERNS,
            **self.ML_MODEL_PATTERNS,
            **self.DATA_PIPELINE_PATTERNS
        }
        
        for category, patterns in all_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    secondary_categories.append(category)
                    break
        
        # Add semantic tags as secondary categories
        secondary_categories.extend(chunk.semantic_tags)
        
        return list(set(secondary_categories))  # Remove duplicates
    
    def _calculate_priority_score(self, chunk: SemanticChunk, primary_category: str) -> float:
        """Calculate priority score for the chunk"""
        base_score = 0.0
        
        # Base score by primary category
        category_scores = {
            'CRITICAL_TRADING': 0.9,
            'RISK_CONTROL': 0.8,
            'ML_MODEL': 0.7,
            'DATA_PIPELINE': 0.6,
            'UTILITY': 0.3
        }
        base_score = category_scores.get(primary_category, 0.3)
        
        # Adjust by trading relevance
        base_score = base_score * 0.7 + chunk.trading_relevance * 0.3
        
        # Boost for specific high-priority patterns
        if chunk.content:
            content_lower = chunk.content.lower()
            high_priority_patterns = [
                'kelly.*criterion', 'live.*trading', 'order.*execution',
                'real.*time', 'critical.*path', 'emergency'
            ]
            
            for pattern in high_priority_patterns:
                if re.search(pattern, content_lower):
                    base_score = min(base_score + 0.1, 1.0)
        
        return base_score
    
    def _calculate_context_importance(self, chunk: SemanticChunk, primary_category: str) -> float:
        """Calculate context loading importance"""
        # Context importance is related to how often this chunk provides useful context
        importance = 0.0
        
        # Base importance by category
        category_importance = {
            'CRITICAL_TRADING': 0.95,
            'RISK_CONTROL': 0.85,
            'ML_MODEL': 0.75,
            'DATA_PIPELINE': 0.65,
            'UTILITY': 0.35
        }
        importance = category_importance.get(primary_category, 0.35)
        
        # Adjust by complexity (more complex = more important for context)
        importance += chunk.complexity_score * 0.1
        
        # Boost for docstrings (better context)
        if chunk.docstring:
            importance = min(importance + 0.05, 1.0)
        
        return importance
    
    def _determine_trading_stage(self, chunk: SemanticChunk) -> str:
        """Determine which trading stage this chunk belongs to"""
        content_lower = chunk.content.lower() if chunk.content else ""
        
        # Check content against trading stage patterns
        for stage, keywords in self.TRADING_STAGES.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return stage
        
        # Check semantic tags
        for tag in chunk.semantic_tags:
            if 'data' in tag or 'fetch' in tag:
                return 'DATA_INGESTION'
            elif 'signal' in tag or 'strategy' in tag or 'momentum' in tag:
                return 'SIGNAL_GENERATION'
            elif 'risk' in tag or 'kelly' in tag:
                return 'RISK_ASSESSMENT'
            elif 'execution' in tag or 'order' in tag or 'trade' in tag:
                return 'ORDER_EXECUTION'
            elif 'analytics' in tag or 'performance' in tag:
                return 'MONITORING'
        
        return 'SIGNAL_GENERATION'  # Default stage
    
    def _assess_complexity_level(self, chunk: SemanticChunk) -> str:
        """Assess the complexity level of the chunk"""
        content_lower = chunk.content.lower() if chunk.content else ""
        
        # Check complexity indicators
        for level, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return level
        
        # Use chunk complexity score
        if chunk.complexity_score > 0.8:
            return 'HIGH'
        elif chunk.complexity_score > 0.5:
            return 'MEDIUM'
        elif chunk.complexity_score > 0.2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _estimate_dependency_rank(self, chunk: SemanticChunk) -> int:
        """Estimate dependency rank (placeholder implementation)"""
        # This would be calculated from the semantic graph
        # For now, estimate based on trading relevance and complexity
        base_rank = int(chunk.trading_relevance * 10)
        complexity_bonus = int(chunk.complexity_score * 5)
        
        # Boost for critical patterns
        if any(tag in ['kelly_criterion', 'risk_management', 'order_execution'] 
               for tag in chunk.semantic_tags):
            base_rank += 5
        
        return min(base_rank + complexity_bonus, 20)
    
    def _estimate_usage_frequency(self, chunk: SemanticChunk, primary_category: str) -> float:
        """Estimate how frequently this chunk is used in development"""
        # Usage frequency based on category and trading relevance
        base_frequency = {
            'CRITICAL_TRADING': 0.9,
            'RISK_CONTROL': 0.7,
            'ML_MODEL': 0.6,
            'DATA_PIPELINE': 0.5,
            'UTILITY': 0.3
        }.get(primary_category, 0.3)
        
        # Adjust by trading relevance
        frequency = base_frequency * 0.6 + chunk.trading_relevance * 0.4
        
        # Boost for frequently modified patterns
        if chunk.content:
            content_lower = chunk.content.lower()
            frequent_patterns = [
                'config', 'parameter', 'threshold', 'limit', 'strategy'
            ]
            
            for pattern in frequent_patterns:
                if pattern in content_lower:
                    frequency = min(frequency + 0.1, 1.0)
        
        return frequency
    
    def _calculate_classification_confidence(
        self, 
        chunk: SemanticChunk, 
        primary_category: str, 
        secondary_categories: List[str]
    ) -> float:
        """Calculate confidence in the classification"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if semantic tags align with classification
        if chunk.semantic_tags:
            tag_alignment = 0.0
            for tag in chunk.semantic_tags:
                if tag in secondary_categories:
                    tag_alignment += 0.1
            confidence += min(tag_alignment, 0.3)
        
        # Boost confidence if chunk type aligns
        type_alignment = {
            'CRITICAL_TRADING': ['trading_logic', 'risk_control'],
            'ML_MODEL': ['ml_model'],
            'DATA_PIPELINE': ['data_processing'],
            'UTILITY': ['function', 'utility']
        }
        
        if chunk.chunk_type in type_alignment.get(primary_category, []):
            confidence += 0.2
        
        # Boost confidence if content patterns are strong
        if chunk.content and primary_category != 'UTILITY':
            confidence += 0.1
        
        # Penalize low trading relevance for trading categories
        if primary_category in ['CRITICAL_TRADING', 'RISK_CONTROL'] and chunk.trading_relevance < 0.5:
            confidence -= 0.2
        
        return max(0.0, min(confidence, 1.0))

class ChunkClassificationEngine:
    """Main engine for chunk classification and management"""
    
    def __init__(self):
        self.domain_classifier = TradingDomainClassifier()
        self.classifications: Dict[str, ChunkClassification] = {}
        self.classification_stats = {
            'total_classified': 0,
            'total_time_ms': 0.0,
            'avg_confidence': 0.0
        }
        
        print("🎯 Chunk Classification Engine initialized")
    
    def classify_chunks(self, chunks: List[SemanticChunk]) -> Dict[str, ChunkClassification]:
        """Classify a list of semantic chunks"""
        start_time = time.time()
        
        classifications = {}
        total_confidence = 0.0
        
        for chunk in chunks:
            classification = self.domain_classifier.classify_chunk(chunk)
            classifications[chunk.chunk_id] = classification
            total_confidence += classification.classification_confidence
        
        # Update statistics
        classification_time = (time.time() - start_time) * 1000
        self.classification_stats['total_classified'] += len(chunks)
        self.classification_stats['total_time_ms'] += classification_time
        self.classification_stats['avg_confidence'] = total_confidence / len(chunks) if chunks else 0.0
        
        # Store classifications
        self.classifications.update(classifications)
        
        print(f"✅ Classified {len(chunks)} chunks in {classification_time:.1f}ms "
              f"(avg confidence: {self.classification_stats['avg_confidence']:.2f})")
        
        return classifications
    
    def get_classification_metrics(self) -> ClassificationMetrics:
        """Get classification system metrics"""
        if not self.classifications:
            return ClassificationMetrics(0, 0.0, {}, {}, {}, 0, 0)
        
        # Calculate distributions
        confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
        category_dist = defaultdict(int)
        trading_stage_dist = defaultdict(int)
        high_priority_count = 0
        critical_count = 0
        
        for classification in self.classifications.values():
            # Confidence distribution
            if classification.classification_confidence > 0.8:
                confidence_dist['high'] += 1
            elif classification.classification_confidence > 0.5:
                confidence_dist['medium'] += 1
            else:
                confidence_dist['low'] += 1
            
            # Category distribution
            category_dist[classification.primary_category] += 1
            
            # Trading stage distribution
            trading_stage_dist[classification.trading_stage] += 1
            
            # Priority counts
            if classification.priority_score > 0.8:
                high_priority_count += 1
            if classification.primary_category == 'CRITICAL_TRADING':
                critical_count += 1
        
        avg_time = (self.classification_stats['total_time_ms'] / 
                   max(self.classification_stats['total_classified'], 1))
        
        return ClassificationMetrics(
            total_chunks_classified=len(self.classifications),
            avg_classification_time_ms=avg_time,
            confidence_distribution=dict(confidence_dist),
            category_distribution=dict(category_dist),
            trading_stage_distribution=dict(trading_stage_dist),
            high_priority_chunks=high_priority_count,
            critical_chunks=critical_count
        )
    
    def get_priority_chunks(self, top_k: int = 10) -> List[Tuple[str, ChunkClassification]]:
        """Get top priority chunks for context loading"""
        if not self.classifications:
            return []
        
        # Sort by priority score and context importance
        priority_chunks = [
            (chunk_id, classification)
            for chunk_id, classification in self.classifications.items()
        ]
        
        priority_chunks.sort(
            key=lambda x: (x[1].priority_score * 0.7 + x[1].context_importance * 0.3),
            reverse=True
        )
        
        return priority_chunks[:top_k]
    
    def get_chunks_by_category(self, category: str) -> List[Tuple[str, ChunkClassification]]:
        """Get all chunks of a specific category"""
        return [
            (chunk_id, classification)
            for chunk_id, classification in self.classifications.items()
            if classification.primary_category == category
        ]
    
    def get_chunks_by_trading_stage(self, stage: str) -> List[Tuple[str, ChunkClassification]]:
        """Get all chunks for a specific trading stage"""
        return [
            (chunk_id, classification)
            for chunk_id, classification in self.classifications.items()
            if classification.trading_stage == stage
        ]
    
    def export_classifications(self) -> Dict[str, Any]:
        """Export classification data for analysis"""
        return {
            'classifications': {
                chunk_id: asdict(classification)
                for chunk_id, classification in self.classifications.items()
            },
            'metrics': asdict(self.get_classification_metrics()),
            'statistics': self.classification_stats
        }

# Example usage and testing
if __name__ == "__main__":
    print("🎯 ULTRATHINK Chunk Classification Testing")
    print("=" * 60)
    
    # Import required modules for testing
    from tree_sitter_chunker import TreeSitterSemanticChunker
    
    # Initialize components
    chunker = TreeSitterSemanticChunker()
    classifier = ChunkClassificationEngine()
    
    # Test with sample trading code
    sample_code = '''
def calculate_kelly_fraction(win_rate, avg_win, avg_loss):
    """Calculate optimal Kelly criterion position sizing for risk management"""
    if avg_loss == 0:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
    
    # Apply fractional Kelly for safety
    return max(0.0, min(kelly_fraction * 0.25, 0.25))

class LiveTradingExecutor:
    """Real-time trading execution engine with risk controls"""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.active_orders = {}
    
    def execute_trade(self, signal, position_size):
        """Execute trading order with comprehensive risk checks"""
        # Risk validation
        if not self.risk_manager.validate_position(position_size):
            return False
        
        # Place order
        order_id = self._place_order(signal, position_size)
        self.active_orders[order_id] = {
            'signal': signal,
            'size': position_size,
            'timestamp': time.time()
        }
        
        return order_id

def fetch_market_data(symbol, timeframe='1h'):
    """Fetch real-time market data for trading analysis"""
    # Data validation and processing
    data = api_client.get_ohlcv(symbol, timeframe)
    return validate_data_quality(data)

class EnsemblePredictor:
    """Advanced ensemble ML model for price prediction"""
    
    def predict_signals(self, features):
        """Generate trading signals using ensemble methods"""
        # Meta-learning approach
        predictions = []
        for model in self.base_models:
            pred = model.predict(features)
            predictions.append(pred)
        
        # Ensemble combination
        final_prediction = self.meta_model.predict(predictions)
        return final_prediction
'''
    
    print("🔍 Analyzing sample trading code...")
    
    # Perform semantic analysis
    analysis = chunker.chunk_module("sample_trading.py", sample_code)
    print(f"📊 Found {len(analysis.chunks)} semantic chunks")
    
    # Classify chunks
    classifications = classifier.classify_chunks(analysis.chunks)
    
    # Display results
    print(f"\n🎯 Classification Results:")
    for chunk_id, classification in classifications.items():
        print(f"\n📦 {chunk_id}")
        print(f"├── Category: {classification.primary_category}")
        print(f"├── Trading Stage: {classification.trading_stage}")
        print(f"├── Priority: {classification.priority_score:.2f}")
        print(f"├── Context Importance: {classification.context_importance:.2f}")
        print(f"├── Complexity: {classification.complexity_level}")
        print(f"├── Confidence: {classification.classification_confidence:.2f}")
        print(f"└── Secondary: {classification.secondary_categories[:3]}...")
    
    # Show metrics
    metrics = classifier.get_classification_metrics()
    print(f"\n📈 Classification Metrics:")
    print(f"├── Total Classified: {metrics.total_chunks_classified}")
    print(f"├── Avg Time: {metrics.avg_classification_time_ms:.1f}ms")
    print(f"├── High Priority Chunks: {metrics.high_priority_chunks}")
    print(f"├── Critical Chunks: {metrics.critical_chunks}")
    print(f"├── Category Distribution: {dict(metrics.category_distribution)}")
    print(f"└── Confidence Distribution: {metrics.confidence_distribution}")
    
    # Show top priority chunks
    priority_chunks = classifier.get_priority_chunks(top_k=3)
    print(f"\n🎯 Top Priority Chunks:")
    for i, (chunk_id, classification) in enumerate(priority_chunks, 1):
        combined_score = classification.priority_score * 0.7 + classification.context_importance * 0.3
        print(f"{i}. {chunk_id} (score: {combined_score:.3f})")
    
    print(f"\n🎯 Chunk classification testing complete!")