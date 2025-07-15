#!/usr/bin/env python3
"""
ULTRATHINK CGRAG Coarse Retrieval Engine
First stage of two-stage intelligent context retrieval

Philosophy: Cast a wide net to capture all potentially relevant context
Performance: < 50ms coarse filtering across 111 modules
Intelligence: Trading-aware broad filtering with semantic understanding
"""

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Import semantic components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic"))
from semantic_integration_manager import SemanticIntegrationManager, SemanticContextBundle
from tree_sitter_chunker import SemanticChunk, SemanticModuleAnalysis
from chunk_classifier import ChunkClassification

@dataclass
class CoarseRetrievalCandidate:
    """Represents a candidate from coarse retrieval"""
    chunk_id: str
    module_path: str
    relevance_score: float
    evidence: List[str]  # Why this chunk is relevant
    semantic_tags: List[str]
    chunk_type: str
    trading_stage: str
    priority_category: str
    content_preview: str  # First few lines for quick assessment

@dataclass
class CoarseRetrievalResult:
    """Result of coarse retrieval stage"""
    query: str
    candidates: List[CoarseRetrievalCandidate]
    total_candidates_found: int
    filtering_time_ms: float
    semantic_patterns_detected: List[str]
    trading_workflows_identified: List[str]
    confidence_score: float
    next_stage_recommendations: Dict[str, Any]

class TradingQueryAnalyzer:
    """Analyzes queries for trading-specific patterns and intent"""
    
    def __init__(self):
        # Define trading query patterns
        self.QUERY_PATTERNS = {
            'kelly_criterion': {
                'keywords': ['kelly', 'criterion', 'position sizing', 'optimal size', 'bet size'],
                'related_concepts': ['risk management', 'portfolio optimization', 'position limits'],
                'trading_stage': 'RISK_ASSESSMENT',
                'priority': 'HIGH'
            },
            'risk_management': {
                'keywords': ['risk', 'stop loss', 'position limit', 'drawdown', 'var', 'cvar'],
                'related_concepts': ['portfolio protection', 'loss control', 'risk control'],
                'trading_stage': 'RISK_ASSESSMENT',
                'priority': 'CRITICAL'
            },
            'ensemble_ml': {
                'keywords': ['ensemble', 'random forest', 'meta learning', 'model combination'],
                'related_concepts': ['prediction', 'machine learning', 'model validation'],
                'trading_stage': 'SIGNAL_GENERATION',
                'priority': 'HIGH'
            },
            'order_execution': {
                'keywords': ['order', 'execution', 'buy', 'sell', 'trade', 'position entry'],
                'related_concepts': ['order management', 'trade execution', 'market orders'],
                'trading_stage': 'ORDER_EXECUTION',
                'priority': 'CRITICAL'
            },
            'live_trading': {
                'keywords': ['live', 'real time', 'paper trading', '24 hour', 'continuous'],
                'related_concepts': ['automated trading', 'live session', 'real-time processing'],
                'trading_stage': 'ORDER_EXECUTION',
                'priority': 'CRITICAL'
            },
            'momentum_strategy': {
                'keywords': ['momentum', 'trend', 'signal', 'strategy', 'indicator'],
                'related_concepts': ['technical analysis', 'signal generation', 'trend following'],
                'trading_stage': 'SIGNAL_GENERATION',
                'priority': 'HIGH'
            },
            'data_processing': {
                'keywords': ['data', 'fetch', 'process', 'clean', 'validate'],
                'related_concepts': ['data pipeline', 'market data', 'data quality'],
                'trading_stage': 'DATA_INGESTION',
                'priority': 'MEDIUM'
            },
            'performance_analytics': {
                'keywords': ['performance', 'analytics', 'metrics', 'backtest', 'evaluation'],
                'related_concepts': ['performance monitoring', 'strategy evaluation', 'analytics'],
                'trading_stage': 'MONITORING',
                'priority': 'MEDIUM'
            }
        }
        
        # Query intent classification patterns
        self.INTENT_PATTERNS = {
            'HOW_DOES_X_WORK': r'how\s+does\s+\w+\s+work',
            'WHAT_IS_RELATIONSHIP': r'what.*relationship.*between',
            'HOW_TO_IMPLEMENT': r'how\s+to\s+implement',
            'OPTIMIZATION_QUERY': r'optim\w+|improv\w+|enhance\w+',
            'CONFIGURATION_QUERY': r'config\w+|setting\w+|parameter\w+',
            'TROUBLESHOOTING_QUERY': r'error\w+|problem\w+|issue\w+|fix\w+',
            'EXPLANATION_QUERY': r'explain\w+|describe\w+|understand\w+'
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for trading patterns and intent"""
        query_lower = query.lower()
        
        # Detect trading patterns
        detected_patterns = []
        related_concepts = set()
        trading_stages = set()
        priority_levels = []
        
        for pattern_name, pattern_info in self.QUERY_PATTERNS.items():
            # Check keywords
            keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                if keyword in query_lower)
            
            if keyword_matches > 0:
                detected_patterns.append(pattern_name)
                related_concepts.update(pattern_info['related_concepts'])
                trading_stages.add(pattern_info['trading_stage'])
                priority_levels.append(pattern_info['priority'])
        
        # Detect query intent
        detected_intent = []
        for intent_name, intent_pattern in self.INTENT_PATTERNS.items():
            if re.search(intent_pattern, query_lower):
                detected_intent.append(intent_name)
        
        # Determine primary focus
        primary_focus = detected_patterns[0] if detected_patterns else 'general_trading'
        
        # Calculate query specificity
        query_words = len(query.split())
        specificity_score = min(query_words / 10.0, 1.0)  # Normalize by expected length
        
        return {
            'detected_patterns': detected_patterns,
            'related_concepts': list(related_concepts),
            'trading_stages': list(trading_stages),
            'priority_levels': priority_levels,
            'detected_intent': detected_intent,
            'primary_focus': primary_focus,
            'specificity_score': specificity_score,
            'query_complexity': 'HIGH' if len(detected_patterns) > 2 else 'MEDIUM' if detected_patterns else 'LOW'
        }

class CoarseRetrievalEngine:
    """First stage of CGRAG: broad context filtering"""
    
    def __init__(self, semantic_manager: SemanticIntegrationManager):
        self.semantic_manager = semantic_manager
        self.query_analyzer = TradingQueryAnalyzer()
        
        # Retrieval configuration
        self.DEFAULT_CANDIDATE_LIMIT = 20
        self.MIN_RELEVANCE_THRESHOLD = 0.1
        self.SEMANTIC_BOOST_FACTOR = 1.5
        self.PRIORITY_BOOST_FACTOR = 2.0
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_processing_time_ms': 0.0,
            'avg_candidates_found': 0.0,
            'cache_hits': 0
        }
        
        print("🔍 CGRAG Coarse Retrieval Engine initialized")
    
    def retrieve_coarse_candidates(self, query: str, max_candidates: int = None) -> CoarseRetrievalResult:
        """Perform coarse retrieval to identify broad candidate chunks"""
        start_time = time.time()
        self.retrieval_stats['total_queries'] += 1
        
        max_candidates = max_candidates or self.DEFAULT_CANDIDATE_LIMIT
        
        # Analyze query for trading patterns
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Ensure semantic analysis is available
        if not self.semantic_manager.semantic_cache:
            self.semantic_manager.analyze_project_semantics()
        
        # Find candidate chunks using multiple strategies
        candidates = []
        
        # Strategy 1: Semantic tag matching
        semantic_candidates = self._find_semantic_candidates(query, query_analysis)
        candidates.extend(semantic_candidates)
        
        # Strategy 2: Content keyword matching
        keyword_candidates = self._find_keyword_candidates(query, query_analysis)
        candidates.extend(keyword_candidates)
        
        # Strategy 3: Trading workflow matching
        workflow_candidates = self._find_workflow_candidates(query, query_analysis)
        candidates.extend(workflow_candidates)
        
        # Strategy 4: Classification-based matching
        classification_candidates = self._find_classification_candidates(query, query_analysis)
        candidates.extend(classification_candidates)
        
        # Remove duplicates and score candidates
        unique_candidates = self._deduplicate_and_score(candidates, query_analysis)
        
        # Filter by relevance threshold
        filtered_candidates = [c for c in unique_candidates if c.relevance_score >= self.MIN_RELEVANCE_THRESHOLD]
        
        # Sort by relevance and limit
        filtered_candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        final_candidates = filtered_candidates[:max_candidates]
        
        # Identify patterns and workflows
        semantic_patterns = query_analysis['detected_patterns']
        trading_workflows = self._identify_trading_workflows(final_candidates)
        
        # Calculate confidence
        confidence_score = self._calculate_coarse_confidence(final_candidates, query_analysis)
        
        # Generate recommendations for fine retrieval
        next_stage_recommendations = self._generate_fine_stage_recommendations(final_candidates, query_analysis)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.retrieval_stats['avg_processing_time_ms'] = (
            (self.retrieval_stats['avg_processing_time_ms'] * (self.retrieval_stats['total_queries'] - 1) + processing_time) /
            self.retrieval_stats['total_queries']
        )
        self.retrieval_stats['avg_candidates_found'] = (
            (self.retrieval_stats['avg_candidates_found'] * (self.retrieval_stats['total_queries'] - 1) + len(final_candidates)) /
            self.retrieval_stats['total_queries']
        )
        
        return CoarseRetrievalResult(
            query=query,
            candidates=final_candidates,
            total_candidates_found=len(filtered_candidates),
            filtering_time_ms=processing_time,
            semantic_patterns_detected=semantic_patterns,
            trading_workflows_identified=trading_workflows,
            confidence_score=confidence_score,
            next_stage_recommendations=next_stage_recommendations
        )
    
    def _find_semantic_candidates(self, query: str, query_analysis: Dict[str, Any]) -> List[CoarseRetrievalCandidate]:
        """Find candidates based on semantic tags"""
        candidates = []
        
        # Get detected patterns and related concepts
        target_patterns = set(query_analysis['detected_patterns'] + query_analysis['related_concepts'])
        
        for module_path, analysis in self.semantic_manager.semantic_cache.items():
            for chunk in analysis.chunks:
                # Check semantic tag overlap
                tag_overlap = set(chunk.semantic_tags) & target_patterns
                
                if tag_overlap:
                    relevance_score = len(tag_overlap) * 0.3 + chunk.trading_relevance * 0.4
                    
                    # Boost for exact pattern matches
                    if any(pattern in chunk.semantic_tags for pattern in query_analysis['detected_patterns']):
                        relevance_score *= self.SEMANTIC_BOOST_FACTOR
                    
                    candidate = CoarseRetrievalCandidate(
                        chunk_id=chunk.chunk_id,
                        module_path=module_path,
                        relevance_score=relevance_score,
                        evidence=[f"Semantic tags: {list(tag_overlap)}"],
                        semantic_tags=chunk.semantic_tags,
                        chunk_type=chunk.chunk_type,
                        trading_stage=self._infer_trading_stage(chunk),
                        priority_category='HIGH' if relevance_score > 0.8 else 'MEDIUM',
                        content_preview=self._get_content_preview(chunk)
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _find_keyword_candidates(self, query: str, query_analysis: Dict[str, Any]) -> List[CoarseRetrievalCandidate]:
        """Find candidates based on keyword matching in content"""
        candidates = []
        query_words = set(query.lower().split())
        
        # Expand with related concepts
        all_keywords = query_words.copy()
        for concept in query_analysis['related_concepts']:
            all_keywords.update(concept.lower().split())
        
        for module_path, analysis in self.semantic_manager.semantic_cache.items():
            for chunk in analysis.chunks:
                if chunk.content:
                    content_lower = chunk.content.lower()
                    
                    # Count keyword matches
                    keyword_matches = sum(1 for keyword in all_keywords if keyword in content_lower)
                    
                    if keyword_matches > 0:
                        relevance_score = (keyword_matches / len(all_keywords)) * 0.5 + chunk.trading_relevance * 0.3
                        
                        candidate = CoarseRetrievalCandidate(
                            chunk_id=chunk.chunk_id,
                            module_path=module_path,
                            relevance_score=relevance_score,
                            evidence=[f"Keyword matches: {keyword_matches}/{len(all_keywords)}"],
                            semantic_tags=chunk.semantic_tags,
                            chunk_type=chunk.chunk_type,
                            trading_stage=self._infer_trading_stage(chunk),
                            priority_category='MEDIUM',
                            content_preview=self._get_content_preview(chunk)
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _find_workflow_candidates(self, query: str, query_analysis: Dict[str, Any]) -> List[CoarseRetrievalCandidate]:
        """Find candidates based on trading workflow patterns"""
        candidates = []
        
        # Focus on trading stages mentioned in query analysis
        target_stages = set(query_analysis['trading_stages'])
        
        if target_stages:
            for module_path, module_classifications in self.semantic_manager.classification_cache.items():
                for chunk_id, classification in module_classifications.items():
                    if classification.trading_stage in target_stages:
                        relevance_score = classification.priority_score * 0.6 + classification.context_importance * 0.4
                        
                        # Find the corresponding chunk
                        chunk = self._find_chunk_by_id(chunk_id)
                        if chunk:
                            candidate = CoarseRetrievalCandidate(
                                chunk_id=chunk_id,
                                module_path=module_path,
                                relevance_score=relevance_score,
                                evidence=[f"Trading stage: {classification.trading_stage}"],
                                semantic_tags=chunk.semantic_tags,
                                chunk_type=chunk.chunk_type,
                                trading_stage=classification.trading_stage,
                                priority_category=classification.primary_category,
                                content_preview=self._get_content_preview(chunk)
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _find_classification_candidates(self, query: str, query_analysis: Dict[str, Any]) -> List[CoarseRetrievalCandidate]:
        """Find candidates based on chunk classifications"""
        candidates = []
        
        # Determine target categories based on query patterns
        target_categories = set()
        for pattern in query_analysis['detected_patterns']:
            if pattern in ['kelly_criterion', 'risk_management']:
                target_categories.add('CRITICAL_TRADING')
                target_categories.add('RISK_CONTROL')
            elif pattern == 'ensemble_ml':
                target_categories.add('ML_MODEL')
            elif pattern in ['order_execution', 'live_trading']:
                target_categories.add('CRITICAL_TRADING')
            elif pattern == 'data_processing':
                target_categories.add('DATA_PIPELINE')
        
        # Default to high-priority categories if no specific patterns detected
        if not target_categories:
            target_categories = {'CRITICAL_TRADING', 'RISK_CONTROL'}
        
        for module_path, module_classifications in self.semantic_manager.classification_cache.items():
            for chunk_id, classification in module_classifications.items():
                if classification.primary_category in target_categories:
                    relevance_score = classification.priority_score
                    
                    # Boost for high-priority classifications
                    if classification.primary_category == 'CRITICAL_TRADING':
                        relevance_score *= self.PRIORITY_BOOST_FACTOR
                    
                    chunk = self._find_chunk_by_id(chunk_id)
                    if chunk:
                        candidate = CoarseRetrievalCandidate(
                            chunk_id=chunk_id,
                            module_path=module_path,
                            relevance_score=relevance_score,
                            evidence=[f"Category: {classification.primary_category}"],
                            semantic_tags=chunk.semantic_tags,
                            chunk_type=chunk.chunk_type,
                            trading_stage=classification.trading_stage,
                            priority_category=classification.primary_category,
                            content_preview=self._get_content_preview(chunk)
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _deduplicate_and_score(self, candidates: List[CoarseRetrievalCandidate], query_analysis: Dict[str, Any]) -> List[CoarseRetrievalCandidate]:
        """Remove duplicates and adjust scoring"""
        # Group by chunk_id and merge evidence
        unique_candidates = {}
        
        for candidate in candidates:
            if candidate.chunk_id in unique_candidates:
                # Merge evidence and take highest score
                existing = unique_candidates[candidate.chunk_id]
                existing.evidence.extend(candidate.evidence)
                existing.relevance_score = max(existing.relevance_score, candidate.relevance_score)
            else:
                unique_candidates[candidate.chunk_id] = candidate
        
        # Apply final scoring adjustments
        for candidate in unique_candidates.values():
            # Boost for query specificity alignment
            if query_analysis['specificity_score'] > 0.7 and candidate.priority_category == 'CRITICAL_TRADING':
                candidate.relevance_score *= 1.2
            
            # Boost for complex queries with matching complexity
            if query_analysis['query_complexity'] == 'HIGH' and len(candidate.semantic_tags) > 2:
                candidate.relevance_score *= 1.1
            
            # Normalize score
            candidate.relevance_score = min(candidate.relevance_score, 1.0)
        
        return list(unique_candidates.values())
    
    def _infer_trading_stage(self, chunk: SemanticChunk) -> str:
        """Infer trading stage from chunk semantic tags"""
        for tag in chunk.semantic_tags:
            if tag in ['data_ingestion', 'data_processing']:
                return 'DATA_INGESTION'
            elif tag in ['kelly_criterion', 'risk_management']:
                return 'RISK_ASSESSMENT'
            elif tag in ['momentum_strategy', 'ensemble_ml']:
                return 'SIGNAL_GENERATION'
            elif tag in ['order_execution', 'live_trading']:
                return 'ORDER_EXECUTION'
            elif tag in ['performance_analytics']:
                return 'MONITORING'
        
        return 'SIGNAL_GENERATION'  # Default
    
    def _find_chunk_by_id(self, chunk_id: str) -> Optional[SemanticChunk]:
        """Find chunk by ID across all modules"""
        for analysis in self.semantic_manager.semantic_cache.values():
            for chunk in analysis.chunks:
                if chunk.chunk_id == chunk_id:
                    return chunk
        return None
    
    def _get_content_preview(self, chunk: SemanticChunk) -> str:
        """Get content preview for quick assessment"""
        if chunk.content:
            lines = chunk.content.split('\n')
            # Take first 3 meaningful lines (not empty or just comments)
            meaningful_lines = []
            for line in lines[:10]:  # Look at first 10 lines
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                    meaningful_lines.append(stripped)
                if len(meaningful_lines) >= 3:
                    break
            
            preview = ' | '.join(meaningful_lines)
            return preview[:100] + '...' if len(preview) > 100 else preview
        
        return f"# {chunk.chunk_type} in {chunk.chunk_id}"
    
    def _identify_trading_workflows(self, candidates: List[CoarseRetrievalCandidate]) -> List[str]:
        """Identify trading workflows represented in candidates"""
        workflows = []
        
        # Group by trading stage
        stage_counts = {}
        for candidate in candidates:
            stage = candidate.trading_stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Identify complete workflows
        complete_workflow_stages = ['DATA_INGESTION', 'SIGNAL_GENERATION', 'RISK_ASSESSMENT', 'ORDER_EXECUTION']
        if all(stage in stage_counts for stage in complete_workflow_stages):
            workflows.append('COMPLETE_TRADING_WORKFLOW')
        
        # Identify partial workflows
        if 'SIGNAL_GENERATION' in stage_counts and 'RISK_ASSESSMENT' in stage_counts:
            workflows.append('STRATEGY_RISK_WORKFLOW')
        
        if 'RISK_ASSESSMENT' in stage_counts and 'ORDER_EXECUTION' in stage_counts:
            workflows.append('RISK_EXECUTION_WORKFLOW')
        
        return workflows
    
    def _calculate_coarse_confidence(self, candidates: List[CoarseRetrievalCandidate], query_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in coarse retrieval results"""
        if not candidates:
            return 0.0
        
        # Base confidence from candidate quality
        avg_relevance = sum(c.relevance_score for c in candidates) / len(candidates)
        
        # Boost for pattern alignment
        pattern_alignment = len(query_analysis['detected_patterns']) / 8.0  # Normalize
        
        # Boost for candidate diversity (different modules/stages)
        modules = set(c.module_path for c in candidates)
        stages = set(c.trading_stage for c in candidates)
        diversity_score = (len(modules) + len(stages)) / (len(candidates) + 5)
        
        # Combined confidence
        confidence = avg_relevance * 0.5 + pattern_alignment * 0.3 + diversity_score * 0.2
        
        return min(confidence, 1.0)
    
    def _generate_fine_stage_recommendations(self, candidates: List[CoarseRetrievalCandidate], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for the fine retrieval stage"""
        recommendations = {
            'priority_candidates': [c.chunk_id for c in candidates if c.relevance_score > 0.8],
            'focus_areas': query_analysis['detected_patterns'],
            'suggested_fine_filters': [],
            'content_depth_needed': 'HIGH' if query_analysis['query_complexity'] == 'HIGH' else 'MEDIUM',
            'estimated_fine_candidates': min(len(candidates) * 2, 50)  # Expect 2x expansion in fine stage
        }
        
        # Add specific fine filter suggestions
        if 'kelly_criterion' in query_analysis['detected_patterns']:
            recommendations['suggested_fine_filters'].append('function_implementations')
            recommendations['suggested_fine_filters'].append('mathematical_formulas')
        
        if 'live_trading' in query_analysis['detected_patterns']:
            recommendations['suggested_fine_filters'].append('real_time_processing')
            recommendations['suggested_fine_filters'].append('error_handling')
        
        return recommendations
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics"""
        return self.retrieval_stats.copy()

# Example usage and testing
if __name__ == "__main__":
    print("🔍 CGRAG Coarse Retrieval Engine Testing")
    print("=" * 60)
    
    # Import semantic integration manager for testing
    sys.path.insert(0, str(Path(__file__).parent.parent / "semantic"))
    from semantic_integration_manager import SemanticIntegrationManager
    
    # Initialize components
    print("🧠 Initializing semantic integration manager...")
    semantic_manager = SemanticIntegrationManager()
    
    print("🔍 Initializing coarse retrieval engine...")
    coarse_engine = CoarseRetrievalEngine(semantic_manager)
    
    # Test queries
    test_queries = [
        "How does Kelly criterion optimize position sizing in risk management?",
        "What is the relationship between ensemble models and live trading execution?",
        "How does momentum filtering prevent overtrading in trading strategies?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing coarse retrieval for: {query[:50]}...")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            result = coarse_engine.retrieve_coarse_candidates(query, max_candidates=10)
            
            query_time = (time.time() - start_time) * 1000
            
            print(f"✅ Coarse Retrieval Results:")
            print(f"├── Processing Time: {result.filtering_time_ms:.1f}ms")
            print(f"├── Candidates Found: {len(result.candidates)}")
            print(f"├── Total Candidates: {result.total_candidates_found}")
            print(f"├── Confidence: {result.confidence_score:.2f}")
            print(f"├── Semantic Patterns: {result.semantic_patterns_detected}")
            print(f"├── Trading Workflows: {result.trading_workflows_identified}")
            print(f"└── Next Stage Recs: {len(result.next_stage_recommendations)} items")
            
            # Show top candidates
            print(f"\n🎯 Top Candidates:")
            for i, candidate in enumerate(result.candidates[:3], 1):
                print(f"{i}. {candidate.chunk_id}")
                print(f"   ├── Score: {candidate.relevance_score:.3f}")
                print(f"   ├── Category: {candidate.priority_category}")
                print(f"   ├── Stage: {candidate.trading_stage}")
                print(f"   ├── Evidence: {candidate.evidence[0]}")
                print(f"   └── Preview: {candidate.content_preview}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show overall statistics
    stats = coarse_engine.get_retrieval_stats()
    print(f"\n📊 Coarse Retrieval Statistics:")
    print(f"├── Total Queries: {stats['total_queries']}")
    print(f"├── Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"├── Avg Candidates Found: {stats['avg_candidates_found']:.1f}")
    print(f"└── Cache Hits: {stats['cache_hits']}")
    
    print(f"\n🔍 Coarse retrieval testing complete!")