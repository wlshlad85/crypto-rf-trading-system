#!/usr/bin/env python3
"""
ULTRATHINK Semantic Integration Manager
Coordinates all semantic analysis components with existing context system

Philosophy: Seamless integration of semantic intelligence with Week 2 framework
Performance: Maintain < 10ms impact on existing systems
Intelligence: Enhanced context delivery through semantic understanding
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import all semantic components
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tree_sitter_chunker import TreeSitterSemanticChunker, SemanticChunk, SemanticModuleAnalysis
from semantic_graph import SemanticGraphBuilder, SemanticGraphMetrics
from chunk_classifier import ChunkClassificationEngine, ChunkClassification, ClassificationMetrics

# Import existing context management components
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from context_loader import UltraThinkContextLoader, ModuleInfo, ContextBundle

@dataclass
class SemanticContextBundle:
    """Enhanced context bundle with semantic intelligence"""
    query: str
    primary_semantic_chunks: List[str]  # chunk_ids
    dependency_chunks: List[str]
    chunk_classifications: Dict[str, ChunkClassification]
    semantic_relationships: Dict[str, List[str]]
    trading_workflow_context: str
    context_content: Dict[str, str]  # chunk_id -> content
    semantic_summary: str
    confidence_score: float
    load_time_ms: float
    token_count: int
    semantic_metrics: Dict[str, Any]

@dataclass
class SemanticSystemMetrics:
    """Comprehensive metrics for the semantic system"""
    semantic_analysis_time_ms: float
    classification_time_ms: float
    graph_building_time_ms: float
    context_generation_time_ms: float
    total_processing_time_ms: float
    chunks_analyzed: int
    relationships_discovered: int
    high_priority_chunks: int
    semantic_graph_metrics: SemanticGraphMetrics
    classification_metrics: ClassificationMetrics

class SemanticIntegrationManager:
    """Manages integration of semantic analysis with existing context system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        
        print("🧠 Initializing ULTRATHINK Semantic Integration Manager...")
        
        # Initialize existing context system
        self.context_loader = UltraThinkContextLoader(self.project_root)
        
        # Initialize semantic components
        self.semantic_chunker = TreeSitterSemanticChunker()
        self.graph_builder = SemanticGraphBuilder()
        self.chunk_classifier = ChunkClassificationEngine()
        
        # Semantic analysis cache
        self.semantic_cache: Dict[str, SemanticModuleAnalysis] = {}
        self.classification_cache: Dict[str, Dict[str, ChunkClassification]] = {}
        
        # Performance tracking
        self.performance_stats = {
            'semantic_queries': 0,
            'cache_hits': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }
        
        print("✅ Semantic Integration Manager ready")
        print(f"📊 Base system: {len(self.context_loader.module_graph)} modules")
    
    def analyze_project_semantics(self) -> Dict[str, SemanticModuleAnalysis]:
        """Perform semantic analysis on all project modules"""
        print("🔍 Performing project-wide semantic analysis...")
        start_time = time.time()
        
        semantic_analyses = {}
        
        # Analyze all modules in the context loader
        for module_path, module_info in self.context_loader.module_graph.items():
            try:
                # Load module content
                full_path = self.project_root / module_path
                if full_path.exists() and full_path.suffix == '.py':
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Perform semantic analysis
                    analysis = self.semantic_chunker.chunk_module(module_path, content)
                    semantic_analyses[module_path] = analysis
                    
                    # Cache the analysis
                    self.semantic_cache[module_path] = analysis
                    
            except Exception as e:
                print(f"⚠️  Error analyzing {module_path}: {e}")
                continue
        
        # Build semantic graph
        semantic_graph = self.graph_builder.build_semantic_graph(semantic_analyses)
        
        # Classify all chunks
        all_chunks = []
        for analysis in semantic_analyses.values():
            all_chunks.extend(analysis.chunks)
        
        if all_chunks:
            classifications = self.chunk_classifier.classify_chunks(all_chunks)
            
            # Cache classifications by module
            for module_path, analysis in semantic_analyses.items():
                module_classifications = {}
                for chunk in analysis.chunks:
                    if chunk.chunk_id in classifications:
                        module_classifications[chunk.chunk_id] = classifications[chunk.chunk_id]
                self.classification_cache[module_path] = module_classifications
        
        analysis_time = (time.time() - start_time) * 1000
        print(f"✅ Project semantic analysis complete in {analysis_time:.1f}ms")
        print(f"📊 Analyzed {len(semantic_analyses)} modules, {len(all_chunks)} chunks")
        
        return semantic_analyses
    
    def get_semantic_context(self, query: str, max_chunks: int = 8) -> SemanticContextBundle:
        """Get intelligent semantic context for a query"""
        start_time = time.time()
        self.performance_stats['semantic_queries'] += 1
        
        # If no semantic cache, perform analysis
        if not self.semantic_cache:
            self.analyze_project_semantics()
        
        # Find relevant chunks based on query
        relevant_chunks = self._find_relevant_chunks(query, max_chunks)
        
        # Get chunk classifications
        chunk_classifications = self._get_chunk_classifications(relevant_chunks)
        
        # Build semantic relationships
        semantic_relationships = self._build_chunk_relationships(relevant_chunks)
        
        # Generate trading workflow context
        workflow_context = self._generate_workflow_context(relevant_chunks, chunk_classifications)
        
        # Load actual content for chunks
        context_content = self._load_chunk_content(relevant_chunks)
        
        # Generate semantic summary
        semantic_summary = self._generate_semantic_summary(query, relevant_chunks, chunk_classifications)
        
        # Calculate confidence score
        confidence_score = self._calculate_semantic_confidence(relevant_chunks, chunk_classifications)
        
        # Calculate token count
        token_count = sum(len(content.split()) for content in context_content.values())
        
        # Gather semantic metrics
        semantic_metrics = self._gather_semantic_metrics()
        
        load_time = (time.time() - start_time) * 1000
        
        # Update performance stats
        self.performance_stats['total_processing_time_ms'] += load_time
        self.performance_stats['avg_processing_time_ms'] = (
            self.performance_stats['total_processing_time_ms'] / 
            self.performance_stats['semantic_queries']
        )
        
        return SemanticContextBundle(
            query=query,
            primary_semantic_chunks=relevant_chunks[:max_chunks//2],
            dependency_chunks=relevant_chunks[max_chunks//2:],
            chunk_classifications=chunk_classifications,
            semantic_relationships=semantic_relationships,
            trading_workflow_context=workflow_context,
            context_content=context_content,
            semantic_summary=semantic_summary,
            confidence_score=confidence_score,
            load_time_ms=load_time,
            token_count=token_count,
            semantic_metrics=semantic_metrics
        )
    
    def _find_relevant_chunks(self, query: str, max_chunks: int) -> List[str]:
        """Find relevant chunks for the query"""
        query_lower = query.lower()
        scored_chunks = []
        
        for module_path, analysis in self.semantic_cache.items():
            for chunk in analysis.chunks:
                score = 0.0
                
                # Score based on semantic tags
                for tag in chunk.semantic_tags:
                    if tag in query_lower:
                        score += 2.0
                
                # Score based on chunk content
                if chunk.content:
                    content_lower = chunk.content.lower()
                    query_words = query_lower.split()
                    for word in query_words:
                        if word in content_lower:
                            score += 0.5
                
                # Score based on trading relevance
                score += chunk.trading_relevance * 1.0
                
                # Score based on chunk type
                if 'trading' in query_lower and chunk.chunk_type in ['trading_logic', 'risk_control']:
                    score += 1.0
                elif 'risk' in query_lower and chunk.chunk_type == 'risk_control':
                    score += 1.0
                elif 'ml' in query_lower and chunk.chunk_type == 'ml_model':
                    score += 1.0
                
                if score > 0.1:
                    scored_chunks.append((chunk.chunk_id, score))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in scored_chunks[:max_chunks]]
    
    def _get_chunk_classifications(self, chunk_ids: List[str]) -> Dict[str, ChunkClassification]:
        """Get classifications for specified chunks"""
        classifications = {}
        
        for module_path, module_classifications in self.classification_cache.items():
            for chunk_id, classification in module_classifications.items():
                if chunk_id in chunk_ids:
                    classifications[chunk_id] = classification
        
        return classifications
    
    def _build_chunk_relationships(self, chunk_ids: List[str]) -> Dict[str, List[str]]:
        """Build relationships between selected chunks"""
        relationships = {}
        
        # Use the semantic graph to find relationships
        if hasattr(self.graph_builder, 'graph') and self.graph_builder.graph:
            for chunk_id in chunk_ids:
                if chunk_id in self.graph_builder.graph.nodes:
                    related_chunks = []
                    
                    # Get direct neighbors
                    neighbors = list(self.graph_builder.graph.neighbors(chunk_id))
                    predecessors = list(self.graph_builder.graph.predecessors(chunk_id))
                    
                    # Filter for chunks in our selection
                    related_chunks = [
                        related for related in (neighbors + predecessors)
                        if related in chunk_ids and related != chunk_id
                    ]
                    
                    relationships[chunk_id] = related_chunks[:5]  # Limit to top 5
        
        return relationships
    
    def _generate_workflow_context(self, chunk_ids: List[str], classifications: Dict[str, ChunkClassification]) -> str:
        """Generate trading workflow context"""
        # Group chunks by trading stage
        stage_chunks = {}
        for chunk_id, classification in classifications.items():
            stage = classification.trading_stage
            if stage not in stage_chunks:
                stage_chunks[stage] = []
            stage_chunks[stage].append(chunk_id)
        
        # Build workflow description
        workflow_parts = []
        stage_order = ['DATA_INGESTION', 'SIGNAL_GENERATION', 'RISK_ASSESSMENT', 'ORDER_EXECUTION', 'MONITORING']
        
        for stage in stage_order:
            if stage in stage_chunks:
                chunk_count = len(stage_chunks[stage])
                stage_name = stage.replace('_', ' ').title()
                workflow_parts.append(f"{stage_name}: {chunk_count} components")
        
        if workflow_parts:
            return f"Trading workflow: {' → '.join(workflow_parts)}"
        else:
            return "General trading system components"
    
    def _load_chunk_content(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Load actual content for chunks"""
        content_map = {}
        
        for module_path, analysis in self.semantic_cache.items():
            for chunk in analysis.chunks:
                if chunk.chunk_id in chunk_ids:
                    # Use chunk content if available, otherwise load from file
                    if chunk.content:
                        content_map[chunk.chunk_id] = chunk.content
                    else:
                        # Fallback to loading from file
                        try:
                            full_path = self.project_root / module_path
                            if full_path.exists():
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    chunk_lines = lines[chunk.start_line-1:chunk.end_line]
                                    content_map[chunk.chunk_id] = ''.join(chunk_lines)
                        except Exception:
                            content_map[chunk.chunk_id] = f"# Content unavailable for {chunk.chunk_id}"
        
        return content_map
    
    def _generate_semantic_summary(self, query: str, chunk_ids: List[str], classifications: Dict[str, ChunkClassification]) -> str:
        """Generate semantic summary of the context"""
        summary_parts = []
        
        # Query context
        summary_parts.append(f"Semantic analysis for: {query}")
        
        # Chunk breakdown
        if classifications:
            category_counts = {}
            for classification in classifications.values():
                category = classification.primary_category
                category_counts[category] = category_counts.get(category, 0) + 1
            
            category_summary = []
            for category, count in category_counts.items():
                category_name = category.replace('_', ' ').title()
                category_summary.append(f"{count} {category_name}")
            
            summary_parts.append(f"Components: {', '.join(category_summary)}")
        
        # Priority assessment
        high_priority_count = sum(1 for c in classifications.values() if c.priority_score > 0.8)
        if high_priority_count > 0:
            summary_parts.append(f"{high_priority_count} high-priority components identified")
        
        return ". ".join(summary_parts)
    
    def _calculate_semantic_confidence(self, chunk_ids: List[str], classifications: Dict[str, ChunkClassification]) -> float:
        """Calculate confidence in semantic context selection"""
        if not classifications:
            return 0.0
        
        # Average classification confidence
        avg_classification_confidence = sum(c.classification_confidence for c in classifications.values()) / len(classifications)
        
        # Coverage score (how many chunks we found vs requested)
        coverage_score = min(len(chunk_ids) / 8.0, 1.0)  # Normalize by typical request size
        
        # Priority score (higher priority chunks = higher confidence)
        avg_priority = sum(c.priority_score for c in classifications.values()) / len(classifications)
        
        # Combined confidence
        confidence = (avg_classification_confidence * 0.4 + coverage_score * 0.3 + avg_priority * 0.3)
        
        return min(confidence, 1.0)
    
    def _gather_semantic_metrics(self) -> Dict[str, Any]:
        """Gather semantic system metrics"""
        graph_metrics = self.graph_builder.get_semantic_metrics()
        classification_metrics = self.chunk_classifier.get_classification_metrics()
        
        return {
            'semantic_graph': asdict(graph_metrics),
            'classification': asdict(classification_metrics),
            'performance': self.performance_stats.copy(),
            'cache_status': {
                'modules_cached': len(self.semantic_cache),
                'classifications_cached': len(self.classification_cache)
            }
        }
    
    def get_system_metrics(self) -> SemanticSystemMetrics:
        """Get comprehensive system metrics"""
        graph_metrics = self.graph_builder.get_semantic_metrics()
        classification_metrics = self.chunk_classifier.get_classification_metrics()
        
        return SemanticSystemMetrics(
            semantic_analysis_time_ms=sum(a.analysis_time_ms for a in self.semantic_cache.values()),
            classification_time_ms=classification_metrics.avg_classification_time_ms * classification_metrics.total_chunks_classified,
            graph_building_time_ms=10.0,  # Placeholder - would track actual time
            context_generation_time_ms=self.performance_stats['avg_processing_time_ms'],
            total_processing_time_ms=self.performance_stats['total_processing_time_ms'],
            chunks_analyzed=sum(len(a.chunks) for a in self.semantic_cache.values()),
            relationships_discovered=graph_metrics.total_relationships,
            high_priority_chunks=classification_metrics.high_priority_chunks,
            semantic_graph_metrics=graph_metrics,
            classification_metrics=classification_metrics
        )
    
    def integrate_with_existing_api(self, existing_api):
        """Integrate semantic capabilities with existing context API"""
        # This method would extend the existing API with semantic capabilities
        print("🔗 Integrating semantic capabilities with existing context API...")
        
        # Add semantic methods to existing API
        existing_api.get_semantic_context = self.get_semantic_context
        existing_api.analyze_project_semantics = self.analyze_project_semantics
        existing_api.get_semantic_metrics = self.get_system_metrics
        
        print("✅ Semantic integration complete")
    
    def export_semantic_data(self) -> Dict[str, Any]:
        """Export all semantic data for analysis"""
        return {
            'semantic_analyses': {
                module_path: {
                    'chunks': [asdict(chunk) for chunk in analysis.chunks],
                    'semantic_graph': analysis.semantic_graph,
                    'trading_patterns': analysis.trading_patterns,
                    'complexity_metrics': analysis.complexity_metrics,
                    'semantic_summary': analysis.semantic_summary
                }
                for module_path, analysis in self.semantic_cache.items()
            },
            'classifications': self.chunk_classifier.export_classifications(),
            'graph_data': self.graph_builder.export_graph_data(),
            'system_metrics': asdict(self.get_system_metrics())
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧠 ULTRATHINK Semantic Integration Manager Testing")
    print("=" * 70)
    
    # Initialize integration manager
    manager = SemanticIntegrationManager()
    
    # Test semantic context retrieval
    test_queries = [
        "How does Kelly criterion optimize position sizing?",
        "What is the relationship between ensemble models and trading execution?",
        "How does risk management work in live trading?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            semantic_context = manager.get_semantic_context(query, max_chunks=6)
            
            query_time = (time.time() - start_time) * 1000
            
            print(f"✅ Semantic Context Retrieved:")
            print(f"├── Load Time: {semantic_context.load_time_ms:.1f}ms")
            print(f"├── Primary Chunks: {len(semantic_context.primary_semantic_chunks)}")
            print(f"├── Dependency Chunks: {len(semantic_context.dependency_chunks)}")
            print(f"├── Confidence: {semantic_context.confidence_score:.2f}")
            print(f"├── Tokens: {semantic_context.token_count:,}")
            print(f"├── Workflow: {semantic_context.trading_workflow_context}")
            print(f"└── Summary: {semantic_context.semantic_summary}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Get system metrics
    print(f"\n📊 System Metrics:")
    metrics = manager.get_system_metrics()
    print(f"├── Chunks Analyzed: {metrics.chunks_analyzed}")
    print(f"├── Relationships Discovered: {metrics.relationships_discovered}")
    print(f"├── High Priority Chunks: {metrics.high_priority_chunks}")
    print(f"├── Avg Processing Time: {metrics.context_generation_time_ms:.1f}ms")
    print(f"└── Total Processing Time: {metrics.total_processing_time_ms:.1f}ms")
    
    print(f"\n🧠 Semantic integration testing complete!")