#!/usr/bin/env python3
"""
ULTRATHINK CGRAG Fine Retrieval Engine
Second stage of two-stage intelligent context retrieval

Philosophy: Precision targeting for exact context needed
Performance: < 20ms fine filtering with line-level precision
Intelligence: Deep code analysis with contextual understanding
"""

import json
import time
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Import coarse retrieval components
import sys
sys.path.insert(0, str(Path(__file__).parent))
from coarse_retrieval import CoarseRetrievalCandidate, CoarseRetrievalResult, CoarseRetrievalEngine

# Import semantic components
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic"))
from tree_sitter_chunker import SemanticChunk
from chunk_classifier import ChunkClassification

@dataclass
class FineRetrievalTarget:
    """Precise target from fine retrieval"""
    chunk_id: str
    module_path: str
    target_type: str  # 'function', 'class', 'method', 'code_block', 'docstring', 'comment'
    start_line: int
    end_line: int
    content: str
    relevance_score: float
    precision_score: float  # How precisely this matches the query
    context_lines: List[str]  # Surrounding context lines
    dependencies: List[str]  # Function/class dependencies within this chunk
    key_variables: List[str]  # Important variables in this target
    complexity_indicators: List[str]  # Complexity patterns found
    trading_relevance_details: Dict[str, Any]  # Specific trading relevance

@dataclass
class FineRetrievalResult:
    """Result of fine retrieval stage"""
    query: str
    targets: List[FineRetrievalTarget]
    total_lines_analyzed: int
    precision_filtering_time_ms: float
    content_depth_achieved: str  # 'SURFACE', 'MEDIUM', 'DEEP'
    token_efficiency_score: float  # Quality of context per token
    final_confidence_score: float
    retrieval_strategy_used: str
    optimization_suggestions: List[str]

class CodePatternAnalyzer:
    """Analyzes code patterns for precise targeting"""
    
    def __init__(self):
        # Define code patterns for different query types
        self.PATTERN_EXTRACTORS = {
            'function_implementation': {
                'patterns': [r'def\s+\w+.*:', r'async\s+def\s+\w+.*:'],
                'context_lines': 3,
                'include_docstring': True
            },
            'class_definition': {
                'patterns': [r'class\s+\w+.*:'],
                'context_lines': 2,
                'include_docstring': True
            },
            'mathematical_formulas': {
                'patterns': [r'[*+\-/]=', r'np\.\w+', r'math\.\w+', r'\*\*\s*\d', r'sqrt\(', r'log\('],
                'context_lines': 2,
                'include_docstring': False
            },
            'error_handling': {
                'patterns': [r'try:', r'except\s+\w+', r'raise\s+\w+', r'assert\s+'],
                'context_lines': 3,
                'include_docstring': False
            },
            'configuration': {
                'patterns': [r'config\w*\s*=', r'setting\w*\s*=', r'parameter\w*\s*=', r'threshold\w*\s*='],
                'context_lines': 1,
                'include_docstring': False
            },
            'real_time_processing': {
                'patterns': [r'async\s+def', r'await\s+', r'real.*time', r'continuous', r'stream'],
                'context_lines': 4,
                'include_docstring': True
            },
            'optimization_logic': {
                'patterns': [r'optim\w+', r'minimize', r'maximize', r'gradient', r'learning_rate'],
                'context_lines': 3,
                'include_docstring': True
            }
        }
        
        # Trading-specific precision patterns
        self.TRADING_PRECISION_PATTERNS = {
            'kelly_criterion': {
                'exact_matches': ['kelly_fraction', 'optimal_f', 'fractional_kelly'],
                'formula_patterns': [r'win_rate.*loss_rate', r'edge.*odds', r'p.*\-.*q'],
                'variable_patterns': ['win_rate', 'avg_win', 'avg_loss', 'kelly_f']
            },
            'risk_management': {
                'exact_matches': ['stop_loss', 'position_limit', 'max_drawdown', 'var', 'cvar'],
                'formula_patterns': [r'risk.*\*.*position', r'position.*\*.*price'],
                'variable_patterns': ['risk_limit', 'position_size', 'max_risk', 'drawdown']
            },
            'ensemble_methods': {
                'exact_matches': ['random_forest', 'meta_model', 'base_models', 'ensemble'],
                'formula_patterns': [r'models\[\w+\]', r'predict.*ensemble'],
                'variable_patterns': ['n_estimators', 'base_models', 'meta_learner']
            },
            'live_trading': {
                'exact_matches': ['execute_trade', 'real_time', 'live_session', 'continuous'],
                'formula_patterns': [r'while.*True', r'time\.sleep', r'schedule'],
                'variable_patterns': ['live_mode', 'real_time', 'active_session']
            }
        }
        
        # Context importance weights
        self.CONTEXT_WEIGHTS = {
            'function_signature': 2.0,
            'docstring': 1.5,
            'variable_assignment': 1.0,
            'formula_calculation': 1.8,
            'error_handling': 0.8,
            'comments': 0.6,
            'import_statements': 0.4
        }
    
    def analyze_code_patterns(self, content: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze code for patterns relevant to query"""
        patterns_found = []
        lines = content.split('\n')
        
        # Determine which extractors to use based on query
        relevant_extractors = self._select_relevant_extractors(query_analysis)
        
        for extractor_name in relevant_extractors:
            extractor = self.PATTERN_EXTRACTORS[extractor_name]
            
            for pattern in extractor['patterns']:
                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        # Extract context around the match
                        context_start = max(0, i - extractor['context_lines'])
                        context_end = min(len(lines), i + extractor['context_lines'] + 1)
                        
                        pattern_info = {
                            'type': extractor_name,
                            'line_number': i + 1,
                            'matched_line': line.strip(),
                            'pattern': pattern,
                            'context_lines': lines[context_start:context_end],
                            'include_docstring': extractor['include_docstring']
                        }
                        patterns_found.append(pattern_info)
        
        return patterns_found
    
    def _select_relevant_extractors(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Select pattern extractors based on query analysis"""
        extractors = []
        
        # Map detected patterns to extractors
        pattern_mapping = {
            'kelly_criterion': ['function_implementation', 'mathematical_formulas'],
            'risk_management': ['function_implementation', 'configuration', 'error_handling'],
            'ensemble_ml': ['class_definition', 'function_implementation'],
            'live_trading': ['real_time_processing', 'error_handling'],
            'data_processing': ['function_implementation', 'error_handling'],
            'performance_analytics': ['function_implementation', 'mathematical_formulas']
        }
        
        for pattern in query_analysis['detected_patterns']:
            if pattern in pattern_mapping:
                extractors.extend(pattern_mapping[pattern])
        
        # Add default extractors for explanation queries
        if 'EXPLANATION_QUERY' in query_analysis['detected_intent']:
            extractors.extend(['function_implementation', 'class_definition'])
        
        return list(set(extractors))  # Remove duplicates

class PrecisionTargeting:
    """Performs precision targeting within code chunks"""
    
    def __init__(self):
        self.pattern_analyzer = CodePatternAnalyzer()
        
        # Precision scoring weights
        self.PRECISION_WEIGHTS = {
            'exact_keyword_match': 3.0,
            'semantic_relevance': 2.0,
            'pattern_complexity': 1.5,
            'context_completeness': 1.2,
            'trading_specificity': 2.5
        }
    
    def extract_precise_targets(self, chunk_content: str, chunk_id: str, module_path: str, 
                              query: str, query_analysis: Dict[str, Any]) -> List[FineRetrievalTarget]:
        """Extract precise targets from chunk content"""
        targets = []
        
        if not chunk_content:
            return targets
        
        # Analyze code patterns
        patterns = self.pattern_analyzer.analyze_code_patterns(chunk_content, query_analysis)
        
        # Extract targets based on patterns
        for pattern in patterns:
            target = self._create_target_from_pattern(
                pattern, chunk_content, chunk_id, module_path, query, query_analysis
            )
            if target:
                targets.append(target)
        
        # Extract additional targets based on query-specific logic
        additional_targets = self._extract_query_specific_targets(
            chunk_content, chunk_id, module_path, query, query_analysis
        )
        targets.extend(additional_targets)
        
        # Score and rank targets
        for target in targets:
            target.precision_score = self._calculate_precision_score(target, query, query_analysis)
            target.trading_relevance_details = self._analyze_trading_relevance(target, query_analysis)
        
        # Sort by precision score
        targets.sort(key=lambda t: t.precision_score, reverse=True)
        
        return targets
    
    def _create_target_from_pattern(self, pattern: Dict[str, Any], chunk_content: str, 
                                  chunk_id: str, module_path: str, query: str, 
                                  query_analysis: Dict[str, Any]) -> Optional[FineRetrievalTarget]:
        """Create a fine retrieval target from a detected pattern"""
        
        # Calculate line range
        start_line = pattern['line_number']
        end_line = start_line
        
        # Extend range for complex patterns
        if pattern['type'] in ['function_implementation', 'class_definition']:
            # Try to find the end of the function/class
            lines = chunk_content.split('\n')
            current_indent = len(pattern['matched_line']) - len(pattern['matched_line'].lstrip())
            
            for i in range(pattern['line_number'], min(len(lines), pattern['line_number'] + 50)):
                line = lines[i]
                if line.strip() and len(line) - len(line.lstrip()) <= current_indent and i > pattern['line_number']:
                    break
                end_line = i + 1
        
        # Extract content
        lines = chunk_content.split('\n')
        target_lines = lines[start_line-1:end_line]
        content = '\n'.join(target_lines)
        
        # Extract additional information
        key_variables = self._extract_key_variables(content)
        dependencies = self._extract_dependencies(content)
        complexity_indicators = self._identify_complexity(content)
        
        # Calculate initial relevance score
        relevance_score = self._calculate_initial_relevance(content, query, query_analysis)
        
        return FineRetrievalTarget(
            chunk_id=chunk_id,
            module_path=module_path,
            target_type=pattern['type'],
            start_line=start_line,
            end_line=end_line,
            content=content,
            relevance_score=relevance_score,
            precision_score=0.0,  # Will be calculated later
            context_lines=pattern['context_lines'],
            dependencies=dependencies,
            key_variables=key_variables,
            complexity_indicators=complexity_indicators,
            trading_relevance_details={}  # Will be populated later
        )
    
    def _extract_query_specific_targets(self, chunk_content: str, chunk_id: str, module_path: str,
                                      query: str, query_analysis: Dict[str, Any]) -> List[FineRetrievalTarget]:
        """Extract targets specific to query patterns"""
        targets = []
        
        # Use trading precision patterns
        for pattern_name in query_analysis['detected_patterns']:
            if pattern_name in self.pattern_analyzer.TRADING_PRECISION_PATTERNS:
                precision_pattern = self.pattern_analyzer.TRADING_PRECISION_PATTERNS[pattern_name]
                
                # Look for exact matches
                for exact_match in precision_pattern['exact_matches']:
                    if exact_match.lower() in chunk_content.lower():
                        target = self._create_exact_match_target(
                            exact_match, chunk_content, chunk_id, module_path, pattern_name
                        )
                        if target:
                            targets.append(target)
        
        return targets
    
    def _create_exact_match_target(self, exact_match: str, chunk_content: str, 
                                 chunk_id: str, module_path: str, pattern_name: str) -> Optional[FineRetrievalTarget]:
        """Create target for exact keyword match"""
        lines = chunk_content.split('\n')
        
        for i, line in enumerate(lines):
            if exact_match.lower() in line.lower():
                # Extract surrounding context
                start_line = max(1, i - 1)
                end_line = min(len(lines), i + 3)
                
                context_lines = lines[start_line-1:end_line]
                content = '\n'.join(context_lines)
                
                return FineRetrievalTarget(
                    chunk_id=chunk_id,
                    module_path=module_path,
                    target_type='exact_match',
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    relevance_score=0.9,  # High relevance for exact matches
                    precision_score=0.0,
                    context_lines=context_lines,
                    dependencies=[],
                    key_variables=self._extract_key_variables(content),
                    complexity_indicators=[],
                    trading_relevance_details={'pattern': pattern_name, 'match_type': 'exact'}
                )
        
        return None
    
    def _extract_key_variables(self, content: str) -> List[str]:
        """Extract key variables from content"""
        variables = []
        
        # Simple variable extraction
        variable_patterns = [
            r'(\w+)\s*=\s*',  # Variable assignments
            r'def\s+\w+\(([^)]+)\)',  # Function parameters
            r'self\.(\w+)',  # Class attributes
        ]
        
        for pattern in variable_patterns:
            matches = re.findall(pattern, content)
            variables.extend(matches)
        
        # Filter out common keywords
        keywords_to_ignore = {'self', 'return', 'if', 'else', 'for', 'while', 'import', 'def', 'class'}
        variables = [v for v in variables if isinstance(v, str) and v not in keywords_to_ignore]
        
        return list(set(variables))[:10]  # Limit to top 10
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract function/class dependencies"""
        dependencies = []
        
        # Function calls
        function_calls = re.findall(r'(\w+)\s*\(', content)
        dependencies.extend(function_calls)
        
        # Attribute access
        attributes = re.findall(r'\.(\w+)', content)
        dependencies.extend(attributes)
        
        # Import statements
        imports = re.findall(r'from\s+\w+\s+import\s+(\w+)', content)
        dependencies.extend(imports)
        
        return list(set(dependencies))[:8]  # Limit to top 8
    
    def _identify_complexity(self, content: str) -> List[str]:
        """Identify complexity indicators in content"""
        indicators = []
        
        complexity_patterns = {
            'loops': r'for\s+\w+\s+in|while\s+',
            'conditionals': r'if\s+|elif\s+|else:',
            'exception_handling': r'try:|except\s+|finally:',
            'async_programming': r'async\s+def|await\s+',
            'list_comprehensions': r'\[.*for.*in.*\]',
            'lambda_functions': r'lambda\s+',
            'decorators': r'@\w+',
            'complex_expressions': r'[+\-*/]{2,}|\*\*|\(\s*[^)]*\s*\)'
        }
        
        for indicator_name, pattern in complexity_patterns.items():
            if re.search(pattern, content):
                indicators.append(indicator_name)
        
        return indicators
    
    def _calculate_initial_relevance(self, content: str, query: str, query_analysis: Dict[str, Any]) -> float:
        """Calculate initial relevance score"""
        score = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Keyword matching
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        keyword_overlap = len(query_words & content_words)
        score += (keyword_overlap / len(query_words)) * 0.4
        
        # Pattern matching
        for pattern in query_analysis['detected_patterns']:
            if pattern in content_lower:
                score += 0.3
        
        # Content quality indicators
        if 'def ' in content_lower:
            score += 0.1
        if '"""' in content or "'''" in content:  # Has docstring
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_precision_score(self, target: FineRetrievalTarget, query: str, query_analysis: Dict[str, Any]) -> float:
        """Calculate precision score for target"""
        score = target.relevance_score * 0.4  # Base from relevance
        
        # Exact keyword matching bonus
        query_lower = query.lower()
        content_lower = target.content.lower()
        
        for word in query_lower.split():
            if word in content_lower:
                score += 0.1
        
        # Pattern-specific bonuses
        if target.target_type == 'function_implementation':
            score += 0.2
        elif target.target_type == 'exact_match':
            score += 0.3
        elif target.target_type == 'mathematical_formulas':
            score += 0.25
        
        # Trading relevance bonus
        trading_patterns = query_analysis['detected_patterns']
        if any(pattern in target.content.lower() for pattern in trading_patterns):
            score += 0.2
        
        # Complexity appropriate to query
        if query_analysis['query_complexity'] == 'HIGH' and len(target.complexity_indicators) > 2:
            score += 0.1
        elif query_analysis['query_complexity'] == 'LOW' and len(target.complexity_indicators) == 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_trading_relevance(self, target: FineRetrievalTarget, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading-specific relevance details"""
        details = {
            'trading_concepts_found': [],
            'mathematical_content': False,
            'real_time_indicators': False,
            'risk_management_patterns': False,
            'execution_patterns': False
        }
        
        content_lower = target.content.lower()
        
        # Check for trading concepts
        trading_concepts = ['kelly', 'risk', 'position', 'trade', 'signal', 'strategy', 'ensemble']
        for concept in trading_concepts:
            if concept in content_lower:
                details['trading_concepts_found'].append(concept)
        
        # Check for mathematical content
        math_patterns = [r'[+\-*/]=', r'np\.\w+', r'math\.\w+', r'\*\*', r'sqrt', r'log']
        if any(re.search(pattern, content_lower) for pattern in math_patterns):
            details['mathematical_content'] = True
        
        # Check for real-time patterns
        realtime_patterns = ['real.*time', 'continuous', 'live', 'async', 'await']
        if any(re.search(pattern, content_lower) for pattern in realtime_patterns):
            details['real_time_indicators'] = True
        
        # Check for risk management
        risk_patterns = ['risk', 'limit', 'stop', 'loss', 'drawdown', 'var']
        if any(pattern in content_lower for pattern in risk_patterns):
            details['risk_management_patterns'] = True
        
        # Check for execution patterns
        execution_patterns = ['execute', 'order', 'buy', 'sell', 'trade']
        if any(pattern in content_lower for pattern in execution_patterns):
            details['execution_patterns'] = True
        
        return details

class FineRetrievalEngine:
    """Second stage of CGRAG: precision context targeting"""
    
    def __init__(self):
        self.precision_targeting = PrecisionTargeting()
        
        # Configuration
        self.MAX_TARGETS_PER_CANDIDATE = 5
        self.MIN_PRECISION_THRESHOLD = 0.3
        self.TOKEN_EFFICIENCY_TARGET = 0.8
        
        # Performance tracking
        self.fine_retrieval_stats = {
            'total_fine_queries': 0,
            'avg_processing_time_ms': 0.0,
            'avg_targets_found': 0.0,
            'avg_precision_score': 0.0
        }
        
        print("🎯 CGRAG Fine Retrieval Engine initialized")
    
    def retrieve_fine_targets(self, coarse_result: CoarseRetrievalResult, 
                            semantic_manager) -> FineRetrievalResult:
        """Perform fine retrieval on coarse candidates"""
        start_time = time.time()
        self.fine_retrieval_stats['total_fine_queries'] += 1
        
        all_targets = []
        total_lines_analyzed = 0
        
        # Process each coarse candidate
        for candidate in coarse_result.candidates:
            # Load full chunk content
            chunk_content = self._load_chunk_content(candidate, semantic_manager)
            
            if chunk_content:
                lines_in_chunk = len(chunk_content.split('\n'))
                total_lines_analyzed += lines_in_chunk
                
                # Analyze query for fine targeting
                from coarse_retrieval import TradingQueryAnalyzer
                query_analyzer = TradingQueryAnalyzer()
                query_analysis = query_analyzer.analyze_query(coarse_result.query)
                
                # Extract precise targets
                targets = self.precision_targeting.extract_precise_targets(
                    chunk_content, candidate.chunk_id, candidate.module_path,
                    coarse_result.query, query_analysis
                )
                
                # Limit targets per candidate
                targets = targets[:self.MAX_TARGETS_PER_CANDIDATE]
                all_targets.extend(targets)
        
        # Filter by precision threshold
        filtered_targets = [t for t in all_targets if t.precision_score >= self.MIN_PRECISION_THRESHOLD]
        
        # Sort by precision score
        filtered_targets.sort(key=lambda t: t.precision_score, reverse=True)
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000
        
        content_depth = self._assess_content_depth(filtered_targets)
        token_efficiency = self._calculate_token_efficiency(filtered_targets, coarse_result.query)
        final_confidence = self._calculate_final_confidence(filtered_targets, coarse_result)
        
        strategy_used = self._determine_strategy_used(coarse_result, query_analysis)
        optimizations = self._suggest_optimizations(filtered_targets, processing_time)
        
        # Update statistics
        self.fine_retrieval_stats['avg_processing_time_ms'] = (
            (self.fine_retrieval_stats['avg_processing_time_ms'] * (self.fine_retrieval_stats['total_fine_queries'] - 1) + processing_time) /
            self.fine_retrieval_stats['total_fine_queries']
        )
        
        if filtered_targets:
            avg_precision = sum(t.precision_score for t in filtered_targets) / len(filtered_targets)
            self.fine_retrieval_stats['avg_targets_found'] = (
                (self.fine_retrieval_stats['avg_targets_found'] * (self.fine_retrieval_stats['total_fine_queries'] - 1) + len(filtered_targets)) /
                self.fine_retrieval_stats['total_fine_queries']
            )
            self.fine_retrieval_stats['avg_precision_score'] = (
                (self.fine_retrieval_stats['avg_precision_score'] * (self.fine_retrieval_stats['total_fine_queries'] - 1) + avg_precision) /
                self.fine_retrieval_stats['total_fine_queries']
            )
        
        return FineRetrievalResult(
            query=coarse_result.query,
            targets=filtered_targets,
            total_lines_analyzed=total_lines_analyzed,
            precision_filtering_time_ms=processing_time,
            content_depth_achieved=content_depth,
            token_efficiency_score=token_efficiency,
            final_confidence_score=final_confidence,
            retrieval_strategy_used=strategy_used,
            optimization_suggestions=optimizations
        )
    
    def _load_chunk_content(self, candidate: CoarseRetrievalCandidate, semantic_manager) -> Optional[str]:
        """Load full content for a chunk"""
        # Try to find chunk in semantic cache
        for module_path, analysis in semantic_manager.semantic_cache.items():
            if module_path == candidate.module_path:
                for chunk in analysis.chunks:
                    if chunk.chunk_id == candidate.chunk_id:
                        return chunk.content
        
        # Fallback: load from file
        try:
            full_path = semantic_manager.project_root / candidate.module_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception:
            pass
        
        return None
    
    def _assess_content_depth(self, targets: List[FineRetrievalTarget]) -> str:
        """Assess the depth of content retrieved"""
        if not targets:
            return 'SURFACE'
        
        # Check for deep indicators
        deep_indicators = 0
        for target in targets:
            if target.target_type in ['function_implementation', 'class_definition']:
                deep_indicators += 1
            if len(target.complexity_indicators) > 2:
                deep_indicators += 1
            if target.trading_relevance_details.get('mathematical_content', False):
                deep_indicators += 1
        
        if deep_indicators >= len(targets) * 0.7:
            return 'DEEP'
        elif deep_indicators >= len(targets) * 0.3:
            return 'MEDIUM'
        else:
            return 'SURFACE'
    
    def _calculate_token_efficiency(self, targets: List[FineRetrievalTarget], query: str) -> float:
        """Calculate token efficiency score"""
        if not targets:
            return 0.0
        
        total_tokens = sum(len(target.content.split()) for target in targets)
        avg_precision = sum(target.precision_score for target in targets) / len(targets)
        
        # Token efficiency = precision per token
        if total_tokens > 0:
            efficiency = (avg_precision * len(targets)) / (total_tokens / 100)  # Normalize by 100 tokens
            return min(efficiency, 1.0)
        
        return 0.0
    
    def _calculate_final_confidence(self, targets: List[FineRetrievalTarget], coarse_result: CoarseRetrievalResult) -> float:
        """Calculate final confidence score"""
        if not targets:
            return 0.0
        
        # Combine coarse and fine confidence
        coarse_confidence = coarse_result.confidence_score
        
        # Fine confidence from precision scores
        avg_precision = sum(t.precision_score for t in targets) / len(targets)
        
        # Coverage confidence (how many of the coarse candidates were useful)
        coverage = len(targets) / max(len(coarse_result.candidates), 1)
        
        # Combined confidence
        final_confidence = (coarse_confidence * 0.3 + avg_precision * 0.5 + coverage * 0.2)
        
        return min(final_confidence, 1.0)
    
    def _determine_strategy_used(self, coarse_result: CoarseRetrievalResult, query_analysis: Dict[str, Any]) -> str:
        """Determine which retrieval strategy was most effective"""
        strategies = []
        
        if len(coarse_result.candidates) > 10:
            strategies.append('BROAD_SEARCH')
        else:
            strategies.append('TARGETED_SEARCH')
        
        if query_analysis['query_complexity'] == 'HIGH':
            strategies.append('DEEP_ANALYSIS')
        elif query_analysis['query_complexity'] == 'LOW':
            strategies.append('SIMPLE_MATCH')
        else:
            strategies.append('BALANCED_RETRIEVAL')
        
        return '_'.join(strategies)
    
    def _suggest_optimizations(self, targets: List[FineRetrievalTarget], processing_time: float) -> List[str]:
        """Suggest optimizations for future retrievals"""
        suggestions = []
        
        if processing_time > 50:
            suggestions.append('Consider reducing candidate set for faster processing')
        
        if not targets:
            suggestions.append('Broaden search criteria or lower precision threshold')
        elif len(targets) > 20:
            suggestions.append('Increase precision threshold to reduce noise')
        
        if targets:
            avg_precision = sum(t.precision_score for t in targets) / len(targets)
            if avg_precision < 0.5:
                suggestions.append('Improve pattern matching for better precision')
        
        return suggestions
    
    def get_fine_retrieval_stats(self) -> Dict[str, Any]:
        """Get fine retrieval performance statistics"""
        return self.fine_retrieval_stats.copy()

# Example usage and testing
if __name__ == "__main__":
    print("🎯 CGRAG Fine Retrieval Engine Testing")
    print("=" * 60)
    
    # This would normally be tested with actual coarse retrieval results
    # For now, create a minimal test
    print("🎯 Fine Retrieval Engine initialized successfully!")
    
    fine_engine = FineRetrievalEngine()
    stats = fine_engine.get_fine_retrieval_stats()
    
    print(f"📊 Initial Statistics:")
    print(f"├── Total Queries: {stats['total_fine_queries']}")
    print(f"├── Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"├── Avg Targets Found: {stats['avg_targets_found']:.1f}")
    print(f"└── Avg Precision Score: {stats['avg_precision_score']:.2f}")
    
    print(f"\n🎯 Fine retrieval engine ready for integration testing!")