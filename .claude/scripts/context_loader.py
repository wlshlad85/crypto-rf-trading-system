#!/usr/bin/env python3
"""
ULTRATHINK Dynamic Context Loading Framework
Optimized for institutional-grade cryptocurrency trading systems

Philosophy: Anticipate developer needs with intelligent context delivery
Performance: < 500ms context loading for any query
Intelligence: Trading-system aware context selection
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import ast
import re

@dataclass
class ModuleInfo:
    """Represents a Python module with its dependencies and metadata"""
    path: str
    name: str
    type: str  # 'trading', 'risk', 'ml', 'data', 'execution', 'analytics'
    priority: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    dependencies: List[str]
    imports: List[str]
    classes: List[str]
    functions: List[str]
    docstring: Optional[str]
    last_modified: float
    lines_of_code: int
    trading_relevance: float  # 0.0-1.0

@dataclass
class ContextBundle:
    """Complete context package for a query"""
    query: str
    primary_context: str
    related_contexts: List[str]
    module_dependencies: List[str]
    performance_metrics: Dict[str, float]
    load_time_ms: float
    relevance_score: float
    token_count: int

class TradingSystemModuleAnalyzer:
    """Analyzes Python modules for trading system specific patterns"""
    
    TRADING_KEYWORDS = {
        'strategy', 'signal', 'trade', 'position', 'order', 'execution',
        'buy', 'sell', 'portfolio', 'return', 'profit', 'loss'
    }
    
    RISK_KEYWORDS = {
        'risk', 'kelly', 'cvar', 'drawdown', 'stop_loss', 'take_profit',
        'volatility', 'sharpe', 'sortino', 'var', 'exposure'
    }
    
    ML_KEYWORDS = {
        'model', 'predict', 'train', 'ensemble', 'feature', 'accuracy',
        'random_forest', 'regression', 'classification', 'cross_validation'
    }
    
    DATA_KEYWORDS = {
        'data', 'fetch', 'validate', 'pipeline', 'ohlcv', 'price',
        'volume', 'indicator', 'technical', 'candle'
    }
    
    def __init__(self):
        self.module_types = {
            'trading': self.TRADING_KEYWORDS,
            'risk': self.RISK_KEYWORDS,
            'ml': self.ML_KEYWORDS,
            'data': self.DATA_KEYWORDS
        }
    
    def analyze_module(self, file_path: Path) -> ModuleInfo:
        """Analyze a Python module and extract trading-relevant metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for structural analysis
            tree = ast.parse(content)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Extract classes and functions
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # Extract module docstring
            docstring = ast.get_docstring(tree)
            
            # Calculate trading relevance
            trading_relevance = self._calculate_trading_relevance(content, classes, functions)
            
            # Determine module type and priority
            module_type = self._determine_module_type(file_path, content)
            priority = self._determine_priority(file_path, module_type, trading_relevance)
            
            # Extract dependencies (simplified - from imports)
            dependencies = self._extract_dependencies(imports, file_path)
            
            return ModuleInfo(
                path=str(file_path.relative_to(Path.cwd())),
                name=file_path.stem,
                type=module_type,
                priority=priority,
                dependencies=dependencies,
                imports=imports,
                classes=classes,
                functions=functions,
                docstring=docstring,
                last_modified=file_path.stat().st_mtime,
                lines_of_code=len(content.splitlines()),
                trading_relevance=trading_relevance
            )
            
        except Exception as e:
            # Return minimal info if parsing fails
            return ModuleInfo(
                path=str(file_path.relative_to(Path.cwd())),
                name=file_path.stem,
                type='unknown',
                priority='LOW',
                dependencies=[],
                imports=[],
                classes=[],
                functions=[],
                docstring=None,
                last_modified=file_path.stat().st_mtime if file_path.exists() else 0,
                lines_of_code=0,
                trading_relevance=0.0
            )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        return imports
    
    def _calculate_trading_relevance(self, content: str, classes: List[str], functions: List[str]) -> float:
        """Calculate how relevant this module is to trading operations"""
        content_lower = content.lower()
        total_score = 0.0
        max_score = 0.0
        
        # Check content for trading keywords
        for category, keywords in self.module_types.items():
            category_score = sum(content_lower.count(keyword) for keyword in keywords)
            weight = {'trading': 2.0, 'risk': 1.8, 'ml': 1.5, 'data': 1.2}.get(category, 1.0)
            total_score += category_score * weight
            max_score += len(keywords) * weight
        
        # Boost score for class/function names with trading terms
        for name in classes + functions:
            name_lower = name.lower()
            for keywords in self.module_types.values():
                if any(keyword in name_lower for keyword in keywords):
                    total_score += 5.0  # Bonus for relevant naming
        
        # Normalize to 0-1 range
        return min(total_score / max(max_score, 1.0), 1.0)
    
    def _determine_module_type(self, file_path: Path, content: str) -> str:
        """Determine the primary type of this module"""
        path_str = str(file_path).lower()
        content_lower = content.lower()
        
        # Path-based classification
        if any(term in path_str for term in ['strategy', 'strategies', 'trading']):
            return 'trading'
        elif any(term in path_str for term in ['risk', 'kelly', 'cvar']):
            return 'risk'
        elif any(term in path_str for term in ['ml', 'model', 'ensemble', 'phase2b']):
            return 'ml'
        elif any(term in path_str for term in ['data', 'fetch', 'pipeline']):
            return 'data'
        elif any(term in path_str for term in ['execution', 'order', 'trader']):
            return 'execution'
        elif any(term in path_str for term in ['analytics', 'performance']):
            return 'analytics'
        
        # Content-based classification
        type_scores = {}
        for module_type, keywords in self.module_types.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            type_scores[module_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _determine_priority(self, file_path: Path, module_type: str, trading_relevance: float) -> str:
        """Determine the priority level of this module"""
        path_str = str(file_path).lower()
        
        # Critical modules for live trading
        critical_patterns = [
            'enhanced_paper_trader_24h', 'advanced_risk_management',
            'ensemble_meta_learning', 'execution'
        ]
        
        if any(pattern in path_str for pattern in critical_patterns):
            return 'CRITICAL'
        
        # High priority based on type and relevance
        if module_type in ['trading', 'risk', 'execution'] and trading_relevance > 0.7:
            return 'HIGH'
        elif module_type in ['ml', 'data'] and trading_relevance > 0.5:
            return 'HIGH'
        elif trading_relevance > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _extract_dependencies(self, imports: List[str], file_path: Path) -> List[str]:
        """Extract local module dependencies from imports"""
        dependencies = []
        project_root = Path.cwd()
        
        for imp in imports:
            # Skip standard library and external packages
            if imp.startswith(('.', 'data.', 'strategies.', 'models.', 'phase', 'ultrathink.')):
                # Convert import to potential file path
                parts = imp.split('.')
                potential_path = project_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py"
                if potential_path.exists():
                    dependencies.append(str(potential_path.relative_to(project_root)))
        
        return dependencies

class UltraThinkContextLoader:
    """
    Intelligent context loading system for trading system development
    
    Core Philosophy:
    - Anticipate developer needs before they know them
    - Provide comprehensive context in minimal time
    - Adapt to trading system patterns and workflows
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.context_dir = self.project_root / ".claude" / "contexts"
        self.cache_dir = self.project_root / ".claude" / "cache"
        self.metrics_dir = self.project_root / ".claude" / "metrics"
        
        # Ensure directories exist
        for directory in [self.cache_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = TradingSystemModuleAnalyzer()
        self.module_graph: Dict[str, ModuleInfo] = {}
        self.context_cache: Dict[str, str] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_loads': 0,
            'cache_hits': 0,
            'avg_load_time_ms': 0.0,
            'total_load_time_ms': 0.0
        }
        
        # Initialize system
        self._build_module_graph()
        self._load_existing_contexts()
    
    def _build_module_graph(self):
        """Build comprehensive graph of all Python modules in the trading system"""
        start_time = time.time()
        
        print("🔍 Building trading system module graph...")
        
        # Find all Python files
        python_files = list(self.project_root.glob("**/*.py"))
        
        # Filter out __pycache__ and other irrelevant files
        python_files = [f for f in python_files if '__pycache__' not in str(f)]
        
        for py_file in python_files:
            try:
                module_info = self.analyzer.analyze_module(py_file)
                self.module_graph[module_info.path] = module_info
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")
        
        build_time = (time.time() - start_time) * 1000
        print(f"✅ Module graph built: {len(self.module_graph)} modules in {build_time:.1f}ms")
        
        # Cache the module graph
        self._cache_module_graph()
    
    def _cache_module_graph(self):
        """Cache the module graph for faster subsequent loads"""
        cache_file = self.cache_dir / "module_graph.json"
        
        # Convert to JSON-serializable format
        serializable_graph = {
            path: asdict(module_info) 
            for path, module_info in self.module_graph.items()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_graph, f, indent=2)
    
    def _load_existing_contexts(self):
        """Load all existing context files into memory"""
        for context_file in self.context_dir.glob("*.md"):
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    self.context_cache[context_file.stem] = f.read()
            except Exception as e:
                print(f"⚠️  Error loading context {context_file}: {e}")
    
    def get_context_for_module(self, module_path: str) -> ContextBundle:
        """Get intelligent context for a specific module"""
        start_time = time.time()
        
        # Normalize module path
        if module_path.startswith('./'):
            module_path = module_path[2:]
        
        module_info = self.module_graph.get(module_path)
        if not module_info:
            return self._create_fallback_context(module_path, start_time)
        
        # Determine primary context based on module type
        primary_context = self._get_primary_context(module_info)
        
        # Get related contexts based on dependencies and type
        related_contexts = self._get_related_contexts(module_info)
        
        # Get module dependencies
        dependencies = self._resolve_dependencies(module_info)
        
        # Calculate metrics
        load_time_ms = (time.time() - start_time) * 1000
        
        context_bundle = ContextBundle(
            query=f"Module context for {module_path}",
            primary_context=primary_context,
            related_contexts=related_contexts,
            module_dependencies=dependencies,
            performance_metrics=self._get_performance_metrics(module_info),
            load_time_ms=load_time_ms,
            relevance_score=module_info.trading_relevance,
            token_count=len(primary_context.split()) + sum(len(ctx.split()) for ctx in related_contexts)
        )
        
        # Update performance metrics
        self._update_performance_metrics(load_time_ms)
        
        return context_bundle
    
    def get_context_for_query(self, query: str) -> ContextBundle:
        """Get intelligent context based on a natural language query"""
        start_time = time.time()
        
        # Analyze query for intent and relevant modules
        relevant_modules = self._analyze_query_intent(query)
        
        # Build comprehensive context
        contexts = []
        dependencies = []
        total_relevance = 0.0
        
        for module_path, relevance in relevant_modules[:5]:  # Top 5 most relevant
            module_info = self.module_graph.get(module_path)
            if module_info:
                context = self._get_primary_context(module_info)
                contexts.append(context)
                dependencies.extend(module_info.dependencies)
                total_relevance += relevance
        
        primary_context = contexts[0] if contexts else "No relevant context found."
        related_contexts = contexts[1:] if len(contexts) > 1 else []
        
        load_time_ms = (time.time() - start_time) * 1000
        
        context_bundle = ContextBundle(
            query=query,
            primary_context=primary_context,
            related_contexts=related_contexts,
            module_dependencies=list(set(dependencies)),
            performance_metrics={'query_analysis_time_ms': load_time_ms},
            load_time_ms=load_time_ms,
            relevance_score=total_relevance / max(len(relevant_modules), 1),
            token_count=sum(len(ctx.split()) for ctx in contexts)
        )
        
        self._update_performance_metrics(load_time_ms)
        return context_bundle
    
    def _get_primary_context(self, module_info: ModuleInfo) -> str:
        """Get the primary context for a module based on its type"""
        context_mapping = {
            'trading': 'trading_strategies',
            'risk': 'risk_management', 
            'ml': 'machine_learning',
            'data': 'data_pipeline',
            'execution': 'order_execution',
            'analytics': 'performance_analytics'
        }
        
        context_name = context_mapping.get(module_info.type, 'trading_strategies')
        return self.context_cache.get(context_name, f"No context found for {context_name}")
    
    def _get_related_contexts(self, module_info: ModuleInfo) -> List[str]:
        """Get related contexts based on module dependencies and type"""
        related = []
        
        # Always include master context for high-priority modules
        if module_info.priority in ['CRITICAL', 'HIGH']:
            master_context = self._load_file_content(self.project_root / "CLAUDE.md")
            if master_context:
                related.append(master_context)
        
        # Add type-specific related contexts
        type_relations = {
            'trading': ['risk_management', 'machine_learning', 'order_execution'],
            'risk': ['trading_strategies', 'performance_analytics'],
            'ml': ['trading_strategies', 'data_pipeline'],
            'data': ['machine_learning', 'trading_strategies'],
            'execution': ['trading_strategies', 'risk_management'],
            'analytics': ['trading_strategies', 'risk_management']
        }
        
        for related_type in type_relations.get(module_info.type, []):
            context_content = self.context_cache.get(related_type)
            if context_content:
                related.append(context_content)
        
        return related[:3]  # Limit to top 3 related contexts
    
    def _resolve_dependencies(self, module_info: ModuleInfo) -> List[str]:
        """Resolve and return dependencies with their context relevance"""
        resolved = []
        for dep_path in module_info.dependencies:
            dep_info = self.module_graph.get(dep_path)
            if dep_info and dep_info.trading_relevance > 0.3:
                resolved.append(dep_path)
        return resolved
    
    def _analyze_query_intent(self, query: str) -> List[tuple]:
        """Analyze query and return ranked list of relevant modules"""
        query_lower = query.lower()
        scored_modules = []
        
        for module_path, module_info in self.module_graph.items():
            score = 0.0
            
            # Score based on trading relevance
            score += module_info.trading_relevance * 0.3
            
            # Score based on query keyword matches
            for keyword in query_lower.split():
                if keyword in module_info.path.lower():
                    score += 0.5
                if keyword in ' '.join(module_info.classes + module_info.functions).lower():
                    score += 0.3
                if module_info.docstring and keyword in module_info.docstring.lower():
                    score += 0.2
            
            # Boost score for high-priority modules
            priority_boost = {'CRITICAL': 0.4, 'HIGH': 0.2, 'MEDIUM': 0.1, 'LOW': 0.0}
            score += priority_boost.get(module_info.priority, 0.0)
            
            if score > 0.1:  # Only include modules with meaningful relevance
                scored_modules.append((module_path, score))
        
        # Return sorted by score (highest first)
        return sorted(scored_modules, key=lambda x: x[1], reverse=True)
    
    def _get_performance_metrics(self, module_info: ModuleInfo) -> Dict[str, float]:
        """Get performance-related metrics for the module"""
        return {
            'trading_relevance': module_info.trading_relevance,
            'lines_of_code': float(module_info.lines_of_code),
            'last_modified_hours_ago': (time.time() - module_info.last_modified) / 3600,
            'dependency_count': float(len(module_info.dependencies))
        }
    
    def _update_performance_metrics(self, load_time_ms: float):
        """Update system performance metrics"""
        self.performance_metrics['total_loads'] += 1
        self.performance_metrics['total_load_time_ms'] += load_time_ms
        self.performance_metrics['avg_load_time_ms'] = (
            self.performance_metrics['total_load_time_ms'] / 
            self.performance_metrics['total_loads']
        )
    
    def _create_fallback_context(self, module_path: str, start_time: float) -> ContextBundle:
        """Create a fallback context when module is not found"""
        load_time_ms = (time.time() - start_time) * 1000
        
        return ContextBundle(
            query=f"Module context for {module_path}",
            primary_context=f"Module not found: {module_path}",
            related_contexts=[],
            module_dependencies=[],
            performance_metrics={},
            load_time_ms=load_time_ms,
            relevance_score=0.0,
            token_count=0
        )
    
    def _load_file_content(self, file_path: Path) -> Optional[str]:
        """Load content from a file safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def get_system_performance(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive system performance metrics"""
        return {
            **self.performance_metrics,
            'modules_analyzed': len(self.module_graph),
            'contexts_cached': len(self.context_cache),
            'critical_modules': len([m for m in self.module_graph.values() if m.priority == 'CRITICAL']),
            'high_priority_modules': len([m for m in self.module_graph.values() if m.priority == 'HIGH']),
            'avg_trading_relevance': sum(m.trading_relevance for m in self.module_graph.values()) / len(self.module_graph)
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧠 ULTRATHINK Context Loader - Dynamic Loading Framework")
    print("=" * 60)
    
    # Initialize the context loader
    loader = UltraThinkContextLoader()
    
    # Example: Get context for a specific module
    print("\n📋 Testing module-specific context loading...")
    context = loader.get_context_for_module("enhanced_paper_trader_24h.py")
    print(f"Load time: {context.load_time_ms:.1f}ms")
    print(f"Relevance: {context.relevance_score:.2f}")
    print(f"Token count: {context.token_count}")
    
    # Example: Get context for a query
    print("\n🔍 Testing query-based context loading...")
    context = loader.get_context_for_query("How do I optimize Kelly criterion position sizing?")
    print(f"Load time: {context.load_time_ms:.1f}ms")
    print(f"Relevance: {context.relevance_score:.2f}")
    print(f"Modules found: {len(context.module_dependencies)}")
    
    # Performance summary
    print("\n📊 System Performance Summary:")
    perf = loader.get_system_performance()
    for key, value in perf.items():
        print(f"  {key}: {value}")