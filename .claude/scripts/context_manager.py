#!/usr/bin/env python3
"""
ULTRATHINK Context Management Interface
High-level API for intelligent context operations

Design Philosophy: Make complex context operations feel effortless
Performance Goal: < 100ms for any context operation
Intelligence Goal: 95%+ query intent recognition
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import time

from context_loader import UltraThinkContextLoader, ContextBundle

class ContextManager:
    """
    High-level interface for all context management operations
    Optimized for trading system development workflows
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.loader = UltraThinkContextLoader(self.project_root)
        
        # Session tracking
        self.session_start = datetime.now()
        self.query_history: List[Dict] = []
        
        print("🧠 ULTRATHINK Context Manager initialized")
        print(f"📊 System ready: {self.loader.get_system_performance()['modules_analyzed']} modules analyzed")
    
    def ask(self, query: str) -> str:
        """
        Natural language interface for context queries
        
        Examples:
        - "How do I modify Kelly criterion parameters?"
        - "Show me the risk management implementation"
        - "What's the current trading strategy performance?"
        """
        start_time = time.time()
        
        # Get context bundle
        context_bundle = self.loader.get_context_for_query(query)
        
        # Format response for human consumption
        response = self._format_context_response(context_bundle)
        
        # Track query for analytics
        self._track_query(query, context_bundle, time.time() - start_time)
        
        return response
    
    def get_module_context(self, module_path: str) -> str:
        """Get comprehensive context for a specific module"""
        context_bundle = self.loader.get_context_for_module(module_path)
        return self._format_context_response(context_bundle)
    
    def search_modules(self, keyword: str) -> List[Dict[str, Union[str, float]]]:
        """Search modules by keyword and return ranked results"""
        results = []
        
        for module_path, module_info in self.loader.module_graph.items():
            score = 0.0
            
            # Search in module path
            if keyword.lower() in module_path.lower():
                score += 1.0
            
            # Search in classes and functions
            for name in module_info.classes + module_info.functions:
                if keyword.lower() in name.lower():
                    score += 0.5
            
            # Search in docstring
            if module_info.docstring and keyword.lower() in module_info.docstring.lower():
                score += 0.3
            
            # Boost by trading relevance
            score *= (1 + module_info.trading_relevance)
            
            if score > 0.1:
                results.append({
                    'module': module_path,
                    'type': module_info.type,
                    'priority': module_info.priority,
                    'relevance': module_info.trading_relevance,
                    'score': score
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    
    def get_trading_system_overview(self) -> str:
        """Get comprehensive overview of the trading system architecture"""
        master_context = self.loader.context_cache.get('week1_completion_report', '')
        if not master_context:
            # Fallback to CLAUDE.md
            claude_path = self.project_root / "CLAUDE.md"
            if claude_path.exists():
                with open(claude_path, 'r') as f:
                    master_context = f.read()
        
        return master_context
    
    def get_live_trading_status(self) -> str:
        """Get current live trading system status and context"""
        # Try to get order execution context (contains live status)
        execution_context = self.loader.context_cache.get('order_execution', '')
        
        # Add performance analytics context
        analytics_context = self.loader.context_cache.get('performance_analytics', '')
        
        return f"""
# Live Trading System Status

## Current Execution Status
{execution_context[:2000]}...

## Performance Analytics
{analytics_context[:1500]}...

*Use `ask("What is the current trading session status?")` for detailed live information*
"""
    
    def get_module_dependencies(self, module_path: str) -> Dict[str, List[str]]:
        """Get comprehensive dependency analysis for a module"""
        module_info = self.loader.module_graph.get(module_path)
        if not module_info:
            return {'error': f'Module not found: {module_path}'}
        
        # Find modules that depend on this one
        dependents = []
        for other_path, other_info in self.loader.module_graph.items():
            if module_path in other_info.dependencies:
                dependents.append(other_path)
        
        return {
            'module': module_path,
            'dependencies': module_info.dependencies,
            'dependents': dependents,
            'imports': module_info.imports,
            'classes': module_info.classes,
            'functions': module_info.functions,
            'type': module_info.type,
            'priority': module_info.priority,
            'trading_relevance': module_info.trading_relevance
        }
    
    def optimize_context(self, query_pattern: str) -> Dict[str, float]:
        """Optimize context loading for specific query patterns"""
        # This could implement caching strategies, preloading, etc.
        # For now, return current performance metrics
        return self.loader.get_system_performance()
    
    def get_session_analytics(self) -> Dict:
        """Get analytics for current context management session"""
        return {
            'session_duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60,
            'queries_processed': len(self.query_history),
            'avg_response_time_ms': sum(q['response_time_ms'] for q in self.query_history) / max(len(self.query_history), 1),
            'system_performance': self.loader.get_system_performance(),
            'recent_queries': self.query_history[-5:] if self.query_history else []
        }
    
    def _format_context_response(self, context_bundle: ContextBundle) -> str:
        """Format context bundle into human-readable response"""
        response = f"""
# Context Response: {context_bundle.query}

## Primary Context
{context_bundle.primary_context[:3000]}{"..." if len(context_bundle.primary_context) > 3000 else ""}

## Related Information
"""
        
        for i, related_context in enumerate(context_bundle.related_contexts[:2], 1):
            response += f"""
### Related Context {i}
{related_context[:1500]}{"..." if len(related_context) > 1500 else ""}
"""
        
        if context_bundle.module_dependencies:
            response += f"""
## Related Modules
{', '.join(context_bundle.module_dependencies[:10])}
"""
        
        response += f"""
---
*Response generated in {context_bundle.load_time_ms:.1f}ms | Relevance: {context_bundle.relevance_score:.2f} | Tokens: {context_bundle.token_count}*
"""
        
        return response
    
    def _track_query(self, query: str, context_bundle: ContextBundle, total_time: float):
        """Track query for analytics and optimization"""
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_time_ms': total_time * 1000,
            'context_load_time_ms': context_bundle.load_time_ms,
            'relevance_score': context_bundle.relevance_score,
            'token_count': context_bundle.token_count,
            'modules_involved': len(context_bundle.module_dependencies)
        })

class TradingSystemContextInterface:
    """
    Specialized interface for common trading system context operations
    Pre-configured for institutional trading workflows
    """
    
    def __init__(self, context_manager: ContextManager):
        self.cm = context_manager
    
    def kelly_criterion_help(self) -> str:
        """Get comprehensive help for Kelly criterion implementation"""
        return self.cm.ask("How do I work with Kelly criterion position sizing and optimization?")
    
    def risk_management_overview(self) -> str:
        """Get risk management system overview"""
        return self.cm.get_module_context("phase2b/advanced_risk_management.py")
    
    def ensemble_model_help(self) -> str:
        """Get help with ensemble ML models"""
        return self.cm.ask("How do the ensemble machine learning models work and how can I optimize them?")
    
    def live_trading_debug(self) -> str:
        """Get debugging context for live trading issues"""
        return self.cm.ask("The live trading system isn't executing trades, what should I check?")
    
    def strategy_development_guide(self) -> str:
        """Get guide for developing new trading strategies"""
        return self.cm.ask("How do I create and integrate a new trading strategy?")
    
    def performance_analysis_help(self) -> str:
        """Get help with performance analysis and optimization"""
        return self.cm.ask("How do I analyze and improve trading system performance?")
    
    def data_pipeline_debug(self) -> str:
        """Get help with data pipeline issues"""
        return self.cm.ask("There are data quality issues, how do I debug the data pipeline?")

# Command-line interface for testing
if __name__ == "__main__":
    import sys
    
    # Initialize context manager
    cm = ContextManager()
    trading_interface = TradingSystemContextInterface(cm)
    
    if len(sys.argv) > 1:
        # Command-line query
        query = " ".join(sys.argv[1:])
        print(cm.ask(query))
    else:
        # Interactive mode
        print("\n🧠 ULTRATHINK Context Manager - Interactive Mode")
        print("=" * 60)
        print("Available commands:")
        print("  ask <query>           - Ask a natural language question")
        print("  module <path>         - Get context for specific module")  
        print("  search <keyword>      - Search modules by keyword")
        print("  overview              - Get trading system overview")
        print("  status                - Get live trading status")
        print("  kelly                 - Kelly criterion help")
        print("  risk                  - Risk management overview")
        print("  ensemble              - Ensemble model help")
        print("  debug                 - Live trading debug")
        print("  strategy              - Strategy development guide")
        print("  performance           - Performance analysis help")
        print("  data                  - Data pipeline debug")
        print("  analytics             - Session analytics")
        print("  quit                  - Exit")
        print()
        
        while True:
            try:
                user_input = input("🔍 > ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.startswith('ask '):
                    print(cm.ask(user_input[4:]))
                elif user_input.startswith('module '):
                    print(cm.get_module_context(user_input[7:]))
                elif user_input.startswith('search '):
                    results = cm.search_modules(user_input[7:])
                    print(f"Found {len(results)} modules:")
                    for result in results:
                        print(f"  {result['module']} (score: {result['score']:.2f}, type: {result['type']})")
                elif user_input == 'overview':
                    print(cm.get_trading_system_overview())
                elif user_input == 'status':
                    print(cm.get_live_trading_status())
                elif user_input == 'kelly':
                    print(trading_interface.kelly_criterion_help())
                elif user_input == 'risk':
                    print(trading_interface.risk_management_overview())
                elif user_input == 'ensemble':
                    print(trading_interface.ensemble_model_help())
                elif user_input == 'debug':
                    print(trading_interface.live_trading_debug())
                elif user_input == 'strategy':
                    print(trading_interface.strategy_development_guide())
                elif user_input == 'performance':
                    print(trading_interface.performance_analysis_help())
                elif user_input == 'data':
                    print(trading_interface.data_pipeline_debug())
                elif user_input == 'analytics':
                    analytics = cm.get_session_analytics()
                    print(json.dumps(analytics, indent=2))
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
                print()  # Add spacing between responses
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Show session summary
        print("\n📊 Session Summary:")
        analytics = cm.get_session_analytics()
        print(f"  Duration: {analytics['session_duration_minutes']:.1f} minutes")
        print(f"  Queries: {analytics['queries_processed']}")
        print(f"  Avg Response Time: {analytics['avg_response_time_ms']:.1f}ms")