#!/usr/bin/env python3
"""
ULTRATHINK Context System Testing & Validation
Week 2 Day 8-9: Validate dynamic context loading implementation
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from context_loader import UltraThinkContextLoader
from context_manager import ContextManager

def test_context_system():
    """Comprehensive test of the context loading system"""
    print("🧪 ULTRATHINK Context System Testing")
    print("=" * 50)
    
    # Test 1: Initialize system
    print("\n📋 Test 1: System Initialization")
    try:
        loader = UltraThinkContextLoader()
        print(f"✅ Loader initialized: {len(loader.module_graph)} modules found")
        print(f"✅ Contexts cached: {len(loader.context_cache)} contexts")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test 2: Module graph analysis
    print("\n📋 Test 2: Module Graph Analysis")
    critical_modules = [m for m in loader.module_graph.values() if m.priority == 'CRITICAL']
    high_modules = [m for m in loader.module_graph.values() if m.priority == 'HIGH']
    
    print(f"✅ Critical modules: {len(critical_modules)}")
    print(f"✅ High priority modules: {len(high_modules)}")
    
    for module in critical_modules[:3]:
        print(f"   - {module.path} (relevance: {module.trading_relevance:.2f})")
    
    # Test 3: Context loading performance
    print("\n📋 Test 3: Context Loading Performance")
    test_modules = [
        "enhanced_paper_trader_24h.py",
        "phase2b/advanced_risk_management.py",
        "strategies/long_short_strategy.py"
    ]
    
    for module_path in test_modules:
        try:
            context = loader.get_context_for_module(module_path)
            print(f"✅ {module_path}: {context.load_time_ms:.1f}ms, relevance: {context.relevance_score:.2f}")
        except Exception as e:
            print(f"❌ {module_path}: {e}")
    
    # Test 4: Query-based context retrieval
    print("\n📋 Test 4: Query-Based Context Retrieval")
    test_queries = [
        "Kelly criterion position sizing",
        "Risk management parameters",
        "Live trading execution",
        "Ensemble model accuracy"
    ]
    
    for query in test_queries:
        try:
            context = loader.get_context_for_query(query)
            print(f"✅ '{query}': {context.load_time_ms:.1f}ms, relevance: {context.relevance_score:.2f}")
        except Exception as e:
            print(f"❌ '{query}': {e}")
    
    # Test 5: High-level Context Manager
    print("\n📋 Test 5: Context Manager Interface")
    try:
        cm = ContextManager()
        
        # Test search functionality
        results = cm.search_modules("trading")
        print(f"✅ Module search: Found {len(results)} trading-related modules")
        
        # Test performance
        perf = cm.get_session_analytics()
        print(f"✅ Session analytics: {perf['queries_processed']} queries processed")
        
    except Exception as e:
        print(f"❌ Context Manager failed: {e}")
    
    # Test 6: Performance Benchmarks
    print("\n📋 Test 6: Performance Benchmarks")
    performance = loader.get_system_performance()
    
    print(f"✅ Average load time: {performance['avg_load_time_ms']:.1f}ms")
    print(f"✅ Modules analyzed: {performance['modules_analyzed']}")
    print(f"✅ Average trading relevance: {performance['avg_trading_relevance']:.2f}")
    
    # Performance targets validation
    targets_met = []
    targets_met.append(("Load time < 500ms", performance['avg_load_time_ms'] < 500))
    targets_met.append(("Modules > 90", performance['modules_analyzed'] > 90))
    targets_met.append(("Trading relevance > 0.8", performance['avg_trading_relevance'] > 0.8))
    
    print("\n🎯 Performance Targets:")
    for target, met in targets_met:
        status = "✅" if met else "❌"
        print(f"   {status} {target}")
    
    # Summary
    print("\n📊 Test Summary:")
    all_passed = all(met for _, met in targets_met)
    print(f"Status: {'✅ ALL TESTS PASSED' if all_passed else '⚠️  SOME TESTS FAILED'}")
    
    return all_passed

def demonstrate_trading_queries():
    """Demonstrate trading-specific query capabilities"""
    print("\n🎯 TRADING SYSTEM QUERY DEMONSTRATION")
    print("=" * 50)
    
    cm = ContextManager()
    
    # Common trading system queries
    demo_queries = [
        "What is the momentum threshold for trading?",
        "How does Kelly criterion work in this system?", 
        "Why isn't the live trader executing trades?",
        "How do I analyze ensemble model performance?",
        "What are the current risk management settings?"
    ]
    
    for query in demo_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        try:
            response = cm.ask(query)
            # Show first 200 characters of response
            preview = response[:200] + "..." if len(response) > 200 else response
            print(preview)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run comprehensive tests
    success = test_context_system()
    
    if success:
        print("\n🚀 System validated! Running trading query demonstration...")
        demonstrate_trading_queries()
    else:
        print("\n⚠️  System validation failed. Check errors above.")
    
    print("\n🧠 ULTRATHINK Week 2 Day 8-9 Testing Complete")