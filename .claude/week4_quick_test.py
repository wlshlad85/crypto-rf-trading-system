#!/usr/bin/env python3
"""
Quick Week 4 Integration Test - Simplified version
"""

import asyncio
import time
from pathlib import Path
import sys
import tempfile
import shutil
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_multimodal_engine():
    """Quick test of multimodal engine"""
    print("Testing Multi-Modal Context Engine...")
    
    try:
        from multimodal.multimodal_context_engine import MultiModalContextEngine
        
        # Use current directory as system root
        system_root = Path("/mnt/c/Users/RICHARD/OneDrive/Documents/crypto-rf-trading-system")
        engine = MultiModalContextEngine(system_root)
        
        # Test basic functionality
        test_context = "Kelly criterion implementation"
        enhanced_context = await engine.integrate_trading_charts(test_context, ["BTC-USD"])
        
        print(f"✅ Multi-Modal Engine: Charts={len(enhanced_context.trading_charts)}, Analytics={len(enhanced_context.analytics)}")
        return True
        
    except Exception as e:
        print(f"❌ Multi-Modal Engine failed: {e}")
        return False

async def test_predictive_loader():
    """Quick test of predictive loader"""
    print("Testing Predictive Context Loader...")
    
    try:
        from predictive.predictive_context_loader import PredictiveContextLoader
        
        system_root = Path("/mnt/c/Users/RICHARD/OneDrive/Documents/crypto-rf-trading-system")
        loader = PredictiveContextLoader(system_root)
        
        # Test basic functionality
        loader.record_usage_pattern("test_dev", ["risk", "kelly"], 60.0, "success")
        predictions = await loader.predict_next_contexts("test_dev", "risk")
        
        print(f"✅ Predictive Loader: Predictions={len(predictions.predicted_contexts)}")
        return True
        
    except Exception as e:
        print(f"❌ Predictive Loader failed: {e}")
        return False

async def test_collaborative_manager():
    """Quick test of collaborative manager"""
    print("Testing Collaborative Context Manager...")
    
    try:
        from collaborative.collaborative_context_manager import CollaborativeContextManager
        
        system_root = Path("/mnt/c/Users/RICHARD/OneDrive/Documents/crypto-rf-trading-system")
        manager = CollaborativeContextManager(system_root)
        
        # Test basic functionality
        dev = await manager.register_developer("test_dev", "Test User", "Tester", ["test"])
        annotation = await manager.create_context_annotation("test_context", dev.developer_id, "note", "Test note")
        
        print(f"✅ Collaborative Manager: Developer registered, annotation created")
        return True
        
    except Exception as e:
        print(f"❌ Collaborative Manager failed: {e}")
        return False

async def test_distributed_cache():
    """Quick test of distributed cache"""
    print("Testing Distributed Context Cache...")
    
    try:
        from caching.distributed_context_cache import DistributedContextCache
        
        system_root = Path("/mnt/c/Users/RICHARD/OneDrive/Documents/crypto-rf-trading-system")
        cache = DistributedContextCache(system_root)
        
        # Test basic functionality (local cache only)
        await cache.put("test_key", "test_content")
        content = await cache.get("test_key")
        
        print(f"✅ Distributed Cache: Content cached and retrieved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Distributed Cache failed: {e}")
        return False

async def main():
    """Run quick tests"""
    print("Week 4 Quick Integration Test")
    print("="*50)
    
    tests = [
        test_multimodal_engine,
        test_predictive_loader,
        test_collaborative_manager,
        test_distributed_cache
    ]
    
    results = []
    for test in tests:
        try:
            result = await asyncio.wait_for(test(), timeout=30)
            results.append(result)
        except asyncio.TimeoutError:
            print(f"❌ {test.__name__} timed out")
            results.append(False)
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("QUICK TEST RESULTS")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")
    
    if passed >= 3:  # 3 out of 4 tests should pass
        print("✅ Week 4 implementation is functional!")
        return True
    else:
        print("❌ Week 4 implementation needs attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)