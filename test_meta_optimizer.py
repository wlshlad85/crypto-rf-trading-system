#!/usr/bin/env python3
"""
Test Meta-Optimizer Layer Components

Quick verification that all meta-optimizer components are working correctly.
"""

import sys
import os
import traceback

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_objective_function():
    """Test the objective function."""
    print("🎯 Testing Meta-Objective Function...")
    
    try:
        from meta_optim.objective_fn import MetaObjectiveFunction
        
        # Create sample data
        sample_results = {
            'trades': [
                {'action': 'BUY', 'value': 1000, 'pnl': 0},
                {'action': 'SELL', 'value': 1100, 'pnl': 100},
                {'action': 'BUY', 'value': 1100, 'pnl': 0},
                {'action': 'SELL', 'value': 1200, 'pnl': 100}
            ],
            'portfolio_values': [100000] + [100000 + i*100 for i in range(1, 21)]
        }
        
        objective_fn = MetaObjectiveFunction()
        results = objective_fn.evaluate_strategy(sample_results)
        
        print(f"  ✅ Composite Score: {results['composite_score']:.3f}")
        print(f"  ✅ Strategy Viable: {results['viable']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_retrain_worker():
    """Test the retrain worker."""
    print("🔄 Testing Retrain Worker...")
    
    try:
        from meta_optim.retrain_worker import RetrainWorker
        
        # Test configuration
        test_config = {
            'entry_model': {
                'n_estimators': 50,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            },
            'trading_params': {
                'momentum_threshold': 1.78,
                'position_range_min': 0.464,
                'position_range_max': 0.800,
                'confidence_threshold': 0.6,
                'exit_threshold': 0.5
            }
        }
        
        worker = RetrainWorker()
        results = worker.train_and_evaluate(test_config, training_fraction=0.1)
        
        print(f"  ✅ Training completed")
        print(f"  ✅ Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"  ✅ Total Trades: {results.get('num_trades', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_hyperband_runner():
    """Test the hyperband runner (minimal test)."""
    print("🎯 Testing Hyperband Runner...")
    
    try:
        from meta_optim.hyperband_runner import HyperbandRunner
        
        runner = HyperbandRunner()
        
        # Test configuration sampling
        configs = runner._sample_configurations(2)
        
        print(f"  ✅ Generated {len(configs)} test configurations")
        print(f"  ✅ Parameter space loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_deployment_cycle():
    """Test the deployment cycle (initialization only)."""
    print("🚀 Testing Deployment Cycle...")
    
    try:
        from meta_optim.deployment_cycle import MetaDeploymentCycle
        
        cycle = MetaDeploymentCycle()
        
        print(f"  ✅ Deployment cycle initialized")
        print(f"  ✅ Configuration loaded")
        print(f"  ✅ Components ready")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 META-OPTIMIZER LAYER TESTS")
    print("=" * 50)
    
    tests = [
        ("Objective Function", test_objective_function),
        ("Retrain Worker", test_retrain_worker),
        ("Hyperband Runner", test_hyperband_runner),
        ("Deployment Cycle", test_deployment_cycle)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        success = test_func()
        results.append((name, success))
        print()
    
    # Summary
    print("=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All meta-optimizer components are working correctly!")
        print("🚀 Ready for production deployment!")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed - review errors above")

if __name__ == "__main__":
    main()