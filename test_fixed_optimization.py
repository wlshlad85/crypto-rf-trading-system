"""Test the fixed optimization with reduced scope to verify it works."""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config, Config
from advanced_hyperparameter_optimization import AdvancedHyperparameterOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_fixed_optimization():
    """Test the fixed optimization with minimal scope."""
    print("\n" + "="*60)
    print("🧪 TESTING FIXED OPTIMIZATION")
    print("="*60 + "\n")
    
    print("📋 Test Configuration:")
    print("   • Trials: 10 (minimal for testing)")
    print("   • Symbols: bitcoin, ethereum (2 symbols)")
    print("   • Data: 30 days")
    print("   • Focus: Verify no -100 penalty returns")
    
    try:
        # Create optimizer with reduced scope
        config = get_default_config()
        config.data.days = 30  # Only 30 days for quick test
        config.data.symbols = ['bitcoin', 'ethereum']  # Only 2 symbols
        
        optimizer = AdvancedHyperparameterOptimizer(config)
        
        # Run minimal optimization
        print("\n🚀 Starting test optimization...")
        results = optimizer.run_advanced_optimization(n_trials=10)  # Very small for testing
        
        # Analyze results
        print("\n" + "="*60)
        print("📊 TEST RESULTS ANALYSIS")
        print("="*60)
        
        # Check model optimization
        if 'model' in optimizer.optimization_results:
            model_score = optimizer.optimization_results['model']['best_score']
            print(f"\n🤖 Model Optimization:")
            print(f"   • Best Score: {model_score:.3f}")
            if model_score > -50:
                print("   ✅ SUCCESS: Model optimization working (no -100 penalty)")
            else:
                print("   ❌ ISSUE: Model optimization still failing")
        
        # Check trading optimization  
        if 'trading' in optimizer.optimization_results:
            trading_score = optimizer.optimization_results['trading']['best_score']
            print(f"\n💼 Trading Optimization:")
            print(f"   • Best Score: {trading_score:.3f}")
            if trading_score > -50:
                print("   ✅ SUCCESS: Trading optimization working")
            else:
                print("   ❌ ISSUE: Trading optimization still failing")
        
        # Check multi-objective
        if 'multi_objective' in optimizer.optimization_results:
            mo_return = optimizer.optimization_results['multi_objective']['best_return']
            mo_sharpe = optimizer.optimization_results['multi_objective']['best_sharpe']
            print(f"\n🎯 Multi-Objective Optimization:")
            print(f"   • Best Return: {mo_return:.2f}%")
            print(f"   • Best Sharpe: {mo_sharpe:.2f}")
            if mo_return > -5 and mo_sharpe > -5:
                print("   ✅ SUCCESS: Multi-objective working")
            else:
                print("   ❌ ISSUE: Multi-objective still failing")
        
        # Check validation results
        if 'validation' in optimizer.optimization_results:
            print(f"\n✅ Validation Results:")
            validation_good = True
            for symbol, result in optimizer.optimization_results['validation'].items():
                if 'total_return' in result:
                    ret = result['total_return']
                    sharpe = result['sharpe_ratio']
                    print(f"   • {symbol.upper()}: {ret:.2f}% return, {sharpe:.2f} Sharpe")
                    # Check for reasonable values (not extreme like 146% or -284%)
                    if abs(ret) > 50 or abs(sharpe) > 10:
                        print(f"     ⚠️ WARNING: Extreme values detected")
                        validation_good = False
                else:
                    print(f"   • {symbol.upper()}: Validation failed")
                    validation_good = False
            
            if validation_good:
                print("   ✅ Validation results look reasonable")
            else:
                print("   ⚠️ Some validation issues detected")
        
        print("\n" + "="*60)
        print("🏁 TEST CONCLUSION")
        print("="*60)
        
        # Overall assessment
        overall_success = True
        if 'model' in optimizer.optimization_results:
            if optimizer.optimization_results['model']['best_score'] <= -50:
                overall_success = False
        
        if 'trading' in optimizer.optimization_results:
            if optimizer.optimization_results['trading']['best_score'] <= -50:
                overall_success = False
        
        if overall_success:
            print("✅ OPTIMIZATION FIX SUCCESSFUL!")
            print("   The numeric column filtering resolved the optimization failures.")
            print("   Ready to run full optimization with larger scope.")
        else:
            print("❌ OPTIMIZATION STILL HAS ISSUES")
            print("   Further debugging may be needed.")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        logger.error(f"Test optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_fixed_optimization()
    
    if success:
        print("\n🎉 Test completed successfully! Optimization is fixed.")
        print("💡 Next step: Run full optimization with more trials and data.")
    else:
        print("\n⚠️ Test revealed remaining issues. Check logs for details.")
    
    return success


if __name__ == "__main__":
    main()