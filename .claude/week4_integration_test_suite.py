#!/usr/bin/env python3
"""
ULTRATHINK Week 4 Integration Test Suite
Comprehensive testing for advanced context intelligence features
"""

import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Week 4 components
from multimodal.multimodal_context_engine import MultiModalContextEngine
from predictive.predictive_context_loader import PredictiveContextLoader
from collaborative.collaborative_context_manager import CollaborativeContextManager
from caching.distributed_context_cache import DistributedContextCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Week4IntegrationTestSuite:
    """Comprehensive integration test suite for Week 4 features"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.temp_dir = None
        self.system_root = None
        
    async def setup_test_environment(self):
        """Setup test environment with temporary directories"""
        self.temp_dir = tempfile.mkdtemp(prefix="ultrathink_test_")
        self.system_root = Path(self.temp_dir)
        
        # Create test data structure
        (self.system_root / "data").mkdir(parents=True)
        (self.system_root / "logs").mkdir(parents=True)
        (self.system_root / "analytics").mkdir(parents=True)
        
        # Create test data files
        await self._create_test_data()
        
        logger.info(f"Test environment setup at: {self.system_root}")
        
    async def _create_test_data(self):
        """Create test data files"""
        # Create sample trading data
        import pandas as pd
        import numpy as np
        
        # Generate sample BTC data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
        np.random.seed(42)
        
        prices = 45000 * (1 + np.random.normal(0, 0.02, 1000)).cumprod()
        
        btc_data = pd.DataFrame({
            'Date': dates,
            'Open': prices * np.random.uniform(0.99, 1.01, 1000),
            'High': prices * np.random.uniform(1.0, 1.05, 1000),
            'Low': prices * np.random.uniform(0.95, 1.0, 1000),
            'Close': prices,
            'Volume': np.random.exponential(1000000, 1000)
        })
        
        btc_data.to_csv(self.system_root / "data" / "btc_data.csv", index=False)
        
        # Create sample log files
        log_content = """
        2024-01-01 10:00:00 - INFO - Trading session started
        2024-01-01 10:01:00 - INFO - Kelly criterion calculated: 0.25
        2024-01-01 10:02:00 - INFO - Position opened: BTC-USD
        """
        
        with open(self.system_root / "logs" / "trading.log", 'w') as f:
            f.write(log_content)
            
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting Week 4 Integration Test Suite...")
        
        await self.setup_test_environment()
        
        try:
            # Test 1: Multi-Modal Context Engine
            await self.test_multimodal_context_engine()
            
            # Test 2: Predictive Context Loader
            await self.test_predictive_context_loader()
            
            # Test 3: Collaborative Context Manager
            await self.test_collaborative_context_manager()
            
            # Test 4: Distributed Context Cache
            await self.test_distributed_context_cache()
            
            # Test 5: Integration Testing
            await self.test_component_integration()
            
            # Test 6: Performance Validation
            await self.test_performance_requirements()
            
            # Test 7: Live Trading Integration
            await self.test_live_trading_integration()
            
            # Test 8: Error Handling and Recovery
            await self.test_error_handling()
            
        finally:
            await self.cleanup_test_environment()
            
        return self._generate_test_report()
        
    async def test_multimodal_context_engine(self):
        """Test multi-modal context engine functionality"""
        logger.info("Testing Multi-Modal Context Engine...")
        
        test_name = "multimodal_context_engine"
        start_time = time.time()
        
        try:
            # Initialize engine
            engine = MultiModalContextEngine(self.system_root)
            
            # Test chart integration
            test_context = "Kelly criterion implementation for position sizing"
            enhanced_context = await engine.integrate_trading_charts(
                test_context, ["BTC-USD"]
            )
            
            # Validate results
            assert enhanced_context.text_content == test_context
            assert len(enhanced_context.trading_charts) > 0
            assert len(enhanced_context.analytics) > 0
            assert enhanced_context.market_data is not None
            assert enhanced_context.confidence_score > 0.5
            
            # Test visual rendering
            visual_response = await engine.render_context_with_visuals(
                test_context, ["BTC-USD"]
            )
            
            assert len(visual_response) > 100  # Should have substantial content
            assert "Trading Charts" in visual_response
            assert "Analytics" in visual_response
            assert "Market Context" in visual_response
            
            # Test performance
            performance_summary = engine.get_performance_summary()
            assert "performance_metrics" in performance_summary
            assert "data_sources" in performance_summary
            
            execution_time = time.time() - start_time
            assert execution_time < 1.0  # Should complete within 1 second
            
            self.test_results[test_name] = {
                "status": "PASS",
                "execution_time": execution_time,
                "charts_generated": len(enhanced_context.trading_charts),
                "analytics_generated": len(enhanced_context.analytics),
                "confidence_score": enhanced_context.confidence_score
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_predictive_context_loader(self):
        """Test predictive context loading system"""
        logger.info("Testing Predictive Context Loader...")
        
        test_name = "predictive_context_loader"
        start_time = time.time()
        
        try:
            # Initialize loader
            loader = PredictiveContextLoader(self.system_root)
            
            # Test usage pattern recording
            developer_id = "test_developer"
            context_sequence = ["risk_management", "kelly_criterion", "position_sizing"]
            
            loader.record_usage_pattern(developer_id, context_sequence, 120.0, "success")
            
            # Test context prediction
            predictions = await loader.predict_next_contexts(
                developer_id, "risk_management", 3
            )
            
            # Validate predictions
            assert len(predictions.predicted_contexts) > 0
            assert len(predictions.confidence_scores) > 0
            assert predictions.prediction_time < 0.5  # Should be fast
            assert predictions.cache_strategy in ["aggressive", "moderate", "conservative", "minimal", "none"]
            
            # Test cache operations
            cache_status = loader.get_cache_status()
            assert "cache_size" in cache_status
            assert "performance_metrics" in cache_status
            
            # Test cache optimization
            await loader.optimize_cache()
            
            execution_time = time.time() - start_time
            
            self.test_results[test_name] = {
                "status": "PASS",
                "execution_time": execution_time,
                "predictions_generated": len(predictions.predicted_contexts),
                "prediction_time": predictions.prediction_time,
                "cache_strategy": predictions.cache_strategy
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_collaborative_context_manager(self):
        """Test collaborative context management system"""
        logger.info("Testing Collaborative Context Manager...")
        
        test_name = "collaborative_context_manager"
        start_time = time.time()
        
        try:
            # Initialize manager
            manager = CollaborativeContextManager(self.system_root)
            
            # Test developer registration
            dev1 = await manager.register_developer(
                "dev1", "Alice", "Senior Developer", ["risk_management", "ml_models"]
            )
            dev2 = await manager.register_developer(
                "dev2", "Bob", "Junior Developer", ["data_pipeline", "backtesting"]
            )
            
            assert dev1.developer_id == "dev1"
            assert dev2.developer_id == "dev2"
            
            # Test team session
            session = await manager.start_team_session("feature_dev", ["dev1", "dev2"])
            assert session.session_id is not None
            assert len(session.active_developers) == 2
            
            # Test context sharing
            success = await manager.share_context_with_team(
                "kelly_criterion", "dev1", ["dev2"]
            )
            assert success
            
            # Test annotations
            annotation = await manager.create_context_annotation(
                "kelly_criterion", "dev1", "note", "Remember validation", ["validation"]
            )
            assert annotation.annotation_id is not None
            
            # Test insights
            insight = await manager.create_context_insight(
                "kelly_criterion", "dev1", "best_practice", "Kelly Best Practices",
                "Use fractional Kelly", ["kelly_fraction = 0.25"]
            )
            assert insight.insight_id is not None
            
            # Test context subscriptions
            await manager.subscribe_to_context("kelly_criterion", "dev2")
            
            # Test analytics
            analytics = await manager.get_team_performance_analytics()
            assert "team_metrics" in analytics
            assert "collaboration_stats" in analytics
            
            # Test collaboration status
            status = manager.get_collaboration_status()
            assert "active_developers" in status
            assert "performance_metrics" in status
            
            execution_time = time.time() - start_time
            
            self.test_results[test_name] = {
                "status": "PASS",
                "execution_time": execution_time,
                "developers_registered": 2,
                "sessions_created": 1,
                "annotations_created": 1,
                "insights_created": 1
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_distributed_context_cache(self):
        """Test distributed context caching system"""
        logger.info("Testing Distributed Context Cache...")
        
        test_name = "distributed_context_cache"
        start_time = time.time()
        
        try:
            # Initialize cache
            cache = DistributedContextCache(self.system_root)
            
            # Test cache initialization (will fall back to local)
            await cache.initialize_distributed_cache()
            
            # Test cache operations
            test_content = "This is test content for Kelly criterion implementation"
            test_key = "kelly_criterion_test"
            
            # Test put operation
            put_success = await cache.put(test_key, test_content, ttl=3600)
            assert put_success
            
            # Test get operation
            retrieved_content = await cache.get(test_key)
            assert retrieved_content == test_content
            
            # Test cache miss
            missing_content = await cache.get("non_existent_key")
            assert missing_content is None
            
            # Test invalidation
            invalidate_success = await cache.invalidate(test_key)
            assert invalidate_success
            
            # Verify invalidation
            invalidated_content = await cache.get(test_key)
            assert invalidated_content is None
            
            # Test cache statistics
            stats = cache.get_cache_statistics()
            assert "node_info" in stats
            assert "performance_metrics" in stats
            assert "statistics" in stats
            
            # Test cache cleanup
            await cache.optimize_cache()
            
            execution_time = time.time() - start_time
            
            # Shutdown cache
            await cache.shutdown()
            
            self.test_results[test_name] = {
                "status": "PASS",
                "execution_time": execution_time,
                "cache_operations": 4,
                "hit_rate": stats["performance_metrics"]["hit_rate"],
                "local_cache_size": stats["node_info"]["local_cache_size"]
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_component_integration(self):
        """Test integration between Week 4 components"""
        logger.info("Testing Component Integration...")
        
        test_name = "component_integration"
        start_time = time.time()
        
        try:
            # Initialize all components
            multimodal_engine = MultiModalContextEngine(self.system_root)
            predictive_loader = PredictiveContextLoader(self.system_root)
            collaborative_manager = CollaborativeContextManager(self.system_root)
            distributed_cache = DistributedContextCache(self.system_root)
            
            await distributed_cache.initialize_distributed_cache()
            
            # Test integrated workflow
            
            # 1. Register developer
            developer = await collaborative_manager.register_developer(
                "integrated_dev", "Integration Test User", "Tester", ["testing"]
            )
            
            # 2. Record usage pattern
            predictive_loader.record_usage_pattern(
                developer.developer_id, 
                ["risk_management", "kelly_criterion"], 
                90.0, 
                "success"
            )
            
            # 3. Generate multi-modal context
            enhanced_context = await multimodal_engine.integrate_trading_charts(
                "Kelly criterion with risk management", ["BTC-USD"]
            )
            
            # 4. Cache the context
            cache_key = "integrated_test_context"
            await distributed_cache.put(
                cache_key, 
                enhanced_context.text_content, 
                ttl=3600
            )
            
            # 5. Predict next contexts
            predictions = await predictive_loader.predict_next_contexts(
                developer.developer_id, "risk_management"
            )
            
            # 6. Create annotation
            annotation = await collaborative_manager.create_context_annotation(
                cache_key, developer.developer_id, "note", "Integration test note", ["test"]
            )
            
            # 7. Retrieve from cache
            cached_content = await distributed_cache.get(cache_key)
            
            # Validate integration
            assert enhanced_context.confidence_score > 0.5
            assert len(predictions.predicted_contexts) > 0
            assert annotation.annotation_id is not None
            assert cached_content is not None
            
            # Test cross-component communication
            analytics = await collaborative_manager.get_team_performance_analytics()
            cache_stats = distributed_cache.get_cache_statistics()
            loader_status = predictive_loader.get_cache_status()
            
            assert "team_metrics" in analytics
            assert "performance_metrics" in cache_stats
            assert "cache_size" in loader_status
            
            execution_time = time.time() - start_time
            
            await distributed_cache.shutdown()
            
            self.test_results[test_name] = {
                "status": "PASS",
                "execution_time": execution_time,
                "components_integrated": 4,
                "workflow_steps": 7,
                "confidence_score": enhanced_context.confidence_score
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_performance_requirements(self):
        """Test performance requirements for Week 4 components"""
        logger.info("Testing Performance Requirements...")
        
        test_name = "performance_requirements"
        start_time = time.time()
        
        try:
            performance_results = {}
            
            # Test multi-modal context performance
            multimodal_engine = MultiModalContextEngine(self.system_root)
            
            multimodal_start = time.time()
            enhanced_context = await multimodal_engine.integrate_trading_charts(
                "Performance test context", ["BTC-USD"]
            )
            multimodal_time = time.time() - multimodal_start
            
            performance_results["multimodal_integration"] = {
                "time": multimodal_time,
                "target": 0.5,  # 500ms target
                "passed": multimodal_time < 0.5
            }
            
            # Test predictive context performance
            predictive_loader = PredictiveContextLoader(self.system_root)
            
            predictive_start = time.time()
            predictions = await predictive_loader.predict_next_contexts(
                "perf_test_dev", "risk_management"
            )
            predictive_time = time.time() - predictive_start
            
            performance_results["predictive_loading"] = {
                "time": predictive_time,
                "target": 0.15,  # 150ms target
                "passed": predictive_time < 0.15
            }
            
            # Test cache performance
            distributed_cache = DistributedContextCache(self.system_root)
            await distributed_cache.initialize_distributed_cache()
            
            cache_start = time.time()
            await distributed_cache.put("perf_test", "Performance test content")
            cache_put_time = time.time() - cache_start
            
            get_start = time.time()
            await distributed_cache.get("perf_test")
            cache_get_time = time.time() - get_start
            
            performance_results["cache_operations"] = {
                "put_time": cache_put_time,
                "get_time": cache_get_time,
                "put_target": 0.2,  # 200ms target
                "get_target": 0.15,  # 150ms target
                "put_passed": cache_put_time < 0.2,
                "get_passed": cache_get_time < 0.15
            }
            
            # Test collaborative operations
            collaborative_manager = CollaborativeContextManager(self.system_root)
            
            collab_start = time.time()
            await collaborative_manager.register_developer(
                "perf_dev", "Performance Test", "Tester", ["performance"]
            )
            collab_time = time.time() - collab_start
            
            performance_results["collaborative_operations"] = {
                "time": collab_time,
                "target": 0.1,  # 100ms target
                "passed": collab_time < 0.1
            }
            
            # Calculate overall performance score
            all_passed = all(
                result.get("passed", result.get("put_passed", False) and result.get("get_passed", False))
                for result in performance_results.values()
            )
            
            execution_time = time.time() - start_time
            
            await distributed_cache.shutdown()
            
            self.test_results[test_name] = {
                "status": "PASS" if all_passed else "FAIL",
                "execution_time": execution_time,
                "performance_results": performance_results,
                "overall_performance": "PASS" if all_passed else "FAIL"
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_live_trading_integration(self):
        """Test integration with live trading system"""
        logger.info("Testing Live Trading Integration...")
        
        test_name = "live_trading_integration"
        start_time = time.time()
        
        try:
            # Test zero-impact operation
            multimodal_engine = MultiModalContextEngine(self.system_root)
            
            # Simulate live trading session
            trading_active = True
            
            # Test context operations during trading
            context_operations = []
            
            for i in range(10):
                op_start = time.time()
                
                # Simulate context request
                enhanced_context = await multimodal_engine.integrate_trading_charts(
                    f"Live trading context {i}", ["BTC-USD"]
                )
                
                op_time = time.time() - op_start
                context_operations.append(op_time)
                
                # Ensure no interference with trading
                assert op_time < 0.1  # Should not impact trading latency
                
            # Test resource usage
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Validate resource constraints
            assert memory_usage < 500  # Should use less than 500MB
            
            # Test concurrent operations
            async def concurrent_context_operation():
                return await multimodal_engine.integrate_trading_charts(
                    "Concurrent context", ["BTC-USD"]
                )
                
            concurrent_start = time.time()
            concurrent_tasks = [concurrent_context_operation() for _ in range(5)]
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_time = time.time() - concurrent_start
            
            # Validate concurrent performance
            assert concurrent_time < 1.0  # Should complete within 1 second
            assert len(concurrent_results) == 5
            
            execution_time = time.time() - start_time
            
            self.test_results[test_name] = {
                "status": "PASS",
                "execution_time": execution_time,
                "average_operation_time": sum(context_operations) / len(context_operations),
                "max_operation_time": max(context_operations),
                "memory_usage_mb": memory_usage,
                "concurrent_operations": len(concurrent_results),
                "concurrent_time": concurrent_time
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def test_error_handling(self):
        """Test error handling and recovery"""
        logger.info("Testing Error Handling and Recovery...")
        
        test_name = "error_handling"
        start_time = time.time()
        
        try:
            error_scenarios = {}
            
            # Test multimodal engine error handling
            multimodal_engine = MultiModalContextEngine(self.system_root)
            
            # Test with invalid symbol
            try:
                await multimodal_engine.integrate_trading_charts(
                    "Test context", ["INVALID-SYMBOL"]
                )
                error_scenarios["invalid_symbol"] = "handled"
            except Exception as e:
                error_scenarios["invalid_symbol"] = f"error: {str(e)}"
                
            # Test predictive loader error handling
            predictive_loader = PredictiveContextLoader(self.system_root)
            
            # Test with invalid developer
            try:
                await predictive_loader.predict_next_contexts(
                    "non_existent_dev", "context"
                )
                error_scenarios["invalid_developer"] = "handled"
            except Exception as e:
                error_scenarios["invalid_developer"] = f"error: {str(e)}"
                
            # Test cache error handling
            distributed_cache = DistributedContextCache(self.system_root)
            
            # Test with invalid key
            try:
                await distributed_cache.get("")
                error_scenarios["invalid_cache_key"] = "handled"
            except Exception as e:
                error_scenarios["invalid_cache_key"] = f"error: {str(e)}"
                
            # Test network failure simulation
            try:
                # Simulate Redis connection failure
                await distributed_cache.initialize_distributed_cache(
                    "invalid_host", 9999
                )
                error_scenarios["network_failure"] = "handled"
            except Exception as e:
                error_scenarios["network_failure"] = f"error: {str(e)}"
                
            # Test recovery mechanisms
            recovery_tests = {}
            
            # Test cache fallback
            cache_fallback_start = time.time()
            await distributed_cache.put("fallback_test", "test content")
            fallback_content = await distributed_cache.get("fallback_test")
            recovery_tests["cache_fallback"] = {
                "success": fallback_content == "test content",
                "time": time.time() - cache_fallback_start
            }
            
            execution_time = time.time() - start_time
            
            # Count handled errors
            handled_errors = sum(1 for status in error_scenarios.values() if status == "handled")
            total_errors = len(error_scenarios)
            
            self.test_results[test_name] = {
                "status": "PASS" if handled_errors >= total_errors * 0.8 else "FAIL",
                "execution_time": execution_time,
                "error_scenarios": error_scenarios,
                "recovery_tests": recovery_tests,
                "error_handling_rate": handled_errors / total_errors if total_errors > 0 else 1.0
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
            
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        total_execution_time = sum(
            result["execution_time"] for result in self.test_results.values()
        )
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_execution_time
            },
            "individual_results": self.test_results,
            "week4_features_status": {
                "multimodal_context": self.test_results.get("multimodal_context_engine", {}).get("status", "UNKNOWN"),
                "predictive_loading": self.test_results.get("predictive_context_loader", {}).get("status", "UNKNOWN"),
                "collaborative_intelligence": self.test_results.get("collaborative_context_manager", {}).get("status", "UNKNOWN"),
                "distributed_caching": self.test_results.get("distributed_context_cache", {}).get("status", "UNKNOWN"),
                "component_integration": self.test_results.get("component_integration", {}).get("status", "UNKNOWN"),
                "performance_validation": self.test_results.get("performance_requirements", {}).get("status", "UNKNOWN"),
                "live_trading_integration": self.test_results.get("live_trading_integration", {}).get("status", "UNKNOWN"),
                "error_handling": self.test_results.get("error_handling", {}).get("status", "UNKNOWN")
            },
            "performance_metrics": {
                "average_execution_time": total_execution_time / total_tests if total_tests > 0 else 0,
                "fastest_test": min(
                    (result["execution_time"] for result in self.test_results.values()),
                    default=0
                ),
                "slowest_test": max(
                    (result["execution_time"] for result in self.test_results.values()),
                    default=0
                )
            },
            "test_environment": {
                "system_root": str(self.system_root) if self.system_root else "N/A",
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        return report

# Main execution
async def main():
    """Run Week 4 integration tests"""
    test_suite = Week4IntegrationTestSuite()
    
    try:
        # Run all tests
        report = await test_suite.run_all_tests()
        
        # Print summary
        print("\n" + "="*80)
        print("WEEK 4 INTEGRATION TEST RESULTS")
        print("="*80)
        
        summary = report["test_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        print("\n" + "-"*80)
        print("FEATURE STATUS")
        print("-"*80)
        
        features = report["week4_features_status"]
        for feature, status in features.items():
            status_symbol = "✅" if status == "PASS" else "❌"
            print(f"{status_symbol} {feature.replace('_', ' ').title()}: {status}")
            
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        
        perf = report["performance_metrics"]
        print(f"Average Execution Time: {perf['average_execution_time']:.3f}s")
        print(f"Fastest Test: {perf['fastest_test']:.3f}s")
        print(f"Slowest Test: {perf['slowest_test']:.3f}s")
        
        # Save detailed report
        report_path = Path("week4_integration_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nDetailed report saved to: {report_path}")
        
        # Return success/failure
        return summary['pass_rate'] >= 0.8
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)