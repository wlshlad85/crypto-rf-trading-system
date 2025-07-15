#!/usr/bin/env python3
"""
24-Hour Meta-Optimization Deployment Cycle

Orchestrates continuous strategy optimization and deployment:
- Runs Hyperband parameter exploration
- Evaluates optimized configurations  
- Deploys best-performing models to production
- Monitors performance and triggers retraining

Usage: python3 deployment_cycle.py [--mode continuous|single]
"""

import os
import sys
import json
import time
import signal
import argparse
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_optim.hyperband_runner import HyperbandRunner
from meta_optim.objective_fn import MetaObjectiveFunction
from meta_optim.retrain_worker import RetrainWorker

class MetaDeploymentCycle:
    """24-hour continuous meta-optimization and deployment cycle."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize deployment cycle with configuration."""
        self.config = config or self._get_default_config()
        
        # Components
        self.hyperband_runner = HyperbandRunner()
        self.objective_fn = MetaObjectiveFunction()
        self.retrain_worker = RetrainWorker()
        
        # State tracking
        self.running = False
        self.current_cycle = 0
        self.deployment_history = []
        self.performance_history = []
        
        # Logging
        self.log_dir = "meta_optim/deployment_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🚀 Meta-Optimization Deployment Cycle Initialized")
        print(f"📁 Log directory: {self.log_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for deployment cycle."""
        return {
            'cycle_duration_hours': 24,         # 24-hour cycles
            'optimization_interval_hours': 6,   # Optimize every 6 hours
            'monitoring_interval_minutes': 30,  # Monitor every 30 minutes
            'performance_threshold': 0.05,      # 5% minimum improvement
            'max_optimization_time_hours': 4,   # Max 4 hours for optimization
            'backup_models_count': 3,           # Keep 3 backup model versions
            'deploy_confidence_threshold': 0.7, # Minimum confidence for deployment
            'emergency_stop_drawdown': -0.25,   # Emergency stop at 25% drawdown
            'retraining_triggers': {
                'performance_degradation': -0.10,    # 10% performance drop
                'sharpe_threshold': 0.5,              # Sharpe ratio below 0.5
                'consecutive_losses': 5,              # 5 consecutive losing trades
                'max_drawdown': -0.20                 # 20% drawdown trigger
            }
        }
    
    def run_continuous_cycle(self):
        """Run continuous 24-hour optimization cycles."""
        print("🔄 Starting Continuous Meta-Optimization Cycle")
        print("=" * 60)
        
        self.running = True
        cycle_start = datetime.now()
        
        try:
            while self.running:
                cycle_duration = timedelta(hours=self.config['cycle_duration_hours'])
                optimization_interval = timedelta(hours=self.config['optimization_interval_hours'])
                
                self.current_cycle += 1
                print(f"\n🎯 CYCLE {self.current_cycle} STARTED")
                print(f"⏰ Start Time: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🏁 End Time: {(cycle_start + cycle_duration).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run optimization phases within cycle
                next_optimization = cycle_start
                cycle_end = cycle_start + cycle_duration
                
                while datetime.now() < cycle_end and self.running:
                    if datetime.now() >= next_optimization:
                        # Run optimization phase
                        self._run_optimization_phase()
                        next_optimization += optimization_interval
                    
                    # Monitor performance
                    self._monitor_performance()
                    
                    # Sleep until next monitoring interval
                    time.sleep(self.config['monitoring_interval_minutes'] * 60)
                
                # End of cycle summary
                self._generate_cycle_summary()
                
                # Prepare for next cycle
                cycle_start = datetime.now()
                
        except KeyboardInterrupt:
            print("\n🛑 Graceful shutdown initiated...")
        except Exception as e:
            print(f"❌ Critical error in deployment cycle: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def run_single_optimization(self):
        """Run a single optimization cycle."""
        print("🎯 Running Single Meta-Optimization Cycle")
        print("=" * 50)
        
        try:
            # Run single optimization
            results = self._run_optimization_phase()
            
            # Generate results
            self._generate_optimization_report(results)
            
            print("✅ Single optimization cycle completed!")
            
        except Exception as e:
            print(f"❌ Error in single optimization: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_optimization_phase(self) -> Dict[str, Any]:
        """Run complete optimization phase."""
        phase_start = datetime.now()
        print(f"\n🔬 OPTIMIZATION PHASE STARTED - {phase_start.strftime('%H:%M:%S')}")
        
        try:
            # Step 1: Run Hyperband optimization
            print("📊 Running Hyperband parameter exploration...")
            optimization_results = self.hyperband_runner.run_hyperband_optimization(
                n_iterations=2  # Reduced for faster cycles
            )
            
            if not optimization_results.get('success', False):
                print("❌ Hyperband optimization failed")
                return {'success': False, 'reason': 'Hyperband failed'}
            
            # Step 2: Evaluate best configuration
            best_config = optimization_results['best_configuration']
            print(f"✅ Best configuration found: Score {best_config['composite_score']:.3f}")
            
            # Step 3: Detailed evaluation
            detailed_results = self._detailed_evaluation(best_config['configuration'])
            
            # Step 4: Deployment decision
            deployment_decision = self._make_deployment_decision(detailed_results)
            
            # Step 5: Deploy if approved
            if deployment_decision['deploy']:
                deployment_results = self._deploy_configuration(best_config['configuration'])
            else:
                deployment_results = {'deployed': False, 'reason': deployment_decision['reason']}
            
            phase_duration = (datetime.now() - phase_start).total_seconds() / 60
            
            results = {
                'success': True,
                'timestamp': phase_start.isoformat(),
                'duration_minutes': phase_duration,
                'optimization_results': optimization_results,
                'detailed_evaluation': detailed_results,
                'deployment_decision': deployment_decision,
                'deployment_results': deployment_results,
                'cycle': self.current_cycle
            }
            
            # Log results
            self._log_optimization_results(results)
            
            print(f"✅ Optimization phase completed in {phase_duration:.1f} minutes")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in optimization phase: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': phase_start.isoformat()
            }
    
    def _detailed_evaluation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed evaluation of configuration."""
        print("🔍 Performing detailed evaluation...")
        
        try:
            # Extended backtest with full dataset
            backtest_results = self.retrain_worker.train_and_evaluate(
                config, 
                training_fraction=1.0
            )
            
            # Comprehensive metrics evaluation
            evaluation = self.objective_fn.evaluate_strategy(backtest_results)
            
            # Additional stability tests
            stability_metrics = self._test_configuration_stability(config)
            
            return {
                'backtest_results': backtest_results,
                'objective_evaluation': evaluation,
                'stability_metrics': stability_metrics,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in detailed evaluation: {e}")
            return {
                'error': str(e),
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def _test_configuration_stability(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test configuration stability across different conditions."""
        
        try:
            # Test with different training fractions
            stability_tests = []
            
            for fraction in [0.6, 0.8, 1.0]:
                test_results = self.retrain_worker.train_and_evaluate(
                    config, 
                    training_fraction=fraction
                )
                
                evaluation = self.objective_fn.evaluate_strategy(test_results)
                
                stability_tests.append({
                    'training_fraction': fraction,
                    'composite_score': evaluation['composite_score'],
                    'viable': evaluation['viable']
                })
            
            # Calculate stability metrics
            scores = [t['composite_score'] for t in stability_tests if t['viable']]
            
            if len(scores) >= 2:
                score_std = np.std(scores) if len(scores) > 1 else 0
                score_mean = np.mean(scores)
                stability_ratio = 1 - (score_std / score_mean) if score_mean > 0 else 0
            else:
                stability_ratio = 0
            
            return {
                'stability_tests': stability_tests,
                'stability_ratio': stability_ratio,
                'consistent_performance': len(scores) == len(stability_tests)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'stability_ratio': 0
            }
    
    def _make_deployment_decision(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent deployment decision based on evaluation."""
        
        try:
            evaluation = detailed_results.get('objective_evaluation', {})
            stability = detailed_results.get('stability_metrics', {})
            
            # Check viability
            if not evaluation.get('viable', False):
                return {
                    'deploy': False,
                    'reason': f"Configuration not viable: {evaluation.get('reason', 'Unknown')}"
                }
            
            # Check performance threshold
            composite_score = evaluation.get('composite_score', 0)
            if composite_score < self.config['deploy_confidence_threshold']:
                return {
                    'deploy': False,
                    'reason': f"Score {composite_score:.3f} below threshold {self.config['deploy_confidence_threshold']}"
                }
            
            # Check stability
            stability_ratio = stability.get('stability_ratio', 0)
            if stability_ratio < 0.7:  # Require 70% stability
                return {
                    'deploy': False,
                    'reason': f"Unstable performance: {stability_ratio:.3f} stability ratio"
                }
            
            # Check improvement over current model
            current_performance = self._get_current_model_performance()
            improvement = composite_score - current_performance.get('composite_score', 0)
            
            if improvement < self.config['performance_threshold']:
                return {
                    'deploy': False,
                    'reason': f"Insufficient improvement: {improvement:.3f} < {self.config['performance_threshold']}"
                }
            
            return {
                'deploy': True,
                'reason': f"Configuration approved: Score {composite_score:.3f}, Stability {stability_ratio:.3f}, Improvement {improvement:.3f}",
                'composite_score': composite_score,
                'stability_ratio': stability_ratio,
                'improvement': improvement
            }
            
        except Exception as e:
            return {
                'deploy': False,
                'reason': f"Deployment decision error: {str(e)}"
            }
    
    def _deploy_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy optimized configuration to production."""
        
        try:
            deployment_timestamp = datetime.now()
            print(f"🚀 Deploying configuration at {deployment_timestamp.strftime('%H:%M:%S')}")
            
            # Backup current configuration
            backup_result = self._backup_current_model()
            
            # Save new configuration
            config_path = "enhanced_models/optimized_config.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump({
                    'config': config,
                    'deployment_timestamp': deployment_timestamp.isoformat(),
                    'cycle': self.current_cycle,
                    'backup_id': backup_result.get('backup_id')
                }, f, indent=2)
            
            # Train models with new configuration
            models = self.retrain_worker._train_models(
                self.retrain_worker.training_data, 
                config
            )
            
            # Save trained models
            model_paths = {}
            for model_name, model_data in models['models'].items():
                model_path = f"enhanced_models/{model_name}_optimized.joblib"
                joblib.dump(model_data, model_path)
                model_paths[model_name] = model_path
            
            # Save scalers
            scaler_paths = {}
            for scaler_name, scaler_data in models['scalers'].items():
                scaler_path = f"enhanced_models/{scaler_name}_scaler.joblib"
                joblib.dump(scaler_data, scaler_path)
                scaler_paths[scaler_name] = scaler_path
            
            deployment_record = {
                'success': True,
                'deployment_timestamp': deployment_timestamp.isoformat(),
                'config_path': config_path,
                'model_paths': model_paths,
                'scaler_paths': scaler_paths,
                'backup_id': backup_result.get('backup_id'),
                'cycle': self.current_cycle
            }
            
            # Update deployment history
            self.deployment_history.append(deployment_record)
            
            print("✅ Configuration deployed successfully!")
            
            return deployment_record
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'deployment_timestamp': datetime.now().isoformat()
            }
    
    def _backup_current_model(self) -> Dict[str, Any]:
        """Backup current model configuration."""
        
        try:
            backup_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = f"enhanced_models/backups/{backup_id}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup configuration file
            config_file = "enhanced_models/optimized_config.json"
            if os.path.exists(config_file):
                import shutil
                shutil.copy2(config_file, f"{backup_dir}/config.json")
            
            # Backup model files
            model_dir = "enhanced_models"
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.joblib'):
                        shutil.copy2(f"{model_dir}/{file}", f"{backup_dir}/{file}")
            
            return {
                'success': True,
                'backup_id': backup_id,
                'backup_dir': backup_dir
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_current_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics of currently deployed model."""
        
        # This would typically load from a performance tracking file
        # For now, return baseline performance
        return {
            'composite_score': 0.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.15,
            'last_updated': datetime.now().isoformat()
        }
    
    def _monitor_performance(self):
        """Monitor current trading performance and trigger alerts."""
        
        try:
            # Check if trading session is running
            log_files = self._find_active_trading_logs()
            
            if not log_files:
                return
            
            # Analyze recent performance
            performance_metrics = self._analyze_recent_performance(log_files)
            
            # Check for retraining triggers
            if self._should_trigger_retraining(performance_metrics):
                print("⚠️ Performance degradation detected - triggering emergency retraining")
                self._emergency_retrain()
            
            # Log performance metrics
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': performance_metrics
            })
            
        except Exception as e:
            print(f"❌ Error in performance monitoring: {e}")
    
    def _find_active_trading_logs(self) -> List[str]:
        """Find currently active trading log files."""
        
        log_dirs = [
            "logs/enhanced_24hr_trading",
            "logs/24hr_trading"
        ]
        
        active_logs = []
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                for file in os.listdir(log_dir):
                    if file.endswith('.log'):
                        file_path = os.path.join(log_dir, file)
                        # Check if file was modified in last hour
                        if os.path.getmtime(file_path) > time.time() - 3600:
                            active_logs.append(file_path)
        
        return active_logs
    
    def _analyze_recent_performance(self, log_files: List[str]) -> Dict[str, Any]:
        """Analyze recent trading performance from log files."""
        
        # Simplified performance analysis
        # In production, this would parse actual log files
        return {
            'recent_return': 0.8,  # Recent return percentage
            'recent_trades': 15,
            'win_rate': 0.6,
            'current_drawdown': -0.05
        }
    
    def _should_trigger_retraining(self, performance_metrics: Dict[str, Any]) -> bool:
        """Check if retraining should be triggered."""
        
        triggers = self.config['retraining_triggers']
        
        # Check drawdown trigger
        if performance_metrics.get('current_drawdown', 0) < triggers['max_drawdown']:
            return True
        
        # Check performance degradation
        if performance_metrics.get('recent_return', 0) < triggers['performance_degradation']:
            return True
        
        return False
    
    def _emergency_retrain(self):
        """Trigger emergency retraining and deployment."""
        
        print("🚨 EMERGENCY RETRAINING INITIATED")
        
        try:
            # Load best previous configuration
            best_config = self.hyperband_runner.load_best_configuration()
            
            if best_config:
                print("♻️ Redeploying best known configuration")
                self._deploy_configuration(best_config['configuration'])
            else:
                print("⚡ Running fast optimization")
                # Run single iteration optimization
                results = self.hyperband_runner.run_hyperband_optimization(n_iterations=1)
                if results.get('success'):
                    self._deploy_configuration(results['best_configuration']['configuration'])
        
        except Exception as e:
            print(f"❌ Emergency retraining failed: {e}")
    
    def _log_optimization_results(self, results: Dict[str, Any]):
        """Log optimization results to file."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'optimization_{timestamp}.json')
        
        try:
            with open(log_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving optimization log: {e}")
    
    def _generate_cycle_summary(self):
        """Generate summary report for completed cycle."""
        
        print(f"\n📊 CYCLE {self.current_cycle} SUMMARY")
        print("=" * 40)
        print(f"Deployments: {len([d for d in self.deployment_history if d.get('success')])}")
        print(f"Performance Checks: {len(self.performance_history)}")
        print(f"Duration: {self.config['cycle_duration_hours']} hours")
        
        # Save cycle summary
        summary = {
            'cycle': self.current_cycle,
            'deployments': len(self.deployment_history),
            'performance_checks': len(self.performance_history),
            'completion_time': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.log_dir, f'cycle_{self.current_cycle}_summary.json')
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving cycle summary: {e}")
    
    def _generate_optimization_report(self, results: Dict[str, Any]):
        """Generate detailed optimization report."""
        
        print("\n📋 OPTIMIZATION REPORT")
        print("=" * 30)
        
        if results.get('success'):
            opt_results = results.get('optimization_results', {})
            print(f"✅ Optimization successful")
            print(f"Best Score: {opt_results.get('best_configuration', {}).get('composite_score', 0):.3f}")
            print(f"Total Evaluations: {opt_results.get('total_evaluations', 0)}")
            print(f"Duration: {results.get('duration_minutes', 0):.1f} minutes")
            
            deployment = results.get('deployment_results', {})
            if deployment.get('success'):
                print(f"🚀 Successfully deployed to production")
            else:
                print(f"⏸️ Deployment skipped: {deployment.get('reason', 'Unknown')}")
        else:
            print(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum} - initiating shutdown...")
        self.running = False
    
    def _cleanup(self):
        """Cleanup resources before shutdown."""
        print("🧹 Cleaning up resources...")
        
        # Save final state
        final_state = {
            'cycles_completed': self.current_cycle,
            'deployments': self.deployment_history,
            'performance_history': self.performance_history[-10:],  # Keep last 10
            'shutdown_time': datetime.now().isoformat()
        }
        
        state_file = os.path.join(self.log_dir, 'final_state.json')
        try:
            with open(state_file, 'w') as f:
                json.dump(final_state, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving final state: {e}")
        
        print("✅ Cleanup completed")

def main():
    """Main entry point for deployment cycle."""
    
    parser = argparse.ArgumentParser(description='Meta-Optimization Deployment Cycle')
    parser.add_argument('--mode', choices=['continuous', 'single'], default='continuous',
                        help='Run mode: continuous 24h cycles or single optimization')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize deployment cycle
    cycle = MetaDeploymentCycle(config)
    
    if args.mode == 'continuous':
        cycle.run_continuous_cycle()
    else:
        cycle.run_single_optimization()

if __name__ == "__main__":
    main()