#!/usr/bin/env python3
"""
Agent01: Project Manager & Meta-Optimizer Controller

Coordinates development, oversees backtests and parameter tuning,
manages meta-optimization lifecycle for the crypto trading system.
"""

import time
import os
import json
import signal
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

COMS_PATH = "coms.md"

class Agent01Controller:
    """Project coordination and meta-optimization control."""
    
    def __init__(self):
        """Initialize Agent01 controller."""
        self.running = False
        self.agents = {
            'agent02': None,  # Data & ML process
            'agent03': None   # Execution engine process
        }
        
        # Performance tracking
        self.session_history = []
        self.current_session = None
        
        # Meta-optimization parameters
        self.target_return = 0.04  # 4% target return
        self.baseline_return = 0.0282  # 2.82% baseline
        self.optimization_interval = 6  # Hours between optimizations
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.log("Agent01 starting project coordination and meta-optimization control.")
    
    def log(self, message: str):
        """Log message to communications file."""
        timestamp = datetime.utcnow().isoformat()
        try:
            with open(COMS_PATH, "a") as f:
                f.write(f"[agent01][{timestamp}] {message}\n")
            print(f"[agent01][{timestamp}] {message}")
        except Exception as e:
            print(f"[agent01] Logging error: {e}")
    
    def start_agent02(self) -> bool:
        """Start Agent02 (Data & ML processor)."""
        try:
            if self.agents['agent02'] is None or self.agents['agent02'].poll() is not None:
                self.log("Starting Agent02 (Data & ML processor)")
                self.agents['agent02'] = subprocess.Popen([
                    'python3', 'agent02_data_ml.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.log("Agent02 started successfully")
                return True
            else:
                self.log("Agent02 already running")
                return True
        except Exception as e:
            self.log(f"Failed to start Agent02: {e}")
            return False
    
    def start_agent03(self) -> bool:
        """Start Agent03 (Execution engine)."""
        try:
            if self.agents['agent03'] is None or self.agents['agent03'].poll() is not None:
                self.log("Starting Agent03 (Execution engine)")
                self.agents['agent03'] = subprocess.Popen([
                    'python3', 'agent03_execution.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.log("Agent03 started successfully")
                return True
            else:
                self.log("Agent03 already running")
                return True
        except Exception as e:
            self.log(f"Failed to start Agent03: {e}")
            return False
    
    def check_agent_health(self) -> Dict[str, bool]:
        """Check health of all agents."""
        health = {}
        
        for agent_name, process in self.agents.items():
            if process is None:
                health[agent_name] = False
                self.log(f"{agent_name} not started")
            elif process.poll() is None:
                health[agent_name] = True
            else:
                health[agent_name] = False
                self.log(f"{agent_name} has terminated (exit code: {process.poll()})")
                
                # Try to get error output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stderr:
                        self.log(f"{agent_name} error: {stderr.decode()}")
                except:
                    pass
        
        return health
    
    def restart_failed_agents(self):
        """Restart any failed agents."""
        health = self.check_agent_health()
        
        if not health.get('agent02', False):
            self.log("Restarting Agent02")
            self.start_agent02()
        
        if not health.get('agent03', False):
            self.log("Restarting Agent03")
            self.start_agent03()
    
    def analyze_session_performance(self) -> Dict[str, Any]:
        """Analyze current and historical session performance."""
        try:
            # Look for recent session files
            session_files = []
            for file in os.listdir('.'):
                if file.startswith('optimized_session_') and file.endswith('.json'):
                    session_files.append(file)
            
            if not session_files:
                self.log("No session files found for analysis")
                return {'current_return': 0, 'trade_count': 0, 'win_rate': 0}
            
            # Get most recent session
            latest_session = max(session_files, key=lambda x: os.path.getmtime(x))
            
            with open(latest_session, 'r') as f:
                session_data = json.load(f)
            
            # Calculate performance metrics
            initial_capital = session_data.get('initial_capital', 100000)
            portfolio_value = session_data.get('portfolio_value', initial_capital)
            current_return = (portfolio_value - initial_capital) / initial_capital
            
            trades = session_data.get('trades', [])
            trade_count = len(trades)
            
            # Calculate win rate
            profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = profitable_trades / trade_count if trade_count > 0 else 0
            
            performance = {
                'current_return': current_return,
                'trade_count': trade_count,
                'win_rate': win_rate,
                'portfolio_value': portfolio_value,
                'session_file': latest_session
            }
            
            self.log(f"Performance analysis: {current_return:.2%} return, {trade_count} trades, {win_rate:.1%} win rate")
            
            return performance
            
        except Exception as e:
            self.log(f"Error analyzing session performance: {e}")
            return {'current_return': 0, 'trade_count': 0, 'win_rate': 0}
    
    def suggest_parameter_tweaks(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameter adjustments based on performance."""
        
        current_return = performance.get('current_return', 0)
        win_rate = performance.get('win_rate', 0)
        trade_count = performance.get('trade_count', 0)
        
        suggestions = {
            'momentum_threshold': 1.780,  # Default optimal
            'position_size_adjustment': 0,
            'confidence_threshold': 0.65,
            'reasoning': []
        }
        
        # Analyze performance vs targets
        if current_return < self.baseline_return:
            suggestions['reasoning'].append("Performance below baseline - increasing aggressiveness")
            suggestions['momentum_threshold'] = 1.5  # Lower threshold for more entries
            suggestions['confidence_threshold'] = 0.6  # Lower confidence for more trades
            
        elif current_return > self.target_return:
            suggestions['reasoning'].append("Exceeding target - maintaining parameters")
            # Keep current optimal parameters
            
        else:
            suggestions['reasoning'].append("Between baseline and target - minor optimization")
            suggestions['momentum_threshold'] = 1.650  # Slight adjustment
        
        # Win rate adjustments
        if win_rate < 0.5:
            suggestions['reasoning'].append("Low win rate - increasing selectivity")
            suggestions['confidence_threshold'] = 0.7  # Higher confidence
            suggestions['momentum_threshold'] = 2.0  # Higher threshold
            
        elif win_rate > 0.8:
            suggestions['reasoning'].append("High win rate - can be more aggressive")
            suggestions['confidence_threshold'] = 0.6  # Lower confidence for more trades
        
        # Trade frequency adjustments
        if trade_count < 20:
            suggestions['reasoning'].append("Low trade frequency - reducing selectivity")
            suggestions['momentum_threshold'] = 1.4
            
        elif trade_count > 100:
            suggestions['reasoning'].append("High trade frequency - increasing selectivity")
            suggestions['momentum_threshold'] = 2.2
        
        self.log(f"Parameter suggestions: {suggestions}")
        return suggestions
    
    def coordinate_meta_optimization(self):
        """Coordinate meta-optimization cycle."""
        self.log("Starting meta-optimization coordination cycle")
        
        try:
            # 1. Analyze current performance
            performance = self.analyze_session_performance()
            
            # 2. Generate parameter suggestions
            suggestions = self.suggest_parameter_tweaks(performance)
            
            # 3. Check if significant improvement needed
            current_return = performance.get('current_return', 0)
            improvement_needed = abs(current_return - self.target_return) > 0.01  # 1% threshold
            
            if improvement_needed:
                self.log("Performance improvement needed - triggering optimization")
                
                # 4. Signal agents for parameter update
                self._signal_parameter_update(suggestions)
                
                # 5. Monitor for completion
                self._monitor_optimization_progress()
                
            else:
                self.log("Performance satisfactory - maintaining current parameters")
            
        except Exception as e:
            self.log(f"Error in meta-optimization coordination: {e}")
    
    def _signal_parameter_update(self, suggestions: Dict[str, Any]):
        """Signal agents to update parameters."""
        
        # Save suggestions to shared file
        suggestions_file = "optimization_suggestions.json"
        try:
            with open(suggestions_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'suggestions': suggestions,
                    'agent01_signal': 'parameter_update'
                }, f, indent=2)
            
            self.log(f"Parameter update signal sent: {suggestions_file}")
            
        except Exception as e:
            self.log(f"Error signaling parameter update: {e}")
    
    def _monitor_optimization_progress(self):
        """Monitor optimization progress from other agents."""
        
        start_time = datetime.now()
        timeout = timedelta(minutes=30)  # 30 minute timeout
        
        while datetime.now() - start_time < timeout:
            try:
                # Check for agent responses
                if os.path.exists("optimization_response.json"):
                    with open("optimization_response.json", 'r') as f:
                        response = json.load(f)
                    
                    self.log(f"Optimization response received: {response.get('status', 'unknown')}")
                    
                    # Cleanup
                    os.remove("optimization_response.json")
                    break
                    
            except Exception as e:
                self.log(f"Error monitoring optimization: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def run_coordination_cycle(self):
        """Run main coordination cycle."""
        self.log("Starting main coordination cycle")
        self.running = True
        
        cycle_count = 0
        last_optimization = datetime.now() - timedelta(hours=self.optimization_interval)
        
        try:
            while self.running:
                cycle_count += 1
                self.log(f"Coordination cycle #{cycle_count}")
                
                # 1. Check agent health and restart if needed
                self.restart_failed_agents()
                
                # 2. Analyze current performance
                performance = self.analyze_session_performance()
                
                # 3. Check if meta-optimization is due
                time_since_optimization = datetime.now() - last_optimization
                if time_since_optimization.total_seconds() >= self.optimization_interval * 3600:
                    self.log("Meta-optimization interval reached")
                    self.coordinate_meta_optimization()
                    last_optimization = datetime.now()
                
                # 4. Log status update
                current_return = performance.get('current_return', 0)
                self.log(f"Current session return: {current_return:.2%}")
                self.log(f"Target return: {self.target_return:.1%}")
                self.log(f"Next optimization in: {self.optimization_interval * 3600 - time_since_optimization.total_seconds():.0f} seconds")
                
                # 5. Wait before next cycle
                time.sleep(300)  # 5 minutes between cycles
                
        except Exception as e:
            self.log(f"Error in coordination cycle: {e}")
            time.sleep(60)  # Cooldown on error
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log(f"Received signal {signum} - shutting down gracefully")
        self.running = False
        
        # Stop all agents
        for agent_name, process in self.agents.items():
            if process and process.poll() is None:
                self.log(f"Stopping {agent_name}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def main(self):
        """Main entry point."""
        try:
            # Start all agents
            self.start_agent02()
            self.start_agent03()
            
            # Run coordination
            self.run_coordination_cycle()
            
        except Exception as e:
            self.log(f"Critical error in Agent01: {e}")
        finally:
            self.log("Agent01 coordination shutdown complete")

def main():
    """Entry point for Agent01."""
    controller = Agent01Controller()
    controller.main()

if __name__ == "__main__":
    main()