#!/usr/bin/env python3
"""
Agent01: Project Manager & Meta-Optimizer Controller

Coordinates development, oversees backtests and parameter tuning,
manages meta-optimization lifecycle for the crypto trading system.
"""

import time
import os
import json
import subprocess
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

COMS_PATH = "crypto_rf_trader/coms.md"
CONFIG_PATH = "crypto_rf_trader/config/system_config.json"

class Agent01Controller:
    """Project coordination and meta-optimization control."""
    
    def __init__(self):
        self.log("Agent01 starting project coordination and meta-optimization control.")
        self.running = True
        self.agents = {}
        self.current_performance = {'return': 0.0, 'trades': 0, 'win_rate': 0.0}
        self.target_return = 0.04  # 4% target
        self.baseline_return = 0.0282  # 2.82% baseline
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def log(self, message):
        """Log message to communications file."""
        timestamp = datetime.utcnow().isoformat()
        with open(COMS_PATH, "a") as f:
            f.write(f"[agent01][{timestamp}] {message}\n")
        print(f"[agent01][{timestamp}] {message}")
    
    def start_agents(self):
        """Start agent02 and agent03 processes."""
        try:
            # Start Agent02 (Data & ML)
            self.log("Starting Agent02 (Data & ML processor)")
            self.agents['agent02'] = subprocess.Popen(
                ['python3', 'crypto_rf_trader/agent02_data_ml.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for models to be ready
            time.sleep(10)
            
            # Start Agent03 (Execution Engine)
            self.log("Starting Agent03 (Execution engine)")
            self.agents['agent03'] = subprocess.Popen(
                ['python3', 'crypto_rf_trader/agent03_execution.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.log("All agents started successfully")
            return True
            
        except Exception as e:
            self.log(f"Error starting agents: {str(e)}")
            return False
    
    def check_agent_health(self):
        """Check if agents are still running."""
        for agent_name, process in self.agents.items():
            if process and process.poll() is not None:
                self.log(f"{agent_name} has stopped (exit code: {process.poll()})")
                # Try to restart
                if agent_name == 'agent02':
                    self.agents['agent02'] = subprocess.Popen(
                        ['python3', 'crypto_rf_trader/agent02_data_ml.py'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                elif agent_name == 'agent03':
                    self.agents['agent03'] = subprocess.Popen(
                        ['python3', 'crypto_rf_trader/agent03_execution.py'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
    
    def analyze_performance(self):
        """Analyze current trading performance."""
        try:
            # Look for session data files
            session_files = [f for f in os.listdir('.') if f.startswith('agent03_session_') and f.endswith('.json')]
            
            if not session_files:
                return self.current_performance
            
            # Get most recent session
            latest_session = max(session_files, key=lambda f: os.path.getmtime(f))
            
            with open(latest_session, 'r') as f:
                session_data = json.load(f)
            
            # Calculate metrics
            initial_capital = session_data.get('initial_capital', 100000)
            current_value = session_data.get('portfolio_value', initial_capital)
            trades = session_data.get('trades', [])
            
            current_return = (current_value - initial_capital) / initial_capital
            trade_count = len(trades)
            
            # Calculate win rate
            profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = profitable_trades / trade_count if trade_count > 0 else 0
            
            self.current_performance = {
                'return': current_return,
                'trades': trade_count,
                'win_rate': win_rate,
                'value': current_value
            }
            
            self.log(f"Performance update: {current_return:.2%} return, {trade_count} trades, {win_rate:.1%} win rate")
            
            return self.current_performance
            
        except Exception as e:
            self.log(f"Error analyzing performance: {str(e)}")
            return self.current_performance
    
    def suggest_parameter_adjustments(self):
        """Suggest parameter tweaks based on performance."""
        perf = self.current_performance
        current_return = perf['return']
        win_rate = perf['win_rate']
        
        suggestions = {
            'momentum_threshold': 1.780,  # Default optimal
            'confidence_threshold': 0.65,
            'position_sizing': 'normal'
        }
        
        # Analyze and adjust
        if current_return < self.baseline_return:
            self.log("Performance below baseline - suggesting more aggressive parameters")
            suggestions['momentum_threshold'] = 1.5
            suggestions['confidence_threshold'] = 0.6
            suggestions['position_sizing'] = 'aggressive'
            
        elif current_return > self.target_return:
            self.log("Target exceeded - maintaining current parameters")
            
        else:
            self.log("Performance between baseline and target - minor adjustments")
            suggestions['momentum_threshold'] = 1.65
        
        # Win rate adjustments
        if win_rate < 0.5:
            suggestions['confidence_threshold'] = 0.7  # Be more selective
            
        # Save suggestions
        self._save_parameter_suggestions(suggestions)
        return suggestions
    
    def _save_parameter_suggestions(self, suggestions):
        """Save parameter suggestions for other agents."""
        try:
            suggestions_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'suggestions': suggestions,
                'current_performance': self.current_performance,
                'source': 'agent01'
            }
            
            with open('crypto_rf_trader/config/parameter_suggestions.json', 'w') as f:
                json.dump(suggestions_data, f, indent=2)
                
            self.log("Parameter suggestions saved")
            
        except Exception as e:
            self.log(f"Error saving suggestions: {str(e)}")
    
    def coordinate_optimization_cycle(self):
        """Run meta-optimization coordination cycle."""
        self.log("Running optimization coordination cycle")
        
        # Analyze current performance
        self.analyze_performance()
        
        # Generate suggestions
        suggestions = self.suggest_parameter_adjustments()
        
        # Check if significant improvement needed
        if abs(self.current_performance['return'] - self.target_return) > 0.01:
            self.log("Triggering model retraining with new parameters")
            # Signal for retraining will be picked up by agent02
            
        # Save status report
        self._save_status_report()
    
    def _save_status_report(self):
        """Save current system status report."""
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'performance': self.current_performance,
                'target_return': self.target_return,
                'baseline_return': self.baseline_return,
                'agents_status': {
                    'agent02': 'running' if self.agents.get('agent02') and self.agents['agent02'].poll() is None else 'stopped',
                    'agent03': 'running' if self.agents.get('agent03') and self.agents['agent03'].poll() is None else 'stopped'
                }
            }
            
            os.makedirs('crypto_rf_trader/reports', exist_ok=True)
            report_file = f"crypto_rf_trader/reports/status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.log(f"Status report saved: {report_file}")
            
        except Exception as e:
            self.log(f"Error saving status report: {str(e)}")
    
    def main_coordination_loop(self):
        """Main coordination loop."""
        self.log("Starting main coordination loop")
        
        # Start all agents
        if not self.start_agents():
            self.log("Failed to start agents, exiting")
            return
        
        cycle_count = 0
        last_optimization = datetime.now()
        
        while self.running:
            try:
                cycle_count += 1
                self.log(f"Coordination cycle #{cycle_count}")
                
                # Check agent health
                self.check_agent_health()
                
                # Run optimization every hour
                if (datetime.now() - last_optimization).total_seconds() >= 3600:
                    self.coordinate_optimization_cycle()
                    last_optimization = datetime.now()
                
                # Analyze performance every 15 minutes
                if cycle_count % 3 == 0:
                    self.analyze_performance()
                
                # Wait 5 minutes before next cycle
                time.sleep(300)
                
            except Exception as e:
                self.log(f"Error in main loop: {str(e)}")
                time.sleep(60)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log(f"Received signal {signum}, shutting down...")
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

def main():
    controller = Agent01Controller()
    try:
        controller.main_coordination_loop()
    except KeyboardInterrupt:
        controller.log("Keyboard interrupt received")
    finally:
        controller.log("Agent01 shutdown complete")

if __name__ == "__main__":
    main()