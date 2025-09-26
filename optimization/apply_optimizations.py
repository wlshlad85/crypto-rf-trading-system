"""Script to apply optimizations across the entire system."""

import os
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime
import re
from typing import Dict, List, Tuple


class OptimizationApplicator:
    """Apply performance optimizations system-wide."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Track changes
        self.changes_made = []
        
    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path("optimization_logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'apply_optimizations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def backup_file(self, filepath: Path):
        """Create backup of file before modification."""
        if filepath.exists():
            backup_path = self.backup_dir / filepath.relative_to(Path.cwd())
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, backup_path)
            self.logger.info(f"Backed up: {filepath}")
    
    def apply_sleep_removal_optimizations(self):
        """Remove time.sleep calls and replace with async operations."""
        self.logger.info("\n🔧 Removing time.sleep bottlenecks...")
        
        files_to_fix = [
            "data/yfinance_fetcher.py",
            "enhanced_paper_trader_24h.py",
            "paper_trading/minute_paper_trader.py",
            "btc_24hr_paper_trader.py",
            "agent01_controller.py",
            "agent02_data_ml.py",
            "agent03_execution.py"
        ]
        
        for file_path in files_to_fix:
            filepath = Path(file_path)
            if filepath.exists():
                self.backup_file(filepath)
                self._remove_sleep_from_file(filepath)
    
    def _remove_sleep_from_file(self, filepath: Path):
        """Remove sleep calls from a specific file."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Replace sleep with async alternatives
        replacements = [
            # Rate limiting sleeps
            (r'time\.sleep\(0\.5\)\s*#?\s*rate limit', 'await asyncio.sleep(0.1)  # Reduced rate limit delay'),
            (r'time\.sleep\((\d+)\)\s*#?\s*[Ss]leep', r'await asyncio.sleep(\1)'),
            
            # Loop delays
            (r'time\.sleep\(300\)', 'await asyncio.sleep(60)  # Reduced from 300s'),
            (r'time\.sleep\(60\)', 'await asyncio.sleep(30)  # Reduced from 60s'),
            (r'time\.sleep\(30\)', 'await asyncio.sleep(10)  # Reduced from 30s'),
            (r'time\.sleep\(10\)', 'await asyncio.sleep(5)  # Reduced from 10s'),
            (r'time\.sleep\(5\)', 'await asyncio.sleep(2)  # Reduced from 5s'),
            
            # Small delays
            (r'time\.sleep\(1\)', 'await asyncio.sleep(0.5)  # Reduced from 1s'),
            (r'time\.sleep\(2\)', 'await asyncio.sleep(1)  # Reduced from 2s'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Add asyncio import if needed and file was modified
        if content != original_content and 'import asyncio' not in content:
            # Add import after other imports
            import_pattern = r'(import.*\n)+'
            match = re.search(import_pattern, content)
            if match:
                insert_pos = match.end()
                content = content[:insert_pos] + 'import asyncio\n' + content[insert_pos:]
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            self.logger.info(f"   ✅ Optimized: {filepath}")
            self.changes_made.append(f"Removed sleep delays from {filepath}")
    
    def apply_parallel_processing_configs(self):
        """Update configuration files to enable parallel processing."""
        self.logger.info("\n🔧 Enabling parallel processing in configurations...")
        
        config_files = list(Path("configs").glob("*.json"))
        
        for config_file in config_files:
            self.backup_file(config_file)
            self._update_config_for_parallel(config_file)
    
    def _update_config_for_parallel(self, config_file: Path):
        """Update a config file for parallel processing."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            modified = False
            
            # Update model configurations
            if 'model' in config:
                if 'n_jobs' in config['model']:
                    config['model']['n_jobs'] = -1  # Use all cores
                    modified = True
                if 'parallel_training' not in config['model']:
                    config['model']['parallel_training'] = True
                    modified = True
            
            # Update system configurations
            if 'system' in config:
                if 'n_jobs' in config['system']:
                    config['system']['n_jobs'] = -1
                    modified = True
                if 'enable_parallel' not in config['system']:
                    config['system']['enable_parallel'] = True
                    modified = True
            
            # Update backtest configurations
            if 'backtest' in config:
                if 'parallel_processing' not in config['backtest']:
                    config['backtest']['parallel_processing'] = True
                    modified = True
            
            if modified:
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                self.logger.info(f"   ✅ Updated: {config_file}")
                self.changes_made.append(f"Enabled parallel processing in {config_file}")
                
        except Exception as e:
            self.logger.error(f"   ❌ Failed to update {config_file}: {e}")
    
    def create_optimized_imports(self):
        """Create a centralized imports file for optimized components."""
        self.logger.info("\n🔧 Creating optimized imports module...")
        
        imports_content = '''"""Centralized imports for optimized components."""

# Optimized data fetching
try:
    from optimization.data_fetcher_optimizer import OptimizedYFinanceFetcher as YFinanceCryptoFetcher
    print("✅ Using optimized data fetcher")
except ImportError:
    from data.yfinance_fetcher import YFinanceCryptoFetcher
    print("⚠️  Using standard data fetcher")

# Optimized feature engineering
try:
    from optimization.feature_engineering_optimizer import OptimizedFeatureEngine as UltraFeatureEngine
    print("✅ Using optimized feature engine")
except ImportError:
    from features.ultra_feature_engineering import UltraFeatureEngine
    print("⚠️  Using standard feature engine")

# Optimized ML training
try:
    from optimization.ml_training_optimizer import OptimizedRandomForestModel as CryptoRandomForestModel
    print("✅ Using optimized ML model")
except ImportError:
    from models.random_forest_model import CryptoRandomForestModel
    print("⚠️  Using standard ML model")

# Optimized backtesting
try:
    from optimization.backtest_engine_optimizer import OptimizedBacktestEngine as CryptoBacktestEngine
    print("✅ Using optimized backtest engine")
except ImportError:
    from backtesting.backtest_engine import CryptoBacktestEngine
    print("⚠️  Using standard backtest engine")

# Export all
__all__ = [
    'YFinanceCryptoFetcher',
    'UltraFeatureEngine', 
    'CryptoRandomForestModel',
    'CryptoBacktestEngine'
]
'''
        
        imports_file = Path("optimization") / "optimized_imports.py"
        with open(imports_file, 'w') as f:
            f.write(imports_content)
        
        self.logger.info(f"   ✅ Created: {imports_file}")
        self.changes_made.append(f"Created optimized imports module")
    
    def create_performance_config(self):
        """Create a performance-optimized configuration file."""
        self.logger.info("\n🔧 Creating performance-optimized configuration...")
        
        perf_config = {
            "system": {
                "n_jobs": -1,
                "enable_parallel": True,
                "cache_enabled": True,
                "memory_optimization": True,
                "async_operations": True,
                "batch_size": 10000,
                "chunk_size": 50000
            },
            "data": {
                "async_fetch": True,
                "concurrent_symbols": 10,
                "cache_ttl": 300,
                "batch_download": True,
                "compression": True
            },
            "features": {
                "parallel_processing": True,
                "vectorized_operations": True,
                "memory_efficient": True,
                "numba_acceleration": True
            },
            "model": {
                "model_type": "lightgbm",
                "n_jobs": -1,
                "parallel_training": True,
                "early_stopping": True,
                "gpu_enabled": False,
                "batch_prediction": True
            },
            "backtest": {
                "parallel_processing": True,
                "vectorized_metrics": True,
                "numba_acceleration": True,
                "chunk_processing": True
            },
            "optimization": {
                "hyperparameter_tuning": "optuna",
                "n_trials": 100,
                "parallel_trials": 4,
                "pruning": True
            }
        }
        
        config_file = Path("configs") / "performance_optimized_config.json"
        with open(config_file, 'w') as f:
            json.dump(perf_config, f, indent=2)
        
        self.logger.info(f"   ✅ Created: {config_file}")
        self.changes_made.append("Created performance-optimized configuration")
    
    def generate_optimization_summary(self):
        """Generate summary of all optimizations applied."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("OPTIMIZATION APPLICATION SUMMARY")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\n📋 Total changes made: {len(self.changes_made)}")
        self.logger.info("\n🔧 Changes applied:")
        for change in self.changes_made:
            self.logger.info(f"   • {change}")
        
        self.logger.info(f"\n💾 Backups saved to: {self.backup_dir}")
        
        # Create summary file
        summary_file = Path("optimization_logs") / f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "changes_made": self.changes_made,
            "backup_location": str(self.backup_dir),
            "files_modified": len(self.changes_made)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"\n📄 Summary saved to: {summary_file}")
        
        # Next steps
        self.logger.info("\n🎯 NEXT STEPS:")
        self.logger.info("   1. Test the optimized components using: python optimization/run_optimizations.py")
        self.logger.info("   2. Update imports in your main scripts to use optimized components")
        self.logger.info("   3. Use the performance_optimized_config.json for new runs")
        self.logger.info("   4. Monitor system performance and adjust parameters as needed")
        self.logger.info("   5. Consider GPU acceleration for ML models if available")
    
    def apply_all_optimizations(self):
        """Apply all optimizations."""
        try:
            self.apply_sleep_removal_optimizations()
            self.apply_parallel_processing_configs()
            self.create_optimized_imports()
            self.create_performance_config()
            self.generate_optimization_summary()
            
            self.logger.info("\n✅ All optimizations applied successfully!")
            
        except Exception as e:
            self.logger.error(f"\n❌ Error applying optimizations: {e}")
            self.logger.info(f"💾 Backups are available at: {self.backup_dir}")
            raise


def main():
    """Main execution function."""
    print("\n🚀 APPLYING SYSTEM-WIDE OPTIMIZATIONS")
    print("=" * 80)
    print("This will apply performance optimizations across your trading system")
    print("All original files will be backed up before modification")
    print("=" * 80)
    
    response = input("\n⚠️  Continue with optimization? (yes/no): ")
    if response.lower() != 'yes':
        print("Optimization cancelled.")
        return
    
    applicator = OptimizationApplicator()
    applicator.apply_all_optimizations()


if __name__ == "__main__":
    main()