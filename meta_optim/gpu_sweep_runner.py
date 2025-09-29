#!/usr/bin/env python3
"""
GPU-Optimized Nightly Parameter Sweep Runner
============================================

Orchestrates nightly GPU-accelerated hyperparameter sweeps for the crypto RF trading system.
Designed for unattended operation with comprehensive logging, checkpointing, and reproducibility.

Features:
- Multi-GPU support with deterministic execution
- Automatic checkpointing and resume capability
- Resource monitoring and optimization
- Comprehensive logging and reporting
- Integration with CatBoost/LightGBM GPU backends

Usage:
    python gpu_sweep_runner.py --config sweep_config.json --gpu-ids 0,1,2,3
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

import numpy as np
import pandas as pd
import psutil
import GPUtil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_optim.hyperband_runner import HyperbandRunner
from meta_optim.crypto_gpu_reproducibility import CryptoGPUReproducibility, setup_crypto_gpu_reproducibility
from meta_optim.objective_fn import MetaObjectiveFunction
from meta_optim.retrain_worker import RetrainWorker


class GPUSweepRunner:
    """Orchestrates GPU-optimized parameter sweeps for crypto trading models."""
    
    def __init__(self, config_path: str, gpu_ids: List[int] = None, 
                 master_seed: int = 123456, log_dir: str = "sweep_logs"):
        """
        Initialize GPU sweep runner.
        
        Args:
            config_path: Path to sweep configuration JSON
            gpu_ids: List of GPU IDs to use (None = all available)
            master_seed: Master seed for reproducibility
            log_dir: Directory for logs and checkpoints
        """
        self.config_path = config_path
        self.master_seed = master_seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup reproducibility
        self.logger.info(f"🌱 Setting up reproducibility with seed: {master_seed}")
        self.reproducibility = CryptoGPUReproducibility(seed=master_seed)
        self.reproducibility.setup_environment(force_deterministic=True)
        
        # GPU configuration
        self.gpu_ids = gpu_ids or self._detect_gpus()
        self.logger.info(f"🖥️  Using GPUs: {self.gpu_ids}")
        
        # Sweep state
        self.sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = self.log_dir / "checkpoints" / self.sweep_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.all_results = []
        self.best_configs = []
        self.resource_usage = []
        
        # Log system info
        self.reproducibility.log_system_info(
            str(self.log_dir / f"{self.sweep_id}_system_info.json"),
            extra_config={
                "sweep_config": self.config,
                "gpu_ids": self.gpu_ids,
                "sweep_id": self.sweep_id
            }
        )
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.log_dir / f"gpu_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("🚀 GPU SWEEP RUNNER INITIALIZED")
        self.logger.info("=" * 80)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load sweep configuration."""
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"📋 Loaded configuration from: {self.config_path}")
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default sweep configuration."""
        return {
            "sweep_type": "hyperband",
            "hyperband": {
                "iterations": 5,
                "max_iter": 81,
                "eta": 3,
                "max_parallel": 4
            },
            "models": {
                "enable_catboost": True,
                "enable_lightgbm": True,
                "enable_xgboost": True,
                "enable_rf": True
            },
            "parameter_ranges": {
                "entry_model": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [6, 8, 10, 12, 15],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15]
                },
                "position_model": {
                    "n_estimators": [100, 150, 200, 300],
                    "max_depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1]
                },
                "exit_model": {
                    "n_estimators": [100, 150, 200, 300],
                    "max_depth": [6, 8, 10, 12],
                    "learning_rate": [0.01, 0.05, 0.1]
                }
            },
            "gpu_settings": {
                "memory_fraction": 0.8,
                "allow_growth": True,
                "per_process_gpu_memory_fraction": 0.25
            },
            "data_settings": {
                "train_months": 12,
                "validation_months": 3,
                "test_months": 1,
                "symbols": ["BTC", "ETH", "ADA"]
            },
            "optimization_targets": {
                "primary": "sharpe_ratio",
                "secondary": ["profit_factor", "max_drawdown", "win_rate"]
            },
            "early_stopping": {
                "patience": 10,
                "min_improvement": 0.001
            }
        }
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        try:
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except:
            pass
        
        try:
            gpus = GPUtil.getGPUs()
            return [gpu.id for gpu in gpus]
        except:
            pass
        
        self.logger.warning("No GPUs detected, running on CPU")
        return []
    
    def run_sweep(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete parameter sweep.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Dictionary with sweep results
        """
        self.logger.info("🎯 Starting GPU parameter sweep")
        self.logger.info(f"Sweep ID: {self.sweep_id}")
        
        start_time = datetime.now()
        
        try:
            # Resume from checkpoint if provided
            if resume_from_checkpoint:
                self._load_checkpoint(resume_from_checkpoint)
            
            # Monitor resources
            self._start_resource_monitoring()
            
            # Run sweep based on type
            if self.config["sweep_type"] == "hyperband":
                results = self._run_hyperband_sweep()
            elif self.config["sweep_type"] == "grid":
                results = self._run_grid_sweep()
            elif self.config["sweep_type"] == "random":
                results = self._run_random_sweep()
            elif self.config["sweep_type"] == "bayesian":
                results = self._run_bayesian_sweep()
            else:
                raise ValueError(f"Unknown sweep type: {self.config['sweep_type']}")
            
            # Process results
            sweep_results = self._process_results(results, start_time)
            
            # Generate reports
            self._generate_reports(sweep_results)
            
            # Save final checkpoint
            self._save_checkpoint("final")
            
            self.logger.info("✅ GPU parameter sweep completed successfully!")
            
            return sweep_results
            
        except Exception as e:
            self.logger.error(f"❌ Sweep failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Save emergency checkpoint
            self._save_checkpoint("emergency")
            
            raise
    
    def _run_hyperband_sweep(self) -> List[Dict[str, Any]]:
        """Run Hyperband optimization sweep."""
        self.logger.info("🔄 Running Hyperband sweep")
        
        results = []
        hyperband_config = self.config.get("hyperband", {})
        
        # Distribute work across GPUs
        if len(self.gpu_ids) > 1:
            results = self._run_distributed_hyperband()
        else:
            # Single GPU or CPU execution
            runner = HyperbandRunner(
                config={
                    "parameter_space": self.config["parameter_ranges"],
                    "max_iter": hyperband_config.get("max_iter", 81),
                    "eta": hyperband_config.get("eta", 3),
                    "max_parallel": hyperband_config.get("max_parallel", 4)
                },
                seed=self.master_seed,
                enable_gpu=len(self.gpu_ids) > 0
            )
            
            sweep_results = runner.run_hyperband_optimization(
                n_iterations=hyperband_config.get("iterations", 5)
            )
            
            if sweep_results["success"]:
                results.extend(sweep_results.get("top_5_configurations", []))
                self.best_configs.extend(sweep_results.get("top_5_configurations", []))
        
        return results
    
    def _run_distributed_hyperband(self) -> List[Dict[str, Any]]:
        """Run Hyperband across multiple GPUs."""
        self.logger.info(f"🌐 Distributing Hyperband across {len(self.gpu_ids)} GPUs")
        
        results = []
        hyperband_config = self.config.get("hyperband", {})
        iterations_per_gpu = hyperband_config.get("iterations", 5) // len(self.gpu_ids)
        
        # Create GPU-specific seeds
        gpu_seeds = self.reproducibility.get_gpu_seeds(len(self.gpu_ids))
        
        with ProcessPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            # Submit jobs to each GPU
            futures = []
            for gpu_id, gpu_seed in zip(self.gpu_ids, gpu_seeds):
                future = executor.submit(
                    self._run_hyperband_on_gpu,
                    gpu_id,
                    gpu_seed,
                    iterations_per_gpu
                )
                futures.append((future, gpu_id))
            
            # Collect results
            for future, gpu_id in futures:
                try:
                    gpu_results = future.result(timeout=3600 * 6)  # 6 hour timeout
                    results.extend(gpu_results)
                    self.logger.info(f"✅ GPU {gpu_id} completed: {len(gpu_results)} results")
                except Exception as e:
                    self.logger.error(f"❌ GPU {gpu_id} failed: {str(e)}")
        
        return results
    
    def _run_hyperband_on_gpu(self, gpu_id: int, seed: int, iterations: int) -> List[Dict[str, Any]]:
        """Run Hyperband on a specific GPU."""
        # Set GPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Create runner with GPU-specific seed
        runner = HyperbandRunner(
            config={
                "parameter_space": self.config["parameter_ranges"],
                "max_iter": self.config["hyperband"].get("max_iter", 81),
                "eta": self.config["hyperband"].get("eta", 3),
                "max_parallel": 1  # Single GPU
            },
            seed=seed,
            enable_gpu=True
        )
        
        # Run optimization
        sweep_results = runner.run_hyperband_optimization(n_iterations=iterations)
        
        if sweep_results["success"]:
            return sweep_results.get("top_5_configurations", [])
        
        return []
    
    def _run_grid_sweep(self) -> List[Dict[str, Any]]:
        """Run grid search sweep."""
        self.logger.info("🔲 Running grid search sweep")
        
        # Generate all parameter combinations
        param_grid = self._generate_parameter_grid()
        self.logger.info(f"Total configurations: {len(param_grid)}")
        
        # Evaluate configurations in batches
        results = []
        batch_size = len(self.gpu_ids) * 4 if self.gpu_ids else 4
        
        for i in range(0, len(param_grid), batch_size):
            batch = param_grid[i:i + batch_size]
            batch_results = self._evaluate_batch(batch)
            results.extend(batch_results)
            
            # Save intermediate checkpoint
            if i % (batch_size * 10) == 0:
                self._save_checkpoint(f"grid_{i}")
        
        return results
    
    def _run_random_sweep(self) -> List[Dict[str, Any]]:
        """Run random search sweep."""
        self.logger.info("🎲 Running random search sweep")
        
        num_samples = self.config.get("random_samples", 1000)
        results = []
        
        # Use reproducible random generator
        rng = np.random.Generator(np.random.PCG64(self.master_seed))
        
        # Generate random configurations
        for i in range(num_samples):
            config = self._sample_random_config(rng)
            
            # Evaluate configuration
            result = self._evaluate_config(config, gpu_id=i % len(self.gpu_ids) if self.gpu_ids else None)
            results.append(result)
            
            # Periodic checkpoint
            if i % 100 == 0:
                self._save_checkpoint(f"random_{i}")
                self.logger.info(f"Progress: {i}/{num_samples} configurations evaluated")
        
        return results
    
    def _run_bayesian_sweep(self) -> List[Dict[str, Any]]:
        """Run Bayesian optimization sweep."""
        self.logger.info("🧠 Running Bayesian optimization sweep")
        
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import HyperbandPruner
        except ImportError:
            self.logger.error("Optuna not installed, falling back to random search")
            return self._run_random_sweep()
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.sweep_id,
            direction="maximize",
            sampler=TPESampler(seed=self.master_seed),
            pruner=HyperbandPruner(),
            storage=f"sqlite:///{self.log_dir}/optuna_{self.sweep_id}.db",
            load_if_exists=True
        )
        
        # Objective function
        def objective(trial):
            config = self._sample_optuna_config(trial)
            result = self._evaluate_config(config)
            
            # Report intermediate values for pruning
            if hasattr(trial, "_trial_id"):
                trial.report(result.get("composite_score", -999), trial._trial_id)
            
            return result.get("composite_score", -999)
        
        # Optimize
        n_trials = self.config.get("bayesian_trials", 500)
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=len(self.gpu_ids) if self.gpu_ids else 1,
            show_progress_bar=True
        )
        
        # Extract results
        results = []
        for trial in study.best_trials[:20]:  # Top 20 trials
            config = trial.params
            config["composite_score"] = trial.value
            results.append(config)
        
        return results
    
    def _evaluate_config(self, config: Dict[str, Any], gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a single configuration."""
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        try:
            # Create retrain worker
            worker = RetrainWorker()
            
            # Train and evaluate
            backtest_results = worker.train_and_evaluate(config)
            
            # Evaluate with objective function
            objective_fn = MetaObjectiveFunction()
            evaluation = objective_fn.evaluate_strategy(backtest_results)
            
            # Add configuration
            evaluation["configuration"] = config
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {
                "configuration": config,
                "composite_score": -999,
                "viable": False,
                "error": str(e)
            }
    
    def _evaluate_batch(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of configurations in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=len(self.gpu_ids) or 1) as executor:
            futures = []
            
            for i, config in enumerate(configs):
                gpu_id = i % len(self.gpu_ids) if self.gpu_ids else None
                future = executor.submit(self._evaluate_config, config, gpu_id)
                futures.append((future, config))
            
            for future, config in futures:
                try:
                    result = future.result(timeout=1800)  # 30 min timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch evaluation failed: {str(e)}")
                    results.append({
                        "configuration": config,
                        "composite_score": -999,
                        "viable": False,
                        "error": str(e)
                    })
        
        return results
    
    def _start_resource_monitoring(self):
        """Start monitoring system resources."""
        def monitor():
            while True:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # GPU usage
                gpu_stats = []
                if self.gpu_ids:
                    try:
                        gpus = GPUtil.getGPUs()
                        for gpu in gpus:
                            if gpu.id in self.gpu_ids:
                                gpu_stats.append({
                                    "id": gpu.id,
                                    "load": gpu.load * 100,
                                    "memory_used": gpu.memoryUsed,
                                    "memory_total": gpu.memoryTotal,
                                    "temperature": gpu.temperature
                                })
                    except:
                        pass
                
                self.resource_usage.append({
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "gpu_stats": gpu_stats
                })
                
                time.sleep(60)  # Monitor every minute
        
        # Start monitoring in background thread
        import threading
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        from itertools import product
        
        param_ranges = self.config["parameter_ranges"]
        
        # Create all combinations
        all_combinations = []
        
        for model_name, model_params in param_ranges.items():
            model_keys = list(model_params.keys())
            model_values = [model_params[k] for k in model_keys]
            
            for values in product(*model_values):
                combo = {model_name: dict(zip(model_keys, values))}
                all_combinations.append(combo)
        
        return all_combinations
    
    def _sample_random_config(self, rng) -> Dict[str, Any]:
        """Sample a random configuration."""
        config = {}
        
        for model_name, model_params in self.config["parameter_ranges"].items():
            config[model_name] = {}
            for param_name, param_values in model_params.items():
                config[model_name][param_name] = rng.choice(param_values)
        
        return config
    
    def _sample_optuna_config(self, trial) -> Dict[str, Any]:
        """Sample configuration using Optuna trial."""
        config = {}
        
        for model_name, model_params in self.config["parameter_ranges"].items():
            config[model_name] = {}
            for param_name, param_values in model_params.items():
                if isinstance(param_values, list):
                    config[model_name][param_name] = trial.suggest_categorical(
                        f"{model_name}_{param_name}", param_values
                    )
                elif isinstance(param_values, dict):
                    # Handle range specifications
                    if "min" in param_values and "max" in param_values:
                        if param_values.get("type") == "int":
                            config[model_name][param_name] = trial.suggest_int(
                                f"{model_name}_{param_name}",
                                param_values["min"],
                                param_values["max"]
                            )
                        else:
                            config[model_name][param_name] = trial.suggest_float(
                                f"{model_name}_{param_name}",
                                param_values["min"],
                                param_values["max"],
                                log=param_values.get("log", False)
                            )
        
        return config
    
    def _process_results(self, results: List[Dict[str, Any]], start_time: datetime) -> Dict[str, Any]:
        """Process and analyze sweep results."""
        self.logger.info("📊 Processing sweep results")
        
        # Filter valid results
        valid_results = [r for r in results if r.get("viable", False)]
        
        if not valid_results:
            return {
                "success": False,
                "reason": "No viable configurations found",
                "total_evaluated": len(results),
                "duration": (datetime.now() - start_time).total_seconds()
            }
        
        # Sort by score
        valid_results.sort(key=lambda x: x.get("composite_score", -999), reverse=True)
        
        # Calculate statistics
        scores = [r["composite_score"] for r in valid_results]
        
        sweep_results = {
            "success": True,
            "sweep_id": self.sweep_id,
            "total_evaluated": len(results),
            "viable_count": len(valid_results),
            "duration": (datetime.now() - start_time).total_seconds(),
            "best_configuration": valid_results[0],
            "top_10_configurations": valid_results[:10],
            "statistics": {
                "best_score": max(scores),
                "worst_score": min(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "median_score": np.median(scores)
            },
            "resource_usage": self._analyze_resource_usage(),
            "parameter_importance": self._analyze_parameter_importance(valid_results)
        }
        
        return sweep_results
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage during sweep."""
        if not self.resource_usage:
            return {}
        
        cpu_usage = [r["cpu_percent"] for r in self.resource_usage]
        memory_usage = [r["memory_percent"] for r in self.resource_usage]
        
        analysis = {
            "cpu": {
                "mean": np.mean(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            },
            "memory": {
                "mean": np.mean(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            }
        }
        
        # GPU analysis
        if self.gpu_ids and self.resource_usage[0].get("gpu_stats"):
            for gpu_id in self.gpu_ids:
                gpu_loads = []
                gpu_memories = []
                
                for r in self.resource_usage:
                    for gpu_stat in r["gpu_stats"]:
                        if gpu_stat["id"] == gpu_id:
                            gpu_loads.append(gpu_stat["load"])
                            gpu_memories.append(
                                gpu_stat["memory_used"] / gpu_stat["memory_total"] * 100
                            )
                
                if gpu_loads:
                    analysis[f"gpu_{gpu_id}"] = {
                        "load_mean": np.mean(gpu_loads),
                        "load_max": max(gpu_loads),
                        "memory_mean": np.mean(gpu_memories),
                        "memory_max": max(gpu_memories)
                    }
        
        return analysis
    
    def _analyze_parameter_importance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter importance from results."""
        if len(results) < 10:
            return {}
        
        # Extract parameters and scores
        param_scores = {}
        
        for result in results:
            config = result.get("configuration", {})
            score = result.get("composite_score", 0)
            
            for model_name, model_params in config.items():
                if isinstance(model_params, dict):
                    for param_name, param_value in model_params.items():
                        key = f"{model_name}.{param_name}"
                        if key not in param_scores:
                            param_scores[key] = []
                        param_scores[key].append((param_value, score))
        
        # Analyze correlations
        importance = {}
        
        for param_key, value_scores in param_scores.items():
            if len(set(v for v, _ in value_scores)) > 1:  # Multiple values
                # Group by value
                value_groups = {}
                for value, score in value_scores:
                    if value not in value_groups:
                        value_groups[value] = []
                    value_groups[value].append(score)
                
                # Calculate mean scores
                value_means = {
                    value: np.mean(scores)
                    for value, scores in value_groups.items()
                }
                
                # Find best value
                best_value = max(value_means.keys(), key=lambda k: value_means[k])
                
                importance[param_key] = {
                    "best_value": best_value,
                    "best_mean_score": value_means[best_value],
                    "value_impact": max(value_means.values()) - min(value_means.values())
                }
        
        # Sort by impact
        sorted_importance = dict(
            sorted(importance.items(), 
                   key=lambda x: x[1]["value_impact"], 
                   reverse=True)
        )
        
        return sorted_importance
    
    def _generate_reports(self, results: Dict[str, Any]):
        """Generate comprehensive sweep reports."""
        self.logger.info("📝 Generating sweep reports")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report = self.log_dir / f"sweep_results_{self.sweep_id}.json"
        with open(json_report, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Human-readable report
        text_report = self.log_dir / f"sweep_report_{self.sweep_id}.txt"
        with open(text_report, 'w') as f:
            f.write(self._format_text_report(results))
        
        # Best configurations
        best_configs = self.log_dir / f"best_configs_{self.sweep_id}.json"
        with open(best_configs, 'w') as f:
            json.dump({
                "sweep_id": self.sweep_id,
                "timestamp": timestamp,
                "configurations": results.get("top_10_configurations", [])
            }, f, indent=2, default=str)
        
        self.logger.info(f"📁 Reports saved to: {self.log_dir}")
    
    def _format_text_report(self, results: Dict[str, Any]) -> str:
        """Format human-readable text report."""
        report = []
        report.append("=" * 80)
        report.append("GPU PARAMETER SWEEP REPORT")
        report.append("=" * 80)
        report.append(f"Sweep ID: {results.get('sweep_id', 'N/A')}")
        report.append(f"Duration: {results.get('duration', 0) / 3600:.2f} hours")
        report.append(f"Total Configurations: {results.get('total_evaluated', 0)}")
        report.append(f"Viable Configurations: {results.get('viable_count', 0)}")
        report.append("")
        
        if results.get("success"):
            report.append("BEST CONFIGURATION")
            report.append("-" * 40)
            best = results.get("best_configuration", {})
            report.append(f"Score: {best.get('composite_score', 0):.4f}")
            report.append(f"Sharpe Ratio: {best.get('individual_metrics', {}).get('sharpe_ratio', 0):.2f}")
            report.append(f"Profit Factor: {best.get('individual_metrics', {}).get('profit_factor', 0):.2f}")
            report.append(f"Max Drawdown: {best.get('individual_metrics', {}).get('max_drawdown', 0):.2f}%")
            report.append("")
            
            report.append("STATISTICS")
            report.append("-" * 40)
            stats = results.get("statistics", {})
            report.append(f"Best Score: {stats.get('best_score', 0):.4f}")
            report.append(f"Mean Score: {stats.get('mean_score', 0):.4f} ± {stats.get('std_score', 0):.4f}")
            report.append(f"Median Score: {stats.get('median_score', 0):.4f}")
            report.append("")
            
            report.append("RESOURCE USAGE")
            report.append("-" * 40)
            resources = results.get("resource_usage", {})
            if resources:
                report.append(f"CPU: {resources.get('cpu', {}).get('mean', 0):.1f}% avg, {resources.get('cpu', {}).get('max', 0):.1f}% max")
                report.append(f"Memory: {resources.get('memory', {}).get('mean', 0):.1f}% avg, {resources.get('memory', {}).get('max', 0):.1f}% max")
                
                for gpu_id in self.gpu_ids:
                    gpu_key = f"gpu_{gpu_id}"
                    if gpu_key in resources:
                        gpu_data = resources[gpu_key]
                        report.append(f"GPU {gpu_id}: {gpu_data.get('load_mean', 0):.1f}% avg load, {gpu_data.get('memory_mean', 0):.1f}% avg memory")
            report.append("")
            
            report.append("TOP 5 PARAMETER IMPACTS")
            report.append("-" * 40)
            importance = results.get("parameter_importance", {})
            for i, (param, data) in enumerate(list(importance.items())[:5]):
                report.append(f"{i+1}. {param}: best={data['best_value']}, impact={data['value_impact']:.4f}")
        
        else:
            report.append(f"SWEEP FAILED: {results.get('reason', 'Unknown error')}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _save_checkpoint(self, tag: str = ""):
        """Save checkpoint for resuming."""
        checkpoint = {
            "sweep_id": self.sweep_id,
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "master_seed": self.master_seed,
            "config": self.config,
            "all_results": self.all_results,
            "best_configs": self.best_configs,
            "resource_usage": self.resource_usage[-100:] if self.resource_usage else []  # Last 100 entries
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume sweep."""
        self.logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.sweep_id = checkpoint["sweep_id"]
        self.all_results = checkpoint["all_results"]
        self.best_configs = checkpoint["best_configs"]
        self.resource_usage = checkpoint["resource_usage"]
        
        self.logger.info(f"✅ Resumed from checkpoint with {len(self.all_results)} results")


def main():
    """Main entry point for GPU sweep runner."""
    parser = argparse.ArgumentParser(description="GPU-Optimized Parameter Sweep Runner")
    parser.add_argument("--config", type=str, default="sweep_config.json", 
                       help="Path to sweep configuration JSON")
    parser.add_argument("--gpu-ids", type=str, default=None,
                       help="Comma-separated GPU IDs to use (e.g., 0,1,2)")
    parser.add_argument("--seed", type=int, default=123456,
                       help="Master seed for reproducibility")
    parser.add_argument("--log-dir", type=str, default="sweep_logs",
                       help="Directory for logs and checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    
    try:
        # Create sweep runner
        runner = GPUSweepRunner(
            config_path=args.config,
            gpu_ids=gpu_ids,
            master_seed=args.seed,
            log_dir=args.log_dir
        )
        
        # Run sweep
        results = runner.run_sweep(resume_from_checkpoint=args.resume)
        
        if results["success"]:
            print(f"\n✅ Sweep completed successfully!")
            print(f"Best score: {results['best_configuration']['composite_score']:.4f}")
            print(f"Total time: {results['duration'] / 3600:.2f} hours")
        else:
            print(f"\n❌ Sweep failed: {results.get('reason', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()