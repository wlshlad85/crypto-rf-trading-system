#!/usr/bin/env python3
"""
Crypto RF Trading GPU Reproducibility Module
============================================

A comprehensive reproducibility kit tailored for the crypto-rf-trading-system's
nightly GPU parameter sweeps. Ensures consistent results across runs for:
- Hyperparameter optimization sweeps
- Model training and evaluation  
- Multi-GPU distributed training
- CatBoost/LightGBM GPU operations

Usage:
    from crypto_gpu_reproducibility import CryptoGPUReproducibility
    
    repro = CryptoGPUReproducibility(seed=123456)
    repro.setup_environment()
    repro.log_system_info("sweep_logs/meta.json")
"""

import os
import sys
import json
import platform
import random
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging

# Optional imports with fallback
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.distributed as dist
    import torch.backends.cudnn as cudnn
except ImportError:
    torch = None
    dist = None
    cudnn = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None


class CryptoGPUReproducibility:
    """
    Comprehensive GPU reproducibility manager for crypto RF trading system.
    """
    
    def __init__(self, seed: int = 123456, log_level: str = "INFO"):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Master seed for all RNGs
            log_level: Logging level (INFO, DEBUG, WARNING)
        """
        self.master_seed = seed
        self.seeds_registry = {}
        self.system_info = {}
        self.is_setup = False
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Detect GPU availability
        self.has_cuda = torch is not None and torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.has_cuda else 0
        
        self.logger.info(f"🎯 CryptoGPUReproducibility initialized with seed={seed}")
        self.logger.info(f"🖥️  CUDA available: {self.has_cuda}, Device count: {self.device_count}")
    
    def setup_environment(self, force_deterministic: bool = True) -> None:
        """
        Setup complete reproducible environment for GPU operations.
        
        Args:
            force_deterministic: If True, forces all operations to be deterministic
        """
        if self.is_setup:
            self.logger.warning("Environment already setup, skipping...")
            return
        
        self.logger.info("🔧 Setting up reproducible environment...")
        
        # 1. Set environment variables BEFORE any CUDA initialization
        self._set_environment_variables()
        
        # 2. Set all random seeds
        self._set_all_seeds()
        
        # 3. Configure PyTorch for determinism
        if torch is not None:
            self._configure_pytorch(force_deterministic)
        
        # 4. Configure other frameworks
        self._configure_other_frameworks()
        
        # 5. Log system information
        self._collect_system_info()
        
        self.is_setup = True
        self.logger.info("✅ Reproducible environment setup complete!")
    
    def _set_environment_variables(self) -> None:
        """Set critical environment variables for reproducibility."""
        env_vars = {
            # Python hash seed
            "PYTHONHASHSEED": str(self.master_seed),
            
            # CUDA determinism
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # For CUDA >= 10.2
            
            # Disable CUDA async operations
            "CUDA_LAUNCH_BLOCKING": "1",
            
            # TensorFlow determinism
            "TF_DETERMINISTIC_OPS": "1",
            "TF_CUDNN_DETERMINISTIC": "1",
            
            # XGBoost reproducibility
            "XGBOOST_USE_CUDA": "1",
            
            # Numba CUDA settings
            "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY": "1",
            
            # Multi-threading determinism
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            self.logger.debug(f"Set {var}={value}")
    
    def _set_all_seeds(self) -> None:
        """Set seeds for all random number generators."""
        # Python's built-in random
        random.seed(self.master_seed)
        self.seeds_registry['python_random'] = self.master_seed
        
        # NumPy
        if np is not None:
            np.random.seed(self.master_seed)
            self.seeds_registry['numpy_legacy'] = self.master_seed
            
            # Modern NumPy Generator
            self.np_generator = np.random.Generator(np.random.PCG64(self.master_seed))
            self.seeds_registry['numpy_generator'] = 'PCG64'
        
        # PyTorch
        if torch is not None:
            torch.manual_seed(self.master_seed)
            if self.has_cuda:
                torch.cuda.manual_seed(self.master_seed)
                torch.cuda.manual_seed_all(self.master_seed)
            self.seeds_registry['torch'] = self.master_seed
        
        # CuPy
        if cp is not None:
            cp.random.seed(self.master_seed)
            self.seeds_registry['cupy'] = self.master_seed
        
        # TensorFlow
        if tf is not None:
            tf.random.set_seed(self.master_seed)
            self.seeds_registry['tensorflow'] = self.master_seed
        
        self.logger.info(f"🌱 Set seeds for: {list(self.seeds_registry.keys())}")
    
    def _configure_pytorch(self, force_deterministic: bool) -> None:
        """Configure PyTorch for reproducibility."""
        if not torch:
            return
        
        # Enable deterministic algorithms
        if force_deterministic:
            torch.use_deterministic_algorithms(True)
            self.logger.info("⚡ PyTorch deterministic algorithms: ENABLED")
        
        # Configure cuDNN
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False  # Disable auto-tuner
            self.logger.info("⚡ cuDNN deterministic: ENABLED, benchmark: DISABLED")
        
        # Set default tensor type
        if self.has_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    def _configure_other_frameworks(self) -> None:
        """Configure other ML frameworks for reproducibility."""
        # Configure Numba CUDA
        if numba_cuda is not None:
            try:
                # Force consistent kernel compilation
                numba_cuda.select_device(0)
                self.logger.info("⚡ Numba CUDA device selected: 0")
            except Exception as e:
                self.logger.warning(f"Failed to configure Numba CUDA: {e}")
        
        # Configure TensorFlow
        if tf is not None:
            try:
                # Limit GPU memory growth
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info("⚡ TensorFlow GPU memory growth: ENABLED")
            except Exception as e:
                self.logger.warning(f"Failed to configure TensorFlow: {e}")
    
    def _collect_system_info(self) -> None:
        """Collect comprehensive system information."""
        self.system_info = {
            "timestamp": datetime.now().isoformat(),
            "master_seed": self.master_seed,
            "seeds": self.seeds_registry,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "python_compiler": platform.python_compiler()
            },
            "environment": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING")
            }
        }
        
        # GPU information
        if self.has_cuda:
            self.system_info["gpu"] = {
                "device_count": self.device_count,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if cudnn else None,
                "devices": []
            }
            
            for i in range(self.device_count):
                device_props = torch.cuda.get_device_properties(i)
                self.system_info["gpu"]["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "multi_processor_count": device_props.multi_processor_count
                })
        
        # Framework versions
        self.system_info["frameworks"] = {
            "numpy": np.__version__ if np else None,
            "torch": torch.__version__ if torch else None,
            "cupy": cp.__version__ if cp else None,
            "tensorflow": tf.__version__ if tf else None,
        }
        
        # Get conda/pip environment info
        try:
            conda_info = subprocess.check_output(['conda', 'info', '--json'], 
                                               stderr=subprocess.DEVNULL).decode()
            self.system_info["conda"] = json.loads(conda_info).get("active_prefix_name")
        except:
            self.system_info["conda"] = None
    
    def get_worker_seeds(self, num_workers: int, base_offset: int = 1000) -> List[int]:
        """
        Generate deterministic seeds for worker processes.
        
        Args:
            num_workers: Number of worker processes
            base_offset: Offset to ensure different seeds from master
            
        Returns:
            List of seeds for each worker
        """
        worker_seeds = []
        for i in range(num_workers):
            # Derive worker seed from master seed
            worker_seed = self.master_seed + base_offset + i
            worker_seeds.append(worker_seed)
        
        self.logger.info(f"🌱 Generated {num_workers} worker seeds: {worker_seeds}")
        return worker_seeds
    
    def get_gpu_seeds(self, num_gpus: Optional[int] = None) -> List[int]:
        """
        Generate deterministic seeds for multi-GPU training.
        
        Args:
            num_gpus: Number of GPUs (defaults to detected count)
            
        Returns:
            List of seeds for each GPU
        """
        if num_gpus is None:
            num_gpus = self.device_count
        
        gpu_seeds = []
        for i in range(num_gpus):
            # Hash-based derivation for GPU seeds
            gpu_string = f"gpu_{i}_seed_{self.master_seed}"
            gpu_seed = int(hashlib.sha256(gpu_string.encode()).hexdigest(), 16) % (2**31)
            gpu_seeds.append(gpu_seed)
        
        self.logger.info(f"🖥️  Generated {num_gpus} GPU seeds: {gpu_seeds}")
        return gpu_seeds
    
    def setup_distributed(self, rank: int, world_size: int) -> None:
        """
        Setup seeds for distributed training.
        
        Args:
            rank: Process rank in distributed training
            world_size: Total number of processes
        """
        if not torch or not dist:
            self.logger.warning("PyTorch distributed not available")
            return
        
        # Derive rank-specific seed
        rank_seed = self.master_seed + rank * 100
        
        # Set seeds for this rank
        torch.manual_seed(rank_seed)
        if self.has_cuda:
            torch.cuda.manual_seed(rank_seed)
        
        self.logger.info(f"🌐 Distributed setup for rank {rank}/{world_size}, seed={rank_seed}")
    
    def setup_data_loader(self, loader, num_workers: int = 0) -> None:
        """
        Configure DataLoader for reproducibility.
        
        Args:
            loader: PyTorch DataLoader instance
            num_workers: Number of worker processes
        """
        if not torch:
            return
        
        # Set worker seeds
        def worker_init_fn(worker_id):
            worker_seed = self.master_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            if torch:
                torch.manual_seed(worker_seed)
        
        loader.worker_init_fn = worker_init_fn
        self.logger.info(f"📊 DataLoader configured with {num_workers} workers")
    
    def log_system_info(self, filepath: str, extra_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Log complete system information to file.
        
        Args:
            filepath: Path to save JSON log file
            extra_config: Additional configuration to log
            
        Returns:
            Complete system information dictionary
        """
        if not self.is_setup:
            self.logger.warning("Environment not setup, call setup_environment() first")
            return {}
        
        # Add extra configuration
        if extra_config:
            self.system_info["extra_config"] = extra_config
        
        # Add optimization sweep metadata
        self.system_info["sweep_metadata"] = {
            "sweep_type": "hyperparameter_optimization",
            "framework": "crypto_rf_trading",
            "optimization_method": "hyperband",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.system_info, f, indent=2, default=str)
        
        self.logger.info(f"📁 System info logged to: {filepath}")
        return self.system_info
    
    def get_catboost_params(self) -> Dict[str, Any]:
        """Get CatBoost parameters for GPU reproducibility."""
        params = {
            'random_state': self.master_seed,
            'thread_count': 1,  # Single thread for determinism
            'task_type': 'GPU' if self.has_cuda else 'CPU',
            'devices': '0' if self.has_cuda else None,
            'gpu_ram_part': 0.5,  # Limit GPU memory usage
            'bootstrap_type': 'Bayesian',  # More stable than default
            'bagging_temperature': 1.0,
            'used_ram_limit': '8gb',
            'allow_writing_files': False,
            'verbose': False
        }
        
        if self.has_cuda:
            params['gpu_cat_features_storage'] = 'GpuRam'
            
        return params
    
    def get_lightgbm_params(self) -> Dict[str, Any]:
        """Get LightGBM parameters for GPU reproducibility."""
        params = {
            'random_state': self.master_seed,
            'deterministic': True,
            'force_row_wise': True,  # Force deterministic histogram building
            'num_threads': 1,
            'device_type': 'gpu' if self.has_cuda else 'cpu',
            'gpu_device_id': 0 if self.has_cuda else -1,
            'gpu_use_dp': False,  # Use float32 for consistency
            'seed': self.master_seed,
            'bagging_seed': self.master_seed,
            'feature_fraction_seed': self.master_seed,
            'drop_seed': self.master_seed,
            'data_random_seed': self.master_seed,
            'objective_seed': self.master_seed,
            'boosting_seed': self.master_seed,
            'verbose': -1
        }
        
        return params
    
    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters for GPU reproducibility."""
        params = {
            'random_state': self.master_seed,
            'seed': self.master_seed,
            'tree_method': 'gpu_hist' if self.has_cuda else 'hist',
            'gpu_id': 0 if self.has_cuda else -1,
            'predictor': 'gpu_predictor' if self.has_cuda else 'cpu_predictor',
            'deterministic_histogram': True,
            'subsample': 1.0,  # No random subsampling
            'sampling_method': 'uniform',
            'n_jobs': 1,  # Single thread
            'verbosity': 0
        }
        
        return params
    
    def checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        """
        Save reproducibility checkpoint for resuming sweeps.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            iteration: Current iteration number
        """
        checkpoint = {
            'iteration': iteration,
            'master_seed': self.master_seed,
            'seeds_registry': self.seeds_registry,
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info
        }
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / f"repro_checkpoint_{iteration}.json"
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.logger.info(f"💾 Reproducibility checkpoint saved: {checkpoint_path}")
    
    def validate_reproducibility(self, test_fn, num_runs: int = 3) -> bool:
        """
        Validate that results are reproducible.
        
        Args:
            test_fn: Function to test reproducibility
            num_runs: Number of runs to validate
            
        Returns:
            True if all runs produce identical results
        """
        results = []
        
        for i in range(num_runs):
            # Reset environment for each run
            self.setup_environment()
            
            # Run test function
            result = test_fn()
            results.append(result)
            
            self.logger.info(f"🧪 Reproducibility test run {i+1}: {result}")
        
        # Check if all results are identical
        all_identical = all(r == results[0] for r in results)
        
        if all_identical:
            self.logger.info("✅ Reproducibility validated: All runs identical!")
        else:
            self.logger.error("❌ Reproducibility failed: Results differ across runs!")
            for i, r in enumerate(results):
                self.logger.error(f"  Run {i+1}: {r}")
        
        return all_identical


# Convenience functions
def setup_crypto_gpu_reproducibility(seed: int = 123456, force_deterministic: bool = True) -> CryptoGPUReproducibility:
    """
    Quick setup function for crypto GPU reproducibility.
    
    Args:
        seed: Master seed for all RNGs
        force_deterministic: Force deterministic algorithms
        
    Returns:
        Configured CryptoGPUReproducibility instance
    """
    repro = CryptoGPUReproducibility(seed=seed)
    repro.setup_environment(force_deterministic=force_deterministic)
    return repro


def get_recommended_sweep_config(seed: int = 123456) -> Dict[str, Any]:
    """
    Get recommended configuration for nightly GPU sweeps.
    
    Returns:
        Dictionary with recommended settings
    """
    repro = CryptoGPUReproducibility(seed=seed)
    
    config = {
        'reproducibility': {
            'master_seed': seed,
            'force_deterministic': True,
            'worker_seeds': repro.get_worker_seeds(4),  # 4 workers
            'gpu_seeds': repro.get_gpu_seeds()
        },
        'catboost_params': repro.get_catboost_params(),
        'lightgbm_params': repro.get_lightgbm_params(),
        'xgboost_params': repro.get_xgboost_params(),
        'environment': {
            'PYTHONHASHSEED': str(seed),
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'CUDA_LAUNCH_BLOCKING': '1'
        }
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    print("🚀 Crypto GPU Reproducibility Module")
    print("=" * 50)
    
    # Setup reproducibility
    repro = setup_crypto_gpu_reproducibility(seed=123456)
    
    # Log system info
    repro.log_system_info("meta_logs/gpu_repro_test.json", extra_config={
        "test": "reproducibility_check",
        "model": "random_forest"
    })
    
    # Get framework-specific parameters
    print("\n📊 Framework Parameters:")
    print(f"CatBoost: {repro.get_catboost_params()}")
    print(f"LightGBM: {repro.get_lightgbm_params()}")
    
    # Test reproducibility
    def test_random():
        return [random.random() for _ in range(5)]
    
    print("\n🧪 Testing reproducibility...")
    is_reproducible = repro.validate_reproducibility(test_random, num_runs=3)
    
    print(f"\n✅ Setup complete! Reproducible: {is_reproducible}")