"""
AdaptiveAlpha Reproducibility Kit
---------------------------------
Drop this file anywhere on your PYTHONPATH and:

    import adaptivealpha_seedkit as aask

    aask.setup_environment()                     # env flags (cuBLAS) + PyTorch deterministic
    g_cpu, children = aask.make_seed_generators( # stable RNGs for CPU / workers / ranks
        root_seed=123456,
        world_size=aask.visible_world_size()
    )
    meta = aask.log_run_extended("runs/2025-09-27/meta.json", 123456, extra_cfg={"lr": 1e-3})

Then wire the helpers into cuDF, XGBoost (GPU), FAISS, Numba CUDA, and DDP using the
functions provided below.
"""

from __future__ import annotations

import os
import sys
import json
import platform
import subprocess
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

# ---------- Light (optional) imports; guard to keep this module importable everywhere ----------
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    dist = None  # type: ignore

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore

try:
    import cudf
except Exception:  # pragma: no cover
    cudf = None  # type: ignore

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from numba import cuda as numba_cuda  # type: ignore
except Exception:  # pragma: no cover
    numba_cuda = None  # type: ignore


# ========= Seed derivation utilities =========

def _hash_to_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31 - 1)


def visible_world_size() -> int:
    """Infer 'world size' from visible CUDA devices (fallback=1)."""
    try:
        if torch and torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 1


def make_seed_generators(root_seed: int, world_size: int = 1):
    """
    Create a process/global CPU rng and a list of per-rank/worker RNGs.
    Returns:
        g_cpu: numpy.Generator for host ops
        children: list[dict] with {"np": np.Generator, "torch": torch.Generator|None, "cupy": cupy.RandomState|None}
    """
    if np is None:
        raise RuntimeError("numpy is required for make_seed_generators")

    # Root seeds for Python's random and NumPy legacy API (some libs still call it)
    random.seed(root_seed)
    np.random.seed(root_seed)

    g_cpu = np.random.Generator(np.random.PCG64(root_seed))
    children = []

    for i in range(world_size):
        child = _hash_to_int(f"{root_seed}-{i}")
        child_np = np.random.Generator(np.random.PCG64(child))
        child_torch = None
        child_cupy = None

        if torch is not None:
            gen = torch.Generator()
            gen.manual_seed(child)
            child_torch = gen

        if cp is not None:
            child_cupy = cp.random.RandomState(child)

        children.append({"np": child_np, "torch": child_torch, "cupy": child_cupy, "seed": child})

    return g_cpu, children


# ========= Environment + deterministic backends =========

def setup_environment():
    """
    One-call environment setup for determinism:
    - cuBLAS workspace config (must be set before first CUDA matmul)
    - PyTorch deterministic algorithms
    """
    # cuBLAS deterministic requirement for matmul/convolutions
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # or ":4096:8" for larger workspaces

    if torch is not None:
        try:
            torch.use_deterministic_algorithms(True)
            # cuDNN autotuner can introduce nondeterminism:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass


# ========= Logging =========

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hw() -> Dict[str, Any]:
    gpu_name = "CPU"
    cuda_runtime = None
    try:
        if torch and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_runtime = torch.version.cuda
    except Exception:
        pass

    return {
        "machine": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "gpu": gpu_name,
        "cuda_runtime": cuda_runtime,
    }


def log_run(path: str, root_seed: int, extra_cfg: Optional[dict] = None) -> Dict[str, Any]:
    info = {
        "root_seed": root_seed,
        "rng": {
            "numpy": "PCG64",
            "python_random": "MersenneTwister",
            "torch": "Philox (Generator)" if torch is not None else None,
            "cupy": "RandomState (Mersenne)" if cp is not None else None,
        },
        "versions": {
            "numpy": getattr(np, "__version__", None),
            "torch": getattr(torch, "__version__", None) if torch else None,
        },
        "env": _hw(),
        "git_sha": _git_sha(),
    }
    if extra_cfg:
        info["config"] = extra_cfg
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def log_run_extended(path: str, root_seed: int, extra_cfg: Optional[dict] = None) -> Dict[str, Any]:
    info = log_run(path, root_seed, extra_cfg)
    # Extend with GPU stack versions
    info["versions"].update({
        "cupy": getattr(cp, "__version__", None) if cp else None,
        "cudf": getattr(cudf, "__version__", None) if cudf else None,
        "xgboost": getattr(xgb, "__version__", None) if xgb else None,
        "faiss": getattr(faiss, "__version__", None) if faiss else None,
        "numba": __import__("numba").__version__ if "numba" in sys.modules or True else None,
    })
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


# ========= cuDF helpers =========

def cudf_shuffle(df, seed: int):
    """
    Deterministic shuffle for cuDF using a seeded CuPy RandomState.
    """
    if cudf is None or cp is None:
        raise RuntimeError("cuDF and CuPy are required for cudf_shuffle")
    rs = cp.random.RandomState(seed)
    idx = rs.permutation(len(df))
    return df.take(idx)


# ========= XGBoost (GPU) helpers =========

def xgb_train_deterministic(dtrain, params: Dict[str, Any], seed: int, num_boost_round: int = 200, evals=()):
    """
    Start an XGBoost training run with deterministic settings on GPU.
    Notes:
      - 'seed_per_iteration' ensures deterministic boosting even with early stopping.
      - 'deterministic_histogram' (if supported by your xgboost version) can further lock GPU histogram building.
    """
    if xgb is None:
        raise RuntimeError("xgboost is required for xgb_train_deterministic")

    p = {
        **(params or {}),
        "seed": seed,
        "seed_per_iteration": True,
        "tree_method": params.get("tree_method", "gpu_hist"),
    }
    # Optional: tighten determinism if available in your XGBoost build
    if "deterministic_histogram" in xgb.core._deprecate._VALID_PARAMETERS:  # defensive check
        p["deterministic_histogram"] = True  # type: ignore

    booster = xgb.train(p, dtrain, num_boost_round=num_boost_round, evals=evals)
    return booster


# ========= FAISS helpers =========

def faiss_train_ivf(index_key: str, dim: int, xb: np.ndarray, nlist: int, seed: int, gpu: bool = True):
    """
    Build an IVF* index deterministically by locking the clustering RNG seed.
    - index_key example: "IVF{nlist},Flat" or "IVF{nlist},PQ64"
    - xb: training vectors (float32, shape: [N, dim])
    """
    if faiss is None:
        raise RuntimeError("faiss is required for faiss_train_ivf")
    if np is None:
        raise RuntimeError("numpy is required for faiss_train_ivf")

    assert xb.dtype == np.float32 and xb.shape[1] == dim

    # Build CPU factory index first
    index_str = index_key.format(nlist=nlist) if "{nlist}" in index_key else index_key
    cpu_index = faiss.index_factory(dim, index_str, faiss.METRIC_L2)

    # Set deterministic clustering seed (used by IVF training)
    clus = faiss.Clustering(dim, nlist)
    clus.seed = seed

    # Train IVF coarse quantizer
    if gpu:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        clus.train(xb, gpu_index)
        trained = faiss.index_gpu_to_cpu(gpu_index)
    else:
        clus.train(xb, cpu_index)
        trained = cpu_index

    # Add data later in your pipeline deterministically (e.g., pre-shuffle with a fixed seed)
    return trained


def faiss_flat_index(dim: int, metric: str = "L2", gpu: bool = True):
    if faiss is None:
        raise RuntimeError("faiss is required for faiss_flat_index")
    m = faiss.METRIC_L2 if metric.upper() == "L2" else faiss.METRIC_INNER_PRODUCT
    idx = faiss.index_factory(dim, "Flat", m)
    if gpu:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, idx)
    return idx


# ========= Numba CUDA helpers =========

def numba_run_kernel_with_seed(host_array: "np.ndarray", seed: int, kernel, blocks: int, threads_per_block: int):
    """
    Example pattern for deterministic numba kernels:
    - Generate randomness on host with numpy.Generator(seed)
    - Copy to device and consume deterministically.
    'kernel' signature must accept (device_in, device_noise, device_out).
    """
    if numba_cuda is None or np is None:
        raise RuntimeError("numba and numpy are required for numba_run_kernel_with_seed")

    rs = np.random.default_rng(seed)
    noise = rs.standard_normal(size=host_array.shape).astype(np.float32)

    d_x = numba_cuda.to_device(host_array.astype(np.float32))
    d_n = numba_cuda.to_device(noise)
    d_out = numba_cuda.device_array_like(d_x)

    kernel[blocks, threads_per_block](d_x, d_n, d_out)
    return d_out.copy_to_host()


# ========= PyTorch DDP helpers =========

def ddp_child_seed(root_seed: int, rank: int) -> int:
    return _hash_to_int(f"{root_seed}-rank-{rank}")


def ddp_setup_rank(root_seed: int, rank: int):
    """
    Set seeds for a single rank/device deterministically.
    Returns a torch.Generator you can pass to DataLoader(generator=...).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for ddp_setup_rank")

    child = ddp_child_seed(root_seed, rank)
    torch.manual_seed(child)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(child)
    g = torch.Generator()
    g.manual_seed(child)
    return g


def dataloader_worker_init(worker_id: int, base_seed: int):
    """
    Plug into DataLoader(worker_init_fn=partial(dataloader_worker_init, base_seed=<seed>))
    Ensures each worker has a deterministic, unique NumPy seed.
    """
    if np is None:
        return
    # Derive a stable child seed per worker from base_seed
    child = _hash_to_int(f"{base_seed}-worker-{worker_id}")
    np.random.seed(child)


# ========= Convenience: one-call bootstrap =========

@dataclass
class Bootstrap:
    root_seed: int
    g_cpu: Any
    children: List[Dict[str, Any]]
    meta: Dict[str, Any]


def setup_run(root_seed: int,
              log_path: Optional[str] = None,
              extra_cfg: Optional[dict] = None,
              world_size: Optional[int] = None) -> Bootstrap:
    """
    Do everything:
      - set env + deterministic PyTorch
      - make seed generators (CPU + per-device/worker)
      - log run (extended if requested)
    """
    setup_environment()
    if world_size is None:
        world_size = visible_world_size()
    if np is None:
        raise RuntimeError("numpy is required for setup_run")

    g_cpu, children = make_seed_generators(root_seed, world_size=world_size)

    if torch is not None:
        # Also set the global manual seed for immediate PyTorch ops on rank 0 process
        torch.manual_seed(root_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(root_seed)

    meta = {}
    if log_path:
        meta = log_run_extended(log_path, root_seed, extra_cfg)

    return Bootstrap(root_seed=root_seed, g_cpu=g_cpu, children=children, meta=meta)