import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Features
    f_num1 = rng.normal(0.0, 1.0, size=n)
    f_num2 = rng.standard_t(df=3, size=n) * 0.5
    f_num3 = rng.uniform(-1.0, 1.0, size=n)
    f_int = rng.integers(0, 10, size=n).astype(float)
    f_miss = rng.normal(0.0, 1.0, size=n)
    # Inject some missing values
    miss_idx = rng.choice(n, size=max(1, n // 25), replace=False)
    f_miss[miss_idx] = np.nan
    # Outliers
    out_idx = rng.choice(n, size=max(1, n // 50), replace=False)
    f_num2[out_idx] *= 10.0

    # Linear combination to form logits
    logits = 0.8 * f_num1 - 0.6 * f_num2 + 0.4 * f_num3 + 0.3 * f_int
    logits += rng.normal(0.0, 0.5, size=n)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0.0, 1.0, size=n) < probs).astype(int)

    df = pd.DataFrame(
        {
            "f_num1": f_num1,
            "f_num2": f_num2,
            "f_num3": f_num3,
            "f_int": f_int,
            "f_miss": f_miss,
            "y": y,
        }
    )
    return df


def run_pipeline(mode: str, input_path: Path) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--mode",
        mode,
        "--input",
        str(input_path),
        "--out",
        out,
    ]
    subprocess.check_call(cmd)
    with open(out) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests" / "data" / "golden.parquet"))
    ap.add_argument("--expectations", default=str(ROOT / "tests" / "expectations.yaml"))
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_dataset(args.rows, args.seed)
    df.to_parquet(out_path, index=False)

    # Compute metrics via pipeline (CPU baseline)
    cpu = run_pipeline("cpu", out_path)
    gpu = run_pipeline("gpu", out_path)

    exp = {
        "metrics": {
            "auc": {"expected": float(cpu["metrics"]["auc"]), "atol": 1.0e-4},
            "sharpe_1d": {"expected": float(cpu["metrics"]["sharpe_1d"]), "atol": 5.0e-3},
            "pnl_sum": {"expected": float(cpu["metrics"]["pnl_sum"]), "atol": 0.50},
        },
        "latency_ms": {
            # Conservative defaults; tune later to be tighter once stabilized
            "cpu_max": 6000,
            "gpu_max": 1200,
        },
        "parity": {
            "yhat_rmse_atol": 5.0e-4,
        },
    }

    exp_path = Path(args.expectations)
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(exp_path, "w") as fh:
        yaml.safe_dump(exp, fh, sort_keys=False)

    print(f"Wrote {out_path} and {exp_path}")
    print("CPU metrics:", cpu["metrics"])    
    print("GPU metrics:", gpu["metrics"])    


if __name__ == "__main__":
    main()
