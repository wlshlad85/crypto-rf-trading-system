import json, os, sys, numpy as np, subprocess, tempfile, yaml
from pathlib import Path


def run(mode: str):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    subprocess.check_call([
        sys.executable,
        "scripts/run_pipeline.py",
        "--mode",
        mode,
        "--input",
        "tests/data/golden.parquet",
        "--out",
        out,
    ])
    with open(out) as fh:
        return json.load(fh)


def test_metrics_and_parity():
    golden = Path("tests/data/golden.parquet")
    exp_path = Path("tests/expectations.yaml")
    assert golden.exists(), "Missing tests/data/golden.parquet. Run: python scripts/generate_golden.py"
    assert exp_path.exists(), "Missing tests/expectations.yaml. Run: python scripts/generate_golden.py"
    exp = yaml.safe_load(open("tests/expectations.yaml"))
    cpu = run("cpu")
    gpu = run("gpu")

    # metrics vs expectations
    for k, cfg in exp["metrics"].items():
        for mode, res in [("cpu", cpu), ("gpu", gpu)]:
            val = res["metrics"][k]
            assert abs(val - cfg["expected"]) <= cfg["atol"], f"{mode} {k} drift: {val} vs {cfg}"

    # parity on predictions
    yhat_cpu = np.array(cpu["predictions"])
    yhat_gpu = np.array(gpu["predictions"])
    rmse = float(np.sqrt(np.mean((yhat_cpu - yhat_gpu) ** 2)))
    assert rmse <= exp["parity"]["yhat_rmse_atol"], f"CPU/GPU mismatch: {rmse}"
