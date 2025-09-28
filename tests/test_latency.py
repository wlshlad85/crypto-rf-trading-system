import json, subprocess, sys, tempfile, time, yaml
from pathlib import Path


def run_and_time(mode: str):
    t0 = time.perf_counter()
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
    dt_ms = (time.perf_counter() - t0) * 1000.0
    with open(out) as fh:
        payload = json.load(fh)
    return dt_ms, bool(payload.get("used_gpu", False))


def test_latency_budget():
    golden = Path("tests/data/golden.parquet")
    exp_path = Path("tests/expectations.yaml")
    assert golden.exists(), "Missing tests/data/golden.parquet. Run: python scripts/generate_golden.py"
    assert exp_path.exists(), "Missing tests/expectations.yaml. Run: python scripts/generate_golden.py"
    exp = yaml.safe_load(open("tests/expectations.yaml"))
    cpu_ms, _ = run_and_time("cpu")
    gpu_ms, used_gpu = run_and_time("gpu")

    assert cpu_ms <= exp["latency_ms"]["cpu_max"], f"CPU slow: {cpu_ms:.1f}ms"
    if used_gpu:
        assert gpu_ms <= exp["latency_ms"]["gpu_max"], f"GPU slow: {gpu_ms:.1f}ms"
