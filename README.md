### GPU/CPU parity CI scaffold

- **Generate golden assets**: create `tests/data/golden.parquet` and `tests/expectations.yaml`.

```bash
python scripts/generate_golden.py
```

- **Run tests**:

```bash
pytest -q
```

- **Notes**:
  - CPU/GPU parity and latency budgets are enforced. GPU latency is only checked if a CUDA-enabled XGBoost build is detected at runtime.
  - If running locally on a system python without venv, create a venv first.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```
