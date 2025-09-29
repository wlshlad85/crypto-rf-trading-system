# Makefile for GPU/CPU regression testing

.PHONY: help install test clean golden baseline

# Default target
help:
	@echo "GPU/CPU Regression Testing"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       Install CPU dependencies"
	@echo "  make install-gpu   Install GPU dependencies (CUDA 11.8)"
	@echo "  make golden        Generate golden dataset"
	@echo "  make test          Run all tests"
	@echo "  make test-cpu      Run CPU tests only"
	@echo "  make test-gpu      Run GPU tests only"
	@echo "  make test-quick    Run quick smoke tests"
	@echo "  make baseline      Update baseline expectations"
	@echo "  make benchmark     Run performance benchmarks"
	@echo "  make clean         Clean up generated files"

# Installation
install:
	pip install -r requirements.txt
	pip install -r requirements-test.txt

install-gpu: install
	pip install -r requirements-gpu-cuda11.8.txt

install-gpu-cu12: install
	pip install -r requirements-gpu-cuda12.1.txt

# Data generation
golden:
	python scripts/generate_golden_data.py

# Testing
test: golden
	pytest tests/ -v

test-cpu: golden
	pytest tests/ -v -k "cpu or not gpu"

test-gpu: golden
	pytest tests/ -v -k "gpu or parity"

test-quick: golden
	pytest tests/test_regression.py::test_golden_data_exists -v
	pytest tests/test_regression.py::TestMetricsRegression::test_cpu_metrics -v

# Benchmarking
benchmark: golden
	pytest tests/test_latency.py -v --benchmark-only

# Baseline management
baseline: golden
	python scripts/update_baseline.py

baseline-no-gpu: golden
	python scripts/update_baseline.py --skip-gpu

# Continuous testing
watch:
	pytest-watch tests/ -v

# Profiling
profile-cpu:
	python -m cProfile -o cpu_profile.prof scripts/run_pipeline.py --mode cpu --input tests/data/golden.parquet --out /tmp/out.json
	python -m pstats cpu_profile.prof

profile-memory:
	mprof run python scripts/run_pipeline.py --mode cpu --input tests/data/golden.parquet --out /tmp/out.json
	mprof plot

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .coverage coverage.xml pytest_results.xml
	rm -f tests/data/golden.parquet tests/data/golden.parquet.stats.json
	rm -f *.prof mprofile_*.dat
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete