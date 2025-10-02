# GPU-Accelerated Quantitative Finance Data Architecture

**Target Hardware**: NVIDIA GeForce RTX 5070 FE (Blackwell-class)  
**Primary Use Cases**: ETL, feature engineering, similarity search on text/news, sub-second research queries  
**Date**: October 1, 2025

---

## Executive Summary

This document outlines a GPU-native data architecture for quantitative finance research, optimized for a single NVIDIA RTX 5070 FE (Blackwell-class) workstation. The approach prioritizes CUDA 12.x-compatible databases, vector stores, and datasets that leverage GPU acceleration for market microstructure analytics, alternative data processing, and semantic search.

---

## Approach Checklist

### 1. Define Workload Classes and SLOs
- **ETL**: Tick data ingestion, normalization, and feature engineering
- **Feature Engineering**: LOB imbalance, OFI, volatility signatures, regime features
- **Similarity Search**: Semantic search on news/filings/alt-data embeddings
- **Research Queries**: Sub-second ad-hoc SQL on multi-billion row datasets
- **Targets**: Size (TB-scale raw → GB-scale features) and latency (ms query, minute-scale ETL) drive component selection

### 2. Score Relevance
- **Asset Coverage**: L1/L2/L3 market depth across equities, futures, FX
- **Time Precision**: Microsecond-to-nanosecond timestamp fidelity
- **Schema Stability**: Consistent schemas for backtesting reproducibility
- **Licensing/Access**: On-premises GPU processing vs. cloud API latency
- **Total Cost**: Institutional subscriptions, compute amortization, storage economics

### 3. Validate Source/Vendor
- **Documentation Depth**: API specs, schema versioning, data dictionaries
- **Version Cadence**: Update frequency, backfill policies, corporate action handling
- **Community Footprint**: Academic citations, GitHub activity, user forums
- **Reliability Guarantees**: SLA uptime, error codes, fill-rate metrics

### 4. CUDA/GPU Fit Check (RTX 5070 FE)
- **Blackwell-class Support**: Confirm compute capability ≥9.0
- **CUDA 12.x Requirements**: Driver ≥525.x, toolkit 12.0+, cuDNN 8.9+
- **Compute Capability Coverage**: Validate each GPU database/library against NVIDIA compatibility matrix
- **Driver Pinning**: Lock to tested driver/runtime versions in production

### 5. Performance Plan
- **GPU Memory Footprint**: RTX 5070 FE VRAM budget (16 GB assumed); partition across SQL buffer pool, vector indexes, and pinned I/O
- **I/O Path**: Parquet/PCAP → cuDF (GPU decode) → GPU SQL/vector DB
- **GPUDirect Options**: GPUDirect Storage for NVMe → GPU bypass (PG-Strom compatible)
- **ANN/SQL Index Choices**: CAGRA vs. IVF_PQ for recall/speed trade-offs; GPU hash joins in SQL
- **Microbenchmarks**: Measure ingest MB/s, query latency (p50/p99), ANN recall@k, and GPU utilization

### 6. Operational Readiness
- **Container Images**: NVIDIA NGC base images with CUDA 12.x runtime
- **Driver/Runtime Pinning**: Lock CUDA driver, cuDNN, and NCCL versions via Dockerfile
- **Fallback CPU Modes**: Graceful degradation when GPU OOM or driver issues arise
- **Observability**: `nvidia-smi` metrics, CUDA profiler (nsight-systems), DB query plans
- **Backup/Export Mechanics**: Parquet snapshots, vector index dumps, incremental backfills

### 7. Pilot/Benchmark Loop
- **Ingest Tests**: CSV/Parquet decode throughput (GB/s), multi-file Dask-cuDF scaling
- **Query Tests**: SQL scan latency, join cardinality, aggregation speedup vs. CPU
- **ANN Latency Tests**: Vector search QPS, recall@10/100, index build time
- **Tuning**: Batch sizes (cuDF chunk_size), pinned memory pools, CAGRA graph degree
- **Repeatable Harness**: Jupyter notebooks + pytest benchmarks under version control

---

## Recommended Data Sources

### GPU-Native Analytical Databases

#### 1. HeavyDB (Heavy.AI)
**Type**: GPU-native SQL Database  
**Relevance**: Columnar, millisecond scans on multi-billion row tick/LOB tables; strong for ad-hoc research queries without pre-aggregation  

**Key Features**:
- CUDA-accelerated SQL execution (scans, joins, aggregations)
- Hybrid CPU/GPU or GPU-only execution modes
- Ideal for time-series analytics on market microstructure data
- Sub-second query response on datasets exceeding 10B rows

**Compatibility (RTX 5070 FE / Blackwell)**:
- **CUDA Requirement**: 12.x driver/toolkit
- **Compute Capability**: ≥6.0 (Blackwell-class fully supported)
- **GPU Memory Controls**: `--gpu-buffer-mem` flag to partition VRAM
- **Platform**: Linux (validated on Ubuntu 22.04 / RHEL 8+)
- **Validation**: Run `nvidia-smi` to confirm driver ≥525.x; test with sample tick data load

**Use Cases**:
- OHLCV aggregations from tick data
- LOB snapshot queries (best bid/ask at timestamp T)
- Cross-venue liquidity analytics
- Volatility surface reconstruction

**Operational Notes**:
- Pin CUDA driver and HeavyDB version in container
- Monitor GPU memory via `heavyai_server --status`
- Export to Parquet for backup/sharing

---

#### 2. PG-Strom (PostgreSQL GPU Extension)
**Type**: PostgreSQL Extension for GPU Acceleration  
**Relevance**: SQL + time-series on Postgres with CUDA pushdown; low vendor lock-in; supports GPUDirect Storage  

**Key Features**:
- Offloads scans, joins, aggregations to CUDA
- Preserves PostgreSQL ACID semantics and ecosystem (pgAdmin, TimescaleDB compatibility)
- GPUDirect Storage for NVMe → GPU zero-copy reads
- Transparent fallback to CPU for unsupported operations

**Compatibility (RTX 5070 FE / Blackwell)**:
- **CUDA Requirement**: 12.x toolkit
- **Compute Capability**: ≥6.0 (Blackwell-class supported)
- **Platform**: Linux; tested on Ubuntu 22.04 / RHEL 8
- **PostgreSQL Version**: 13+ recommended
- **Validation**: Check `pg_strom.enabled` in `EXPLAIN ANALYZE` output; confirm GPU execution

**Use Cases**:
- Time-series tick data queries with PostgreSQL tools
- JOIN-heavy feature engineering (trade ⋈ quote ⋈ order book)
- Regulatory reporting queries with GPU acceleration
- Hybrid CPU/GPU workloads (metadata on CPU, analytics on GPU)

**Operational Notes**:
- Tune `shared_buffers` and `pg_strom.gpu_memory_size`
- Use EXPLAIN to verify GPU pushdown
- Combine with TimescaleDB for hypertable partitioning

---

### GPU-Native Vector Databases

#### 3. Milvus (GPU Indexes: CAGRA / IVF_FLAT / IVF_PQ / BRUTE_FORCE)
**Type**: Vector Database with GPU-Accelerated ANN  
**Relevance**: High-QPS, high-recall ANN search over embeddings of news, filings, earnings calls, and alternative data  

**Key Features**:
- GPU indexes built with RAPIDS RAFT (CAGRA = graph-based ANN)
- IVF_FLAT / IVF_PQ for memory-speed trade-offs
- BRUTE_FORCE for exact search baseline
- Horizontal scaling with Milvus clustering (future-proof for multi-GPU)

**Compatibility (RTX 5070 FE / Blackwell)**:
- **CUDA Requirement**: 12.x (via official GPU Docker images)
- **Compute Capability**: Validate against Milvus release notes (typically ≥7.0 for GPU builds; Blackwell-class supported)
- **Fallback**: CPU HNSW index if GPU build unavailable for new compute capability
- **Validation**: `nvidia-smi` inside container; run CAGRA build test on sample embeddings

**Use Cases**:
- Semantic search: "Find filings similar to Enron 10-K (2001)"
- Regime matching: Retrieve historical periods with similar volatility/correlation structure
- News clustering: Group related events for sentiment aggregation
- Pattern retrieval: Nearest-neighbor LOB shapes for microstructure signals

**Operational Notes**:
- Use NVIDIA Container Toolkit (`docker --gpus all`)
- Tune CAGRA `intermediate_graph_degree` and `graph_degree` for recall/speed
- Monitor GPU memory via Milvus metrics endpoint
- Export collections to Parquet for version control

---

#### 4. Qdrant (GPU Build, v1.13+)
**Type**: Vector Database with GPU Acceleration  
**Relevance**: Simple REST/gRPC API; GPU-enabled Docker images; good for single-workstation RTX deployments  

**Key Features**:
- GPU-accelerated vector search (exact and HNSW)
- Payload filtering (e.g., "news published after 2023-01-01 AND sector=tech")
- RESTful API for easy integration with Python/R workflows
- Lightweight compared to Milvus (easier ops for small teams)

**Compatibility (RTX 5070 FE / Blackwell)**:
- **CUDA Requirement**: 12.x drivers (via GPU Docker images)
- **Compute Capability**: Blackwell-class supported in v1.13+ GPU builds
- **Fallback**: Default CPU build if GPU image unavailable
- **Validation**: Run `docker run --gpus all qdrant/qdrant:v1.13-gpu` and check logs for GPU initialization

**Use Cases**:
- Embedding search for news/filings (smaller scale than Milvus)
- Local RTX workstation prototyping
- Filtered vector search (e.g., "top 10 similar stocks in energy sector")

**Operational Notes**:
- Reserve GPU VRAM for embeddings (coordinate with HeavyDB/PG-Strom)
- Use Qdrant snapshots for backup
- Monitor via Prometheus metrics

---

### High-Fidelity Market Microstructure Datasets

#### 5. NYSE TAQ (WRDS) — U.S. Consolidated Trades & Quotes
**Type**: Dataset (Tick-Level Market Data)  
**Relevance**: Market-standard microstructure dataset; microsecond resolution; huge volume stresses ETL and query layers  

**Key Features**:
- Consolidated U.S. equity trades and NBBO quotes
- Microsecond timestamps (TAQ 3 Millisecond, TAQ Pillar for sub-microsecond)
- Full depth of book for selected securities
- Decades of history (1993+)

**Compatibility (RTX 5070 FE / Blackwell)**:
- **No Native GPU**: Leverage RAPIDS cuDF for GPU-accelerated CSV/Parquet ingestion
- **ETL Pipeline**: WRDS API → Parquet → cuDF (CUDA 12.x) → HeavyDB/PG-Strom
- **Validation**: Benchmark cuDF `read_csv` vs. pandas on 1-day TAQ file; expect 5-10× speedup

**Use Cases**:
- VWAP/TWAP calculations
- Spread dynamics analysis
- Market impact models
- Trade classification (Lee-Ready, EMO)

**Operational Notes**:
- Requires WRDS subscription (institutional access)
- Download monthly Parquet files to NVMe SSD
- Use Dask-cuDF for multi-file parallel ingest
- Store aggregated features in GPU SQL database

---

#### 6. LOBSTER — NASDAQ Limit Order Book (Event-Level)
**Type**: Dataset (Order Book Reconstruction)  
**Relevance**: High-fidelity LOB events (L1–L10+); perfect for microstructure alpha and simulator training; extremely I/O heavy  

**Key Features**:
- Event-level order book messages (add, cancel, execute, replace)
- Reconstructed L1-L10 depth snapshots
- Nanosecond timestamps
- Institutional-quality data for academic research

**Compatibility (RTX 5070 FE / Blackwell)**:
- **No Native GPU**: Use cuDF CSV/Parquet readers for decode
- **ETL Pipeline**: LOBSTER CSV → cuDF → feature engineering (GPU kernels) → HeavyDB
- **Validation**: Test cuDF `read_csv` on multi-GB LOBSTER file; monitor GPU memory

**Use Cases**:
- Order flow imbalance (OFI) features
- Queue position modeling
- Limit order arrival/cancellation hazards
- Market maker inventory simulation

**Operational Notes**:
- Download per-symbol CSVs from LOBSTER portal
- Use Dask-cuDF for parallel processing (chunk by date/symbol)
- Compute rolling features (imbalance, spread, depth) with cuDF rolling windows
- Persist features in GPU SQL for fast backtesting

---

#### 7. LSEG Refinitiv Tick History (PCAP / Query Service)
**Type**: Dataset (Global Multi-Asset Tick Data)  
**Relevance**: Petabyte-scale tick/quote/market-depth history; consistent schemas; institutional reliability; global coverage  

**Key Features**:
- Multi-asset: equities, futures, FX, fixed income, commodities
- Nanosecond precision (via PCAP feeds)
- Normalized schemas across venues
- REST API and PCAP file delivery

**Compatibility (RTX 5070 FE / Blackwell)**:
- **No Native GPU**: Use RAPIDS/cuDF (CUDA 12.x) for ingest/transform
- **ETL Pipeline**: Refinitiv API → Parquet → cuDF → normalization → HeavyDB/Milvus
- **Vector Search**: Embed text fields (headlines, analyst notes) with GPU transformers → Milvus/Qdrant
- **Validation**: Benchmark cuDF on sample tick history Parquet; test FAISS-GPU for local ANN

**Use Cases**:
- Cross-venue liquidity studies
- Global event studies (earnings, macro announcements)
- Multi-asset correlation regimes
- Alternative data enrichment (tick data + news embeddings)

**Operational Notes**:
- Subscription-based access (on-prem or cloud)
- Use GPUDirect Storage where available (PG-Strom + NVMe)
- Embed news/filings with GPU-accelerated transformers (Hugging Face + CUDA)
- Store embeddings in Milvus (CAGRA index) or Qdrant (GPU build)

---

## Ordering Rationale

Data sources are ordered by **GPU-native analytical impact**:

1. **HeavyDB, PG-Strom** (GPU SQL): Maximum leverage of RTX 5070 FE for sub-second research queries on billion-row tables
2. **Milvus, Qdrant** (GPU vector search): Native GPU ANN for semantic search and pattern retrieval
3. **TAQ, LOBSTER, Refinitiv** (Heavy datasets): ETL/feature engineering benefits from CUDA via cuDF/Spark RAPIDS; output feeds GPU SQL/vector DBs

**Validation**: RTX 5070 FE (Blackwell-class) supports CUDA 12.x and compute capability ≥9.0. All listed GPU databases/libraries either explicitly target NVIDIA CUDA or provide current GPU builds. Datasets lack native GPU but align with RAPIDS/cuDF ingestion on CUDA 12.x. If Milvus GPU builds lag newer compute capabilities, fall back to CPU HNSW or pin to a tested container version.

---

## Performance Planning

### GPU Memory Budget (RTX 5070 FE, 16 GB VRAM assumed)

| Component | Allocation | Notes |
|-----------|-----------|-------|
| HeavyDB Buffer Pool | 8 GB | `--gpu-buffer-mem 8192` |
| Milvus CAGRA Index | 4 GB | Embeddings (1M vectors × 768 dims × fp16) |
| cuDF Pinned Memory | 2 GB | I/O staging for ingest |
| Reserved | 2 GB | Kernel overhead, temporary buffers |

**Tuning**:
- Monitor `nvidia-smi` during concurrent SQL + vector search
- Adjust HeavyDB buffer size based on working set
- Use Milvus IVF_PQ (quantized) if CAGRA exceeds VRAM

### I/O Path

```
NVMe SSD (Parquet/PCAP)
    ↓
cuDF GPU Reader (CUDA 12.x)
    ↓
GPU DataFrame (cuDF/Dask-cuDF)
    ↓
Feature Engineering (GPU kernels, RAPIDS)
    ↓
├─→ HeavyDB (SQL analytics)
└─→ Milvus/Qdrant (vector search)
```

**GPUDirect Storage**: Enable for PG-Strom to bypass CPU on NVMe → GPU transfers (requires compatible NVMe controller + driver).

### ANN/SQL Index Choices

| Use Case | Index Type | Trade-Off |
|----------|-----------|-----------|
| Exact search baseline | BRUTE_FORCE / IVF_FLAT | Recall=1.0, slow for >1M vectors |
| High-recall ANN | CAGRA (graph_degree=64) | Recall ≈0.95, fast GPU build |
| Memory-constrained ANN | IVF_PQ (m=16, nbits=8) | 16× compression, recall ≈0.85 |
| SQL scans | GPU Hash Join + Columnar Scan | Leverage HeavyDB/PG-Strom query planner |

**Microbenchmarks**:
- Measure Milvus CAGRA recall@10 vs. IVF_PQ on 1M news embeddings
- Profile HeavyDB query latency (p50/p95/p99) on 10B-row tick table
- Track cuDF ingest throughput (MB/s) on LOBSTER CSVs

---

## Operational Readiness

### Container Image Stack

```dockerfile
FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Pin CUDA toolkit, cuDNN, NCCL versions
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-2 \
    libcudnn8=8.9.5.29-1+cuda12.2 \
    libnccl2=2.18.5-1+cuda12.2

# Install RAPIDS cuDF (CUDA 12.x wheel)
RUN pip install --no-cache-dir \
    cudf-cu12==24.10.* \
    dask-cudf==24.10.*

# Install HeavyDB client / PG-Strom drivers
# (vendor-specific instructions)

# Install Milvus/Qdrant clients
RUN pip install pymilvus qdrant-client

# Lock versions in requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt
```

**Validation**:
- `nvidia-smi` confirms driver ≥525.x
- `python -c "import cudf; print(cudf.__version__)"` succeeds
- `docker run --gpus all` passes GPU passthrough test

### Fallback CPU Modes

- **HeavyDB**: Runs in CPU-only mode if `--num-gpus 0` (slower, but functional)
- **PG-Strom**: Graceful fallback to CPU executor for unsupported operations
- **Milvus/Qdrant**: Switch to CPU HNSW if GPU build unavailable
- **cuDF**: Use `pandas` fallback in ETL scripts (10× slower, but avoids crashes)

### Observability

| Metric | Tool | Threshold |
|--------|------|-----------|
| GPU Utilization | `nvidia-smi dmon` | >80% during queries |
| GPU Memory | `nvidia-smi --query-gpu=memory.used` | <14 GB (leave 2 GB headroom) |
| HeavyDB Query Time | `heavysql` EXPLAIN | p95 <500ms for research queries |
| Milvus Search QPS | Prometheus `/metrics` | >100 QPS at recall≥0.9 |
| cuDF Ingest Throughput | Custom logger | >1 GB/s on Parquet decode |

### Backup/Export Mechanics

- **HeavyDB**: `COPY (SELECT * FROM ticks) TO '/backup/ticks.parquet' WITH (format='parquet')`
- **Milvus**: Collection export to JSON/Parquet via PyMilvus SDK
- **Qdrant**: Snapshot API (`POST /collections/{name}/snapshots`)
- **cuDF**: Write features to Parquet with `df.to_parquet(..., compression='snappy')`

---

## Pilot/Benchmark Loop

### 1. Ingest Tests
**Objective**: Validate cuDF GPU ingest throughput on TAQ/LOBSTER CSVs

```python
import cudf
import time

start = time.time()
df = cudf.read_csv("TAQ_2024_01_03.csv", parse_dates=["timestamp"])
elapsed = time.time() - start
print(f"Read {len(df):,} rows in {elapsed:.2f}s ({len(df)/elapsed:,.0f} rows/s)")
```

**Success Criteria**:
- >1M rows/s on RTX 5070 FE
- GPU memory usage <4 GB for single-day TAQ file

### 2. Query Tests
**Objective**: Measure HeavyDB/PG-Strom SQL latency on billion-row tick table

```sql
-- HeavyDB: Sub-second VWAP over 1B trades
SELECT symbol, 
       DATE_TRUNC(minute, timestamp) AS minute,
       SUM(price * size) / SUM(size) AS vwap
FROM trades
WHERE DATE(timestamp) = '2024-01-03'
GROUP BY 1, 2
ORDER BY 1, 2;
```

**Success Criteria**:
- p95 latency <500ms on RTX 5070 FE
- GPU utilization >70% during scan

### 3. ANN Latency Tests
**Objective**: Milvus CAGRA search QPS and recall on 1M news embeddings

```python
from pymilvus import Collection

collection = Collection("news_embeddings")
collection.load()

# Search 1000 queries, k=10
results = collection.search(
    data=query_vectors,  # shape: (1000, 768)
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"search_width": 64}},
    limit=10
)
```

**Success Criteria**:
- QPS >200 on RTX 5070 FE
- Recall@10 ≥0.95 (vs. BRUTE_FORCE ground truth)

### 4. Tuning
- **cuDF**: Increase `chunk_size` in `read_csv` for larger files
- **HeavyDB**: Tune `--gpu-buffer-mem` to maximize cache hit rate
- **Milvus CAGRA**: Adjust `graph_degree` (32/64/128) and `search_width` for recall/speed

### 5. Repeatable Harness
```bash
# Run benchmark suite
pytest tests/benchmarks/ --benchmark-only --gpu-device=0

# Generate report
jupyter nbconvert --execute benchmark_report.ipynb --to html
```

**Version Control**:
- Store benchmark notebooks, pytest scripts, and Docker Compose configs in Git
- Track CUDA driver, DB versions, and dataset snapshots in `versions.txt`

---

## Next Steps

1. **Environment Setup**:
   - Install CUDA 12.x drivers (≥525.x) on RTX 5070 FE workstation
   - Pull NVIDIA NGC base image and install RAPIDS cuDF
   - Deploy HeavyDB/PG-Strom via Docker Compose

2. **Data Acquisition**:
   - Obtain WRDS credentials for NYSE TAQ access
   - Purchase LOBSTER sample dataset (1 month, 10 symbols)
   - Evaluate Refinitiv Tick History trial subscription

3. **Pilot Workload**:
   - Ingest 1 day of TAQ data via cuDF → HeavyDB
   - Run VWAP/TWAP queries and measure p95 latency
   - Embed 100k news articles (Hugging Face transformers + GPU) → Milvus CAGRA
   - Execute semantic search benchmark (QPS, recall)

4. **Production Hardening**:
   - Lock CUDA driver and DB versions in Dockerfile
   - Set up `nvidia-smi` monitoring + Prometheus alerts
   - Implement Parquet backup cron jobs
   - Document runbooks for GPU OOM recovery

---

## References

- **HeavyDB**: [https://docs.heavy.ai/](https://docs.heavy.ai/)
- **PG-Strom**: [https://heterodb.github.io/pg-strom/](https://heterodb.github.io/pg-strom/)
- **Milvus GPU Indexes**: [https://milvus.io/docs/gpu_index.md](https://milvus.io/docs/gpu_index.md)
- **Qdrant GPU**: [https://qdrant.tech/documentation/guides/gpu/](https://qdrant.tech/documentation/guides/gpu/)
- **RAPIDS cuDF**: [https://docs.rapids.ai/api/cudf/stable/](https://docs.rapids.ai/api/cudf/stable/)
- **NVIDIA CUDA Compatibility**: [https://docs.nvidia.com/deploy/cuda-compatibility/](https://docs.nvidia.com/deploy/cuda-compatibility/)

---

**Document Version**: 1.0  
**Last Updated**: October 1, 2025  
**Maintainer**: Quantitative Research Infrastructure Team
