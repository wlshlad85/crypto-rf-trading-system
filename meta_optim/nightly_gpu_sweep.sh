#!/bin/bash
#
# Nightly GPU Parameter Sweep Deployment Script
# =============================================
#
# Runs automated GPU-accelerated hyperparameter sweeps for the crypto RF trading system.
# Designed for cron execution with comprehensive logging and error handling.
#
# Usage:
#   ./nightly_gpu_sweep.sh [--config CONFIG_FILE] [--gpus GPU_IDS] [--seed SEED]
#
# Cron example (runs daily at 2 AM):
#   0 2 * * * /workspace/meta_optim/nightly_gpu_sweep.sh >> /var/log/crypto_sweep.log 2>&1

set -euo pipefail

# ===== Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SWEEP_CONFIG="${SWEEP_CONFIG:-$SCRIPT_DIR/sweep_config.json}"
LOG_BASE_DIR="${LOG_BASE_DIR:-$PROJECT_ROOT/sweep_logs}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv}"
MASTER_SEED="${MASTER_SEED:-123456}"
GPU_IDS="${GPU_IDS:-}"  # Empty = auto-detect
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"  # Optional Slack notifications
EMAIL_RECIPIENT="${EMAIL_RECIPIENT:-}"  # Optional email notifications

# ===== Functions =====

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error_exit() {
    log "ERROR: $1"
    send_notification "ERROR" "$1"
    exit 1
}

send_notification() {
    local status="$1"
    local message="$2"
    
    # Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Crypto GPU Sweep ${status}: ${message}\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
    
    # Email notification
    if [[ -n "$EMAIL_RECIPIENT" ]] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "Crypto GPU Sweep ${status}" "$EMAIL_RECIPIENT" || true
    fi
}

check_gpu_availability() {
    log "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits || true
        
        # Check if GPUs are busy
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')
        if (( $(echo "$gpu_util > 80" | bc -l) )); then
            log "WARNING: Average GPU utilization is high (${gpu_util}%)"
        fi
    else
        log "nvidia-smi not found, assuming CPU-only execution"
    fi
}

setup_environment() {
    log "Setting up environment..."
    
    # Set CUDA environment variables
    export CUDA_LAUNCH_BLOCKING=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export TF_DETERMINISTIC_OPS=1
    export PYTHONHASHSEED=$MASTER_SEED
    
    # Limit CPU threads for determinism
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    
    # Activate virtual environment if exists
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
        log "Activating virtual environment: $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    else
        log "No virtual environment found, using system Python"
    fi
    
    # Verify Python and required packages
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || \
        error_exit "PyTorch not properly installed"
}

create_sweep_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        log "Creating default sweep configuration..."
        
        cat > "$config_file" <<EOF
{
    "sweep_type": "hyperband",
    "hyperband": {
        "iterations": 10,
        "max_iter": 81,
        "eta": 3,
        "max_parallel": 4
    },
    "models": {
        "enable_catboost": true,
        "enable_lightgbm": true,
        "enable_xgboost": false,
        "enable_rf": true
    },
    "parameter_ranges": {
        "entry_model": {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [6, 8, 10, 12, 15, 20],
            "min_samples_split": [5, 10, 15, 20, 30],
            "min_samples_leaf": [2, 5, 8, 10, 15],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
        },
        "position_model": {
            "n_estimators": [100, 150, 200, 250, 300],
            "max_depth": [4, 6, 8, 10, 12],
            "min_samples_split": [5, 10, 15, 20],
            "min_samples_leaf": [2, 5, 8, 10]
        },
        "exit_model": {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [6, 8, 10, 12, 15],
            "min_samples_split": [5, 10, 15, 25],
            "min_samples_leaf": [2, 5, 8, 12]
        },
        "profit_model": {
            "n_estimators": [100, 150, 200, 250, 300],
            "max_depth": [8, 10, 12, 15, 18],
            "min_samples_split": [8, 12, 16, 20],
            "min_samples_leaf": [3, 6, 9, 12]
        },
        "trading_params": {
            "momentum_threshold": [1.2, 1.5, 1.78, 2.0, 2.5],
            "position_range_min": [0.3, 0.4, 0.464, 0.5],
            "position_range_max": [0.7, 0.8, 0.85, 0.9],
            "confidence_threshold": [0.5, 0.6, 0.65, 0.7],
            "exit_threshold": [0.4, 0.5, 0.55, 0.6]
        }
    },
    "gpu_settings": {
        "memory_fraction": 0.8,
        "allow_growth": true,
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
EOF
    fi
}

run_sweep() {
    local config_file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_dir="$LOG_BASE_DIR/nightly_${timestamp}"
    
    mkdir -p "$log_dir"
    
    log "Starting GPU parameter sweep..."
    log "Configuration: $config_file"
    log "Log directory: $log_dir"
    log "Master seed: $MASTER_SEED"
    
    # Build command
    local cmd="python3 $SCRIPT_DIR/gpu_sweep_runner.py"
    cmd="$cmd --config $config_file"
    cmd="$cmd --seed $MASTER_SEED"
    cmd="$cmd --log-dir $log_dir"
    
    if [[ -n "$GPU_IDS" ]]; then
        cmd="$cmd --gpu-ids $GPU_IDS"
    fi
    
    # Check for existing checkpoints
    local latest_checkpoint=$(find "$LOG_BASE_DIR" -name "checkpoint_*.pkl" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -n "$latest_checkpoint" && -f "$latest_checkpoint" ]]; then
        log "Found checkpoint: $latest_checkpoint"
        read -p "Resume from checkpoint? (y/N) " -n 1 -r -t 10 || true
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cmd="$cmd --resume $latest_checkpoint"
        fi
    fi
    
    # Run the sweep
    local start_time=$(date +%s)
    
    if $cmd 2>&1 | tee "$log_dir/sweep_output.log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        
        log "Sweep completed successfully in ${hours}h ${minutes}m"
        send_notification "SUCCESS" "GPU sweep completed in ${hours}h ${minutes}m. Logs: $log_dir"
        
        # Copy best configurations to production location
        if [[ -f "$log_dir/best_configs_"*.json ]]; then
            cp "$log_dir/best_configs_"*.json "$PROJECT_ROOT/models/best_configs_latest.json"
            log "Best configurations copied to production"
        fi
        
        return 0
    else
        log "Sweep failed with exit code $?"
        send_notification "FAILURE" "GPU sweep failed. Check logs: $log_dir/sweep_output.log"
        return 1
    fi
}

cleanup_old_logs() {
    local days_to_keep="${DAYS_TO_KEEP:-7}"
    
    log "Cleaning up logs older than $days_to_keep days..."
    
    find "$LOG_BASE_DIR" -type f -name "*.log" -mtime +$days_to_keep -delete 2>/dev/null || true
    find "$LOG_BASE_DIR" -type f -name "*.json" -mtime +$days_to_keep -delete 2>/dev/null || true
    
    # Keep checkpoints for 30 days
    find "$LOG_BASE_DIR" -type f -name "checkpoint_*.pkl" -mtime +30 -delete 2>/dev/null || true
}

generate_summary_report() {
    local log_dir="$1"
    
    log "Generating summary report..."
    
    python3 - <<EOF
import json
import glob
import os

log_dir = "$log_dir"

# Find results file
results_files = glob.glob(os.path.join(log_dir, "sweep_results_*.json"))
if not results_files:
    print("No results file found")
    exit(1)

with open(results_files[0], 'r') as f:
    results = json.load(f)

if results.get('success'):
    print(f"Best Score: {results['best_configuration']['composite_score']:.4f}")
    print(f"Total Configurations: {results['total_evaluated']}")
    print(f"Viable Configurations: {results['viable_count']}")
    print(f"Duration: {results['duration'] / 3600:.2f} hours")
    
    # Top parameters
    print("\nTop Parameter Impacts:")
    for i, (param, data) in enumerate(list(results.get('parameter_importance', {}).items())[:5]):
        print(f"{i+1}. {param}: best={data['best_value']}, impact={data['value_impact']:.4f}")
else:
    print(f"Sweep failed: {results.get('reason', 'Unknown')}")
EOF
}

# ===== Main Script =====

main() {
    log "===== Crypto GPU Parameter Sweep Starting ====="
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                SWEEP_CONFIG="$2"
                shift 2
                ;;
            --gpus)
                GPU_IDS="$2"
                shift 2
                ;;
            --seed)
                MASTER_SEED="$2"
                shift 2
                ;;
            *)
                error_exit "Unknown argument: $1"
                ;;
        esac
    done
    
    # Pre-flight checks
    check_gpu_availability
    setup_environment
    create_sweep_config "$SWEEP_CONFIG"
    
    # Run the sweep
    if run_sweep "$SWEEP_CONFIG"; then
        # Post-processing
        local latest_log_dir=$(ls -td "$LOG_BASE_DIR"/nightly_* 2>/dev/null | head -1)
        if [[ -n "$latest_log_dir" ]]; then
            generate_summary_report "$latest_log_dir"
        fi
        
        # Cleanup
        cleanup_old_logs
        
        log "===== Crypto GPU Parameter Sweep Completed Successfully ====="
        exit 0
    else
        log "===== Crypto GPU Parameter Sweep Failed ====="
        exit 1
    fi
}

# Run main function
main "$@"