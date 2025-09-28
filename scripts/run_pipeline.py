#!/usr/bin/env python3
"""
Main pipeline script with CPU/GPU mode switching.
Supports pandas/cuDF and sklearn/cuML/XGBoost.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_df(path, use_gpu):
    """Load dataframe in CPU or GPU mode."""
    if use_gpu:
        try:
            import cudf
            df = cudf.read_parquet(path)
            print(f"Loaded data to GPU: {df.shape}")
            return df
        except ImportError:
            print("WARNING: cuDF not available, falling back to pandas")
            return pd.read_parquet(path)
    return pd.read_parquet(path)

def preprocess_data(df, use_gpu):
    """Preprocess data: handle NaNs, scale features."""
    # Get feature columns (all except target)
    target_col = 'y'
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Fill NaNs with median (GPU-aware)
    if use_gpu and hasattr(df, 'fillna'):
        # cuDF fillna
        for col in feature_cols:
            if df[col].isnull().any():
                median_val = float(df[col].median())
                df[col] = df[col].fillna(median_val)
    else:
        # Pandas fillna
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    # Clip outliers (99th percentile)
    for col in feature_cols:
        if 'return' in col or 'ratio' in col:
            if use_gpu and hasattr(df, 'to_pandas'):
                # Convert to pandas for percentile calculation
                vals = df[col].to_pandas() if hasattr(df[col], 'to_pandas') else df[col]
                p99 = np.percentile(vals.dropna(), 99)
                p1 = np.percentile(vals.dropna(), 1)
                df[col] = df[col].clip(p1, p99)
            else:
                p99 = df[col].quantile(0.99)
                p1 = df[col].quantile(0.01)
                df[col] = df[col].clip(p1, p99)
    
    return df

def model_fit_predict(df, use_gpu, model_type='xgboost'):
    """Fit model and predict. Supports XGBoost and cuML."""
    target_col = 'y'
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Prepare data
    if use_gpu and hasattr(df, 'to_pandas'):
        # cuDF to pandas for XGBoost (it handles GPU internally)
        X = df[feature_cols].to_pandas()
        y = df[target_col].to_pandas().values
    else:
        X = df[feature_cols]
        y = df[target_col].values
    
    # Model selection
    if model_type == 'xgboost':
        import xgboost as xgb
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Parameters with GPU support
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': 42,
            'nthread': -1,
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'predictor': 'gpu_predictor' if use_gpu else 'auto',
            'gpu_id': 0 if use_gpu else -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0
        }
        
        # Train model
        print(f"Training XGBoost ({'GPU' if use_gpu else 'CPU'} mode)...")
        bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        
        # Predict
        yhat = bst.predict(dtrain)
        
    elif model_type == 'cuml' and use_gpu:
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            
            print("Training cuML RandomForest (GPU mode)...")
            model = cuRF(
                n_estimators=100,
                max_depth=10,
                n_bins=32,
                random_state=42
            )
            model.fit(X, y)
            yhat = model.predict_proba(X)[:, 1]
            
        except ImportError:
            print("WARNING: cuML not available, falling back to XGBoost")
            return model_fit_predict(df, use_gpu, model_type='xgboost')
    
    else:
        # Fallback to sklearn
        from sklearn.ensemble import RandomForestClassifier
        
        print("Training sklearn RandomForest (CPU mode)...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        yhat = model.predict_proba(X)[:, 1]
    
    return y, yhat

def compute_metrics(y, yhat):
    """Compute evaluation metrics."""
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
    
    # Basic ML metrics
    auc = float(roc_auc_score(y, yhat))
    acc = float(accuracy_score(y, yhat > 0.5))
    logloss = float(log_loss(y, yhat))
    
    # Trading metrics
    # Simple PnL: long if yhat > median(yhat)
    threshold = np.median(yhat)
    positions = np.where(yhat > threshold, 1, -1)
    
    # Convert y to returns: 1 -> +1%, 0 -> -1%
    returns = (y * 2 - 1) * 0.01  # ±1% returns
    pnl = positions * returns
    
    # Cumulative metrics
    cum_pnl = np.cumsum(pnl)
    pnl_sum = float(cum_pnl[-1] * 10000)  # Scale to basis points
    
    # Sharpe ratio (annualized, 252 trading days)
    daily_returns = pnl
    sharpe = float(
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)
    )
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = (cum_pnl - running_max) / (running_max + 1e-9)
    max_dd = float(np.min(drawdown))
    
    # Win rate
    winning_trades = np.sum(pnl > 0)
    total_trades = len(pnl)
    win_rate = float(winning_trades / total_trades)
    
    return {
        'auc': auc,
        'accuracy': acc,
        'logloss': logloss,
        'sharpe_1d': sharpe,
        'pnl_sum': pnl_sum,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'n_trades': total_trades
    }

def main():
    parser = argparse.ArgumentParser(description='Run ML pipeline in CPU or GPU mode')
    parser.add_argument('--mode', choices=['cpu', 'gpu'], required=True,
                        help='Execution mode')
    parser.add_argument('--input', required=True,
                        help='Input data path (parquet)')
    parser.add_argument('--out', required=True,
                        help='Output JSON path for results')
    parser.add_argument('--model', choices=['xgboost', 'cuml', 'sklearn'],
                        default='xgboost',
                        help='Model type to use')
    
    args = parser.parse_args()
    
    use_gpu = args.mode == 'gpu'
    
    print(f"Starting pipeline in {args.mode.upper()} mode...")
    start_time = time.time()
    
    # Load data
    df = load_df(args.input, use_gpu)
    
    # Preprocess
    df = preprocess_data(df, use_gpu)
    
    # Train and predict
    y, yhat = model_fit_predict(df, use_gpu, model_type=args.model)
    
    # Compute metrics
    metrics = compute_metrics(y, yhat)
    
    # Execution time
    execution_time = time.time() - start_time
    
    # Prepare output
    output = {
        'mode': args.mode,
        'model_type': args.model,
        'execution_time_ms': float(execution_time * 1000),
        'metrics': metrics,
        'predictions': [float(v) for v in yhat[:1000]],  # First 1000 for parity check
        'data_shape': list(df.shape),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Pipeline completed in {execution_time:.2f}s")
    print(f"Results saved to: {args.out}")
    print(f"Metrics: AUC={metrics['auc']:.4f}, Sharpe={metrics['sharpe_1d']:.2f}")

if __name__ == '__main__':
    main()