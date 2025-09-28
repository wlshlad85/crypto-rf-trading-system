import argparse, json, os
import numpy as np
import pandas as pd


def set_determinism(seed: int = 42):
    # Stabilize as much as possible across runs
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    try:
        import xgboost as xgb  # noqa: F401
        # XGBoost picks up determinism from params; set later
    except Exception:
        pass


def load_df(path: str, use_gpu: bool):
    if use_gpu:
        try:
            import cudf  # type: ignore
            return cudf.read_parquet(path), "cudf"
        except Exception:
            # Fallback to pandas if GPU stack not available; still allows CPU mode tests
            return pd.read_parquet(path), "pandas"
    return pd.read_parquet(path), "pandas"


def to_numpy(series_or_array):
    if hasattr(series_or_array, "to_pandas"):
        return series_or_array.to_pandas().values
    if hasattr(series_or_array, "values"):
        return series_or_array.values
    return np.asarray(series_or_array)


def to_2d_frame(df_like, columns):
    if hasattr(df_like, "to_pandas"):
        return df_like[columns].to_pandas()
    return df_like[columns]


def xgb_gpu_available() -> bool:
    """Return True if the installed XGBoost build supports CUDA/GPU."""
    try:
        import xgboost as xgb  # type: ignore
        info_fn = getattr(xgb, "build_info", None)
        if callable(info_fn):
            bi = info_fn()
            # bi may be dict or string depending on version
            if isinstance(bi, dict):
                val = str(bi.get("USE_CUDA", "")).upper()
                return val == "ON" or val == "TRUE"
            s = str(bi).upper()
            return "USE_CUDA" in s and ("ON" in s or "TRUE" in s)
    except Exception:
        pass
    return False


def model_fit_predict(df, use_gpu: bool, seed: int = 42):
    import xgboost as xgb
    target = "y"
    features = [c for c in df.columns if c != target]

    X_df = to_2d_frame(df, features)
    y_np = to_numpy(df[target])

    dtrain = xgb.DMatrix(X_df, label=y_np)

    gpu_ok = bool(use_gpu and xgb_gpu_available())
    tree_method = "gpu_hist" if gpu_ok else "hist"
    predictor = "gpu_predictor" if gpu_ok else "auto"

    params = dict(
        max_depth=6,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        nthread=0,
        tree_method=tree_method,
        predictor=predictor,
        eval_metric="auc",
        seed=seed,
    )

    bst = xgb.train(params, dtrain, num_boost_round=80, verbose_eval=False)
    yhat = bst.predict(dtrain)
    return y_np, yhat


def compute_metrics(y: np.ndarray, yhat: np.ndarray):
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, yhat))
    thr = float(np.median(yhat))
    pos = np.where(yhat > thr, 1.0, -1.0)
    ret = pos * (y * 2.0 - 1.0)
    pnl_sum = float(ret.sum())
    sharpe_1d = float(np.mean(ret) / (np.std(ret) + 1e-12) * np.sqrt(252.0))
    return dict(auc=auc, pnl_sum=pnl_sum, sharpe_1d=sharpe_1d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_determinism(args.seed)

    use_gpu = args.mode == "gpu"
    df, reader = load_df(args.input, use_gpu)
    y, yhat = model_fit_predict(df, use_gpu, seed=args.seed)
    out = dict(
        metrics=compute_metrics(y, yhat),
        predictions=[float(v) for v in yhat],
        used_gpu=bool(use_gpu and xgb_gpu_available()),
        reader=reader,
        requested_mode=args.mode,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
