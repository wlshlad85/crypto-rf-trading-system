"""Model comparison harness to evaluate multiple models on a common dataset.

This script compares three candidates:
  - minute_rf: MinuteRandomForestModel (single, multi-horizon)
  - minute_rf_ensemble: EnsembleMinuteRandomForest (ensemble of minute models)
  - basic_rf: CryptoRandomForestModel (single-horizon regression)

It generates a synthetic minute-level OHLCV series, builds light features,
trains each model on the same training split, generates out-of-sample
predictions, converts predictions to signals via a percentile threshold
computed on the train split, simulates a simple long-only strategy, and
computes key performance metrics.

Usage:
  python -m analytics.model_comparison --models minute_rf basic_rf minute_rf_ensemble --minutes 3000 --horizon 5

Outputs a concise summary per model and a JSON blob of metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Local imports
from analytics.minute_performance_analytics import (
    MinutePerformanceAnalytics,
    generate_performance_summary,
)
from models.minute_random_forest_model import (
    MinuteRandomForestModel,
    EnsembleMinuteRandomForest,
)
from models.random_forest_model import CryptoRandomForestModel


SUPPORTED_MODELS = {"minute_rf", "minute_rf_ensemble", "basic_rf"}


@dataclass
class ComparisonResult:
    model_name: str
    summary: Dict[str, float]
    full_metrics: Dict[str, Any]


def generate_synthetic_minute_data(
    minutes: int = 3000, seed: int = 42
) -> pd.DataFrame:
    """Create a synthetic minute-level OHLCV series with basic realism."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=minutes, freq="1T")

    price = 50000.0 + rng.normal(0.0, 15.0, minutes).cumsum()
    noise = rng.normal(0.0, 5.0, minutes)

    df = pd.DataFrame(
        {
            "open": price + noise,
            "high": price + np.abs(rng.normal(0.0, 10.0, minutes)) + 5.0,
            "low": price - np.abs(rng.normal(0.0, 10.0, minutes)) - 5.0,
            "close": price,
            "volume": rng.exponential(1_000.0, minutes),
        },
        index=idx,
    )

    # Enforce OHLC invariants
    df["high"] = np.maximum.reduce([df["open"], df["close"], df["high"]])
    df["low"] = np.minimum.reduce([df["open"], df["close"], df["low"]])
    return df


def build_light_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight, fast feature set shared across models."""
    features = pd.DataFrame(index=df.index)

    # Returns and volatility
    ret1 = df["close"].pct_change()
    features["ret_1m"] = ret1
    features["ret_5m"] = df["close"].pct_change(5)
    features["ret_15m"] = df["close"].pct_change(15)

    for w in (5, 15, 30, 60):
        features[f"close_ma_{w}"] = df["close"].rolling(w).mean()
        features[f"close_std_{w}"] = df["close"].rolling(w).std()
        features[f"vol_ma_{w}"] = df["volume"].rolling(w).mean()
        features[f"vol_std_{w}"] = df["volume"].rolling(w).std()

    # Price positioning
    hl_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    features["price_pos"] = (df["close"] - df["low"]) / hl_range

    # Intraday seasonality
    features["hour"] = df.index.hour
    features["minute"] = df.index.minute
    features["sin_time"] = np.sin(2 * np.pi * (features["hour"] * 60 + features["minute"]) / (24 * 60))
    features["cos_time"] = np.cos(2 * np.pi * (features["hour"] * 60 + features["minute"]) / (24 * 60))

    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    return features


def compute_targets(close: pd.Series, horizons: List[int]) -> pd.DataFrame:
    """Create forward return targets for requested horizons (in minutes)."""
    targets = pd.DataFrame(index=close.index)
    for h in horizons:
        targets[f"target_{h}m"] = close.pct_change(h).shift(-h)
    return targets


def to_portfolio_history(
    returns: pd.Series,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """Convert per-minute returns into a portfolio history time series."""
    cumulative = (1.0 + returns.fillna(0.0)).cumprod()
    total_value = initial_capital * cumulative
    portfolio = pd.DataFrame(
        {
            "total_value": total_value,
            "cash": 0.0,  # simplified
            "position_value": total_value,
            "num_positions": 1,
        },
        index=returns.index,
    )
    return portfolio


def simulate_long_only_strategy(
    close: pd.Series,
    predictions: pd.Series,
    train_predictions: pd.Series,
    threshold_quantile: float = 0.7,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate signals from predictions and simulate a simple long-only strategy.

    - Signal rule: buy (1) when pred > q-quantile of train preds; else 0
    - Position applied with one-period lag to avoid lookahead
    - Returns are 1-minute close-to-close returns
    """
    thresh = float(np.nanquantile(train_predictions, threshold_quantile)) if len(train_predictions) else 0.0
    signals = (predictions > thresh).astype(float)

    # One-minute returns and apply previous signal (no lookahead)
    rets_1m = close.pct_change()
    strat_rets = signals.shift(1).fillna(0.0) * rets_1m

    portfolio = to_portfolio_history(strat_rets)
    return portfolio, {"threshold": thresh}


def evaluate_portfolio(portfolio: pd.DataFrame) -> Dict[str, Any]:
    analyzer = MinutePerformanceAnalytics()
    results = analyzer.analyze_portfolio_performance(portfolio)
    summary = generate_performance_summary(results)
    return {"summary": summary, "full": results}


def _train_minute_rf(
    features: pd.DataFrame,
    targets_multi: pd.DataFrame,
    symbol: str,
    horizon: int,
) -> Tuple[MinuteRandomForestModel, pd.Series]:
    model = MinuteRandomForestModel()
    model.train_multi_horizon_models(features, targets_multi, [symbol])
    # Prepare train predictions for thresholding
    preds = model.predict_multi_horizon(features, [symbol])
    pred_col = f"{symbol}_{horizon}min_pred"
    train_pred = preds[pred_col] if pred_col in preds.columns else preds.iloc[:, 0]
    return model, train_pred


def _train_minute_rf_ensemble(
    features: pd.DataFrame,
    targets_multi: pd.DataFrame,
    symbol: str,
    horizon: int,
) -> Tuple[EnsembleMinuteRandomForest, pd.Series]:
    ensemble = EnsembleMinuteRandomForest(n_models=3)
    ensemble.train(features, targets_multi, [symbol])
    preds = ensemble.predict_multi_horizon(features, [symbol])
    pred_col = f"{symbol}_{horizon}min_pred"
    train_pred = preds[pred_col] if pred_col in preds.columns else preds.iloc[:, 0]
    return ensemble, train_pred


def _train_basic_rf(
    features: pd.DataFrame, target: pd.Series
) -> Tuple[CryptoRandomForestModel, pd.Series]:
    from utils.config import ModelConfig

    cfg = ModelConfig()
    basic = CryptoRandomForestModel(cfg)
    # Train on full training features; model handles internal split for validation but fits scaler
    basic.train(features, target)
    # Use in-sample predictions for thresholding
    train_pred = pd.Series(basic.predict(features), index=features.index)
    return basic, train_pred


def compare_models(
    models: List[str], minutes: int = 3000, horizon: int = 5
) -> Dict[str, ComparisonResult]:
    if not set(models).issubset(SUPPORTED_MODELS):
        raise ValueError(f"Unsupported models requested. Supported: {sorted(SUPPORTED_MODELS)}")

    symbol = "BTC-USD"
    data = generate_synthetic_minute_data(minutes=minutes)
    features = build_light_features(data)

    # Targets for minute models (multi-horizon) and basic (single horizon)
    targets_multi = pd.DataFrame(index=features.index)
    targets_multi[f"{symbol}_{horizon}min_target"] = data["close"].pct_change(horizon).shift(-horizon)
    basic_target = targets_multi[f"{symbol}_{horizon}min_target"].rename("target")

    # Train/test split by time
    split_idx = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    close_train, close_test = data["close"].iloc[:split_idx], data["close"].iloc[split_idx:]
    y_multi_train = targets_multi.iloc[:split_idx]
    y_multi_test_index = targets_multi.index[split_idx:]
    y_basic_train = basic_target.iloc[:split_idx]

    results: Dict[str, ComparisonResult] = {}

    for model_name in models:
        if model_name == "minute_rf":
            # Train
            m, train_pred = _train_minute_rf(X_train, y_multi_train, symbol, horizon)
            # Predict OOS
            preds = m.predict_multi_horizon(X_test, [symbol])
            pred_col = f"{symbol}_{horizon}min_pred"
            oos_pred = preds[pred_col] if pred_col in preds.columns else preds.iloc[:, 0]

        elif model_name == "minute_rf_ensemble":
            m, train_pred = _train_minute_rf_ensemble(X_train, y_multi_train, symbol, horizon)
            preds = m.predict_multi_horizon(X_test, [symbol])
            pred_col = f"{symbol}_{horizon}min_pred"
            oos_pred = preds[pred_col] if pred_col in preds.columns else preds.iloc[:, 0]

        elif model_name == "basic_rf":
            m, train_pred = _train_basic_rf(X_train, y_basic_train)
            oos_pred = pd.Series(m.predict(X_test), index=X_test.index)

        else:
            raise AssertionError("Unreachable: unsupported model")

        # Align for simulation
        portfolio, sim_meta = simulate_long_only_strategy(close_test, oos_pred, train_pred)
        eval_res = evaluate_portfolio(portfolio)
        results[model_name] = ComparisonResult(
            model_name=model_name,
            summary=eval_res["summary"],
            full_metrics=eval_res["full"],
        )

    return results


def _format_pct(x: float) -> str:
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return "n/a"


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models on a synthetic minute dataset")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["minute_rf", "basic_rf", "minute_rf_ensemble"],
        help=f"Models to compare. Supported: {sorted(SUPPORTED_MODELS)}",
    )
    parser.add_argument("--minutes", type=int, default=3000, help="Number of minutes to simulate")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in minutes")
    parser.add_argument("--json_out", type=str, help="Optional path to write JSON results")

    args = parser.parse_args()
    results = compare_models(args.models, args.minutes, args.horizon)

    print("\nMODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        s = res.summary
        print(
            f"{name}: total_return={_format_pct(s.get('total_return', 0.0))}, "
            f"annualized_return={_format_pct(s.get('annualized_return', 0.0))}, "
            f"sharpe={s.get('sharpe_ratio', 0.0):.2f}, "
            f"max_drawdown={_format_pct(s.get('max_drawdown', 0.0))}, "
            f"win_rate={_format_pct(s.get('win_rate', 0.0))}"
        )

    if args.json_out:
        serializable = {
            k: {
                "summary": v.summary,
                "analysis_timestamp": v.full_metrics.get("analysis_timestamp"),
                "basic_metrics": v.full_metrics.get("basic_metrics", {}),
                "risk_metrics": v.full_metrics.get("risk_metrics", {}),
            }
            for k, v in results.items()
        }
        with open(args.json_out, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nSaved JSON results to {args.json_out}")


if __name__ == "__main__":
    main()

