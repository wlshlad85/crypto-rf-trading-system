#!/usr/bin/env python3
"""Phase 2B: Ensemble Signal Experiments with Meta-Signal Generation.

This module creates *meta-signals* that combine ensemble model disagreements with
Hidden Markov Model (HMM) regime shifts. The goal is to surface tradable
inefficiencies that appear when the model ensemble loses consensus while the
market transitions between regimes.

Core ideas implemented here:

* Quantify disagreement intensity across ensemble members.
* Detect regime shifts using the Phase 2B HMM regime detector.
* Fuse both signals into actionable meta-signals that highlight
  momentum and mean-reversion opportunities.
* Provide analytics to understand where inefficiencies come from.

The implementation is intentionally self-contained so it can be dropped into
notebooks, research pipelines, or the live system without heavy dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from phase2b.hmm_regime_detection import HMMConfig, HMMRegimeDetector


@dataclass
class MetaSignalConfig:
    """Configuration options for :class:`EnsembleMetaSignalGenerator`."""

    # Columns that contain directional ensemble signals (e.g. -1, 0, 1).
    model_signal_columns: Optional[List[str]] = None

    # Optional columns with probability/confidence style outputs.
    model_probability_columns: Optional[List[str]] = None

    # Disagreement thresholds.
    disagreement_threshold: float = 0.45
    signal_range_threshold: float = 1.5

    # Probability spread thresholds (if probability columns are provided).
    probability_spread_threshold: float = 0.30

    # Minimum absolute mean signal required to treat a consensus as actionable.
    min_consensus_strength: float = 0.40

    # How strong a regime shift needs to be (based on probability jump) to
    # participate in meta-signal construction.
    regime_shift_threshold: float = 0.35
    regime_confidence_floor: float = 0.55

    # Rolling window used to smooth the meta-signal strength.
    smoothing_window: int = 3

    # Weighting between disagreement- and regime-based components when building
    # the final meta-signal score.
    disagreement_weight: float = 0.6
    regime_weight: float = 0.4

    # Lookback for computing short-term market returns (used to determine
    # whether we should lean into momentum or mean-reversion when the ensemble
    # is split).
    lookback_return: int = 3
    return_column: str = "Close"

    # Mapping between regime names and risk-on/risk-off context. Defaults cover
    # the naming convention used in :class:`HMMConfig` but can be overridden.
    risk_on_states: Iterable[str] = field(
        default_factory=lambda: ("Bull", "Expansion", "Growth")
    )
    risk_off_states: Iterable[str] = field(
        default_factory=lambda: ("Bear", "Contraction", "Stress", "Crash")
    )

    # Meta-signal labelling.
    hold_action: str = "HOLD"
    hedge_action: str = "HEDGE"
    exploit_action: str = "EXPLOIT"
    momentum_action: str = "MOMENTUM"
    mean_reversion_action: str = "MEAN_REVERSION"

    # Minimum absolute value for the meta-signal confidence score.
    meta_signal_floor: float = 0.10

    def resolve_signal_columns(self, predictions: pd.DataFrame) -> List[str]:
        """Return the list of model signal columns to use."""

        if self.model_signal_columns:
            missing = [col for col in self.model_signal_columns if col not in predictions.columns]
            if missing:
                raise ValueError(
                    "Configured model signal columns are missing from predictions: "
                    f"{missing}"
                )
            return list(self.model_signal_columns)

        inferred = [col for col in predictions.columns if col.endswith("_signal")]
        if not inferred:
            raise ValueError(
                "Unable to infer model signal columns. Provide them explicitly via "
                "MetaSignalConfig.model_signal_columns."
            )
        return inferred

    def resolve_probability_columns(self, predictions: pd.DataFrame) -> List[str]:
        """Return probability-style columns if present and valid."""

        if self.model_probability_columns:
            missing = [col for col in self.model_probability_columns if col not in predictions.columns]
            if missing:
                raise ValueError(
                    "Configured probability columns are missing from predictions: "
                    f"{missing}"
                )
            return list(self.model_probability_columns)

        inferred = [col for col in predictions.columns if col.endswith("_prob")]
        return inferred


class EnsembleMetaSignalGenerator:
    """Generate meta-signals using ensemble disagreements and HMM regimes."""

    def __init__(
        self,
        config: Optional[MetaSignalConfig] = None,
        regime_detector: Optional[HMMRegimeDetector] = None,
    ) -> None:
        self.config = config or MetaSignalConfig()
        self.regime_detector = regime_detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_meta_signals(
        self,
        predictions: pd.DataFrame,
        market_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create meta-signals from model outputs and regime information.

        Args:
            predictions: DataFrame containing ensemble predictions. Must include
                the directional model signal columns defined in the config.
            market_data: Market OHLCV data aligned with ``predictions``.
            regime_data: Optional pre-computed HMM regime dataframe. If not
                provided, the internal ``regime_detector`` will be used.

        Returns:
            DataFrame with meta-signal diagnostics and actionable labels.
        """

        predictions = self._ensure_datetime_index(predictions.copy())
        market_data = self._ensure_datetime_index(market_data.copy())
        disagreement_metrics = self._compute_disagreement_metrics(predictions)
        regime_metrics = self._compute_regime_metrics(market_data, regime_data)
        regime_metrics = regime_metrics.reindex(predictions.index)
        regime_metrics = regime_metrics.ffill().bfill()
        price_metrics = self._compute_price_metrics(market_data, predictions.index)

        combined = predictions.join(disagreement_metrics, how="left")
        combined = combined.join(regime_metrics, how="left")
        combined = combined.join(price_metrics, how="left")

        combined["meta_signal_score"] = self._combine_scores(combined)
        combined["meta_signal_strength"] = (
            combined["meta_signal_score"].rolling(self.config.smoothing_window).mean()
        )
        combined["meta_signal_strength"].fillna(combined["meta_signal_score"], inplace=True)
        combined["meta_signal_confidence"] = combined["meta_signal_strength"].clip(
            lower=self.config.meta_signal_floor
        ).clip(upper=1.0)

        actions, triggers = self._label_actions(combined)
        combined["meta_signal_action"] = actions
        combined["inefficiency_trigger"] = triggers

        return combined

    def summarize_meta_signals(self, meta_signals: pd.DataFrame) -> Dict[str, Any]:
        """Provide quick analytics over generated meta-signals."""

        signal_mask = meta_signals["meta_signal_action"] != self.config.hold_action
        active_signals = meta_signals[signal_mask]

        summary = {
            "total_observations": int(len(meta_signals)),
            "active_meta_signals": int(signal_mask.sum()),
            "average_strength": float(active_signals["meta_signal_strength"].mean()
                                      if not active_signals.empty else 0.0),
            "average_confidence": float(active_signals["meta_signal_confidence"].mean()
                                         if not active_signals.empty else 0.0),
            "trigger_counts": meta_signals["inefficiency_trigger"].value_counts().to_dict(),
            "action_counts": meta_signals["meta_signal_action"].value_counts().to_dict(),
        }

        if "lookback_return" in meta_signals.columns and not active_signals.empty:
            summary["avg_return_during_signals"] = float(
                active_signals["lookback_return"].mean()
            )

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_disagreement_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        model_cols = self.config.resolve_signal_columns(predictions)
        signals = predictions[model_cols].astype(float)

        disagreement_std = signals.std(axis=1)
        disagreement_range = signals.max(axis=1) - signals.min(axis=1)
        consensus_signal = signals.mean(axis=1)
        consensus_strength = consensus_signal.abs()
        consensus_direction = np.sign(consensus_signal).replace({0.0: 0})

        bullish_support = (signals > 0).sum(axis=1)
        bearish_support = (signals < 0).sum(axis=1)
        neutral_support = (signals == 0).sum(axis=1)

        rolling_mean = disagreement_std.rolling(window=25, min_periods=5).mean()
        rolling_std = disagreement_std.rolling(window=25, min_periods=5).std()
        disagreement_z = (disagreement_std - rolling_mean) / rolling_std.replace(0, np.nan)
        disagreement_z = disagreement_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        disagreement_norm = (disagreement_std - disagreement_std.min())
        denom = disagreement_std.max() - disagreement_std.min()
        if denom and not np.isclose(denom, 0.0):
            disagreement_norm = disagreement_norm / denom
        else:
            disagreement_norm = pd.Series(0.0, index=signals.index)
        disagreement_norm = disagreement_norm.fillna(0.0)

        probability_cols = self.config.resolve_probability_columns(predictions)
        probability_spread = None
        if probability_cols:
            probabilities = predictions[probability_cols].astype(float)
            probability_spread = probabilities.max(axis=1) - probabilities.min(axis=1)
        else:
            probability_spread = pd.Series(index=signals.index, data=np.nan)

        if probability_spread.notna().any():
            probability_condition = probability_spread >= self.config.probability_spread_threshold
        else:
            probability_condition = pd.Series(False, index=signals.index)

        disagreement_flag = (
            (disagreement_std >= self.config.disagreement_threshold)
            | (disagreement_range >= self.config.signal_range_threshold)
            | probability_condition
        )

        metrics = pd.DataFrame(
            {
                "disagreement_std": disagreement_std,
                "disagreement_range": disagreement_range,
                "disagreement_z": disagreement_z,
                "disagreement_norm": disagreement_norm,
                "consensus_signal": consensus_signal,
                "consensus_strength": consensus_strength,
                "consensus_direction": consensus_direction,
                "bullish_support": bullish_support,
                "bearish_support": bearish_support,
                "neutral_support": neutral_support,
                "probability_spread": probability_spread,
                "disagreement_flag": disagreement_flag.astype(int),
            }
        )

        return metrics

    def _compute_regime_metrics(
        self,
        market_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if regime_data is None:
            if self.regime_detector is None:
                raise ValueError(
                    "regime_data not supplied and no regime_detector configured."
                )
            if not getattr(self.regime_detector, "is_fitted", False):
                raise ValueError(
                    "regime_detector must be fitted before generating meta-signals."
                )
            regime_data = self.regime_detector.predict_regimes(market_data)
        else:
            regime_data = regime_data.copy()

        regime_data = self._ensure_datetime_index(regime_data)

        if "regime_name" not in regime_data.columns and "regime_state" in regime_data.columns:
            if isinstance(self.regime_detector, HMMRegimeDetector):
                names = self.regime_detector.config.state_names
                regime_data["regime_name"] = [names[int(state)] for state in regime_data["regime_state"]]
            else:
                regime_data["regime_name"] = regime_data["regime_state"].astype(str)

        prob_columns = [col for col in regime_data.columns if col.startswith("regime_prob_")]
        if prob_columns:
            probability_matrix = regime_data[prob_columns].fillna(0.0)
            dominant_probability = probability_matrix.max(axis=1)
            sorted_probs = np.sort(probability_matrix.values, axis=1)
            top_two_diff = (
                sorted_probs[:, -1] - sorted_probs[:, -2]
                if probability_matrix.shape[1] >= 2
                else sorted_probs[:, -1]
            )
            regime_pressure = pd.Series(top_two_diff, index=probability_matrix.index)
        else:
            dominant_probability = pd.Series(1.0, index=regime_data.index)
            regime_pressure = pd.Series(0.0, index=regime_data.index)

        regime_state = regime_data.get("regime_state")
        if regime_state is not None:
            regime_shift = regime_state.ne(regime_state.shift()).fillna(False)
        else:
            regime_shift = pd.Series(False, index=regime_data.index, dtype=bool)

        regime_shift_strength = dominant_probability.diff().abs().fillna(0.0)
        if not prob_columns:
            regime_shift_strength = regime_shift_strength.where(~regime_shift, 1.0)
        regime_shift_strength = regime_shift_strength.clip(upper=1.0)
        regime_shift_strength = regime_shift_strength * regime_shift.astype(int)

        regime_bias = self._map_regime_bias(regime_data.get("regime_name"), regime_data.index)

        metrics = pd.DataFrame(
            {
                "regime_state": regime_state,
                "regime_name": regime_data.get("regime_name"),
                "regime_confidence": dominant_probability,
                "regime_pressure": regime_pressure,
                "regime_shift": regime_shift.astype(int),
                "regime_shift_strength": regime_shift_strength,
                "regime_bias": regime_bias,
            }
        )

        low_conf_mask = metrics["regime_confidence"] < self.config.regime_confidence_floor
        metrics.loc[low_conf_mask, "regime_shift"] = 0
        metrics.loc[low_conf_mask, "regime_shift_strength"] = 0.0

        weak_shift_mask = metrics["regime_shift_strength"] < self.config.regime_shift_threshold
        metrics.loc[weak_shift_mask, "regime_shift"] = 0
        metrics.loc[weak_shift_mask, "regime_shift_strength"] = 0.0

        return metrics

    def _compute_price_metrics(
        self, market_data: pd.DataFrame, target_index: pd.Index
    ) -> pd.DataFrame:
        market_data = market_data.copy()
        market_data = self._ensure_datetime_index(market_data)

        price_series = self._extract_price_series(market_data)
        lookback_return = price_series.pct_change(self.config.lookback_return)
        lookback_return = lookback_return.reindex(target_index, method="ffill")
        lookback_return = lookback_return.fillna(0.0)

        metrics = pd.DataFrame({"lookback_return": lookback_return}, index=target_index)
        return metrics

    def _combine_scores(self, combined: pd.DataFrame) -> pd.Series:
        disagreement_component = combined["disagreement_norm"].fillna(0.0)
        regime_component = combined["regime_shift_strength"].fillna(0.0)
        pressure_component = combined.get("regime_pressure", pd.Series(0.0, index=combined.index))

        meta_score = (
            self.config.disagreement_weight * disagreement_component
            + self.config.regime_weight * regime_component
            + 0.1 * pressure_component.fillna(0.0)
        )
        return meta_score.clip(lower=0.0, upper=1.5)

    def _label_actions(self, combined: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        actions = pd.Series(self.config.hold_action, index=combined.index, dtype="object")
        triggers = pd.Series("NONE", index=combined.index, dtype="object")

        for idx, row in combined.iterrows():
            disagreement_active = bool(row.get("disagreement_flag", 0))
            regime_shift = bool(row.get("regime_shift", 0))
            consensus_strength = float(row.get("consensus_strength", 0.0))
            consensus_direction = float(row.get("consensus_direction", 0.0))
            lookback_return = float(row.get("lookback_return", 0.0))
            regime_bias = float(row.get("regime_bias", 0.0))

            action = self.config.hold_action
            trigger = "NONE"

            if disagreement_active and regime_shift:
                trigger = "DISAGREEMENT_AND_REGIME_SHIFT"
                if consensus_strength >= self.config.min_consensus_strength:
                    if consensus_direction >= 0:
                        action = f"{self.config.momentum_action}_LONG"
                    else:
                        action = f"{self.config.momentum_action}_SHORT"
                else:
                    if lookback_return <= 0:
                        action = f"{self.config.mean_reversion_action}_LONG"
                    else:
                        action = f"{self.config.mean_reversion_action}_SHORT"
            elif disagreement_active:
                trigger = "PURE_DISAGREEMENT"
                if consensus_direction > 0:
                    action = f"{self.config.hedge_action}_LONG"
                elif consensus_direction < 0:
                    action = f"{self.config.hedge_action}_SHORT"
                else:
                    action = self.config.hedge_action
            elif regime_shift:
                trigger = "REGIME_SHIFT_ONLY"
                if regime_bias > 0:
                    action = f"{self.config.exploit_action}_LONG"
                elif regime_bias < 0:
                    action = f"{self.config.exploit_action}_SHORT"
                else:
                    action = self.config.exploit_action

            actions.at[idx] = action
            triggers.at[idx] = trigger

        return actions, triggers

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex)):
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
            elif "datetime" in df.columns:
                df.index = pd.to_datetime(df["datetime"])
        df.sort_index(inplace=True)
        return df

    def _extract_price_series(self, market_data: pd.DataFrame) -> pd.Series:
        candidates = [
            self.config.return_column,
            "close",
            "Close",
            "adj_close",
            "Adj Close",
            "price",
        ]
        for column in candidates:
            if column in market_data.columns:
                return pd.to_numeric(market_data[column], errors="coerce").fillna(method="ffill")
        raise ValueError(
            "No suitable price column found for market_data. Provide one via MetaSignalConfig.return_column."
        )

    def _map_regime_bias(self, regime_names: Optional[pd.Series], index: pd.Index) -> pd.Series:
        if regime_names is None:
            return pd.Series(0.0, index=index)

        risk_on = set(state.lower() for state in self.config.risk_on_states)
        risk_off = set(state.lower() for state in self.config.risk_off_states)

        def mapper(name: Any) -> float:
            if name is None or (isinstance(name, float) and math.isnan(name)):
                return 0.0
            lower = str(name).lower()
            if lower in risk_on:
                return 1.0
            if lower in risk_off:
                return -1.0
            return 0.0

        return regime_names.reindex(index).map(mapper).fillna(0.0).astype(float)


def run_demo() -> None:
    """Demonstration showcasing the meta-signal pipeline."""

    np.random.seed(42)
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="H")

    # Simulated market data
    price = 30000 + np.cumsum(np.random.randn(periods)) * 50
    volume = 1000 + np.random.randn(periods) * 25
    market_df = pd.DataFrame({"timestamp": index, "Close": price, "Volume": volume})

    # Mock ensemble predictions (four models generating directional scores)
    predictions = pd.DataFrame(
        {
            "timestamp": index,
            "rf_signal": np.random.choice([-1, 0, 1], size=periods, p=[0.25, 0.5, 0.25]),
            "xgb_signal": np.random.choice([-1, 0, 1], size=periods, p=[0.3, 0.4, 0.3]),
            "lgbm_signal": np.random.choice([-1, 0, 1], size=periods, p=[0.2, 0.6, 0.2]),
            "nn_signal": np.random.choice([-1, 0, 1], size=periods, p=[0.2, 0.5, 0.3]),
        }
    )

    # Create a simple HMM detector and fit it on the synthetic data.
    hmm_config = HMMConfig(n_states=3)
    detector = HMMRegimeDetector(hmm_config)
    detector.fit(market_df)
    regime_df = detector.predict_regimes(market_df)

    generator = EnsembleMetaSignalGenerator(config=MetaSignalConfig(), regime_detector=detector)
    meta_signals = generator.generate_meta_signals(
        predictions=predictions,
        market_data=market_df,
        regime_data=regime_df,
    )

    summary = generator.summarize_meta_signals(meta_signals)

    print("\n🚀 Meta-Signal Generation Demo")
    print("=" * 40)
    print(meta_signals[[
        "disagreement_std",
        "regime_shift",
        "meta_signal_score",
        "meta_signal_action",
        "meta_signal_confidence",
    ]].tail())

    print("\n📊 Summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    run_demo()
