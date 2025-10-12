"""Microstructure analysis for detecting hidden institutional order flow."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class HiddenOrderSequence:
    """Summary of a suspected hidden institutional parent order."""

    sequence_id: int
    side: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    child_count: int
    duration_ms: float
    total_size: float
    total_notional: float
    avg_clip_size: float
    clip_size_cv: float
    median_time_gap_ms: float
    price_drift_bps: float
    venue_count: int
    venue_rotation: List[str]
    pattern_tags: List[str]

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["start_time"] = self.start_time.isoformat()
        payload["end_time"] = self.end_time.isoformat()
        return payload


def load_trade_log(path: Path) -> pd.DataFrame:
    """Load the microstructure trade log and compute derived features."""

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        raise ValueError("Trade log is empty; expected at least one trade")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["notional"] = df["price"] * df["size"]
    df["signed_notional"] = np.where(df["side"].str.upper() == "BUY", df["notional"], -df["notional"])
    df["time_delta_us"] = df["timestamp"].diff().dt.total_seconds().fillna(0) * 1_000_000
    df.loc[df.index[0], "time_delta_us"] = 0.0
    df["price_change_bps"] = df["price"].pct_change().fillna(0.0) * 10_000
    df["rolling_vwap"] = (
        (df["price"] * df["size"]).rolling(window=50, min_periods=1).sum()
        / df["size"].rolling(window=50, min_periods=1).sum()
    )
    df["price_vs_vwap_bps"] = (df["price"] - df["rolling_vwap"]) / df["rolling_vwap"] * 10_000
    return df


def _sequence_indices(df: pd.DataFrame,
                      max_gap_us: float = 250_000,
                      size_tolerance: float = 0.28,
                      price_tolerance_bps: float = 12.0,
                      min_length: int = 8) -> List[pd.Index]:
    """Group consecutive trades that resemble order slicing."""

    sequences: List[List[int]] = []
    current: List[int] = [0]
    for idx in range(1, len(df)):
        prev_idx = current[-1]
        prev = df.iloc[prev_idx]
        row = df.iloc[idx]

        same_side = row["side"] == prev["side"]
        within_gap = row["time_delta_us"] <= max_gap_us
        within_size = prev["size"] > 0 and abs(row["size"] - prev["size"]) <= size_tolerance * prev["size"]
        price_diff_bps = abs(row["price"] - prev["price"]) / prev["price"] * 10_000
        stable_price = price_diff_bps <= price_tolerance_bps

        if same_side and within_gap and within_size and stable_price:
            current.append(idx)
        else:
            if len(current) >= min_length:
                sequences.append(pd.Index(current))
            current = [idx]

    if len(current) >= min_length:
        sequences.append(pd.Index(current))

    # Merge sequences that are separated by small gaps but flagged due to venue rotation or slight clip drift
    merged: List[pd.Index] = []
    for seq in sequences:
        if not merged:
            merged.append(seq)
            continue
        previous_seq = merged[-1]
        gap_us = (df.iloc[seq[0]]["timestamp"] - df.iloc[previous_seq[-1]]["timestamp"]).total_seconds() * 1_000_000
        same_side = df.iloc[seq[0]]["side"] == df.iloc[previous_seq[-1]]["side"]
        if same_side and gap_us <= max_gap_us * 2:
            merged[-1] = previous_seq.append(seq)
        else:
            merged.append(seq)

    return merged


def _sequence_tags(subset: pd.DataFrame) -> List[str]:
    tags: List[str] = []
    median_gap = subset["time_delta_us"].median() / 1_000
    size_cv = subset["size"].std(ddof=0) / subset["size"].mean() if subset["size"].mean() else 0.0
    price_changes = subset["price"].diff().dropna()
    monotonicity = (np.sign(price_changes) == np.sign(price_changes.iloc[0])).mean() if not price_changes.empty else 0
    price_drift = (subset["price"].iloc[-1] - subset["price"].iloc[0]) / subset["price"].iloc[0] * 10_000
    vwap_drift = subset["price_vs_vwap_bps"].abs().median()

    if median_gap <= 120.0:
        tags.append("high-frequency slicing")
    if size_cv <= 0.12:
        tags.append("iceberg-like constant clip")
    if vwap_drift <= 4.0:
        tags.append("VWAP shadowing")
    if monotonicity >= 0.8:
        tags.append("price stepping")
    if len(subset["venue"].unique()) >= 3:
        tags.append("venue rotation")

    first_half = subset.head(len(subset) // 2)
    second_half = subset.tail(len(subset) // 2)
    if not first_half.empty and not second_half.empty:
        first_gap = first_half["time_delta_us"].mean()
        second_gap = second_half["time_delta_us"].mean()
        if np.isfinite(first_gap) and np.isfinite(second_gap) and second_gap < 0.65 * first_gap:
            tags.append("execution acceleration")
    if abs(price_drift) >= 15:
        tags.append("momentum participation")
    return sorted(set(tags))


def detect_hidden_sequences(df: pd.DataFrame) -> List[HiddenOrderSequence]:
    raw_sequences = _sequence_indices(df)
    sequences: List[HiddenOrderSequence] = []
    for seq_id, indices in enumerate(raw_sequences, start=1):
        subset = df.iloc[indices]
        child_count = len(subset)
        duration_ms = (
            (subset["timestamp"].iloc[-1] - subset["timestamp"].iloc[0]).total_seconds() * 1_000
            if child_count > 1
            else 0.0
        )
        total_size = float(subset["size"].sum())
        total_notional = float(subset["notional"].sum())
        avg_clip = float(subset["size"].mean())
        clip_cv = float(subset["size"].std(ddof=0) / avg_clip) if avg_clip else 0.0
        median_gap_ms = float(subset["time_delta_us"].median() / 1_000)
        price_drift_bps = float(
            (subset["price"].iloc[-1] - subset["price"].iloc[0]) / subset["price"].iloc[0] * 10_000
            if child_count > 1
            else 0.0
        )
        venue_rotation = subset["venue"].tolist()[: min(child_count, 10)]
        tags = _sequence_tags(subset)

        sequences.append(
            HiddenOrderSequence(
                sequence_id=seq_id,
                side=subset["side"].iloc[0],
                start_time=subset["timestamp"].iloc[0],
                end_time=subset["timestamp"].iloc[-1],
                child_count=child_count,
                duration_ms=duration_ms,
                total_size=total_size,
                total_notional=total_notional,
                avg_clip_size=avg_clip,
                clip_size_cv=clip_cv,
                median_time_gap_ms=median_gap_ms,
                price_drift_bps=price_drift_bps,
                venue_count=int(subset["venue"].nunique()),
                venue_rotation=venue_rotation,
                pattern_tags=tags,
            )
        )
    return sequences


def aggregate_microstructure_metrics(df: pd.DataFrame, sequences: Sequence[HiddenOrderSequence]) -> Dict[str, object]:
    total_notional = float(df["notional"].sum())
    hidden_notional = float(sum(seq.total_notional for seq in sequences))
    total_trades = int(len(df))
    total_hidden_trades = int(sum(seq.child_count for seq in sequences))
    buy_share = float(df.loc[df["side"] == "BUY", "notional"].sum() / total_notional)
    signed_notional = float(df["signed_notional"].sum())

    top_sequences = sorted(sequences, key=lambda seq: seq.total_notional, reverse=True)[:5]

    return {
        "total_trades": total_trades,
        "total_hidden_trades": total_hidden_trades,
        "hidden_trade_ratio": float(total_hidden_trades / total_trades),
        "hidden_notional_ratio": float(hidden_notional / total_notional),
        "net_signed_notional": signed_notional,
        "dominant_flow": "BUY" if signed_notional > 0 else "SELL",
        "buy_notional_share": buy_share,
        "median_trade_size": float(df["size"].median()),
        "median_time_gap_ms": float(df["time_delta_us"].median() / 1_000),
        "top_sequences": [seq.as_dict() for seq in top_sequences],
    }


def generate_markdown_report(metrics: Dict[str, object], sequences: Sequence[HiddenOrderSequence]) -> str:
    lines = [
        "# Microstructure Hidden Flow Analysis",
        "",
        f"- **Total trades analysed:** {metrics['total_trades']}",
        f"- **Hidden-flow trade share:** {metrics['hidden_trade_ratio'] * 100:.1f}%",
        f"- **Hidden notional share:** {metrics['hidden_notional_ratio'] * 100:.1f}%",
        f"- **Dominant flow direction:** {metrics['dominant_flow']} (net signed notional {metrics['net_signed_notional']:.2f})",
        f"- **Buy-side notional share:** {metrics['buy_notional_share'] * 100:.1f}%",
        f"- **Median clip size:** {metrics['median_trade_size']:.5f}",
        f"- **Median inter-trade gap:** {metrics['median_time_gap_ms']:.2f} ms",
        "",
        "## Top Hidden Flow Sequences",
    ]

    for seq in sequences:
        lines.extend(
            [
                f"### Sequence {seq.sequence_id} ({seq.side})",
                f"- Duration: {seq.duration_ms:.2f} ms with {seq.child_count} clips",
                f"- Total size: {seq.total_size:.4f}, avg clip {seq.avg_clip_size:.5f} (CV {seq.clip_size_cv:.2f})",
                f"- Price drift: {seq.price_drift_bps:.2f} bps, median gap {seq.median_time_gap_ms:.2f} ms",
                f"- Venues: {', '.join(seq.venue_rotation)}",
                f"- Tags: {', '.join(seq.pattern_tags) if seq.pattern_tags else '—'}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def run_analysis(input_path: Path, json_output: Path | None, markdown_output: Path | None) -> Dict[str, object]:
    df = load_trade_log(input_path)
    sequences = detect_hidden_sequences(df)
    metrics = aggregate_microstructure_metrics(df, sequences)

    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with json_output.open("w", encoding="utf-8") as fh:
            payload = {
                "metrics": metrics,
                "sequences": [seq.as_dict() for seq in sequences],
            }
            json.dump(payload, fh, indent=2)

    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        top_sequences = sorted(sequences, key=lambda seq: seq.total_notional, reverse=True)[:5]
        report = generate_markdown_report(metrics, top_sequences)
        markdown_output.write_text(report, encoding="utf-8")

    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect hidden institutional order flow from microsecond trade logs")
    parser.add_argument("input", type=Path, help="Path to the CSV trade log")
    parser.add_argument("--json-output", type=Path, help="Optional path for JSON summary")
    parser.add_argument("--markdown-output", type=Path, help="Optional path for Markdown report")
    args = parser.parse_args(argv)

    run_analysis(args.input, args.json_output, args.markdown_output)


if __name__ == "__main__":
    main()
