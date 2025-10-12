"""Microstructure pattern analysis utilities.

This module provides functionality to analyse granular trade data for
microstructure patterns such as refreshing (iceberg) orders and bursts of
activity that may indicate institutional trading footprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RefreshingOrderSignal:
    """Summary of an inferred refreshing order pattern."""

    side: str
    price: float
    refresh_count: int
    total_quantity: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    mean_size: float
    median_size: float
    size_cv: float
    cluster_id: int

    @property
    def duration_seconds(self) -> float:
        """Return the duration in seconds for which the order refreshed."""

        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON serialisable representation of the signal."""

        return {
            "side": self.side,
            "price": self.price,
            "refresh_count": self.refresh_count,
            "total_quantity": self.total_quantity,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "mean_size": self.mean_size,
            "median_size": self.median_size,
            "size_cv": self.size_cv,
            "cluster_id": self.cluster_id,
            "duration_seconds": self.duration_seconds,
        }


class MicrostructurePatternAnalyzer:
    """Analyse microstructure trade data to surface institutional patterns.

    Parameters
    ----------
    refresh_time_threshold_us:
        Maximum gap (in microseconds) between trades to be considered part of
        the same refreshing sequence.
    min_refresh_count:
        Minimum number of executions at the same price/side required before we
        classify the activity as a refreshing (iceberg-style) order.
    price_precision:
        Number of decimal places to round the price to when grouping trades.
        This helps combine executions that may have fractional tick rounding.
    size_similarity_tolerance:
        Relative tolerance used to consider trade sizes as part of the same
        refreshing order. Expressed as a fraction of the mean size.
    cluster_gap_us:
        Microsecond gap used to segment the tape into time clusters for higher
        level burst analysis.
    """

    def __init__(
        self,
        refresh_time_threshold_us: int = 500_000,
        min_refresh_count: int = 5,
        price_precision: int = 2,
        size_similarity_tolerance: float = 0.25,
        cluster_gap_us: int = 1_000_000,
    ) -> None:
        self.refresh_time_threshold_us = refresh_time_threshold_us
        self.min_refresh_count = min_refresh_count
        self.price_precision = price_precision
        self.size_similarity_tolerance = size_similarity_tolerance
        self.cluster_gap_us = cluster_gap_us

    def analyze(self, trades: pd.DataFrame) -> Dict[str, object]:
        """Run the full microstructure analysis pipeline."""

        normalized = self._prepare_trades(trades)
        normalized = self._assign_cluster_ids(normalized)
        clusters = self._cluster_trades(normalized)
        refreshing_orders = self._detect_refreshing_orders(normalized)
        bursts = self._detect_bursts(clusters)

        return {
            "trade_count": len(normalized),
            "clusters": clusters,
            "refreshing_orders": refreshing_orders,
            "bursts": bursts,
        }

    def _prepare_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise the trade dataset."""

        if trades.empty:
            raise ValueError("Trade dataset is empty; cannot run analysis.")

        required_columns = {"timestamp", "price", "quantity"}
        missing = required_columns - set(trades.columns)
        if missing:
            raise ValueError(
                "Trade dataset missing required columns: " + ", ".join(sorted(missing))
            )

        normalized = trades.copy()
        normalized["timestamp"] = pd.to_datetime(
            normalized["timestamp"], utc=True, errors="coerce"
        )

        if normalized["timestamp"].isna().any():
            raise ValueError("Trade dataset contains invalid timestamps.")

        # Standardise trade side nomenclature.
        if "side" in normalized.columns:
            normalized["side"] = normalized["side"].str.lower().map(
                {
                    "buy": "buy",
                    "bid": "buy",
                    "b": "buy",
                    "sell": "sell",
                    "ask": "sell",
                    "s": "sell",
                }
            )
        else:
            normalized["side"] = "buy"

        normalized = normalized.sort_values("timestamp").reset_index(drop=True)
        normalized["price_bucket"] = normalized["price"].round(self.price_precision)
        normalized["timestamp_us"] = (
            normalized["timestamp"].astype("int64") // 1_000
        )

        return normalized

    def _assign_cluster_ids(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Assign cluster identifiers to each trade."""

        timestamps = trades["timestamp_us"].to_numpy()
        gap = np.diff(timestamps, prepend=timestamps[0])
        cluster_breaks = gap > self.cluster_gap_us
        trades = trades.assign(cluster_id=np.cumsum(cluster_breaks))
        return trades

    def _cluster_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Segment the tape into bursts of activity using a time gap heuristic."""

        cluster_stats = (
            trades.groupby("cluster_id")
            .agg(
                start_time=("timestamp", "min"),
                end_time=("timestamp", "max"),
                trade_count=("timestamp", "count"),
                total_quantity=("quantity", "sum"),
            )
            .reset_index()
        )
        cluster_stats["duration_seconds"] = (
            cluster_stats["end_time"] - cluster_stats["start_time"]
        ).dt.total_seconds()

        return cluster_stats

    def _detect_refreshing_orders(self, trades: pd.DataFrame) -> List[RefreshingOrderSignal]:
        """Detect refreshing (iceberg-style) orders."""

        signals: List[RefreshingOrderSignal] = []
        for (side, price_bucket), group in trades.groupby(["side", "price_bucket"]):
            if len(group) < self.min_refresh_count:
                continue

            timestamps = group["timestamp_us"].to_numpy()
            gaps = np.diff(timestamps, prepend=timestamps[0])
            sequence_ids = np.cumsum(gaps > self.refresh_time_threshold_us)
            group = group.assign(sequence_id=sequence_ids)

            for _, sequence in group.groupby("sequence_id"):
                if len(sequence) < self.min_refresh_count:
                    continue

                size_mean = sequence["quantity"].mean()
                if np.isclose(size_mean, 0.0):
                    continue

                relative_deviation = (
                    np.std(sequence["quantity"], ddof=0) / max(size_mean, 1e-9)
                )
                if relative_deviation > self.size_similarity_tolerance:
                    continue

                cluster_id = int(sequence["cluster_id"].mode().iat[0])
                signal = RefreshingOrderSignal(
                    side=side,
                    price=float(price_bucket),
                    refresh_count=len(sequence),
                    total_quantity=float(sequence["quantity"].sum()),
                    start_time=sequence["timestamp"].iloc[0],
                    end_time=sequence["timestamp"].iloc[-1],
                    mean_size=float(size_mean),
                    median_size=float(sequence["quantity"].median()),
                    size_cv=float(relative_deviation),
                    cluster_id=cluster_id,
                )
                signals.append(signal)

        return signals

    def _detect_bursts(self, clusters: pd.DataFrame) -> pd.DataFrame:
        """Identify clusters with abnormally high activity."""

        if clusters.empty:
            return clusters

        volume_z = (
            clusters["total_quantity"] - clusters["total_quantity"].mean()
        ) / (clusters["total_quantity"].std(ddof=0) + 1e-9)
        clusters = clusters.assign(volume_z=volume_z)
        clusters["is_burst"] = clusters["volume_z"].gt(1.5)
        return clusters


def load_trades(path: Path | str) -> pd.DataFrame:
    """Load trade data from CSV or JSON into a DataFrame."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Trade data file not found: {resolved}")

    if resolved.suffix.lower() == ".csv":
        df = pd.read_csv(resolved)
    elif resolved.suffix.lower() == ".json":
        df = pd.read_json(resolved)
    else:
        raise ValueError("Unsupported trade data format. Use CSV or JSON.")

    return df


def run_analysis(
    path: Path | str,
    output: Optional[Path | str] = None,
    analyzer: Optional[MicrostructurePatternAnalyzer] = None,
) -> Dict[str, object]:
    """Convenience wrapper to execute the analysis and optionally save output."""

    analyzer = analyzer or MicrostructurePatternAnalyzer()
    trades = load_trades(path)
    results = analyzer.analyze(trades)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        serialisable = {
            "trade_count": results["trade_count"],
            "clusters": results["clusters"].to_dict(orient="records"),
            "refreshing_orders": [signal.to_dict() for signal in results["refreshing_orders"]],
            "bursts": results["bursts"].to_dict(orient="records"),
        }
        output_path.write_text(json.dumps(serialisable, indent=2))

    return results


def _build_cli_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(description="Microstructure pattern analysis")
    parser.add_argument("input", help="Path to trade data (CSV or JSON)")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--min-refresh-count",
        type=int,
        default=5,
        help="Minimum number of executions for an iceberg detection",
    )
    parser.add_argument(
        "--refresh-gap-us",
        type=int,
        default=500_000,
        help="Maximum microsecond gap between fills for refreshing orders",
    )
    parser.add_argument(
        "--cluster-gap-us",
        type=int,
        default=1_000_000,
        help="Gap (microseconds) for time clustering",
    )
    parser.add_argument(
        "--size-tolerance",
        type=float,
        default=0.25,
        help="Relative tolerance for trade size similarity",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Entry-point for command line execution."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    analyzer = MicrostructurePatternAnalyzer(
        refresh_time_threshold_us=args.refresh_gap_us,
        min_refresh_count=args.min_refresh_count,
        cluster_gap_us=args.cluster_gap_us,
        size_similarity_tolerance=args.size_tolerance,
    )

    return run_analysis(args.input, args.output, analyzer)


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
