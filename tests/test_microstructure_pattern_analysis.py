import pandas as pd
import pytest

from analysis.microstructure_pattern_analysis import (
    MicrostructurePatternAnalyzer,
    RefreshingOrderSignal,
)


def _build_sample_trades() -> pd.DataFrame:
    base_time = pd.Timestamp("2025-07-14T12:00:00.000000Z")
    records = []
    # Refreshing buy order at price 100.0 repeated 6 times.
    for i in range(6):
        records.append(
            {
                "timestamp": base_time + pd.Timedelta(microseconds=1000 * i),
                "price": 100.0,
                "quantity": 1.01 + 0.01 * ((-1) ** i),
                "side": "buy",
            }
        )
    # Non-refreshing activity.
    for i in range(4):
        records.append(
            {
                "timestamp": base_time + pd.Timedelta(seconds=2, microseconds=500 * i),
                "price": 100.5,
                "quantity": 0.5 + 0.05 * i,
                "side": "sell",
            }
        )

    return pd.DataFrame.from_records(records)


def test_prepare_and_cluster_assignments():
    trades = _build_sample_trades()
    analyzer = MicrostructurePatternAnalyzer(cluster_gap_us=500_000)
    normalized = analyzer._prepare_trades(trades)
    with_clusters = analyzer._assign_cluster_ids(normalized)

    # Expect at least two clusters due to the 2 second gap.
    assert with_clusters["cluster_id"].nunique() >= 2


def test_refreshing_order_detection():
    trades = _build_sample_trades()
    analyzer = MicrostructurePatternAnalyzer(
        refresh_time_threshold_us=200_000,
        min_refresh_count=5,
        size_similarity_tolerance=0.2,
    )
    normalized = analyzer._assign_cluster_ids(analyzer._prepare_trades(trades))
    signals = analyzer._detect_refreshing_orders(normalized)

    assert len(signals) == 1
    signal = signals[0]
    assert isinstance(signal, RefreshingOrderSignal)
    assert signal.side == "buy"
    assert pytest.approx(signal.price, rel=1e-3) == 100.0
    assert signal.refresh_count == 6
    assert signal.total_quantity > 0


def test_full_analysis_pipeline():
    trades = _build_sample_trades()
    analyzer = MicrostructurePatternAnalyzer(
        refresh_time_threshold_us=200_000,
        min_refresh_count=5,
        size_similarity_tolerance=0.2,
        cluster_gap_us=500_000,
    )
    results = analyzer.analyze(trades)

    assert results["trade_count"] == len(trades)
    assert not results["clusters"].empty
    assert any(signal.side == "buy" for signal in results["refreshing_orders"])
