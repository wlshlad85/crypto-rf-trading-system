"""Tests for cross-market and psychological overlays in the strategy ensemble."""

import numpy as np
import pandas as pd

from strategies.minute_trading_strategies import MinuteStrategyEnsemble


def _build_time_index(length: int = 120) -> pd.DatetimeIndex:
    """Helper to create a standard minute-level index for tests."""

    return pd.date_range("2024-01-01", periods=length, freq="T")


def test_cross_market_influence_aligns_altcoin_signals():
    """BTC momentum should bias positively correlated altcoins."""

    index = _build_time_index()
    btc_signal = np.linspace(0.0, 0.5, len(index))

    signals = pd.DataFrame(
        {
            "BTC-USD_signal": btc_signal,
            "ETH-USD_signal": np.zeros(len(index)),
        },
        index=index,
    )

    market_data = pd.DataFrame(
        {
            "BTC-USD_close": 30000 + np.linspace(0, 1000, len(index)),
            "ETH-USD_close": 2000 + np.linspace(0, 200, len(index)),
        },
        index=index,
    )

    ensemble = MinuteStrategyEnsemble(strategies=[])
    adjusted = ensemble._apply_cross_market_psychological_overlays(
        signals, market_data, ["BTC-USD", "ETH-USD"]
    )

    # BTC strength should pull ETH signals positive over time
    assert adjusted["ETH-USD_signal"].iloc[-1] > 0.15


def test_low_momentum_contrarian_overlay_dampens_signal():
    """Flat markets should trigger contrarian dampening."""

    index = _build_time_index()

    signals = pd.DataFrame(
        {
            "BTC-USD_signal": np.linspace(0.0, 0.3, len(index)),
            "ETH-USD_signal": np.full(len(index), 0.3),
        },
        index=index,
    )

    market_data = pd.DataFrame(
        {
            "BTC-USD_close": 30000 + np.linspace(0, 500, len(index)),
            "ETH-USD_close": np.full(len(index), 2000.0),
        },
        index=index,
    )

    ensemble = MinuteStrategyEnsemble(strategies=[])
    adjusted = ensemble._apply_cross_market_psychological_overlays(
        signals, market_data, ["BTC-USD", "ETH-USD"]
    )

    original = signals["ETH-USD_signal"].iloc[-1]
    updated = adjusted["ETH-USD_signal"].iloc[-1]

    # Low momentum environment should reduce conviction
    assert abs(updated) < abs(original)


def test_failed_pattern_flips_signal_contrarian():
    """Sustained losses against a long bias should flip the signal."""

    index = _build_time_index()

    signals = pd.DataFrame(
        {
            "BTC-USD_signal": np.linspace(0.0, 0.4, len(index)),
            "ETH-USD_signal": np.full(len(index), 0.4),
        },
        index=index,
    )

    eth_prices = 2000 - np.linspace(0, 150, len(index))
    market_data = pd.DataFrame(
        {
            "BTC-USD_close": 30000 + np.linspace(0, 500, len(index)),
            "ETH-USD_close": eth_prices,
        },
        index=index,
    )

    ensemble = MinuteStrategyEnsemble(strategies=[])
    adjusted = ensemble._apply_cross_market_psychological_overlays(
        signals, market_data, ["BTC-USD", "ETH-USD"]
    )

    # The ETH signal should be forced negative after repeated adverse moves
    assert adjusted["ETH-USD_signal"].iloc[-1] < 0
