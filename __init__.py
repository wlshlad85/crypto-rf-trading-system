"""Cryptocurrency Random Forest Trading System.

A sophisticated cryptocurrency trading system using Random Forest machine learning
for multi-asset portfolio management with long/short strategies and monthly rebalancing.
"""

__version__ = "1.0.0"
__author__ = "Crypto RF Trading System"
__description__ = "Random Forest-based cryptocurrency trading system"

import importlib
from typing import Any

__all__ = ["CryptoRFTradingSystem"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    if name != "CryptoRFTradingSystem":
        raise AttributeError(name)

    if __package__:
        module = importlib.import_module(".main", __package__)
    else:
        module = importlib.import_module("main")
    return module.CryptoRFTradingSystem
