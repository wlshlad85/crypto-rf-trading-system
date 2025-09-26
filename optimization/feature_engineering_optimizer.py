"""Optimized feature engineering with vectorized operations and parallel processing."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from numba import jit, prange
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
warnings.filterwarnings('ignore')


class OptimizedFeatureEngine:
    """Highly optimized feature engineering using vectorization and parallel processing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for parallel feature generation
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Pre-compute common values
        self._precomputed = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'enable_all_features': True,
            'momentum_periods': np.array([5, 10, 20, 50]),
            'volatility_periods': np.array([10, 20, 50]),
            'volume_periods': np.array([5, 10, 20]),
            'ma_periods': np.array([5, 10, 20, 50, 100, 200]),
            'ema_periods': np.array([12, 26, 50]),
            'parallel_processing': True
        }
    
    def generate_features_optimized(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Generate features using optimized vectorized operations."""
        self.logger.debug(f"Generating optimized features for {symbol or 'unknown symbol'}")
        
        # Don't copy unless necessary - work with views when possible
        df = data
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns")
        
        # Pre-compute commonly used values
        self._precompute_values(df)
        
        if self.config['parallel_processing']:
            # Generate features in parallel
            df = self._generate_features_parallel(df)
        else:
            # Sequential generation
            df = self._generate_features_sequential(df)
        
        # Clean features efficiently
        df = self._clean_features_vectorized(df)
        
        self.logger.debug(f"Feature generation complete: {df.shape}")
        return df
    
    def _precompute_values(self, df: pd.DataFrame):
        """Pre-compute commonly used values for efficiency."""
        self._precomputed['close'] = df['close'].values
        self._precomputed['high'] = df['high'].values
        self._precomputed['low'] = df['low'].values
        self._precomputed['open'] = df['open'].values
        self._precomputed['volume'] = df['volume'].values
        self._precomputed['returns'] = np.diff(self._precomputed['close']) / self._precomputed['close'][:-1]
        self._precomputed['returns'] = np.concatenate([[np.nan], self._precomputed['returns']])
    
    def _generate_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features using parallel processing."""
        # Define feature generation tasks
        tasks = [
            (self._add_price_features_vectorized, df),
            (self._add_technical_indicators_vectorized, df),
            (self._add_momentum_features_vectorized, df),
            (self._add_volatility_features_vectorized, df),
            (self._add_volume_features_vectorized, df),
            (self._add_pattern_features_vectorized, df)
        ]
        
        # Execute tasks in parallel
        futures = []
        for func, data in tasks:
            future = self.executor.submit(func, data)
            futures.append(future)
        
        # Combine results
        feature_dfs = []
        for future in futures:
            feature_df = future.result()
            feature_dfs.append(feature_df)
        
        # Merge all features
        for feature_df in feature_dfs[1:]:
            # Only add new columns
            new_cols = [col for col in feature_df.columns if col not in df.columns]
            df[new_cols] = feature_df[new_cols]
        
        return df
    
    def _generate_features_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features sequentially."""
        df = self._add_price_features_vectorized(df)
        df = self._add_technical_indicators_vectorized(df)
        df = self._add_momentum_features_vectorized(df)
        df = self._add_volatility_features_vectorized(df)
        df = self._add_volume_features_vectorized(df)
        df = self._add_pattern_features_vectorized(df)
        return df
    
    def _add_price_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price features using vectorized operations."""
        close = self._precomputed['close']
        high = self._precomputed['high']
        low = self._precomputed['low']
        open_price = self._precomputed['open']
        
        # Vectorized price features
        df['price_range'] = (high - low) / close
        df['price_gap'] = np.concatenate([[np.nan], (open_price[1:] - close[:-1]) / close[:-1]])
        df['body_size'] = np.abs(close - open_price) / close
        df['upper_shadow'] = (high - np.maximum(open_price, close)) / close
        df['lower_shadow'] = (np.minimum(open_price, close) - low) / close
        
        # Vectorized returns calculation
        for period in [1, 3, 6, 12, 24]:
            df[f'return_{period}h'] = self._calculate_returns_vectorized(close, period)
            df[f'log_return_{period}h'] = self._calculate_log_returns_vectorized(close, period)
        
        # Cumulative returns
        df['return_cumulative_5d'] = self._calculate_returns_vectorized(close, 120)
        df['return_cumulative_30d'] = self._calculate_returns_vectorized(close, 720)
        
        return df
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_returns_vectorized(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate returns using Numba for speed."""
        n = len(prices)
        returns = np.full(n, np.nan)
        
        for i in prange(period, n):
            if prices[i-period] != 0:
                returns[i] = (prices[i] - prices[i-period]) / prices[i-period]
        
        return returns
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_log_returns_vectorized(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate log returns using Numba."""
        n = len(prices)
        returns = np.full(n, np.nan)
        
        for i in prange(period, n):
            if prices[i-period] > 0 and prices[i] > 0:
                returns[i] = np.log(prices[i] / prices[i-period])
        
        return returns
    
    def _add_technical_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using vectorized operations."""
        close = self._precomputed['close']
        high = self._precomputed['high']
        low = self._precomputed['low']
        
        # Moving averages - vectorized using rolling
        for period in self.config['ma_periods']:
            df[f'ma_{period}'] = pd.Series(close).rolling(period, min_periods=1).mean()
            df[f'ma_{period}_ratio'] = close / df[f'ma_{period}'].values
        
        # EMA - vectorized
        for period in self.config['ema_periods']:
            df[f'ema_{period}'] = pd.Series(close).ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_ratio'] = close / df[f'ema_{period}'].values
        
        # Bollinger Bands - vectorized
        for period in [20, 50]:
            ma = pd.Series(close).rolling(period, min_periods=1).mean()
            std = pd.Series(close).rolling(period, min_periods=1).std()
            df[f'bb_upper_{period}'] = ma + (2 * std)
            df[f'bb_lower_{period}'] = ma - (2 * std)
            
            # Avoid division by zero
            bb_range = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
            df[f'bb_position_{period}'] = np.where(bb_range > 0, 
                                                   (close - df[f'bb_lower_{period}']) / bb_range, 
                                                   0.5)
            df[f'bb_width_{period}'] = np.where(ma > 0, bb_range / ma, 0)
        
        # RSI - vectorized
        returns = self._precomputed['returns']
        for period in [14, 30]:
            df[f'rsi_{period}'] = self._calculate_rsi_vectorized(returns, period)
        
        # MACD - vectorized
        ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ATR - vectorized
        for period in [14, 21]:
            df[f'atr_{period}'] = self._calculate_atr_vectorized(high, low, close, period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / close
        
        return df
    
    @staticmethod
    def _calculate_rsi_vectorized(returns: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI using vectorized operations."""
        # Separate gains and losses
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        # Calculate average gains and losses
        avg_gains = pd.Series(gains).rolling(period, min_periods=1).mean()
        avg_losses = pd.Series(losses).rolling(period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr_vectorized(high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR using Numba."""
        n = len(high)
        tr = np.full(n, np.nan)
        
        # Calculate True Range
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR
        atr = np.full(n, np.nan)
        for i in range(period, n):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        return atr
    
    def _add_momentum_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features using vectorized operations."""
        close = self._precomputed['close']
        high = self._precomputed['high']
        low = self._precomputed['low']
        
        # ROC - vectorized
        for period in self.config['momentum_periods']:
            df[f'roc_{period}'] = self._calculate_returns_vectorized(close, period)
        
        # Momentum - vectorized
        for period in [10, 20]:
            df[f'momentum_{period}'] = close / np.roll(close, period)
        
        # Williams %R - vectorized
        for period in [14, 21]:
            high_max = pd.Series(high).rolling(period, min_periods=1).max()
            low_min = pd.Series(low).rolling(period, min_periods=1).min()
            
            range_hl = high_max - low_min
            df[f'williams_r_{period}'] = np.where(range_hl > 0,
                                                  -100 * (high_max - close) / range_hl,
                                                  -50)
        
        # Price acceleration - vectorized
        df['price_acceleration'] = np.gradient(np.gradient(close))
        
        return df
    
    def _add_volatility_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features using vectorized operations."""
        returns = self._precomputed['returns']
        close = self._precomputed['close']
        high = self._precomputed['high']
        low = self._precomputed['low']
        open_price = self._precomputed['open']
        
        # Rolling volatility - vectorized
        for period in self.config['volatility_periods']:
            vol = pd.Series(returns).rolling(period, min_periods=1).std() * np.sqrt(24)
            df[f'volatility_{period}'] = vol
            
            # Volatility ratio
            vol_ma = vol.rolling(period*2, min_periods=1).mean()
            df[f'volatility_ratio_{period}'] = vol / (vol_ma + 1e-10)
        
        # Garman-Klass volatility - vectorized
        for period in [20, 50]:
            ln_hl = np.log(high / (low + 1e-10))
            ln_co = np.log(close / (open_price + 1e-10))
            gk_vol = ln_hl * (ln_hl - ln_co) + ln_co**2
            df[f'gk_volatility_{period}'] = pd.Series(gk_vol).rolling(period, min_periods=1).mean()
        
        # Volatility of volatility
        if 'volatility_20' in df.columns:
            df['vol_of_vol'] = df['volatility_20'].rolling(20, min_periods=1).std()
        
        # Realized volatility
        df['realized_vol_5d'] = pd.Series(returns).rolling(120, min_periods=1).std() * np.sqrt(24 * 365)
        df['realized_vol_30d'] = pd.Series(returns).rolling(720, min_periods=1).std() * np.sqrt(24 * 365)
        
        return df
    
    def _add_volume_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features using vectorized operations."""
        volume = self._precomputed['volume']
        close = self._precomputed['close']
        high = self._precomputed['high']
        low = self._precomputed['low']
        
        # Volume moving averages - vectorized
        for period in self.config['volume_periods']:
            vol_ma = pd.Series(volume).rolling(period, min_periods=1).mean()
            df[f'volume_ma_{period}'] = vol_ma
            df[f'volume_ratio_{period}'] = volume / (vol_ma + 1e-10)
        
        # VWAP - vectorized
        typical_price = (high + low + close) / 3
        for period in [20, 50]:
            vwap_num = (typical_price * volume)
            vwap_rolling_sum = pd.Series(vwap_num).rolling(period, min_periods=1).sum()
            vol_rolling_sum = pd.Series(volume).rolling(period, min_periods=1).sum()
            vwap = vwap_rolling_sum / (vol_rolling_sum + 1e-10)
            df[f'vwap_{period}'] = vwap
            df[f'vwap_ratio_{period}'] = close / (vwap + 1e-10)
        
        # OBV - vectorized
        price_direction = np.sign(np.diff(close))
        price_direction = np.concatenate([[0], price_direction])
        obv = np.cumsum(price_direction * volume)
        df['obv'] = obv
        df['obv_ma_20'] = pd.Series(obv).rolling(20, min_periods=1).mean()
        
        # Volume ROC - vectorized
        for period in [5, 10]:
            df[f'volume_roc_{period}'] = self._calculate_returns_vectorized(volume, period)
        
        return df
    
    def _add_pattern_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern features using vectorized operations."""
        close = self._precomputed['close']
        high = self._precomputed['high']
        low = self._precomputed['low']
        
        # Support and resistance - vectorized
        for period in [20, 50]:
            resistance = pd.Series(high).rolling(period, min_periods=1).max()
            support = pd.Series(low).rolling(period, min_periods=1).min()
            
            df[f'resistance_{period}'] = resistance
            df[f'support_{period}'] = support
            df[f'resistance_distance_{period}'] = (resistance - close) / (close + 1e-10)
            df[f'support_distance_{period}'] = (close - support) / (close + 1e-10)
            
            # Channel position
            channel_range = resistance - support
            df[f'channel_position_{period}'] = np.where(channel_range > 0,
                                                        (close - support) / channel_range,
                                                        0.5)
        
        # Fractal patterns - vectorized
        high_shifted_back = np.roll(high, 1)
        high_shifted_forward = np.roll(high, -1)
        low_shifted_back = np.roll(low, 1)
        low_shifted_forward = np.roll(low, -1)
        
        df['fractal_high'] = ((high > high_shifted_back) & (high > high_shifted_forward)).astype(int)
        df['fractal_low'] = ((low < low_shifted_back) & (low < low_shifted_forward)).astype(int)
        
        # Trend strength - vectorized
        for period in [10, 20]:
            price_changes = np.diff(close)
            price_changes = np.concatenate([[0], price_changes])
            
            up_moves = np.where(price_changes > 0, price_changes, 0)
            down_moves = np.where(price_changes < 0, -price_changes, 0)
            
            up_sum = pd.Series(up_moves).rolling(period, min_periods=1).sum()
            down_sum = pd.Series(down_moves).rolling(period, min_periods=1).sum()
            
            total_moves = up_sum + down_sum
            df[f'trend_strength_{period}'] = np.where(total_moves > 0,
                                                      (up_sum - down_sum) / total_moves,
                                                      0)
        
        return df
    
    def _clean_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features using vectorized operations."""
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinities with NaN in one operation
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Calculate bounds for all columns at once
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std()
        
        # Cap extreme values vectorized
        for col in numeric_cols:
            if stds[col] > 0:  # Avoid division by zero
                upper_bound = means[col] + 5 * stds[col]
                lower_bound = means[col] - 5 * stds[col]
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Forward fill with limit
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill', limit=5)
        
        # Fill remaining NaN with 0
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)