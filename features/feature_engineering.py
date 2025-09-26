"""Comprehensive feature engineering for cryptocurrency trading."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

from utils.config import FeatureConfig


class CryptoFeatureEngine:
    """Advanced feature engineering for cryptocurrency trading."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        self.feature_importance = {}
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all features for the cryptocurrency data."""
        self.logger.info("Starting feature generation")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Get list of symbols from column names
        symbols = self._extract_symbols(df.columns)
        
        # Generate features for each symbol
        total_symbols = len(symbols)
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Generating features for {symbol} ({i}/{total_symbols})")
            
            # Technical indicators
            df = self._add_technical_indicators(df, symbol)
            
            # Price and return features
            df = self._add_price_features(df, symbol)
            
            # Volume features
            df = self._add_volume_features(df, symbol)
            
            # Volatility features
            df = self._add_volatility_features(df, symbol)
            
            # Momentum features
            df = self._add_momentum_features(df, symbol)
        
        # Cross-asset features
        df = self._add_cross_asset_features(df, symbols)
        
        # Temporal features
        df = self._add_temporal_features(df)
        
        # Market regime features
        df = self._add_market_regime_features(df, symbols)
        
        # Clean up features
        df = self._clean_features(df)
        
        self.logger.info(f"Generated {len(df.columns)} features")
        
        return df
    
    def _extract_symbols(self, columns: List[str]) -> List[str]:
        """Extract unique symbols from column names."""
        symbols = set()
        for col in columns:
            if '_' in col:
                symbol = col.split('_')[0]
                symbols.add(symbol)
        return sorted(list(symbols))
    
    def _add_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add technical indicators for a symbol."""
        # Price columns
        close_col = f"{symbol}_close"
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        volume_col = f"{symbol}_volume"
        
        if close_col not in df.columns:
            return df
        
        close = df[close_col]
        high = df[high_col] if high_col in df.columns else close
        low = df[low_col] if low_col in df.columns else close
        volume = df[volume_col] if volume_col in df.columns else pd.Series(index=df.index, data=0)
        
        # RSI
        df[f"{symbol}_rsi"] = self._calculate_rsi(close, self.config.rsi_period)
        
        # MACD
        macd, signal, histogram = self._calculate_macd(
            close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        df[f"{symbol}_macd"] = macd
        df[f"{symbol}_macd_signal"] = signal
        df[f"{symbol}_macd_histogram"] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            close, self.config.bb_period, self.config.bb_std
        )
        df[f"{symbol}_bb_upper"] = bb_upper
        df[f"{symbol}_bb_middle"] = bb_middle
        df[f"{symbol}_bb_lower"] = bb_lower
        df[f"{symbol}_bb_width"] = (bb_upper - bb_lower) / bb_middle
        df[f"{symbol}_bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df[f"{symbol}_atr"] = self._calculate_atr(high, low, close, self.config.atr_period)
        
        # Moving Averages
        for period in self.config.ema_periods:
            df[f"{symbol}_ema_{period}"] = close.ewm(span=period).mean()
        
        for period in self.config.sma_periods:
            df[f"{symbol}_sma_{period}"] = close.rolling(window=period).mean()
        
        # Stochastic Oscillator
        df[f"{symbol}_stoch_k"], df[f"{symbol}_stoch_d"] = self._calculate_stochastic(
            high, low, close, 14, 3
        )
        
        # Williams %R
        df[f"{symbol}_williams_r"] = self._calculate_williams_r(high, low, close, 14)
        
        # Commodity Channel Index
        df[f"{symbol}_cci"] = self._calculate_cci(high, low, close, 20)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add price-based features."""
        close_col = f"{symbol}_close"
        
        if close_col not in df.columns:
            return df
        
        close = df[close_col]
        
        # Returns
        for period in self.config.return_periods:
            df[f"{symbol}_return_{period}h"] = close.pct_change(period)
            df[f"{symbol}_log_return_{period}h"] = np.log(close / close.shift(period))
        
        # Price ratios
        df[f"{symbol}_price_ratio_20"] = close / close.rolling(20).mean()
        df[f"{symbol}_price_ratio_50"] = close / close.rolling(50).mean()
        
        # Price momentum
        df[f"{symbol}_momentum_12"] = close / close.shift(12)
        df[f"{symbol}_momentum_24"] = close / close.shift(24)
        
        # Price acceleration
        df[f"{symbol}_acceleration_6"] = (close - close.shift(6)) - (close.shift(6) - close.shift(12))
        
        # Support and resistance levels
        df[f"{symbol}_support_20"] = close.rolling(20).min()
        df[f"{symbol}_resistance_20"] = close.rolling(20).max()
        df[f"{symbol}_support_distance"] = (close - df[f"{symbol}_support_20"]) / close
        df[f"{symbol}_resistance_distance"] = (df[f"{symbol}_resistance_20"] - close) / close
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add volume-based features."""
        close_col = f"{symbol}_close"
        volume_col = f"{symbol}_volume"
        
        if close_col not in df.columns or volume_col not in df.columns:
            return df
        
        close = df[close_col]
        volume = df[volume_col]
        
        # Volume moving averages
        df[f"{symbol}_volume_sma"] = volume.rolling(self.config.volume_sma_period).mean()
        df[f"{symbol}_volume_ratio"] = volume / df[f"{symbol}_volume_sma"]
        
        # Volume-weighted average price
        df[f"{symbol}_vwap"] = (close * volume).cumsum() / volume.cumsum()
        
        # On-balance volume
        df[f"{symbol}_obv"] = self._calculate_obv(close, volume)
        
        # Volume rate of change
        df[f"{symbol}_volume_roc"] = volume.pct_change(12)
        
        # Price-volume trend
        df[f"{symbol}_pvt"] = self._calculate_pvt(close, volume)
        
        # Volume-price confirmation
        returns = close.pct_change()
        df[f"{symbol}_volume_price_confirm"] = (
            (returns > 0) & (volume > df[f"{symbol}_volume_sma"])
        ).astype(int) - (
            (returns < 0) & (volume > df[f"{symbol}_volume_sma"])
        ).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add volatility-based features."""
        close_col = f"{symbol}_close"
        
        if close_col not in df.columns:
            return df
        
        close = df[close_col]
        returns = close.pct_change()
        
        # Rolling volatility
        for period in self.config.volatility_periods:
            df[f"{symbol}_volatility_{period}h"] = returns.rolling(period).std()
        
        # Volatility ratios
        df[f"{symbol}_vol_ratio_short_long"] = (
            df[f"{symbol}_volatility_24h"] / df[f"{symbol}_volatility_168h"]
        )
        
        # Volatility momentum
        df[f"{symbol}_vol_momentum"] = (
            df[f"{symbol}_volatility_24h"] / df[f"{symbol}_volatility_24h"].shift(24)
        )
        
        # Volatility rank
        df[f"{symbol}_vol_rank"] = (
            df[f"{symbol}_volatility_24h"].rolling(720).rank(pct=True)
        )
        
        # Garman-Klass volatility (if OHLC available)
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        open_col = f"{symbol}_open"
        
        if all(col in df.columns for col in [high_col, low_col, open_col]):
            high = df[high_col]
            low = df[low_col]
            open_price = df[open_col]
            
            df[f"{symbol}_gk_volatility"] = np.sqrt(
                np.log(high / close) * np.log(high / open_price) +
                np.log(low / close) * np.log(low / open_price)
            )
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add momentum-based features."""
        close_col = f"{symbol}_close"
        
        if close_col not in df.columns:
            return df
        
        close = df[close_col]
        
        # Rate of change
        df[f"{symbol}_roc_12"] = (close - close.shift(12)) / close.shift(12)
        df[f"{symbol}_roc_24"] = (close - close.shift(24)) / close.shift(24)
        
        # Momentum oscillator
        df[f"{symbol}_momentum_osc"] = close - close.shift(12)
        
        # Trend strength
        df[f"{symbol}_trend_strength"] = (
            (close > close.shift(1)).rolling(20).sum() / 20
        )
        
        # Directional movement
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        
        if high_col in df.columns and low_col in df.columns:
            high = df[high_col]
            low = df[low_col]
            
            df[f"{symbol}_adx"] = self._calculate_adx(high, low, close, 14)
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Add cross-asset features."""
        if len(symbols) < 2:
            return df
        
        # Get close prices for all symbols
        close_prices = {}
        for symbol in symbols:
            close_col = f"{symbol}_close"
            if close_col in df.columns:
                close_prices[symbol] = df[close_col]
        
        if len(close_prices) < 2:
            return df
        
        # Correlation features
        for i, symbol1 in enumerate(symbols):
            if symbol1 not in close_prices:
                continue
                
            for j, symbol2 in enumerate(symbols):
                if j <= i or symbol2 not in close_prices:
                    continue
                
                # Rolling correlation
                correlation = close_prices[symbol1].rolling(
                    self.config.correlation_period
                ).corr(close_prices[symbol2])
                
                df[f"{symbol1}_{symbol2}_correlation"] = correlation
        
        # Market dominance
        total_market_cap = pd.Series(index=df.index, data=0)
        for symbol in symbols:
            market_cap_col = f"{symbol}_market_cap"
            if market_cap_col in df.columns:
                total_market_cap += df[market_cap_col].fillna(0)
        
        for symbol in symbols:
            market_cap_col = f"{symbol}_market_cap"
            if market_cap_col in df.columns:
                df[f"{symbol}_market_dominance"] = (
                    df[market_cap_col] / total_market_cap
                )
        
        # Relative strength
        btc_close = close_prices.get('bitcoin')
        if btc_close is not None:
            for symbol in symbols:
                if symbol != 'bitcoin' and symbol in close_prices:
                    df[f"{symbol}_btc_relative_strength"] = (
                        close_prices[symbol] / btc_close
                    )
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        # Hour of day
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Add market regime features."""
        # Use Bitcoin as market proxy
        btc_close_col = 'bitcoin_close'
        
        if btc_close_col not in df.columns:
            return df
        
        btc_close = df[btc_close_col]
        btc_returns = btc_close.pct_change()
        
        # Bull/bear market indicator
        df['market_regime'] = (
            btc_close > btc_close.rolling(200).mean()
        ).astype(int)
        
        # Market volatility regime
        vol_20 = btc_returns.rolling(20).std()
        vol_200 = btc_returns.rolling(200).std()
        df['volatility_regime'] = (vol_20 > vol_200).astype(int)
        
        # Trend strength
        df['market_trend_strength'] = (
            (btc_close > btc_close.shift(1)).rolling(20).sum() / 20
        )
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with too many NaN values
        threshold = len(df) * 0.8  # At least 80% of data must be present
        df = df.dropna(thresh=threshold, axis=1)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        return df
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        # Vectorized mean absolute deviation around rolling mean
        mad = (tp - sma_tp).abs().rolling(window=period).mean()
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        close_diff = close.diff()
        flow = np.where(close_diff > 0, volume, np.where(close_diff < 0, -volume, 0))
        obv = pd.Series(flow, index=close.index)
        if len(obv) > 0:
            obv.iloc[0] = volume.iloc[0]
        return obv.cumsum()
    
    def _calculate_pvt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Price-Volume Trend."""
        increment = (volume * close.pct_change()).fillna(0)
        base = volume.iloc[0] if len(volume) > 0 else 0.0
        return increment.cumsum() + base
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        # Calculate True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        # Calculate directional movements (vectorized)
        up_move = high.diff()
        down_move = low.shift(1) - low
        dm_plus = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
        dm_minus = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
        
        # Calculate directional indicators
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names
    
    def select_features(self, df: pd.DataFrame, target: pd.Series, method: str = 'importance') -> pd.DataFrame:
        """Select the most important features."""
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        if method == 'importance':
            # Use Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Clean data for fitting
            clean_df = df.dropna()
            clean_target = target.loc[clean_df.index].dropna()
            
            # Align indices
            common_index = clean_df.index.intersection(clean_target.index)
            clean_df = clean_df.loc[common_index]
            clean_target = clean_target.loc[common_index]
            
            rf.fit(clean_df, clean_target)
            
            # Get feature importance
            importance = pd.Series(rf.feature_importances_, index=clean_df.columns)
            
            # Select top features
            top_features = importance.nlargest(self.config.max_features).index
            
            self.feature_importance = importance.to_dict()
            
            return df[top_features]
        
        elif method == 'statistical':
            # Use statistical feature selection
            selector = SelectKBest(score_func=f_regression, k=self.config.max_features)
            
            # Clean data for fitting
            clean_df = df.dropna()
            clean_target = target.loc[clean_df.index].dropna()
            
            # Align indices
            common_index = clean_df.index.intersection(clean_target.index)
            clean_df = clean_df.loc[common_index]
            clean_target = clean_target.loc[common_index]
            
            selector.fit(clean_df, clean_target)
            
            # Get selected features
            selected_features = clean_df.columns[selector.get_support()]
            
            return df[selected_features]
        
        else:
            return df