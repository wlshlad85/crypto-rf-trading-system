"""Ultra Feature Engineering: Enhanced feature generation for UltraThink system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class UltraFeatureEngine:
    """Enhanced feature engineering with advanced technical indicators."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'enable_all_features': True,
            'momentum_periods': [5, 10, 20, 50],
            'volatility_periods': [10, 20, 50],
            'volume_periods': [5, 10, 20],
            'cross_asset_features': True,
            'regime_features': True,
            'microstructure_features': True
        }
    
    def generate_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Generate comprehensive features from market data."""
        
        self.logger.debug(f"Generating features for {symbol or 'unknown symbol'}")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Generate all feature categories
        df = self._add_price_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_pattern_features(df)
        df = self._add_regime_features(df)
        df = self._add_microstructure_features(df)
        
        # Remove any infinite or extremely large values
        df = self._clean_features(df)
        
        self.logger.debug(f"Feature generation complete: {df.shape}")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        
        # Basic price features
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Returns at multiple horizons
        for period in [1, 3, 6, 12, 24]:
            df[f'return_{period}h'] = df['close'].pct_change(period)
            df[f'log_return_{period}h'] = np.log(df['close'] / df['close'].shift(period))
        
        # Cumulative returns
        df['return_cumulative_5d'] = (df['close'] / df['close'].shift(120) - 1)  # 5 days * 24h
        df['return_cumulative_30d'] = (df['close'] / df['close'].shift(720) - 1)  # 30 days * 24h
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']
        
        # Exponential moving averages
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # Bollinger Bands
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = ma + (2 * std)
            df[f'bb_lower_{period}'] = ma - (2 * std)
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
        
        # RSI
        for period in [14, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
        
        # Average True Range (ATR)
        for period in [14, 21]:
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        
        # Rate of Change (ROC)
        for period in self.config['momentum_periods']:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
        
        # Williams %R
        for period in [14, 21]:
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        for period in [20, 50]:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            ma_tp = typical_price.rolling(period).mean()
            mad = (typical_price - ma_tp).abs().rolling(period).mean()
            df[f'cci_{period}'] = (typical_price - ma_tp) / (0.015 * mad)
        
        # Price acceleration
        df['price_acceleration'] = df['close'].diff().diff()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        
        # Rolling volatility (standard deviation of returns)
        returns = df['close'].pct_change()
        for period in self.config['volatility_periods']:
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(24)  # Annualized hourly vol
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(period*2).mean()
        
        # Garman-Klass volatility estimator
        for period in [20, 50]:
            ln_hl = np.log(df['high'] / df['low'])
            ln_co = np.log(df['close'] / df['open'])
            gk_vol = ln_hl * (ln_hl - ln_co) + ln_co**2
            df[f'gk_volatility_{period}'] = gk_vol.rolling(period).mean()
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20'].rolling(20).std()
        
        # Realized volatility
        df['realized_vol_5d'] = returns.rolling(120).std() * np.sqrt(24 * 365)  # 5 days
        df['realized_vol_30d'] = returns.rolling(720).std() * np.sqrt(24 * 365)  # 30 days
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        
        # Volume moving averages
        for period in self.config['volume_periods']:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # Volume-weighted average price (VWAP)
        for period in [20, 50]:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_num = (typical_price * df['volume']).rolling(period).sum()
            vwap_den = df['volume'].rolling(period).sum()
            df[f'vwap_{period}'] = vwap_num / vwap_den
            df[f'vwap_ratio_{period}'] = df['close'] / df[f'vwap_{period}']
        
        # On-Balance Volume (OBV)
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        df['obv'] = obv.cumsum()
        df['obv_ma_20'] = df['obv'].rolling(20).mean()
        
        # Volume Rate of Change
        for period in [5, 10]:
            df[f'volume_roc_{period}'] = df['volume'].pct_change(period)
        
        # Chaikin Money Flow
        for period in [20, 50]:
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df[f'cmf_{period}'] = money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition features."""
        
        # Support and resistance levels
        for period in [20, 50]:
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'resistance_distance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
            df[f'support_distance_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']
        
        # Fractal patterns (simplified)
        df['fractal_high'] = ((df['high'] > df['high'].shift(1)) & 
                             (df['high'] > df['high'].shift(-1))).astype(int)
        df['fractal_low'] = ((df['low'] < df['low'].shift(1)) & 
                            (df['low'] < df['low'].shift(-1))).astype(int)
        
        # Price channels
        for period in [20, 50]:
            df[f'channel_position_{period}'] = ((df['close'] - df[f'support_{period}']) / 
                                               (df[f'resistance_{period}'] - df[f'support_{period}']))
        
        # Trend strength
        for period in [10, 20]:
            price_changes = df['close'].diff()
            up_moves = price_changes.where(price_changes > 0, 0).rolling(period).sum()
            down_moves = abs(price_changes.where(price_changes < 0, 0)).rolling(period).sum()
            df[f'trend_strength_{period}'] = (up_moves - down_moves) / (up_moves + down_moves)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        
        if not self.config['regime_features']:
            return df
        
        returns = df['close'].pct_change()
        
        # Regime indicators
        for period in [50, 100]:
            # Trend regime
            trend_strength = returns.rolling(period).mean() / returns.rolling(period).std()
            df[f'trend_regime_{period}'] = np.where(trend_strength > 0.5, 1, 
                                                   np.where(trend_strength < -0.5, -1, 0))
            
            # Volatility regime
            current_vol = returns.rolling(20).std()
            historical_vol = returns.rolling(period).std()
            vol_ratio = current_vol / historical_vol
            df[f'vol_regime_{period}'] = np.where(vol_ratio > 1.5, 1, 
                                                 np.where(vol_ratio < 0.7, -1, 0))
        
        # Market state indicators
        df['bull_bear_indicator'] = np.where(df['close'] > df['ma_50'], 1, -1)
        df['volatility_state'] = np.where(df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8), 1,
                                         np.where(df['volatility_20'] < df['volatility_20'].rolling(50).quantile(0.2), -1, 0))
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        
        if not self.config['microstructure_features']:
            return df
        
        # Bid-ask spread proxy (using high-low range)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_ma_20'] = df['spread_proxy'].rolling(20).mean()
        
        # Price efficiency measures
        returns = df['close'].pct_change()
        
        # Auto-correlation of returns (momentum vs mean reversion) - vectorized
        for lag in [1, 5]:
            df[f'return_autocorr_{lag}'] = returns.rolling(50).corr(returns.shift(lag))
        
        # Variance ratio test (random walk hypothesis)
        def _variance_ratio_window(arr: np.ndarray, k: int) -> float:
            # Expect raw numpy array from rolling(..., raw=True)
            if arr.size < k * 10:
                return np.nan
            var_1 = np.var(arr)
            if var_1 == 0:
                return np.nan
            # Compute k-period summed returns via sliding window sum
            # Use convolution for efficiency
            kernel = np.ones(k, dtype=np.float64)
            k_returns = np.convolve(arr, kernel, mode='valid')
            var_k = np.var(k_returns) / k
            return float(var_k / var_1)

        df['variance_ratio_2'] = returns.rolling(100).apply(lambda x: _variance_ratio_window(x, 2), raw=True)
        df['variance_ratio_5'] = returns.rolling(100).apply(lambda x: _variance_ratio_window(x, 5), raw=True)
        
        # Round number effects
        df['round_number'] = (df['close'] % 1000 == 0).astype(int)  # For BTC-like prices
        df['near_round_number'] = (abs(df['close'] % 1000) < 50).astype(int)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling infinities and extreme values."""
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values (beyond 5 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].std() > 0:  # Avoid division by zero
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Cap at 5 standard deviations
                upper_bound = mean_val + 5 * std_val
                lower_bound = mean_val - 5 * std_val
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Forward fill remaining NaN values (limited)
        df = df.fillna(method='ffill', limit=5)
        
        return df