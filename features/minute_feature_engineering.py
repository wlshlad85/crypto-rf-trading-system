"""Specialized feature engineering for minute-level cryptocurrency data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .ultra_feature_engineering import UltraFeatureEngine


class MinuteFeatureEngine:
    """Enhanced feature engineering optimized for minute-level trading data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Use the ultra feature engine as base
        self.base_engine = UltraFeatureEngine(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for minute data."""
        return {
            'enable_all_features': True,
            'high_frequency_features': True,
            'microstructure_features': True,
            'intraday_seasonality': True,
            'volume_profile': True,
            'order_flow_features': True,
            
            # Minute-specific periods
            'momentum_periods_minutes': [1, 2, 5, 10, 15, 30, 60],
            'volatility_periods_minutes': [5, 10, 15, 30, 60, 120],
            'volume_periods_minutes': [5, 10, 15, 30, 60],
            
            # Intraday periods  
            'intraday_periods_minutes': [60, 120, 240, 480],  # 1h, 2h, 4h, 8h
            
            # Market microstructure
            'tick_features': True,
            'spread_features': True,
            'market_impact_features': True,
            
            # Performance optimizations
            'vectorized_operations': True,
            'memory_efficient': True,
            'parallel_processing': False  # Can be enabled for very large datasets
        }
    
    def generate_minute_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Generate comprehensive features optimized for minute-level data."""
        
        self.logger.info(f"Generating minute-level features for {symbol or 'unknown symbol'}")
        start_time = datetime.now()
        
        # Start with base feature engineering
        df = self.base_engine.generate_features(data, symbol)
        
        # Add minute-specific features
        df = self._add_high_frequency_features(df)
        df = self._add_microstructure_features(df)
        df = self._add_intraday_seasonality_features(df)
        df = self._add_volume_profile_features(df)
        df = self._add_order_flow_approximation_features(df)
        df = self._add_market_regime_minute_features(df)
        df = self._add_cross_timeframe_features(df)
        df = self._add_minute_momentum_features(df)
        df = self._add_volatility_clustering_features(df)
        
        # Final cleaning and optimization
        df = self._optimize_memory_usage(df)
        df = self._final_feature_cleaning(df)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Minute feature generation completed in {duration:.2f}s: {df.shape}")
        
        return df
    
    def _add_high_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to high-frequency minute data."""
        
        if not self.config['high_frequency_features']:
            return df
        
        # Tick-by-tick approximations using OHLC
        df['true_range_pct'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ]) / df['close']
        
        # High-frequency price movements
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['close'] / df['open']
        df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Minute-level price gaps
        df['price_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_filled'] = ((df['price_gap_pct'] > 0) & (df['low'] <= df['close'].shift(1))).astype(int)
        
        # Intraday price position
        df['price_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['price_position_in_range'] = df['price_position_in_range'].fillna(0.5)
        
        # High-frequency momentum (1-5 minute windows)
        for period in [1, 2, 3, 5]:
            df[f'hf_momentum_{period}m'] = df['close'].pct_change(period)
            df[f'hf_volume_momentum_{period}m'] = df['volume'].pct_change(period)
            
            # Rolling correlation between price and volume
            df[f'pv_corr_{period}m'] = df['close'].rolling(period).corr(df['volume'])
        
        # Minute-level volatility
        for period in [5, 10, 15]:
            returns = df['close'].pct_change()
            df[f'realized_vol_{period}m'] = returns.rolling(period).std() * np.sqrt(period)
            df[f'vol_skew_{period}m'] = returns.rolling(period).skew()
            df[f'vol_kurt_{period}m'] = returns.rolling(period).kurt()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features for minute data."""
        
        if not self.config['microstructure_features']:
            return df
        
        # Bid-ask spread proxies using OHLC
        df['spread_proxy_hl'] = (df['high'] - df['low']) / df['close']
        df['spread_proxy_oc'] = abs(df['open'] - df['close']) / df['close']
        
        # Rolling spread metrics
        for period in [5, 10, 15]:
            df[f'avg_spread_{period}m'] = df['spread_proxy_hl'].rolling(period).mean()
            df[f'spread_volatility_{period}m'] = df['spread_proxy_hl'].rolling(period).std()
        
        # Price impact approximations
        df['price_impact_proxy'] = abs(df['close'].pct_change()) / (df['volume'] + 1)
        df['volume_weighted_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Market depth approximation (using volume)
        for period in [5, 10, 15]:
            df[f'market_depth_{period}m'] = df['volume'].rolling(period).sum()
            df[f'volume_imbalance_{period}m'] = (
                df['volume'].rolling(period).sum() / 
                df['volume'].rolling(period * 2).sum()
            )
        
        # Liquidity measures
        df['volume_per_price_move'] = df['volume'] / (abs(df['close'].pct_change()) + 1e-8)
        df['turnover_rate'] = df['volume'] / df['volume'].rolling(60).mean()  # vs 1-hour average
        
        # Quote instability (using price volatility as proxy)
        for period in [5, 10]:
            price_changes = df['close'].diff()
            nonzero = (price_changes != 0).astype(int)
            df[f'quote_instability_{period}m'] = nonzero.rolling(period).sum() / period
        
        # Effective spread estimation
        df['effective_spread_proxy'] = 2 * abs(
            df['close'] - (df['high'] + df['low']) / 2
        ) / df['close']
        
        return df
    
    def _add_intraday_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday seasonality and time-based features."""
        
        if not self.config['intraday_seasonality']:
            return df
        
        # Time-based features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Intraday session features (crypto trades 24/7 but still has patterns)
        # Asian session: 0-8 UTC, London: 8-16 UTC, New York: 16-24 UTC
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        # Session overlap periods (higher volatility)
        df['london_ny_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        df['asian_london_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
        
        # Time since market events (simplified for crypto)
        df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']
        df['sin_time'] = np.sin(2 * np.pi * df['minutes_since_midnight'] / (24 * 60))
        df['cos_time'] = np.cos(2 * np.pi * df['minutes_since_midnight'] / (24 * 60))
        
        # Weekly seasonality
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Volatility patterns by time of day
        for period in [60, 240]:  # 1h, 4h windows
            vol_by_time = df.groupby(['hour'])['close'].pct_change().rolling(period).std()
            df[f'time_vol_pattern_{period}m'] = df['hour'].map(
                vol_by_time.groupby('hour').mean()
            )
        
        return df
    
    def _add_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume profile and volume-price analysis features."""
        
        if not self.config['volume_profile']:
            return df
        
        # Volume-weighted indicators
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        for period in [15, 30, 60, 120]:
            # VWAP and deviations
            vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            df[f'vwap_{period}m'] = vwap
            df[f'price_vs_vwap_{period}m'] = (df['close'] - vwap) / vwap
            
            # Volume concentration
            total_volume = df['volume'].rolling(period).sum()
            df[f'volume_concentration_{period}m'] = df['volume'] / total_volume
            
            # Volume-price trend
            df[f'vpt_{period}m'] = (df['volume'] * df['close'].pct_change()).rolling(period).sum()
        
        # On-Balance Volume variations
        price_change = df['close'].diff()
        volume_direction = np.where(price_change > 0, df['volume'], 
                                  np.where(price_change < 0, -df['volume'], 0))
        df['obv'] = volume_direction.cumsum()
        
        # Volume rate of change
        for period in [5, 10, 15, 30]:
            df[f'volume_roc_{period}m'] = df['volume'].pct_change(period)
            df[f'volume_sma_{period}m'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}m'] = df['volume'] / df[f'volume_sma_{period}m']
        
        # Volume spikes detection
        volume_ma_60 = df['volume'].rolling(60).mean()
        volume_std_60 = df['volume'].rolling(60).std()
        df['volume_spike'] = (
            (df['volume'] > volume_ma_60 + 2 * volume_std_60)
        ).astype(int)
        
        return df
    
    def _add_order_flow_approximation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add order flow approximation features using OHLCV data."""
        
        if not self.config['order_flow_features']:
            return df
        
        # Buy/sell pressure approximation
        # Assumption: closing near high suggests buying pressure
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['buy_pressure'] = df['buy_pressure'].fillna(0.5)
        df['sell_pressure'] = df['sell_pressure'].fillna(0.5)
        
        # Volume-weighted buy/sell approximation
        df['buy_volume_approx'] = df['volume'] * df['buy_pressure']
        df['sell_volume_approx'] = df['volume'] * df['sell_pressure']
        
        # Rolling order flow balance
        for period in [5, 10, 15, 30]:
            df[f'buy_sell_ratio_{period}m'] = (
                df['buy_volume_approx'].rolling(period).sum() / 
                (df['sell_volume_approx'].rolling(period).sum() + 1e-8)
            )
            
            df[f'net_order_flow_{period}m'] = (
                df['buy_volume_approx'].rolling(period).sum() - 
                df['sell_volume_approx'].rolling(period).sum()
            )
        
        # Price-volume relationship (market impact approximation)
        price_change_abs = abs(df['close'].pct_change())
        df['price_impact_per_volume'] = price_change_abs / (df['volume'] + 1e-8)
        
        # Cumulative order flow approximation
        net_flow = df['buy_volume_approx'] - df['sell_volume_approx']
        df['cumulative_order_flow'] = net_flow.cumsum()
        
        # Order flow momentum
        for period in [5, 10, 15]:
            df[f'order_flow_momentum_{period}m'] = (
                df['cumulative_order_flow'].diff(period)
            )
        
        return df
    
    def _add_market_regime_minute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features for minute-level data."""
        
        returns = df['close'].pct_change()
        
        # Short-term regime indicators
        for period in [15, 30, 60, 120]:
            # Trend regime
            rolling_return = returns.rolling(period).mean()
            rolling_vol = returns.rolling(period).std()
            
            df[f'trend_strength_{period}m'] = rolling_return / (rolling_vol + 1e-8)
            df[f'regime_score_{period}m'] = np.where(
                df[f'trend_strength_{period}m'] > 0.5, 1,
                np.where(df[f'trend_strength_{period}m'] < -0.5, -1, 0)
            )
            
            # Volatility regime
            current_vol = returns.rolling(10).std()
            historical_vol = returns.rolling(period).std()
            vol_ratio = current_vol / (historical_vol + 1e-8)
            
            df[f'vol_regime_{period}m'] = np.where(
                vol_ratio > 1.5, 1,  # High vol
                np.where(vol_ratio < 0.7, -1, 0)  # Low vol
            )
        
        # Market state transitions
        df['regime_change'] = df['regime_score_60m'].diff().abs()
        df['vol_regime_change'] = df['vol_regime_60m'].diff().abs()
        
        # Persistence measures
        for period in [30, 60]:
            score = df[f'regime_score_{period}m']
            same_as_last = (score == score.shift(1)).astype(int)
            # Count of consecutive equal signs in window approximated by sum of equals
            df[f'trend_persistence_{period}m'] = same_as_last.rolling(period).mean()
        
        return df
    
    def _add_cross_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that combine multiple timeframes."""
        
        # Multi-timeframe momentum alignment
        timeframes = [5, 15, 30, 60]
        momentum_cols = []
        
        for tf in timeframes:
            col_name = f'momentum_{tf}m'
            df[col_name] = df['close'].pct_change(tf)
            momentum_cols.append(col_name)
        
        # Momentum alignment score
        momentum_signs = df[momentum_cols].apply(np.sign, axis=1)
        df['momentum_alignment'] = momentum_signs.sum(axis=1) / len(timeframes)
        
        # Volatility term structure
        vol_timeframes = [5, 15, 30, 60]
        for i, tf in enumerate(vol_timeframes[:-1]):
            next_tf = vol_timeframes[i + 1]
            vol_short = df['close'].pct_change().rolling(tf).std()
            vol_long = df['close'].pct_change().rolling(next_tf).std()
            df[f'vol_term_structure_{tf}_{next_tf}m'] = vol_short / (vol_long + 1e-8)
        
        # Cross-timeframe volume analysis
        for tf in [15, 30, 60]:
            volume_ma = df['volume'].rolling(tf).mean()
            df[f'volume_vs_{tf}m_avg'] = df['volume'] / (volume_ma + 1e-8)
        
        return df
    
    def _add_minute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features optimized for minute-level data."""
        
        # Ultra-short-term momentum
        for period in [1, 2, 3, 5, 10]:
            # Price momentum
            df[f'price_mom_{period}m'] = df['close'].pct_change(period)
            
            # Accelerating momentum
            mom = df['close'].pct_change()
            df[f'mom_acceleration_{period}m'] = mom.diff(period)
            
            # Volume-adjusted momentum
            volume_ma = df['volume'].rolling(period).mean()
            df[f'vol_adj_mom_{period}m'] = (
                df[f'price_mom_{period}m'] * volume_ma / df['volume'].mean()
            )
        
        # Momentum persistence
        for period in [5, 10, 15]:
            returns = df['close'].pct_change()
            sign = np.sign(returns)
            same_direction = (sign == sign.shift(1)).astype(int)
            df[f'momentum_persistence_{period}m'] = same_direction.rolling(period).mean()
        
        # Momentum mean reversion signals
        for period in [10, 15, 30]:
            momentum = df['close'].pct_change(period)
            momentum_ma = momentum.rolling(period).mean()
            momentum_std = momentum.rolling(period).std()
            
            df[f'momentum_zscore_{period}m'] = (
                (momentum - momentum_ma) / (momentum_std + 1e-8)
            )
        
        return df
    
    def _add_volatility_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering and GARCH-like features."""
        
        returns = df['close'].pct_change()
        squared_returns = returns ** 2
        
        # Simple volatility clustering measures
        for period in [5, 10, 15, 30]:
            # Rolling volatility
            vol = returns.rolling(period).std()
            df[f'volatility_{period}m'] = vol
            
            # Volatility of volatility
            df[f'vol_of_vol_{period}m'] = vol.rolling(period).std()
            
            # Volatility persistence
            vol_change = vol.diff()
            same_direction_vol = (np.sign(vol_change) == np.sign(vol_change.shift(1))).astype(int)
            df[f'vol_persistence_{period}m'] = same_direction_vol.rolling(period).mean()
        
        # GARCH-like approximations
        # Simple EWMA volatility
        for alpha in [0.06, 0.12, 0.25]:  # Different decay rates
            ewma_vol = squared_returns.ewm(alpha=alpha).mean().apply(np.sqrt)
            df[f'ewma_vol_alpha_{int(alpha*100)}'] = ewma_vol
        
        # Volatility regime changes
        vol_5m = returns.rolling(5).std()
        vol_30m = returns.rolling(30).std()
        df['vol_regime_ratio'] = vol_5m / (vol_30m + 1e-8)
        df['vol_shock'] = (df['vol_regime_ratio'] > 2.0).astype(int)
        
        return df
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage for large minute-level datasets."""
        
        if not self.config['memory_efficient']:
            return df
        
        # Convert float64 to float32 where precision loss is acceptable
        float_cols = df.select_dtypes(include=[np.float64]).columns
        
        for col in float_cols:
            # Check if values are within float32 range
            col_min, col_max = df[col].min(), df[col].max()
            if (not np.isnan(col_min) and not np.isnan(col_max) and
                col_min > np.finfo(np.float32).min and 
                col_max < np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)
        
        # Convert int64 to smaller int types where possible
        int_cols = df.select_dtypes(include=[np.int64]).columns
        
        for col in int_cols:
            col_min, col_max = df[col].min(), df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype(np.int32)
        
        return df
    
    def _final_feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning and validation of minute-level features."""
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme outliers at 99.9th percentile
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].std() > 0:
                p999 = df[col].quantile(0.999)
                p001 = df[col].quantile(0.001)
                df[col] = df[col].clip(lower=p001, upper=p999)
        
        # Forward fill NaN values (limited to avoid lookahead bias)
        df = df.fillna(method='ffill', limit=3)
        
        # Drop columns with too many NaN values
        nan_threshold = 0.95
        cols_to_drop = []
        for col in df.columns:
            nan_pct = df[col].isna().sum() / len(df)
            if nan_pct > nan_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self.logger.warning(f"Dropping {len(cols_to_drop)} columns with >{nan_threshold:.1%} NaN values")
            df = df.drop(columns=cols_to_drop)
        
        # Final NaN fill with median for remaining columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis and selection."""
        # This would analyze the generated features and group them
        # Implementation would examine column names and categorize them
        return {
            'price_features': ['Open', 'High', 'Low', 'Close'],
            'volume_features': ['Volume'],
            'high_frequency': [col for col in [] if 'hf_' in col or '_1m' in col or '_2m' in col],
            'microstructure': [col for col in [] if 'spread' in col or 'depth' in col or 'impact' in col],
            'seasonality': [col for col in [] if 'hour' in col or 'session' in col or 'sin_' in col],
            'volume_profile': [col for col in [] if 'vwap' in col or 'obv' in col or 'volume' in col],
            'order_flow': [col for col in [] if 'buy_' in col or 'sell_' in col or 'flow' in col],
            'regime': [col for col in [] if 'regime' in col or 'trend_strength' in col],
            'momentum': [col for col in [] if 'mom' in col or 'momentum' in col],
            'volatility': [col for col in [] if 'vol' in col or 'volatility' in col]
        }


# Utility functions

def create_minute_feature_engine(config: Dict[str, Any] = None) -> MinuteFeatureEngine:
    """Create a configured minute feature engine."""
    return MinuteFeatureEngine(config)


def generate_minute_features_for_symbol(data: pd.DataFrame, symbol: str,
                                       config: Dict[str, Any] = None) -> pd.DataFrame:
    """Generate minute-level features for a single symbol."""
    engine = create_minute_feature_engine(config)
    return engine.generate_minute_features(data, symbol)


def batch_generate_minute_features(data_dict: Dict[str, pd.DataFrame],
                                  config: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
    """Generate minute-level features for multiple symbols."""
    engine = create_minute_feature_engine(config)
    
    result = {}
    for symbol, data in data_dict.items():
        try:
            features = engine.generate_minute_features(data, symbol)
            result[symbol] = features
            print(f"✓ Generated {features.shape[1]} features for {symbol}")
        except Exception as e:
            print(f"✗ Error generating features for {symbol}: {e}")
            continue
    
    return result


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample minute data
    dates = pd.date_range('2024-01-01', '2024-01-02', freq='1T')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Open': 50000 + np.random.randn(len(dates)).cumsum() * 100,
        'High': 50000 + np.random.randn(len(dates)).cumsum() * 100 + 50,
        'Low': 50000 + np.random.randn(len(dates)).cumsum() * 100 - 50,
        'Close': 50000 + np.random.randn(len(dates)).cumsum() * 100,
        'Volume': np.random.exponential(1000, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationships are valid
    sample_data['High'] = np.maximum.reduce([
        sample_data['Open'], sample_data['Close'], sample_data['High']
    ])
    sample_data['Low'] = np.minimum.reduce([
        sample_data['Open'], sample_data['Close'], sample_data['Low']
    ])
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Generate features
    engine = create_minute_feature_engine()
    features = engine.generate_minute_features(sample_data, 'BTC-USD')
    
    print(f"Generated features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)[:20]}...")  # Show first 20 columns