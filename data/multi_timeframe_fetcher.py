"""Ultra-optimized multi-timeframe data fetcher for maximum prediction accuracy."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

from utils.config import DataConfig


class UltraMultiTimeframeFetcher:
    """Advanced multi-timeframe data fetcher with sophisticated alignment and feature generation."""
    
    SYMBOL_MAPPING = {
        'bitcoin': 'BTC-USD',
        'ethereum': 'ETH-USD',
        'tether': 'USDT-USD',
        'solana': 'SOL-USD',
        'binancecoin': 'BNB-USD',
        'usd-coin': 'USDC-USD',
        'ripple': 'XRP-USD',
        'dogecoin': 'DOGE-USD',
        'cardano': 'ADA-USD'
    }
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define multiple timeframes for comprehensive analysis
        self.timeframes = {
            'primary': self.config.interval,  # Main prediction timeframe
            'higher': self._get_higher_timeframe(self.config.interval),
            'lower': self._get_lower_timeframe(self.config.interval) if self.config.interval != '1m' else None
        }
        
        self.logger.info(f"Initialized multi-timeframe fetcher: {self.timeframes}")
    
    def _get_higher_timeframe(self, interval: str) -> str:
        """Get the next higher timeframe for context."""
        timeframe_hierarchy = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        if interval in timeframe_hierarchy:
            idx = timeframe_hierarchy.index(interval)
            if idx < len(timeframe_hierarchy) - 1:
                return timeframe_hierarchy[idx + 1]
        
        # Default mapping
        mapping = {
            '1m': '5m',
            '5m': '1h', 
            '15m': '1h',
            '30m': '4h',
            '1h': '4h',
            '4h': '1d'
        }
        return mapping.get(interval, '1h')
    
    def _get_lower_timeframe(self, interval: str) -> Optional[str]:
        """Get the next lower timeframe for microstructure analysis."""
        timeframe_hierarchy = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        if interval in timeframe_hierarchy:
            idx = timeframe_hierarchy.index(interval)
            if idx > 0:
                return timeframe_hierarchy[idx - 1]
        
        return None
    
    def _get_period_for_timeframe(self, timeframe: str) -> str:
        """Determine optimal period based on timeframe and data availability."""
        if timeframe in ['1m']:
            return '7d'
        elif timeframe in ['5m', '15m', '30m']:
            return '60d'
        elif timeframe in ['1h']:
            return '2y'
        elif timeframe in ['4h']:
            return '5y'
        else:  # daily and above
            return 'max'
    
    def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data across multiple timeframes for comprehensive analysis."""
        yf_ticker = self.SYMBOL_MAPPING.get(symbol, f"{symbol.upper()}-USD")
        multi_data = {}
        
        for tf_name, tf_interval in self.timeframes.items():
            if tf_interval is None:
                continue
                
            try:
                period = self._get_period_for_timeframe(tf_interval)
                
                self.logger.info(f"Fetching {symbol} {tf_name} timeframe: {tf_interval}, period: {period}")
                
                ticker = yf.Ticker(yf_ticker)
                df = ticker.history(period=period, interval=tf_interval)
                
                if not df.empty:
                    # Add timeframe identifier
                    df = self._process_timeframe_data(df, symbol, tf_name, tf_interval)
                    multi_data[tf_name] = df
                    self.logger.info(f"Fetched {len(df)} {tf_name} records for {symbol}")
                else:
                    self.logger.warning(f"No {tf_name} data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {tf_name} data for {symbol}: {e}")
        
        return multi_data
    
    def _process_timeframe_data(self, df: pd.DataFrame, symbol: str, tf_name: str, tf_interval: str) -> pd.DataFrame:
        """Process and standardize data for a specific timeframe."""
        # Standardize column names
        columns_mapping = {
            'Open': f'{symbol}_{tf_name}_open',
            'High': f'{symbol}_{tf_name}_high', 
            'Low': f'{symbol}_{tf_name}_low',
            'Close': f'{symbol}_{tf_name}_close',
            'Volume': f'{symbol}_{tf_name}_volume'
        }
        
        df = df.rename(columns=columns_mapping)
        
        # Add timeframe-specific features
        df = self._add_timeframe_features(df, symbol, tf_name, tf_interval)
        
        # Ensure timezone-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != 'UTC':
            df.index = df.index.tz_convert('UTC')
        
        return df
    
    def _add_timeframe_features(self, df: pd.DataFrame, symbol: str, tf_name: str, tf_interval: str) -> pd.DataFrame:
        """Add timeframe-specific technical indicators and features."""
        close_col = f'{symbol}_{tf_name}_close'
        high_col = f'{symbol}_{tf_name}_high'
        low_col = f'{symbol}_{tf_name}_low'
        volume_col = f'{symbol}_{tf_name}_volume'
        
        # Basic price features
        df[f'{symbol}_{tf_name}_returns'] = df[close_col].pct_change()
        df[f'{symbol}_{tf_name}_log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
        
        # Volatility (using appropriate lookback for timeframe)
        vol_window = self._get_volatility_window(tf_interval)
        df[f'{symbol}_{tf_name}_volatility'] = df[f'{symbol}_{tf_name}_returns'].rolling(vol_window).std()
        
        # Momentum indicators
        mom_window = self._get_momentum_window(tf_interval)
        df[f'{symbol}_{tf_name}_momentum'] = df[close_col] / df[close_col].shift(mom_window) - 1
        
        # Trend indicators  
        sma_window = self._get_trend_window(tf_interval)
        df[f'{symbol}_{tf_name}_sma'] = df[close_col].rolling(sma_window).mean()
        df[f'{symbol}_{tf_name}_trend'] = (df[close_col] / df[f'{symbol}_{tf_name}_sma'] - 1)
        
        # Range-based features
        df[f'{symbol}_{tf_name}_hl_ratio'] = (df[high_col] - df[low_col]) / df[close_col]
        df[f'{symbol}_{tf_name}_co_ratio'] = (df[close_col] - df[f'{symbol}_{tf_name}_open']) / df[f'{symbol}_{tf_name}_open']
        
        # Volume features (if available)
        if volume_col in df.columns and not df[volume_col].isna().all():
            df[f'{symbol}_{tf_name}_volume_sma'] = df[volume_col].rolling(vol_window).mean()
            df[f'{symbol}_{tf_name}_volume_ratio'] = df[volume_col] / df[f'{symbol}_{tf_name}_volume_sma']
            df[f'{symbol}_{tf_name}_volume_price'] = df[volume_col] * df[close_col]
        
        return df
    
    def _get_volatility_window(self, tf_interval: str) -> int:
        """Get appropriate volatility calculation window for timeframe."""
        windows = {
            '1m': 60,    # 1 hour
            '5m': 288,   # 24 hours  
            '15m': 96,   # 24 hours
            '30m': 48,   # 24 hours
            '1h': 24,    # 24 hours
            '4h': 168,   # 7 days
            '1d': 30     # 30 days
        }
        return windows.get(tf_interval, 24)
    
    def _get_momentum_window(self, tf_interval: str) -> int:
        """Get appropriate momentum calculation window for timeframe."""
        windows = {
            '1m': 15,    # 15 minutes
            '5m': 12,    # 1 hour
            '15m': 16,   # 4 hours  
            '30m': 24,   # 12 hours
            '1h': 24,    # 24 hours
            '4h': 42,    # 7 days
            '1d': 14     # 14 days
        }
        return windows.get(tf_interval, 12)
    
    def _get_trend_window(self, tf_interval: str) -> int:
        """Get appropriate trend calculation window for timeframe."""
        windows = {
            '1m': 60,    # 1 hour
            '5m': 60,    # 5 hours
            '15m': 32,   # 8 hours
            '30m': 24,   # 12 hours  
            '1h': 24,    # 24 hours
            '4h': 42,    # 7 days
            '1d': 20     # 20 days
        }
        return windows.get(tf_interval, 20)
    
    def align_timeframes(self, multi_data: Dict[str, pd.DataFrame], primary_tf: str = 'primary') -> pd.DataFrame:
        """Align multiple timeframes to primary timeframe with forward-fill for higher timeframes."""
        if primary_tf not in multi_data:
            raise ValueError(f"Primary timeframe {primary_tf} not found in data")
        
        primary_df = multi_data[primary_tf].copy()
        self.logger.info(f"Aligning timeframes to {primary_tf}: {primary_df.shape}")
        
        # Align other timeframes
        for tf_name, tf_df in multi_data.items():
            if tf_name == primary_tf:
                continue
            
            self.logger.info(f"Aligning {tf_name} timeframe: {tf_df.shape}")
            
            # Reindex higher timeframe data to primary timeframe with forward fill
            aligned_df = tf_df.reindex(primary_df.index, method='ffill')
            
            # Merge aligned data (handle overlapping columns)
            primary_df = primary_df.join(aligned_df, how='left', rsuffix='_dup')
            
            # Remove duplicate columns
            dup_columns = [col for col in primary_df.columns if col.endswith('_dup')]
            primary_df = primary_df.drop(columns=dup_columns)
        
        # Add cross-timeframe features
        primary_df = self._add_cross_timeframe_features(primary_df, multi_data.keys())
        
        self.logger.info(f"Final aligned dataset: {primary_df.shape}")
        return primary_df
    
    def _add_cross_timeframe_features(self, df: pd.DataFrame, timeframe_names: List[str]) -> pd.DataFrame:
        """Add sophisticated cross-timeframe features for enhanced prediction."""
        symbol = self.config.symbols[0] if self.config.symbols else 'bitcoin'
        
        # Trend alignment features
        if 'primary' in timeframe_names and 'higher' in timeframe_names:
            primary_trend = f'{symbol}_primary_trend'
            higher_trend = f'{symbol}_higher_trend'
            
            if primary_trend in df.columns and higher_trend in df.columns:
                # Trend alignment score
                df[f'{symbol}_trend_alignment'] = np.sign(df[primary_trend]) * np.sign(df[higher_trend])
                
                # Trend strength convergence
                df[f'{symbol}_trend_convergence'] = abs(df[primary_trend]) * abs(df[higher_trend])
        
        # Volatility regime detection
        volatility_cols = [col for col in df.columns if 'volatility' in col and symbol in col]
        if len(volatility_cols) >= 2:
            # Multi-timeframe volatility average
            df[f'{symbol}_vol_regime'] = df[volatility_cols].mean(axis=1)
            
            # Volatility expansion/contraction
            if len(volatility_cols) >= 2:
                df[f'{symbol}_vol_expansion'] = df[volatility_cols[0]] / df[volatility_cols[1]]
        
        # Momentum persistence
        momentum_cols = [col for col in df.columns if 'momentum' in col and symbol in col]
        if len(momentum_cols) >= 2:
            # Momentum alignment across timeframes
            df[f'{symbol}_momentum_persistence'] = np.prod(np.sign(df[momentum_cols]), axis=1)
            
            # Momentum strength
            df[f'{symbol}_momentum_strength'] = df[momentum_cols].abs().mean(axis=1)
        
        return df
    
    def fetch_all_symbols_multi_timeframe(self) -> pd.DataFrame:
        """Fetch multi-timeframe data for all configured symbols."""
        if not self.config.symbols:
            raise ValueError("No symbols configured")
        
        all_data = []
        
        for symbol in self.config.symbols:
            self.logger.info(f"Fetching multi-timeframe data for {symbol}")
            
            # Fetch multi-timeframe data
            multi_data = self.fetch_multi_timeframe_data(symbol)
            
            if multi_data:
                # Align timeframes
                aligned_data = self.align_timeframes(multi_data)
                all_data.append(aligned_data)
            else:
                self.logger.warning(f"No data fetched for {symbol}")
        
        if not all_data:
            raise ValueError("No data fetched for any symbols")
        
        # Combine all symbols
        if len(all_data) == 1:
            combined_data = all_data[0]
        else:
            # Join on index (timestamp)
            combined_data = all_data[0]
            for data in all_data[1:]:
                combined_data = combined_data.join(data, how='outer')
        
        # Clean and validate
        combined_data = self._clean_combined_data(combined_data)
        
        self.logger.info(f"Final multi-timeframe dataset: {combined_data.shape}")
        return combined_data
    
    def _clean_combined_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the combined multi-timeframe dataset."""
        # Remove columns with too many NaN values
        nan_threshold = 0.8
        df = df.loc[:, df.isnull().mean() < nan_threshold]
        
        # Forward fill missing values (appropriate for financial time series)
        df = df.fillna(method='ffill')
        
        # Remove any remaining rows with NaN values
        df = df.dropna()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').dropna()
        
        return df