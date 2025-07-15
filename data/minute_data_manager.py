"""High-frequency minute-level data management system for 6-month backtesting."""

import yfinance as yf
import pandas as pd
import numpy as np
import h5py
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .yfinance_fetcher import YFinanceCryptoFetcher
from utils.config import DataConfig


class MinuteDataManager:
    """Manages high-frequency minute-level cryptocurrency data for extended backtesting."""
    
    # YFinance minute data limitations
    MINUTE_DATA_LIMITS = {
        '1m': 7,      # 7 days max for 1-minute data
        '2m': 60,     # 60 days max for 2-minute data  
        '5m': 60,     # 60 days max for 5-minute data
        '15m': 60,    # 60 days max for 15-minute data
        '30m': 60,    # 60 days max for 30-minute data
        '60m': 730,   # 730 days max for 1-hour data
        '90m': 60,    # 60 days max for 90-minute data
        '1h': 730,    # 730 days max for 1-hour data
        '1d': 3650,   # 10 years max for daily data
    }
    
    def __init__(self, config: DataConfig = None, cache_dir: str = None):
        self.config = config or self._get_default_config()
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache_minute')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # HDF5 storage for efficient large dataset handling
        self.hdf5_path = self.cache_dir / 'minute_data.h5'
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Data quality tracking
        self.data_quality_log = []
        
    def _get_default_config(self) -> DataConfig:
        """Get default configuration for minute data."""
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'MATIC-USD']
        config.interval = '1m'
        config.days = 180  # 6 months
        config.cache_dir = 'data/cache_minute'  # Default cache directory
        return config
    
    def fetch_6_month_minute_data(self, symbols: List[str] = None, 
                                 interval: str = '1m') -> Dict[str, pd.DataFrame]:
        """
        Fetch 6 months of minute-level data by stitching multiple periods.
        
        Args:
            symbols: List of symbols to fetch (e.g., ['BTC-USD', 'ETH-USD'])
            interval: Data interval ('1m', '5m', '15m', '30m', '1h')
            
        Returns:
            Dictionary mapping symbols to their complete 6-month DataFrames
        """
        if symbols is None:
            symbols = self.config.symbols
            
        self.logger.info(f"Starting 6-month {interval} data collection for {len(symbols)} symbols")
        
        all_data = {}
        
        for symbol in symbols:
            try:
                self.logger.info(f"Fetching {symbol} {interval} data...")
                symbol_data = self._fetch_symbol_6_months(symbol, interval)
                
                if symbol_data is not None and not symbol_data.empty:
                    all_data[symbol] = symbol_data
                    self.logger.info(f"✓ {symbol}: {len(symbol_data):,} data points from "
                                   f"{symbol_data.index[0]} to {symbol_data.index[-1]}")
                    
                    # Save to HDF5 cache
                    self._save_to_hdf5(symbol, symbol_data, interval)
                    
                else:
                    self.logger.warning(f"✗ No data retrieved for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"✗ Error fetching {symbol}: {e}")
                continue
        
        self.logger.info(f"6-month data collection completed. Retrieved {len(all_data)} symbols")
        
        # Generate data quality report
        self._generate_data_quality_report(all_data, interval)
        
        return all_data
    
    def _fetch_symbol_6_months(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch 6 months of data for a single symbol by stitching periods."""
        
        # Check cache first
        cached_data = self._load_from_cache(symbol, interval)
        if cached_data is not None:
            self.logger.debug(f"Using cached data for {symbol}")
            return cached_data
        
        # Determine how to split the requests based on interval limits
        max_days = self.MINUTE_DATA_LIMITS.get(interval, 7)
        target_days = 180  # 6 months
        
        if max_days >= target_days:
            # Can fetch all data in one request
            return self._fetch_single_period(symbol, interval, f"{target_days}d")
        
        # Need to stitch multiple periods
        return self._fetch_multiple_periods(symbol, interval, target_days, max_days)
    
    def _fetch_single_period(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Fetch data for a single period."""
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=True)
            
            if data.empty:
                self.logger.warning(f"Empty data returned for {symbol} {period} {interval}")
                return pd.DataFrame()
            
            # Clean and validate data
            data = self._clean_data(data, symbol)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching single period for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_multiple_periods(self, symbol: str, interval: str, 
                              target_days: int, max_days_per_request: int) -> pd.DataFrame:
        """Fetch data across multiple overlapping periods and stitch together."""
        
        # Calculate end date (today) and start date (6 months ago)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=target_days)
        
        all_periods_data = []
        current_date = start_date
        
        self.logger.info(f"Fetching {symbol} in {max_days_per_request}-day chunks from "
                        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        while current_date < end_date:
            # Calculate period end (ensuring we don't exceed available data limit)
            period_end = min(current_date + timedelta(days=max_days_per_request - 1), end_date)
            
            # Fetch this period
            period_data = self._fetch_date_range(symbol, interval, current_date, period_end)
            
            if period_data is not None and not period_data.empty:
                all_periods_data.append(period_data)
                self.logger.debug(f"  ✓ {current_date.strftime('%Y-%m-%d')} to "
                                f"{period_end.strftime('%Y-%m-%d')}: {len(period_data)} points")
            else:
                self.logger.warning(f"  ✗ No data for period {current_date.strftime('%Y-%m-%d')} to "
                                  f"{period_end.strftime('%Y-%m-%d')}")
            
            # Move to next period (with small overlap to ensure no gaps)
            current_date = period_end - timedelta(days=1)
            
            # Rate limiting between requests
            time.sleep(0.5)
        
        if not all_periods_data:
            self.logger.error(f"No data periods successfully fetched for {symbol}")
            return pd.DataFrame()
        
        # Combine all periods and remove duplicates
        combined_data = pd.concat(all_periods_data, axis=0)
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        combined_data = combined_data.sort_index()
        
        self.logger.info(f"Successfully stitched {len(all_periods_data)} periods for {symbol}: "
                        f"{len(combined_data):,} total data points")
        
        return combined_data
    
    def _fetch_date_range(self, symbol: str, interval: str, 
                         start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data for a specific date range."""
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Use start and end parameters for precise date ranges
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Clean data
            data = self._clean_data(data, symbol)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error fetching {symbol} {start_date} to {end_date}: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate minute-level data."""
        if data.empty:
            return data
        
        original_length = len(data)
        
        # Remove rows with missing OHLCV data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Remove rows with zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            data = data[data[col] > 0]
        
        # Remove rows with impossible price relationships
        data = data[
            (data['High'] >= data['Low']) &
            (data['High'] >= data['Open']) &
            (data['High'] >= data['Close']) &
            (data['Low'] <= data['Open']) &
            (data['Low'] <= data['Close'])
        ]
        
        # Remove extreme outliers (price changes > 50% in one minute)
        if len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            data = data[price_changes <= 0.5]
        
        # Volume validation (remove zero volume where inappropriate)
        if 'Volume' in data.columns:
            # For minute data, some zero volume periods are normal (off-hours)
            # But we'll flag suspicious patterns
            zero_volume_pct = (data['Volume'] == 0).sum() / len(data)
            if zero_volume_pct > 0.5:
                self.logger.warning(f"{symbol}: High zero-volume percentage: {zero_volume_pct:.1%}")
        
        cleaned_length = len(data)
        removed_count = original_length - cleaned_length
        
        if removed_count > 0:
            self.logger.debug(f"{symbol}: Cleaned data - removed {removed_count:,} invalid points "
                            f"({removed_count/original_length:.1%})")
        
        # Log data quality metrics
        self.data_quality_log.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'removed_count': removed_count,
            'removal_rate': removed_count / original_length if original_length > 0 else 0
        })
        
        return data
    
    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _save_to_hdf5(self, symbol: str, data: pd.DataFrame, interval: str):
        """Save data to HDF5 for efficient storage and retrieval."""
        try:
            with h5py.File(self.hdf5_path, 'a') as hf:
                dataset_name = f"{symbol}_{interval}"
                
                # Remove existing dataset if it exists
                if dataset_name in hf:
                    del hf[dataset_name]
                
                # Convert DataFrame to numpy array for HDF5 storage
                # Store as structured array with column information
                dt = np.dtype([
                    ('timestamp', 'i8'),  # Unix timestamp
                    ('open', 'f8'),
                    ('high', 'f8'), 
                    ('low', 'f8'),
                    ('close', 'f8'),
                    ('volume', 'f8')
                ])
                
                # Prepare data array
                timestamps = data.index.astype(np.int64) // 10**9  # Convert to Unix timestamp
                structured_array = np.zeros(len(data), dtype=dt)
                structured_array['timestamp'] = timestamps
                structured_array['open'] = data['Open'].values
                structured_array['high'] = data['High'].values
                structured_array['low'] = data['Low'].values
                structured_array['close'] = data['Close'].values
                structured_array['volume'] = data['Volume'].values
                
                # Create dataset
                hf.create_dataset(dataset_name, data=structured_array, compression='gzip')
                
                # Store metadata
                hf[dataset_name].attrs['symbol'] = symbol
                hf[dataset_name].attrs['interval'] = interval
                hf[dataset_name].attrs['start_date'] = str(data.index[0])
                hf[dataset_name].attrs['end_date'] = str(data.index[-1])
                hf[dataset_name].attrs['length'] = len(data)
                hf[dataset_name].attrs['cached_at'] = datetime.now().isoformat()
                
            self.logger.debug(f"Saved {symbol} {interval} data to HDF5 cache")
            
        except Exception as e:
            self.logger.warning(f"Failed to save {symbol} to HDF5 cache: {e}")
    
    def _load_from_cache(self, symbol: str, interval: str, 
                        max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Load data from HDF5 cache if available and recent."""
        if not self.hdf5_path.exists():
            return None
        
        try:
            with h5py.File(self.hdf5_path, 'r') as hf:
                dataset_name = f"{symbol}_{interval}"
                
                if dataset_name not in hf:
                    return None
                
                dataset = hf[dataset_name]
                
                # Check cache age
                cached_at_str = dataset.attrs.get('cached_at', '')
                if cached_at_str:
                    cached_at = datetime.fromisoformat(cached_at_str)
                    age_hours = (datetime.now() - cached_at).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        self.logger.debug(f"Cache too old for {symbol} {interval}: {age_hours:.1f}h")
                        return None
                
                # Load data
                structured_data = dataset[:]
                
                # Convert back to DataFrame
                timestamps = pd.to_datetime(structured_data['timestamp'], unit='s')
                df = pd.DataFrame({
                    'Open': structured_data['open'],
                    'High': structured_data['high'],
                    'Low': structured_data['low'],
                    'Close': structured_data['close'],
                    'Volume': structured_data['volume']
                }, index=timestamps)
                
                self.logger.debug(f"Loaded {symbol} {interval} from cache: {len(df):,} points")
                return df
                
        except Exception as e:
            self.logger.warning(f"Failed to load {symbol} from cache: {e}")
            return None
    
    def get_data_info(self, symbol: str, interval: str) -> Dict[str, any]:
        """Get information about cached data."""
        if not self.hdf5_path.exists():
            return {"cached": False}
        
        try:
            with h5py.File(self.hdf5_path, 'r') as hf:
                dataset_name = f"{symbol}_{interval}"
                
                if dataset_name not in hf:
                    return {"cached": False}
                
                dataset = hf[dataset_name]
                attrs = dict(dataset.attrs)
                attrs['cached'] = True
                attrs['size_mb'] = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
                
                return attrs
                
        except Exception as e:
            return {"cached": False, "error": str(e)}
    
    def clear_cache(self, symbol: str = None, interval: str = None):
        """Clear cache for specific symbol/interval or all data."""
        if not self.hdf5_path.exists():
            return
        
        try:
            if symbol is None and interval is None:
                # Clear all cache
                self.hdf5_path.unlink()
                self.logger.info("Cleared all cached data")
            else:
                # Clear specific dataset
                with h5py.File(self.hdf5_path, 'a') as hf:
                    dataset_name = f"{symbol}_{interval}"
                    if dataset_name in hf:
                        del hf[dataset_name]
                        self.logger.info(f"Cleared cache for {symbol} {interval}")
                        
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def _generate_data_quality_report(self, data_dict: Dict[str, pd.DataFrame], interval: str):
        """Generate comprehensive data quality report."""
        if not data_dict:
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'interval': interval,
            'symbols_processed': len(data_dict),
            'total_data_points': sum(len(df) for df in data_dict.values()),
            'symbol_summary': {},
            'data_quality_metrics': {}
        }
        
        for symbol, df in data_dict.items():
            if df.empty:
                continue
                
            # Basic statistics
            summary = {
                'data_points': len(df),
                'date_range': {
                    'start': str(df.index[0]),
                    'end': str(df.index[-1]),
                    'duration_days': (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
                },
                'price_statistics': {
                    'min_close': float(df['Close'].min()),
                    'max_close': float(df['Close'].max()),
                    'avg_close': float(df['Close'].mean()),
                    'volatility': float(df['Close'].pct_change().std())
                },
                'volume_statistics': {
                    'avg_volume': float(df['Volume'].mean()),
                    'max_volume': float(df['Volume'].max()),
                    'zero_volume_pct': float((df['Volume'] == 0).sum() / len(df))
                }
            }
            
            # Data completeness check
            expected_points = self._calculate_expected_data_points(
                df.index[0], df.index[-1], interval
            )
            completeness = len(df) / expected_points if expected_points > 0 else 0
            summary['completeness'] = float(completeness)
            
            report['symbol_summary'][symbol] = summary
        
        # Save report
        report_path = self.cache_dir / f"data_quality_report_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        # Log summary
        total_points = report['total_data_points']
        avg_completeness = np.mean([s['completeness'] for s in report['symbol_summary'].values()])
        
        self.logger.info(f"Data Quality Report:")
        self.logger.info(f"  Total data points: {total_points:,}")
        self.logger.info(f"  Average completeness: {avg_completeness:.1%}")
        self.logger.info(f"  Report saved: {report_path}")
    
    def _calculate_expected_data_points(self, start_time: pd.Timestamp, 
                                      end_time: pd.Timestamp, interval: str) -> int:
        """Calculate expected number of data points for a time range."""
        total_minutes = (end_time - start_time).total_seconds() / 60
        
        interval_minutes = {
            '1m': 1,
            '2m': 2, 
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '90m': 90
        }
        
        minutes_per_point = interval_minutes.get(interval, 1)
        expected_points = int(total_minutes / minutes_per_point)
        
        # Adjust for market hours if needed (crypto trades 24/7 so no adjustment needed)
        return expected_points
    
    def get_cache_summary(self) -> Dict[str, any]:
        """Get summary of all cached data."""
        if not self.hdf5_path.exists():
            return {"cache_exists": False}
        
        summary = {
            "cache_exists": True,
            "cache_file": str(self.hdf5_path),
            "cache_size_mb": self.hdf5_path.stat().st_size / (1024 * 1024),
            "datasets": {}
        }
        
        try:
            with h5py.File(self.hdf5_path, 'r') as hf:
                for dataset_name in hf.keys():
                    dataset = hf[dataset_name]
                    attrs = dict(dataset.attrs)
                    attrs['size_mb'] = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
                    summary["datasets"][dataset_name] = attrs
                    
        except Exception as e:
            summary["error"] = str(e)
        
        return summary


# Utility functions for minute data management

def create_minute_data_manager(symbols: List[str] = None, 
                             interval: str = '1m',
                             cache_dir: str = None) -> MinuteDataManager:
    """Create a configured minute data manager."""
    from types import SimpleNamespace
    
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'MATIC-USD']
    
    config = SimpleNamespace()
    config.symbols = symbols
    config.interval = interval
    config.days = 180
    config.cache_dir = cache_dir or 'data/cache_minute'
    
    return MinuteDataManager(config, cache_dir)


def fetch_6_month_data_for_backtesting(symbols: List[str] = None,
                                     interval: str = '1m') -> Dict[str, pd.DataFrame]:
    """Convenience function to fetch 6 months of minute data for backtesting."""
    manager = create_minute_data_manager(symbols, interval)
    return manager.fetch_6_month_minute_data(symbols, interval)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test the system
    symbols = ['BTC-USD', 'ETH-USD']
    interval = '5m'  # Use 5m for testing (less data, faster)
    
    manager = create_minute_data_manager(symbols, interval)
    
    print(f"Fetching {interval} data for: {symbols}")
    data = manager.fetch_6_month_minute_data(symbols, interval)
    
    for symbol, df in data.items():
        print(f"{symbol}: {len(df):,} data points from {df.index[0]} to {df.index[-1]}")
    
    # Show cache summary
    print("\nCache Summary:")
    cache_summary = manager.get_cache_summary()
    print(f"Cache file: {cache_summary.get('cache_file', 'None')}")
    print(f"Cache size: {cache_summary.get('cache_size_mb', 0):.2f} MB")
    print(f"Datasets: {len(cache_summary.get('datasets', {}))}")