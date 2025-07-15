"""Multi-source cryptocurrency data fetcher."""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
import time
from pathlib import Path
import pickle

from utils.config import Config, DataConfig


class CryptoDataFetcher:
    """Fetches cryptocurrency data from multiple sources."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Base URLs
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.coinmarketcap_base_url = "https://pro-api.coinmarketcap.com/v1"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60 / config.requests_per_minute
        
        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_cache_path(self, symbol: str, days: int) -> str:
        """Get cache file path for a symbol."""
        return os.path.join(
            self.config.cache_dir,
            f"{symbol}_{days}d_{self.config.interval}.pkl"
        )
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        expiry_time = datetime.now() - timedelta(hours=self.config.cache_expiry_hours)
        
        return file_time > expiry_time
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: str):
        """Save data to cache file."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    async def fetch_coingecko_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Fetch historical data from CoinGecko API."""
        days = days or self.config.days
        
        # Check cache first
        cache_path = self._get_cache_path(f"cg_{symbol}", days)
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                self.logger.info(f"Loaded {symbol} data from cache")
                return cached_data
        
        await self._rate_limit()
        
        # Construct API URL
        url = f"{self.coingecko_base_url}/coins/{symbol}/market_chart"
        params = {
            'vs_currency': self.config.vs_currency,
            'days': days,
            'interval': self.config.interval
        }
        
        # Add API key if available
        if self.config.coingecko_api_key:
            params['x_cg_pro_api_key'] = self.config.coingecko_api_key
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = self._parse_coingecko_response(data, symbol)
                    
                    # Save to cache
                    self._save_to_cache(df, cache_path)
                    
                    self.logger.info(f"Fetched {len(df)} records for {symbol} from CoinGecko")
                    return df
                else:
                    self.logger.error(f"CoinGecko API error {response.status} for {symbol}")
                    return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from CoinGecko: {e}")
            return pd.DataFrame()
    
    def _parse_coingecko_response(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse CoinGecko API response into DataFrame."""
        try:
            # Extract price, market cap, and volume data
            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            volumes = data.get('total_volumes', [])
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': [p[0] for p in prices],
                'price': [p[1] for p in prices],
                'market_cap': [m[1] for m in market_caps],
                'volume': [v[1] for v in volumes]
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Create OHLC data (approximate from price data)
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']
            df['close'] = df['price']
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing CoinGecko response for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_coinmarketcap_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Fetch historical data from CoinMarketCap API."""
        days = days or self.config.days
        
        # Check cache first
        cache_path = self._get_cache_path(f"cmc_{symbol}", days)
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                self.logger.info(f"Loaded {symbol} data from CMC cache")
                return cached_data
        
        # Note: CoinMarketCap Pro API is required for historical data
        if not self.config.coinmarketcap_api_key:
            self.logger.warning("CoinMarketCap API key not provided, skipping CMC data")
            return pd.DataFrame()
        
        await self._rate_limit()
        
        # Map symbol to CMC symbol (this is a simplified mapping)
        cmc_symbol_map = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'tether': 'USDT',
            'solana': 'SOL',
            'binancecoin': 'BNB',
            'usd-coin': 'USDC',
            'ripple': 'XRP',
            'dogecoin': 'DOGE',
            'cardano': 'ADA'
        }
        
        cmc_symbol = cmc_symbol_map.get(symbol, symbol.upper())
        
        # Construct API URL
        url = f"{self.coinmarketcap_base_url}/cryptocurrency/quotes/historical"
        headers = {
            'X-CMC_PRO_API_KEY': self.config.coinmarketcap_api_key
        }
        
        # Calculate time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'symbol': cmc_symbol,
            'time_start': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'time_end': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'interval': '1h' if self.config.interval == 'hourly' else '1d'
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = self._parse_coinmarketcap_response(data, symbol)
                    
                    # Save to cache
                    self._save_to_cache(df, cache_path)
                    
                    self.logger.info(f"Fetched {len(df)} records for {symbol} from CoinMarketCap")
                    return df
                else:
                    self.logger.error(f"CoinMarketCap API error {response.status} for {symbol}")
                    return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from CoinMarketCap: {e}")
            return pd.DataFrame()
    
    def _parse_coinmarketcap_response(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse CoinMarketCap API response into DataFrame."""
        try:
            # Extract quotes data
            quotes = data.get('data', {}).get('quotes', [])
            
            records = []
            for quote in quotes:
                record = {
                    'timestamp': quote.get('timestamp'),
                    'price': quote.get('quote', {}).get('USD', {}).get('price'),
                    'volume': quote.get('quote', {}).get('USD', {}).get('volume_24h'),
                    'market_cap': quote.get('quote', {}).get('USD', {}).get('market_cap'),
                    'symbol': symbol
                }
                records.append(record)
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create OHLC data (approximate from price data)
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']
            df['close'] = df['price']
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing CoinMarketCap response for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_all_symbols(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured symbols."""
        symbols = symbols or self.config.symbols
        
        self.logger.info(f"Fetching data for {len(symbols)} symbols")
        
        # Create tasks for all symbols
        tasks = []
        for symbol in symbols:
            # Primary source: CoinGecko
            task = self.fetch_coingecko_data(symbol)
            tasks.append((symbol, task))
        
        # Execute all tasks concurrently
        results = {}
        for symbol, task in tasks:
            try:
                df = await task
                if not df.empty:
                    results[symbol] = df
                else:
                    self.logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
        
        self.logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple symbols into a single DataFrame."""
        if not data_dict:
            return pd.DataFrame()
        
        # Create a list to store all dataframes
        dfs = []
        
        for symbol, df in data_dict.items():
            # Add symbol prefix to columns (except timestamp)
            df_copy = df.copy()
            df_copy.columns = [f"{symbol}_{col}" if col != 'symbol' else col for col in df_copy.columns]
            dfs.append(df_copy)
        
        # Combine all dataframes on timestamp index
        combined_df = pd.concat(dfs, axis=1, sort=True)
        
        # Fill missing values with forward fill then backward fill
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def get_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data for analysis."""
        # Remove rows with too many missing values
        threshold = len(data.columns) * 0.5  # At least 50% of columns must have data
        data = data.dropna(thresh=threshold)
        
        # Remove duplicate timestamps
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        self.logger.info(f"Clean data shape: {data.shape}")
        
        return data


async def main():
    """Test the data fetcher."""
    from utils.config import get_default_config
    
    config = get_default_config()
    
    async with CryptoDataFetcher(config.data) as fetcher:
        # Fetch data for a few symbols
        test_symbols = ['bitcoin', 'ethereum', 'solana']
        data_dict = await fetcher.fetch_all_symbols(test_symbols)
        
        # Combine data
        combined_data = fetcher.combine_data(data_dict)
        
        # Clean data
        clean_data = fetcher.get_clean_data(combined_data)
        
        print(f"Fetched data for {len(data_dict)} symbols")
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Clean data shape: {clean_data.shape}")
        
        # Display first few rows
        print("\nFirst few rows:")
        print(clean_data.head())


if __name__ == "__main__":
    asyncio.run(main())