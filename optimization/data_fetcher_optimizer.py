"""Optimized data fetcher with async operations and efficient batch processing."""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import functools
from asyncio import Semaphore

from utils.config import DataConfig


class OptimizedYFinanceFetcher:
    """Optimized cryptocurrency data fetcher with async operations."""
    
    # Mapping from CoinGecko-style names to Yahoo Finance tickers
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
        
        # Rate limiting with semaphore instead of sleep
        self.rate_limiter = Semaphore(10)  # Max 10 concurrent requests
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    def get_yf_ticker(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance ticker format."""
        return self.SYMBOL_MAPPING.get(symbol, f"{symbol.upper()}-USD")
    
    async def fetch_all_symbols_async(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols asynchronously."""
        symbols = symbols or self.config.symbols
        
        self.logger.info(f"Fetching data for {len(symbols)} symbols asynchronously")
        
        # Create async tasks for all symbols
        tasks = []
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                task = self._fetch_symbol_async(session, symbol)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching {symbol}: {result}")
            elif result is not None and not result.empty:
                data_dict[symbol] = result
            else:
                self.logger.warning(f"No data retrieved for {symbol}")
        
        self.logger.info(f"Successfully fetched data for {len(data_dict)} symbols")
        return data_dict
    
    async def _fetch_symbol_async(self, session: aiohttp.ClientSession, symbol: str) -> pd.DataFrame:
        """Fetch single symbol data asynchronously."""
        async with self.rate_limiter:
            # Check cache first
            cache_key = f"{symbol}_{self.config.days}_{self.config.interval}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if (datetime.now() - timestamp).seconds < self._cache_ttl:
                    return cached_data
            
            # Fetch data in thread pool (yfinance is not async)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor,
                functools.partial(self._fetch_symbol_sync, symbol)
            )
            
            # Cache the result
            if data is not None and not data.empty:
                self._cache[cache_key] = (data, datetime.now())
            
            return data
    
    def _fetch_symbol_sync(self, symbol: str) -> pd.DataFrame:
        """Synchronous fetch for a single symbol."""
        try:
            yf_ticker = self.get_yf_ticker(symbol)
            ticker = yf.Ticker(yf_ticker)
            
            # Determine period based on config
            days = self.config.days
            if days <= 7:
                period = "7d"
            elif days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            else:
                period = "2y"
            
            # Map interval
            interval_map = {
                "1m": "1m", "minute": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "hourly": "1h",
                "daily": "1d"
            }
            interval = interval_map.get(self.config.interval, "1h")
            
            # Fetch historical data
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return pd.DataFrame()
            
            # Process data efficiently
            return self._process_data_vectorized(df, symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_data_vectorized(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process yfinance data using vectorized operations."""
        # Rename columns efficiently
        df.columns = df.columns.str.lower()
        
        # Add columns using vectorized operations
        df['symbol'] = symbol
        df['price'] = df['close']
        df['market_cap'] = df['volume'] * df['close']
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Handle NaN values efficiently
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
        
        return df
    
    def fetch_batch_data_optimized(self, symbols: List[str], period: str = "2y", 
                                  interval: str = "1h") -> Dict[str, pd.DataFrame]:
        """Optimized batch data fetching."""
        # Convert symbols to yfinance tickers
        tickers = [self.get_yf_ticker(symbol) for symbol in symbols]
        ticker_string = " ".join(tickers)
        
        self.logger.info(f"Batch fetching {len(symbols)} symbols")
        
        try:
            # Download all at once with threading enabled
            data = yf.download(
                tickers=ticker_string,
                period=period,
                interval=interval,
                group_by='ticker',
                threads=True,
                progress=False
            )
            
            if data.empty:
                return {}
            
            # Process data efficiently for multiple tickers
            return self._process_batch_data_vectorized(data, symbols, tickers)
            
        except Exception as e:
            self.logger.error(f"Error in batch fetch: {e}")
            return {}
    
    def _process_batch_data_vectorized(self, data: pd.DataFrame, symbols: List[str], 
                                      tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Process batch data using vectorized operations."""
        processed_data = {}
        
        if len(tickers) > 1:
            # Multi-ticker processing
            for symbol, ticker in zip(symbols, tickers):
                # Extract data for this ticker using vectorized operations
                ticker_data = pd.DataFrame()
                
                metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
                for metric in metrics:
                    if (ticker, metric) in data.columns:
                        ticker_data[metric.lower()] = data[(ticker, metric)]
                
                if not ticker_data.empty:
                    # Add derived columns
                    ticker_data['symbol'] = symbol
                    ticker_data['price'] = ticker_data['close']
                    ticker_data['market_cap'] = ticker_data['volume'] * ticker_data['close']
                    
                    # Handle NaN values
                    ticker_data = ticker_data.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
                    
                    processed_data[symbol] = ticker_data
        else:
            # Single ticker - process directly
            data.columns = data.columns.str.lower()
            data['symbol'] = symbols[0]
            data['price'] = data['close']
            data['market_cap'] = data['volume'] * data['close']
            
            # Handle NaN values
            data = data.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
            
            processed_data[symbols[0]] = data
        
        return processed_data
    
    def combine_data_optimized(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data efficiently using vectorized operations."""
        if not data_dict:
            return pd.DataFrame()
        
        # Pre-allocate list for DataFrames
        dfs = []
        
        # Process each symbol's data
        for symbol, df in data_dict.items():
            # Create a copy for modification
            df_prefixed = df.copy()
            
            # Vectorized column renaming
            rename_dict = {col: f"{symbol}_{col}" for col in df.columns if col != 'symbol'}
            df_prefixed = df_prefixed.rename(columns=rename_dict)
            
            # Keep original symbol column
            if 'symbol' in df.columns:
                df_prefixed['symbol'] = df['symbol']
            
            dfs.append(df_prefixed)
        
        # Efficient concatenation
        combined_df = pd.concat(dfs, axis=1, sort=True)
        
        # Remove duplicate columns efficiently
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        # Fill missing values using optimized method
        combined_df = combined_df.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
        
        self.logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)