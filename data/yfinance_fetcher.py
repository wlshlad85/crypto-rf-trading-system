"""yfinance-based cryptocurrency data fetcher for free real-time and historical data."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
import time

from utils.config import DataConfig


class YFinanceCryptoFetcher:
    """Fetches cryptocurrency data using yfinance (Yahoo Finance)."""
    
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
        
        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)
    
    def get_yf_ticker(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance ticker format."""
        return self.SYMBOL_MAPPING.get(symbol, f"{symbol.upper()}-USD")
    
    def fetch_crypto_data(self, symbol: str, period: str = None, interval: str = None) -> pd.DataFrame:
        """Fetch cryptocurrency data from Yahoo Finance."""
        yf_ticker = self.get_yf_ticker(symbol)
        
        # Determine period and interval
        if period is None:
            # Convert days to period string
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
            elif days <= 730:
                period = "2y"
            else:
                period = "5y"
        
        if interval is None:
            # Map config interval to yfinance interval
            if self.config.interval == "1m" or self.config.interval == "minute":
                interval = "1m"
                # Auto-adjust period for minute data constraints
                if period in ["2y", "1y", "6mo", "3mo"]:
                    period = "7d"  # 1m data limited to 7 days
                    self.logger.warning("1-minute data limited to 7 days, adjusting period")
            elif self.config.interval == "5m":
                interval = "5m"
                # Auto-adjust period for 5-minute data constraints
                if period in ["2y", "1y", "6mo", "3mo"]:
                    period = "60d"  # 5m data limited to ~60 days
                    self.logger.warning("5-minute data limited to 60 days, adjusting period")
            elif self.config.interval == "15m":
                interval = "15m"
                # Auto-adjust period for 15-minute data constraints
                if period in ["2y", "1y", "6mo", "3mo"]:
                    period = "60d"  # 15m data limited to ~60 days
                    self.logger.warning("15-minute data limited to 60 days, adjusting period")
            elif self.config.interval == "30m":
                interval = "30m"
                # Auto-adjust period for 30-minute data constraints
                if period in ["2y", "1y", "6mo", "3mo"]:
                    period = "60d"  # 30m data limited to ~60 days
                    self.logger.warning("30-minute data limited to 60 days, adjusting period")
            elif self.config.interval == "hourly":
                interval = "1h"
            elif self.config.interval == "daily":
                interval = "1d"
            else:
                interval = "1h"  # Default to hourly
        
        self.logger.info(f"Fetching {symbol} ({yf_ticker}) data - period: {period}, interval: {interval}")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(yf_ticker)
            
            # Fetch historical data
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                self.logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Process and standardize data
            df = self._process_yfinance_data(df, symbol)
            
            self.logger.info(f"Fetched {len(df)} records for {symbol}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from yfinance: {e}")
            return pd.DataFrame()
    
    def _process_yfinance_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process yfinance data to match expected format."""
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Missing column {col} for {symbol}")
                df[col] = 0
        
        # Add price column (same as close)
        df['price'] = df['close']
        
        # Add market cap (approximate - would need shares outstanding)
        # For now, using volume * price as a proxy
        df['market_cap'] = df['volume'] * df['close']
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Handle any NaN values
        df = df.ffill().bfill()
        
        return df
    
    def fetch_all_symbols(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured symbols."""
        symbols = symbols or self.config.symbols
        
        self.logger.info(f"Fetching data for {len(symbols)} symbols using yfinance")
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                # Add small delay to avoid rate limiting
                if i > 0:
                    time.sleep(0.5)
                
                df = self.fetch_crypto_data(symbol)
                
                if not df.empty:
                    results[symbol] = df
                else:
                    self.logger.warning(f"No data retrieved for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
        
        self.logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def fetch_batch_data(self, symbols: List[str], period: str = "2y", interval: str = "1h") -> pd.DataFrame:
        """Fetch data for multiple symbols in batch and return combined DataFrame."""
        # Convert symbols to yfinance tickers
        tickers = [self.get_yf_ticker(symbol) for symbol in symbols]
        ticker_string = " ".join(tickers)
        
        self.logger.info(f"Batch fetching {len(symbols)} symbols: {ticker_string}")
        
        try:
            # Download all at once
            data = yf.download(
                tickers=ticker_string,
                period=period,
                interval=interval,
                group_by='ticker',
                threads=True,
                progress=False
            )
            
            if data.empty:
                self.logger.warning("No data retrieved in batch fetch")
                return pd.DataFrame()
            
            # Process multi-level columns if multiple tickers
            if len(tickers) > 1:
                # Reorganize data by symbol
                processed_data = {}
                
                for i, (symbol, ticker) in enumerate(zip(symbols, tickers)):
                    symbol_data = pd.DataFrame()
                    
                    # Extract data for this ticker
                    for metric in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if (ticker, metric) in data.columns:
                            symbol_data[metric.lower()] = data[(ticker, metric)]
                        elif metric in data.columns:  # Single ticker case
                            symbol_data[metric.lower()] = data[metric]
                    
                    if not symbol_data.empty:
                        symbol_data['symbol'] = symbol
                        symbol_data['price'] = symbol_data['close']
                        symbol_data['market_cap'] = symbol_data['volume'] * symbol_data['close']
                        processed_data[symbol] = symbol_data
                
                return processed_data
            else:
                # Single ticker - process directly
                data.columns = data.columns.str.lower()
                data['symbol'] = symbols[0]
                data['price'] = data['close']
                data['market_cap'] = data['volume'] * data['close']
                return {symbols[0]: data}
                
        except Exception as e:
            self.logger.error(f"Error in batch fetch: {e}")
            return {}
    
    def get_latest_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get latest prices for symbols."""
        symbols = symbols or self.config.symbols
        prices = {}
        
        for symbol in symbols:
            yf_ticker = self.get_yf_ticker(symbol)
            
            try:
                ticker = yf.Ticker(yf_ticker)
                info = ticker.info
                
                # Try different price fields
                price = info.get('regularMarketPrice') or info.get('price') or info.get('previousClose')
                
                if price:
                    prices[symbol] = float(price)
                    self.logger.info(f"{symbol}: ${price:.2f}")
                else:
                    self.logger.warning(f"Could not get price for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error getting price for {symbol}: {e}")
        
        return prices
    
    def combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple symbols into a single DataFrame."""
        if not data_dict:
            return pd.DataFrame()
        
        # Create a list to store all dataframes
        dfs = []
        
        for symbol, df in data_dict.items():
            # Add symbol prefix to columns (except timestamp and symbol)
            df_copy = df.copy()
            
            # Rename columns with symbol prefix
            for col in df_copy.columns:
                if col not in ['symbol']:
                    df_copy.rename(columns={col: f"{symbol}_{col}"}, inplace=True)
            
            # Keep symbol column without prefix
            if 'symbol' in df.columns:
                df_copy['symbol'] = df['symbol']
            
            dfs.append(df_copy)
        
        # Combine all dataframes on timestamp index
        combined_df = pd.concat(dfs, axis=1, sort=True)
        
        # Remove duplicate symbol columns
        symbol_cols = [col for col in combined_df.columns if col == 'symbol']
        if len(symbol_cols) > 1:
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        # Fill missing values with forward fill then backward fill
        combined_df = combined_df.ffill().bfill()
        
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


def test_yfinance_fetcher():
    """Test the yfinance fetcher with real data."""
    from utils.config import get_default_config
    
    config = get_default_config()
    fetcher = YFinanceCryptoFetcher(config.data)
    
    # Test single symbol fetch
    print("\n=== Testing single symbol fetch ===")
    btc_data = fetcher.fetch_crypto_data('bitcoin', period='1mo', interval='1h')
    if not btc_data.empty:
        print(f"BTC data shape: {btc_data.shape}")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"Latest BTC price: ${btc_data['close'].iloc[-1]:,.2f}")
        print("\nFirst few rows:")
        print(btc_data.head())
    
    # Test batch fetch
    print("\n=== Testing batch fetch ===")
    test_symbols = ['bitcoin', 'ethereum', 'solana']
    batch_data = fetcher.fetch_batch_data(test_symbols, period='7d', interval='1h')
    
    for symbol, df in batch_data.items():
        print(f"\n{symbol}: {df.shape[0]} records")
        if not df.empty:
            print(f"Latest {symbol} price: ${df['close'].iloc[-1]:,.2f}")
    
    # Test latest prices
    print("\n=== Testing latest prices ===")
    prices = fetcher.get_latest_prices(test_symbols)
    for symbol, price in prices.items():
        print(f"{symbol}: ${price:,.2f}")
    
    # Test full data fetch
    print("\n=== Testing full data fetch ===")
    all_data = fetcher.fetch_all_symbols(test_symbols)
    
    if all_data:
        combined = fetcher.combine_data(all_data)
        clean = fetcher.get_clean_data(combined)
        
        print(f"\nCombined data shape: {combined.shape}")
        print(f"Clean data shape: {clean.shape}")
        print(f"Columns: {list(clean.columns[:10])}...")


if __name__ == "__main__":
    test_yfinance_fetcher()