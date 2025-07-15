#!/usr/bin/env python3
"""
4-Hour Cryptocurrency Data Fetcher and Cleaner

Fetches extensive 4-hour interval cryptocurrency data for Random Forest model training.
Includes comprehensive data cleaning, feature engineering, and preprocessing pipeline.

Usage: python3 data_fetcher_4h.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Technical analysis libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class CryptoDataFetcher4H:
    """Fetches and cleans 4-hour cryptocurrency data for ML training."""
    
    def __init__(self, output_dir: str = "data/4h_training"):
        self.output_dir = output_dir
        self.symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD',
            'LINK-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD', 'XLM-USD',
            'BCH-USD', 'FIL-USD', 'TRX-USD', 'ETC-USD', 'XMR-USD'
        ]
        self.interval = '4h'
        self.lookback_months = 12  # Fetch 12 months of 4h data
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_crypto_data(self, symbol: str, months_back: int = 12) -> pd.DataFrame:
        """Fetch 4-hour data for a specific cryptocurrency."""
        print(f"📈 Fetching {symbol} 4h data ({months_back} months)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=self.interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                print(f"❌ No data retrieved for {symbol}")
                return pd.DataFrame()
                
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol identifier
            data['symbol'] = symbol.replace('-USD', '')
            
            # Remove any rows with missing essential data
            data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            print(f"✅ {symbol}: {len(data)} 4h candles retrieved")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        print("🔧 Calculating technical indicators...")
        
        df = data.copy()
        
        # Price-based indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_1'] = df['close'].pct_change(1)
        df['momentum_4'] = df['close'].pct_change(4)
        df['momentum_12'] = df['close'].pct_change(12)
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['atr'] = self._calculate_atr(df)
        
        # Support/Resistance levels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['range_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Advanced indicators using TA-Lib if available
        if TALIB_AVAILABLE:
            try:
                # Stochastic
                df['stoch_k'], df['stoch_d'] = talib.STOCH(
                    df['high'].values,
                    df['low'].values, 
                    df['close'].values
                )
                
                # Williams %R
                df['williams_r'] = talib.WILLR(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values
                )
                
                # Commodity Channel Index
                df['cci'] = talib.CCI(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values
                )
                
                # Average Directional Index
                df['adx'] = talib.ADX(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values
                )
                
            except Exception as e:
                print(f"⚠️ TA-Lib indicators failed: {e}")
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return true_range.rolling(period).mean()
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for ML training."""
        print("⚙️ Engineering ML features...")
        
        df = data.copy()
        
        # Price patterns
        df['doji'] = ((np.abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & 
                       ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
        
        # Trend strength
        df['trend_strength'] = np.abs(df['sma_10'] - df['sma_20']) / df['close']
        
        # Multi-timeframe momentum
        for period in [2, 6, 12, 24]:
            df[f'momentum_{period}h'] = df['close'].pct_change(period)
            df[f'volume_change_{period}h'] = df['volume'].pct_change(period)
        
        # Price position indicators
        df['price_vs_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['market_structure'] = df['higher_high'] + df['higher_low']  # 0=bearish, 1=mixed, 2=bullish
        
        # Volume profile
        df['volume_percentile'] = df['volume'].rolling(50).rank(pct=True)
        df['price_volume_correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        return df
    
    def create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for Random Forest training."""
        print("🎯 Creating target variables...")
        
        df = data.copy()
        
        # Future price movements (classification targets)
        for horizon in [1, 2, 4, 8]:  # 4h, 8h, 16h, 32h ahead
            df[f'price_up_{horizon}h'] = (df['close'].shift(-horizon) > df['close']).astype(int)
            df[f'price_change_{horizon}h'] = (df['close'].shift(-horizon) / df['close'] - 1)
            
            # Categorize price movements
            df[f'movement_{horizon}h'] = pd.cut(
                df[f'price_change_{horizon}h'],
                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                labels=['strong_down', 'down', 'sideways', 'up', 'strong_up']
            )
        
        # Volatility targets
        df['future_volatility'] = df['close'].rolling(4).std().shift(-4)
        df['high_volatility'] = (df['future_volatility'] > df['volatility'].rolling(20).quantile(0.8)).astype(int)
        
        # Trend continuation
        df['trend_continues'] = (
            (df['close'].shift(-4) > df['close']) == (df['close'] > df['close'].shift(-4))
        ).astype(int)
        
        return df
    
    def clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        print("🧹 Cleaning and validating data...")
        
        df = data.copy()
        
        # Remove rows with excessive missing values
        missing_threshold = 0.3
        df = df.dropna(thresh=int(len(df.columns) * (1 - missing_threshold)))
        
        # Remove outliers using IQR method for key price features
        price_features = ['open', 'high', 'low', 'close', 'volume']
        for feature in price_features:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
        
        # Forward fill remaining missing values for indicators
        indicator_columns = [col for col in df.columns if col not in price_features + ['symbol']]
        df[indicator_columns] = df[indicator_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        print(f"✅ Data cleaned: {len(df)} rows remaining")
        return df
    
    def fetch_all_symbols(self) -> pd.DataFrame:
        """Fetch data for all cryptocurrency symbols."""
        print(f"🚀 Fetching 4h data for {len(self.symbols)} cryptocurrencies")
        print("=" * 60)
        
        all_data = []
        
        for symbol in self.symbols:
            try:
                # Fetch raw data
                raw_data = self.fetch_crypto_data(symbol, self.lookback_months)
                
                if raw_data.empty:
                    continue
                
                # Calculate indicators
                data_with_indicators = self.calculate_technical_indicators(raw_data)
                
                # Engineer features
                data_with_features = self.engineer_features(data_with_indicators)
                
                # Create targets
                data_with_targets = self.create_target_variables(data_with_features)
                
                # Clean data
                clean_data = self.clean_and_validate_data(data_with_targets)
                
                if not clean_data.empty:
                    all_data.append(clean_data)
                    print(f"✅ {symbol}: {len(clean_data)} clean records")
                
            except Exception as e:
                print(f"❌ Failed to process {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data was successfully fetched")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()
        
        print("=" * 60)
        print(f"🎉 Combined dataset: {len(combined_data)} total records")
        print(f"📊 Features: {len(combined_data.columns)} columns")
        print(f"📅 Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        return combined_data
    
    def save_dataset(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save the processed dataset."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'crypto_4h_dataset_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath)
        
        print(f"💾 Dataset saved: {filepath}")
        print(f"📊 Shape: {data.shape}")
        
        # Save feature info
        feature_info = {
            'total_features': len(data.columns),
            'price_features': [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'volume']],
            'technical_indicators': [col for col in data.columns if any(x in col for x in ['sma', 'ema', 'rsi', 'macd', 'bb_'])],
            'engineered_features': [col for col in data.columns if any(x in col for x in ['momentum', 'trend', 'volatility'])],
            'target_variables': [col for col in data.columns if any(x in col for x in ['price_up', 'movement', 'high_volatility'])],
            'symbols': data['symbol'].unique().tolist() if 'symbol' in data.columns else [],
            'date_range': {
                'start': str(data.index.min()),
                'end': str(data.index.max()),
                'total_periods': len(data)
            }
        }
        
        info_filepath = os.path.join(self.output_dir, filename.replace('.csv', '_info.json'))
        import json
        with open(info_filepath, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"📋 Feature info saved: {info_filepath}")
        return filepath
    
    def run_complete_pipeline(self) -> str:
        """Run the complete data fetching and processing pipeline."""
        print("🚀 Starting Complete 4H Crypto Data Pipeline")
        print("=" * 70)
        
        start_time = datetime.now()
        
        try:
            # Fetch and process all data
            dataset = self.fetch_all_symbols()
            
            # Save dataset
            filepath = self.save_dataset(dataset)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("=" * 70)
            print(f"✅ Pipeline completed successfully!")
            print(f"⏱️ Duration: {duration}")
            print(f"📁 Output: {filepath}")
            print(f"🎯 Ready for Random Forest training!")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main function to run the data fetching pipeline."""
    fetcher = CryptoDataFetcher4H()
    
    try:
        dataset_path = fetcher.run_complete_pipeline()
        print(f"\n🎉 Success! Dataset ready for Random Forest training:")
        print(f"📄 File: {dataset_path}")
        
    except Exception as e:
        print(f"\n❌ Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()