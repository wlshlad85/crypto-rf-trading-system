#!/usr/bin/env python3
"""
Phase 2A: Advanced Technical Indicators Engine
ULTRATHINK Implementation - Institutional-Grade Technical Analysis

Implements sophisticated technical indicators used by professional trading firms:
- Ichimoku Cloud System (complete analysis)
- Volume Profile Analysis
- On-Balance Volume (OBV) with trends
- Advanced momentum and volatility indicators
- Multi-timeframe signal aggregation

Designed to enhance feature quality and reduce overfitting risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    # Ichimoku parameters
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    
    # Volume Profile parameters
    volume_profile_bins: int = 20
    volume_profile_window: int = 100
    
    # OBV parameters
    obv_smooth_window: int = 10
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Stochastic parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30

class AdvancedTechnicalIndicators:
    """
    Professional technical indicators engine for cryptocurrency trading.
    
    Implements institutional-grade technical analysis with advanced indicators
    used by professional trading firms for enhanced feature engineering.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize advanced technical indicators engine.
        
        Args:
            config: Configuration for indicator parameters
        """
        self.config = config or IndicatorConfig()
        self.indicator_results = {}
        
        print("🔧 Advanced Technical Indicators Engine Initialized")
        print(f"📊 Ichimoku: Tenkan({self.config.ichimoku_tenkan}), Kijun({self.config.ichimoku_kijun}), Senkou B({self.config.ichimoku_senkou_b})")
        print(f"📈 Volume Profile: {self.config.volume_profile_bins} bins, {self.config.volume_profile_window} period window")
        print(f"🎯 OBV Smoothing: {self.config.obv_smooth_window} periods")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced technical indicators for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        print("\n🔧 Calculating Advanced Technical Indicators")
        print("=" * 50)
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = df.copy()
        
        # 1. Ichimoku Cloud System
        print("📊 Calculating Ichimoku Cloud System...")
        ichimoku_indicators = self.calculate_ichimoku_cloud(df)
        for name, values in ichimoku_indicators.items():
            result_df[name] = values
        
        # 2. Volume Profile Analysis
        print("📈 Calculating Volume Profile...")
        volume_profile_indicators = self.calculate_volume_profile(df)
        for name, values in volume_profile_indicators.items():
            result_df[name] = values
        
        # 3. On-Balance Volume (OBV)
        print("🎯 Calculating On-Balance Volume...")
        obv_indicators = self.calculate_obv_analysis(df)
        for name, values in obv_indicators.items():
            result_df[name] = values
        
        # 4. Bollinger Bands with additional analysis
        print("📊 Calculating Enhanced Bollinger Bands...")
        bb_indicators = self.calculate_bollinger_bands(df)
        for name, values in bb_indicators.items():
            result_df[name] = values
        
        # 5. Advanced MACD
        print("📈 Calculating Advanced MACD...")
        macd_indicators = self.calculate_advanced_macd(df)
        for name, values in macd_indicators.items():
            result_df[name] = values
        
        # 6. Stochastic Oscillator
        print("🎯 Calculating Stochastic Oscillator...")
        stoch_indicators = self.calculate_stochastic(df)
        for name, values in stoch_indicators.items():
            result_df[name] = values
        
        # 7. Enhanced RSI
        print("📊 Calculating Enhanced RSI...")
        rsi_indicators = self.calculate_enhanced_rsi(df)
        for name, values in rsi_indicators.items():
            result_df[name] = values
        
        # 8. Volatility Indicators
        print("📈 Calculating Volatility Indicators...")
        vol_indicators = self.calculate_volatility_indicators(df)
        for name, values in vol_indicators.items():
            result_df[name] = values
        
        # 9. Momentum Indicators
        print("🚀 Calculating Momentum Indicators...")
        momentum_indicators = self.calculate_momentum_indicators(df)
        for name, values in momentum_indicators.items():
            result_df[name] = values
        
        # Count new indicators
        original_columns = len(df.columns)
        new_columns = len(result_df.columns)
        indicators_added = new_columns - original_columns
        
        print(f"\n✅ Advanced Technical Indicators Complete")
        print(f"📊 Original Features: {original_columns}")
        print(f"🔧 Indicators Added: {indicators_added}")
        print(f"📈 Total Features: {new_columns}")
        
        return result_df
    
    def calculate_ichimoku_cloud(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate complete Ichimoku Cloud system."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(self.config.ichimoku_tenkan).max()
        tenkan_low = low.rolling(self.config.ichimoku_tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(self.config.ichimoku_kijun).max()
        kijun_low = low.rolling(self.config.ichimoku_kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.config.ichimoku_kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(self.config.ichimoku_senkou_b).max()
        senkou_low = low.rolling(self.config.ichimoku_senkou_b).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.config.ichimoku_kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-self.config.ichimoku_kijun)
        
        # Cloud thickness and position
        cloud_thickness = abs(senkou_span_a - senkou_span_b)
        cloud_top = np.maximum(senkou_span_a, senkou_span_b)
        cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
        
        # Price position relative to cloud
        price_above_cloud = (close > cloud_top).astype(int)
        price_below_cloud = (close < cloud_bottom).astype(int)
        price_in_cloud = ((close >= cloud_bottom) & (close <= cloud_top)).astype(int)
        
        # Ichimoku signals
        bullish_tk_cross = ((tenkan_sen > kijun_sen) & (tenkan_sen.shift(1) <= kijun_sen.shift(1))).astype(int)
        bearish_tk_cross = ((tenkan_sen < kijun_sen) & (tenkan_sen.shift(1) >= kijun_sen.shift(1))).astype(int)
        
        return {
            'ichimoku_tenkan_sen': tenkan_sen,
            'ichimoku_kijun_sen': kijun_sen,
            'ichimoku_senkou_span_a': senkou_span_a,
            'ichimoku_senkou_span_b': senkou_span_b,
            'ichimoku_chikou_span': chikou_span,
            'ichimoku_cloud_thickness': cloud_thickness,
            'ichimoku_price_above_cloud': price_above_cloud,
            'ichimoku_price_below_cloud': price_below_cloud,
            'ichimoku_price_in_cloud': price_in_cloud,
            'ichimoku_bullish_tk_cross': bullish_tk_cross,
            'ichimoku_bearish_tk_cross': bearish_tk_cross
        }
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Volume Profile analysis."""
        close = df['Close']
        volume = df['Volume']
        
        # Initialize result series
        vp_poc = pd.Series(index=df.index, dtype=float)  # Point of Control
        vp_value_area_high = pd.Series(index=df.index, dtype=float)
        vp_value_area_low = pd.Series(index=df.index, dtype=float)
        vp_volume_at_price = pd.Series(index=df.index, dtype=float)
        
        window = self.config.volume_profile_window
        
        for i in range(window, len(df)):
            # Get window data
            window_data = df.iloc[i-window:i]
            
            if len(window_data) < window:
                continue
            
            # Create price bins
            price_min = window_data['Low'].min()
            price_max = window_data['High'].max()
            
            if price_max == price_min:
                continue
            
            price_bins = np.linspace(price_min, price_max, self.config.volume_profile_bins)
            volume_by_price = np.zeros(len(price_bins) - 1)
            
            # Distribute volume across price levels
            for _, row in window_data.iterrows():
                # Simple approximation: distribute volume evenly across OHLC
                prices = [row['Open'], row['High'], row['Low'], row['Close']]
                vol_per_price = row['Volume'] / 4
                
                for price in prices:
                    bin_idx = np.digitize(price, price_bins) - 1
                    if 0 <= bin_idx < len(volume_by_price):
                        volume_by_price[bin_idx] += vol_per_price
            
            # Find Point of Control (highest volume price level)
            poc_idx = np.argmax(volume_by_price)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_by_price)
            target_volume = total_volume * 0.7
            
            # Find value area bounds
            cumulative_volume = 0
            value_area_indices = []
            
            # Start from POC and expand outward
            indices = list(range(len(volume_by_price)))
            indices.sort(key=lambda x: volume_by_price[x], reverse=True)
            
            for idx in indices:
                value_area_indices.append(idx)
                cumulative_volume += volume_by_price[idx]
                if cumulative_volume >= target_volume:
                    break
            
            if value_area_indices:
                va_low_idx = min(value_area_indices)
                va_high_idx = max(value_area_indices)
                va_low_price = price_bins[va_low_idx]
                va_high_price = price_bins[va_high_idx + 1]
            else:
                va_low_price = price_min
                va_high_price = price_max
            
            # Store results
            current_idx = df.index[i]
            vp_poc.loc[current_idx] = poc_price
            vp_value_area_high.loc[current_idx] = va_high_price
            vp_value_area_low.loc[current_idx] = va_low_price
            
            # Volume at current price level
            current_price = close.iloc[i]
            price_bin_idx = np.digitize(current_price, price_bins) - 1
            if 0 <= price_bin_idx < len(volume_by_price):
                vp_volume_at_price.loc[current_idx] = volume_by_price[price_bin_idx]
        
        # Additional volume profile indicators
        vp_price_position = (close - vp_value_area_low) / (vp_value_area_high - vp_value_area_low)
        vp_above_poc = (close > vp_poc).astype(int)
        vp_in_value_area = ((close >= vp_value_area_low) & (close <= vp_value_area_high)).astype(int)
        
        return {
            'vp_point_of_control': vp_poc,
            'vp_value_area_high': vp_value_area_high,
            'vp_value_area_low': vp_value_area_low,
            'vp_volume_at_price': vp_volume_at_price,
            'vp_price_position': vp_price_position,
            'vp_above_poc': vp_above_poc,
            'vp_in_value_area': vp_in_value_area
        }
    
    def calculate_obv_analysis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate On-Balance Volume with additional analysis."""
        close = df['Close']
        volume = df['Volume']
        
        # Basic OBV calculation
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # OBV smoothed
        obv_smoothed = obv.rolling(self.config.obv_smooth_window).mean()
        
        # OBV momentum
        obv_momentum = obv.pct_change(periods=5)
        
        # OBV divergence signals
        price_change = close.pct_change(periods=5)
        obv_change = obv.pct_change(periods=5)
        
        # Bullish divergence: price down, OBV up
        bullish_divergence = ((price_change < 0) & (obv_change > 0)).astype(int)
        
        # Bearish divergence: price up, OBV down
        bearish_divergence = ((price_change > 0) & (obv_change < 0)).astype(int)
        
        # OBV trend
        obv_trend = np.where(obv > obv.shift(1), 1, np.where(obv < obv.shift(1), -1, 0))
        
        # OBV accumulation/distribution
        obv_normalized = (obv - obv.rolling(50).min()) / (obv.rolling(50).max() - obv.rolling(50).min())
        
        return {
            'obv': obv,
            'obv_smoothed': obv_smoothed,
            'obv_momentum': obv_momentum,
            'obv_bullish_divergence': bullish_divergence,
            'obv_bearish_divergence': bearish_divergence,
            'obv_trend': obv_trend,
            'obv_normalized': obv_normalized
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate enhanced Bollinger Bands with additional metrics."""
        close = df['Close']
        
        # Standard Bollinger Bands
        bb_middle = close.rolling(self.config.bb_period).mean()
        bb_std = close.rolling(self.config.bb_period).std()
        bb_upper = bb_middle + (bb_std * self.config.bb_std)
        bb_lower = bb_middle - (bb_std * self.config.bb_std)
        
        # Bollinger Band Width
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # %B (position within bands)
        bb_percent = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Bollinger Band squeeze (low volatility)
        bb_squeeze = bb_width < bb_width.rolling(20).mean() * 0.8
        
        # Band touches
        bb_upper_touch = (close >= bb_upper * 0.99).astype(int)
        bb_lower_touch = (close <= bb_lower * 1.01).astype(int)
        
        # Band breakouts
        bb_upper_breakout = (close > bb_upper).astype(int)
        bb_lower_breakout = (close < bb_lower).astype(int)
        
        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'bb_percent': bb_percent,
            'bb_squeeze': bb_squeeze.astype(int),
            'bb_upper_touch': bb_upper_touch,
            'bb_lower_touch': bb_lower_touch,
            'bb_upper_breakout': bb_upper_breakout,
            'bb_lower_breakout': bb_lower_breakout
        }
    
    def calculate_advanced_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate advanced MACD with additional analysis."""
        close = df['Close']
        
        # MACD calculation
        ema_fast = close.ewm(span=self.config.macd_fast).mean()
        ema_slow = close.ewm(span=self.config.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.config.macd_signal).mean()
        macd_histogram = macd_line - macd_signal
        
        # MACD crossovers
        macd_bullish_cross = ((macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))).astype(int)
        macd_bearish_cross = ((macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))).astype(int)
        
        # MACD divergence
        price_change = close.pct_change(periods=10)
        macd_change = macd_line.pct_change(periods=10)
        
        macd_bullish_divergence = ((price_change < 0) & (macd_change > 0)).astype(int)
        macd_bearish_divergence = ((price_change > 0) & (macd_change < 0)).astype(int)
        
        # MACD momentum
        macd_momentum = macd_histogram.pct_change()
        
        # MACD normalized
        macd_normalized = (macd_line - macd_line.rolling(50).mean()) / macd_line.rolling(50).std()
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'macd_bullish_cross': macd_bullish_cross,
            'macd_bearish_cross': macd_bearish_cross,
            'macd_bullish_divergence': macd_bullish_divergence,
            'macd_bearish_divergence': macd_bearish_divergence,
            'macd_momentum': macd_momentum,
            'macd_normalized': macd_normalized
        }
    
    def calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator with additional analysis."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Stochastic %K
        lowest_low = low.rolling(self.config.stoch_k_period).min()
        highest_high = high.rolling(self.config.stoch_k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Stochastic %D
        stoch_d = stoch_k.rolling(self.config.stoch_d_period).mean()
        
        # Stochastic signals
        stoch_overbought = (stoch_k > 80).astype(int)
        stoch_oversold = (stoch_k < 20).astype(int)
        
        # Stochastic crossovers
        stoch_bullish_cross = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
        stoch_bearish_cross = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)
        
        # Stochastic momentum
        stoch_momentum = stoch_k.pct_change()
        
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_overbought': stoch_overbought,
            'stoch_oversold': stoch_oversold,
            'stoch_bullish_cross': stoch_bullish_cross,
            'stoch_bearish_cross': stoch_bearish_cross,
            'stoch_momentum': stoch_momentum
        }
    
    def calculate_enhanced_rsi(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate enhanced RSI with additional analysis."""
        close = df['Close']
        
        # RSI calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # RSI conditions
        rsi_overbought = (rsi > self.config.rsi_overbought).astype(int)
        rsi_oversold = (rsi < self.config.rsi_oversold).astype(int)
        
        # RSI divergence
        price_change = close.pct_change(periods=10)
        rsi_change = rsi.pct_change(periods=10)
        
        rsi_bullish_divergence = ((price_change < 0) & (rsi_change > 0)).astype(int)
        rsi_bearish_divergence = ((price_change > 0) & (rsi_change < 0)).astype(int)
        
        # RSI momentum
        rsi_momentum = rsi.pct_change()
        
        # RSI smoothed
        rsi_smoothed = rsi.rolling(5).mean()
        
        return {
            'rsi': rsi,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'rsi_bullish_divergence': rsi_bullish_divergence,
            'rsi_bearish_divergence': rsi_bearish_divergence,
            'rsi_momentum': rsi_momentum,
            'rsi_smoothed': rsi_smoothed
        }
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various volatility indicators."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Average True Range
        atr = true_range.rolling(14).mean()
        
        # Normalized ATR
        atr_normalized = atr / close
        
        # ATR-based volatility bands
        atr_upper = close + (atr * 2)
        atr_lower = close - (atr * 2)
        
        # Keltner Channels
        keltner_middle = close.rolling(20).mean()
        keltner_upper = keltner_middle + (atr * 2)
        keltner_lower = keltner_middle - (atr * 2)
        
        # Volatility breakouts
        vol_breakout_up = (close > close.shift(1) + atr).astype(int)
        vol_breakout_down = (close < close.shift(1) - atr).astype(int)
        
        # Price volatility
        price_volatility = close.pct_change().rolling(20).std()
        
        return {
            'atr': atr,
            'atr_normalized': atr_normalized,
            'atr_upper': atr_upper,
            'atr_lower': atr_lower,
            'keltner_upper': keltner_upper,
            'keltner_middle': keltner_middle,
            'keltner_lower': keltner_lower,
            'vol_breakout_up': vol_breakout_up,
            'vol_breakout_down': vol_breakout_down,
            'price_volatility': price_volatility
        }
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various momentum indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Rate of Change (ROC)
        roc_5 = close.pct_change(periods=5) * 100
        roc_10 = close.pct_change(periods=10) * 100
        roc_20 = close.pct_change(periods=20) * 100
        
        # Momentum
        momentum_5 = close / close.shift(5)
        momentum_10 = close / close.shift(10)
        
        # Williams %R
        highest_high_14 = high.rolling(14).max()
        lowest_low_14 = low.rolling(14).min()
        williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
        
        # Commodity Channel Index (CCI)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        # Price momentum acceleration
        price_acceleration = close.pct_change().pct_change()
        
        return {
            'roc_5': roc_5,
            'roc_10': roc_10,
            'roc_20': roc_20,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'williams_r': williams_r,
            'cci': cci,
            'price_acceleration': price_acceleration
        }
    
    def get_feature_importance_analysis(self, df_with_indicators: pd.DataFrame) -> Dict:
        """Analyze the importance and quality of generated indicators."""
        indicator_columns = [col for col in df_with_indicators.columns 
                           if any(prefix in col for prefix in [
                               'ichimoku', 'vp_', 'obv', 'bb_', 'macd', 'stoch', 'rsi', 'atr', 'roc', 'momentum', 'williams', 'cci'
                           ])]
        
        analysis = {
            'total_indicators': len(indicator_columns),
            'indicator_categories': {
                'ichimoku': len([c for c in indicator_columns if 'ichimoku' in c]),
                'volume_profile': len([c for c in indicator_columns if 'vp_' in c]),
                'obv': len([c for c in indicator_columns if 'obv' in c]),
                'bollinger_bands': len([c for c in indicator_columns if 'bb_' in c]),
                'macd': len([c for c in indicator_columns if 'macd' in c]),
                'stochastic': len([c for c in indicator_columns if 'stoch' in c]),
                'rsi': len([c for c in indicator_columns if 'rsi' in c]),
                'volatility': len([c for c in indicator_columns if any(v in c for v in ['atr', 'vol_', 'keltner'])]),
                'momentum': len([c for c in indicator_columns if any(m in c for m in ['roc', 'momentum', 'williams', 'cci'])])
            },
            'data_quality': {
                'total_samples': len(df_with_indicators),
                'complete_samples': len(df_with_indicators.dropna()),
                'completeness_ratio': len(df_with_indicators.dropna()) / len(df_with_indicators)
            }
        }
        
        return analysis

def main():
    """Demonstrate advanced technical indicators engine."""
    print("🔧 PHASE 2A: Advanced Technical Indicators Engine")
    print("ULTRATHINK Implementation - Institutional-Grade Features")
    print("=" * 60)
    
    # Load sample data from Phase 1
    data_dir = "phase1/data/processed"
    import glob
    import os
    
    data_files = glob.glob(f"{data_dir}/BTC-USD_*.csv")
    if not data_files:
        print("❌ No data files found. Run Phase 1A first.")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"📂 Loading data from: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    print(f"📊 Data loaded: {len(df)} samples")
    print(f"📈 Columns: {list(df.columns)}")
    
    # Initialize advanced technical indicators engine
    config = IndicatorConfig()
    engine = AdvancedTechnicalIndicators(config)
    
    # Calculate all indicators
    try:
        enhanced_df = engine.calculate_all_indicators(df)
        
        # Feature importance analysis
        importance_analysis = engine.get_feature_importance_analysis(enhanced_df)
        
        print(f"\n📊 FEATURE ENHANCEMENT SUMMARY:")
        print(f"   Total Indicators: {importance_analysis['total_indicators']}")
        print(f"   Data Completeness: {importance_analysis['data_quality']['completeness_ratio']:.1%}")
        
        print(f"\n🔧 INDICATOR CATEGORIES:")
        for category, count in importance_analysis['indicator_categories'].items():
            print(f"   {category.replace('_', ' ').title()}: {count}")
        
        # Show sample of new features
        indicator_cols = [col for col in enhanced_df.columns if col not in df.columns]
        print(f"\n📈 SAMPLE INDICATORS (showing first 10):")
        for i, col in enumerate(indicator_cols[:10]):
            print(f"   {i+1}. {col}")
        
        if len(indicator_cols) > 10:
            print(f"   ... and {len(indicator_cols) - 10} more")
        
        # Save enhanced dataset
        output_file = "phase2/enhanced_features_with_advanced_indicators.csv"
        enhanced_df.to_csv(output_file)
        print(f"\n💾 Enhanced dataset saved: {output_file}")
        
        print(f"\n🚀 Phase 2A Advanced Technical Indicators: COMPLETE")
        print(f"📊 Feature Enhancement: {len(df.columns)} → {len(enhanced_df.columns)} features")
        print(f"🎯 Ready for Phase 2A Next Step: On-Chain Features")
        
    except Exception as e:
        print(f"❌ Error calculating indicators: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()