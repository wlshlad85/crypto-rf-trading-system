#!/usr/bin/env python3
"""
Enhanced Momentum Feature Engineering for Random Forest

Creates sophisticated momentum and pattern recognition features based on
successful trading patterns: 1.780%/hour threshold, 0.6-1.2h cycles, optimal position sizing.

Usage: from enhanced_momentum_features import MomentumFeatureEngineer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class MomentumFeatureEngineer:
    """Enhanced momentum feature engineering based on successful trading patterns."""
    
    def __init__(self, pattern_insights: Dict = None):
        self.pattern_insights = pattern_insights or self._get_default_patterns()
        self.momentum_threshold = 1.780  # From pattern analysis
        self.optimal_position_range = (0.464, 0.800)  # BTC range
        self.optimal_duration_range = (0.6, 1.2)  # Hours
        
    def _get_default_patterns(self) -> Dict:
        """Default pattern insights from successful trading analysis."""
        return {
            'momentum_threshold': 1.780,
            'optimal_position_size': {'min': 0.464, 'max': 0.800, 'median': 0.588},
            'optimal_duration': {'min': 0.6, 'max': 1.2, 'median': 0.9},
            'best_start_hours': [3],
            'avg_momentum_strength': 1.230,
            'success_rate': 0.636
        }
    
    def create_enhanced_features(self, data: pd.DataFrame, symbol: str = 'BTC') -> pd.DataFrame:
        """Create comprehensive momentum and pattern features."""
        print(f"🚀 Creating enhanced momentum features for {symbol}...")
        
        df = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Basic momentum features
        df = self._add_price_momentum_features(df)
        
        # Volume momentum features
        df = self._add_volume_momentum_features(df)
        
        # Pattern recognition features
        df = self._add_pattern_recognition_features(df)
        
        # Market structure features
        df = self._add_market_structure_features(df)
        
        # Volatility regime features
        df = self._add_volatility_regime_features(df)
        
        # Time-based momentum features
        df = self._add_time_momentum_features(df)
        
        # Position optimization features
        df = self._add_position_optimization_features(df)
        
        # Success probability features
        df = self._add_success_probability_features(df)
        
        print(f"✅ Enhanced features created: {len(df.columns)} total columns")
        return df
    
    def _add_price_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sophisticated price momentum features."""
        
        # Multi-timeframe momentum (based on successful 0.6-1.2h cycles)
        for periods in [2, 4, 6, 12, 24]:  # 8h, 16h, 24h, 48h, 96h for 4h data
            df[f'momentum_{periods}p'] = df['close'].pct_change(periods) * 100
            df[f'momentum_strength_{periods}p'] = df[f'momentum_{periods}p'] / periods  # %/period
        
        # Exponential momentum (gives more weight to recent price action)
        df['ema_momentum_fast'] = df['close'].ewm(span=3).mean().pct_change() * 100
        df['ema_momentum_slow'] = df['close'].ewm(span=12).mean().pct_change() * 100
        df['ema_momentum_diff'] = df['ema_momentum_fast'] - df['ema_momentum_slow']
        
        # Rate of change acceleration
        df['roc_2'] = df['close'].pct_change(2) * 100
        df['roc_4'] = df['close'].pct_change(4) * 100
        df['momentum_acceleration'] = df['roc_2'] - df['roc_4']
        
        # Momentum strength classification (based on 1.780%/hour threshold)
        df['is_high_momentum'] = (df['momentum_strength_2p'] > 1.780).astype(int)
        df['momentum_regime'] = pd.cut(
            df['momentum_strength_2p'],
            bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, 1.780, np.inf],
            labels=['strong_down', 'down', 'weak', 'up', 'strong', 'explosive']
        )
        
        # Price velocity and acceleration
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1) * 100
        df['price_acceleration'] = df['price_velocity'].diff()
        
        return df
    
    def _add_volume_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based momentum features."""
        
        # Volume rate of change
        df['volume_roc_2'] = df['volume'].pct_change(2) * 100
        df['volume_roc_4'] = df['volume'].pct_change(4) * 100
        df['volume_momentum'] = df['volume_roc_2'] - df['volume_roc_4']
        
        # Volume-price relationship
        df['volume_price_correlation'] = df['close'].rolling(12).corr(df['volume'])
        df['volume_weighted_momentum'] = df['momentum_2p'] * (df['volume'] / df['volume'].rolling(20).mean())
        
        # On-Balance Volume momentum
        df['obv'] = (df['volume'] * ((df['close'] > df['close'].shift(1)).astype(int) * 2 - 1)).cumsum()
        df['obv_momentum'] = df['obv'].pct_change(4) * 100
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(6).mean() / df['volume'].rolling(12).mean()
        df['is_volume_increasing'] = (df['volume_trend'] > 1.0).astype(int)
        
        return df
    
    def _add_pattern_recognition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick and price pattern features."""
        
        # Candlestick patterns
        df['body_size'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
        
        # Doji patterns (indecision)
        df['is_doji'] = (df['body_size'] < 0.1).astype(int)
        
        # Hammer/shooting star patterns
        df['is_hammer'] = ((df['lower_shadow'] > 0.6) & (df['upper_shadow'] < 0.1) & (df['body_size'] < 0.3)).astype(int)
        df['is_shooting_star'] = ((df['upper_shadow'] > 0.6) & (df['lower_shadow'] < 0.1) & (df['body_size'] < 0.3)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &  # Current green
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous red
            (df['open'] < df['close'].shift(1)) &  # Open below previous close
            (df['close'] > df['open'].shift(1))  # Close above previous open
        ).astype(int)
        
        # Price gaps
        df['gap_up'] = ((df['open'] > df['high'].shift(1)) & (df['momentum_2p'] > 0)).astype(int)
        df['gap_down'] = ((df['open'] < df['low'].shift(1)) & (df['momentum_2p'] < 0)).astype(int)
        
        # Trend continuation patterns
        df['trend_continuation'] = (
            (df['momentum_2p'] > 0) & (df['momentum_4p'] > 0) & (df['momentum_6p'] > 0)
        ).astype(int)
        
        return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and trend features."""
        
        # Higher highs and higher lows
        df['higher_high'] = (df['high'] > df['high'].rolling(4).max().shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].rolling(4).min().shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].rolling(4).max().shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].rolling(4).min().shift(1)).astype(int)
        
        # Market structure score
        df['bullish_structure'] = df['higher_high'] + df['higher_low']
        df['bearish_structure'] = df['lower_high'] + df['lower_low']
        df['market_structure_score'] = df['bullish_structure'] - df['bearish_structure']
        
        # Trend strength
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        df['trend_alignment'] = (
            (df['close'] > df['sma_5']) & 
            (df['sma_5'] > df['sma_10']) & 
            (df['sma_10'] > df['sma_20'])
        ).astype(int)
        
        # Support and resistance levels
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_level'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close'] * 100
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close'] * 100
        
        return df
    
    def _add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime classification features."""
        
        # True Range and ATR
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['true_range'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Volatility regimes
        df['volatility_regime'] = pd.cut(
            df['atr_pct'],
            bins=[0, 2, 4, 6, np.inf],
            labels=['low', 'medium', 'high', 'extreme']
        )
        
        # Volatility momentum
        df['volatility_momentum'] = df['atr_pct'].pct_change(4) * 100
        df['is_vol_expanding'] = (df['volatility_momentum'] > 0).astype(int)
        
        # Price efficiency (how much price moves relative to volatility)
        df['price_efficiency'] = np.abs(df['momentum_4p']) / df['atr_pct']
        df['is_efficient_move'] = (df['price_efficiency'] > 1.0).astype(int)
        
        return df
    
    def _add_time_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based momentum features."""
        
        # Time of day effects (based on successful 3 AM pattern)
        df['hour'] = df.index.hour
        df['is_optimal_hour'] = (df['hour'] == 3).astype(int)
        df['is_asian_session'] = df['hour'].between(0, 8).astype(int)
        df['is_european_session'] = df['hour'].between(8, 16).astype(int)
        df['is_us_session'] = df['hour'].between(16, 24).astype(int)
        
        # Day of week effects
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Session momentum
        session_windows = {'asian': (0, 8), 'european': (8, 16), 'us': (16, 24)}
        for session, (start, end) in session_windows.items():
            mask = df['hour'].between(start, end)
            df[f'{session}_session_momentum'] = np.where(
                mask, 
                df['momentum_2p'],
                np.nan
            )
            df[f'{session}_session_momentum'] = df[f'{session}_session_momentum'].fillna(method='ffill')
        
        return df
    
    def _add_position_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for optimal position sizing based on successful patterns."""
        
        # Position size indicators (based on 0.464-0.800 BTC optimal range)
        df['optimal_position_signal'] = 0
        
        # Momentum-based position sizing
        df['momentum_position_size'] = np.where(
            df['is_high_momentum'] == 1,
            0.800,  # Max position for high momentum
            np.where(
                df['momentum_strength_2p'] > 0.5,
                0.588,  # Median position for medium momentum
                0.464   # Min position for low momentum
            )
        )
        
        # Volatility-adjusted position sizing
        df['vol_adjusted_position'] = np.where(
            df['atr_pct'] < 2.0,
            0.800,  # Larger position in low volatility
            np.where(
                df['atr_pct'] < 4.0,
                0.588,  # Medium position in medium volatility
                0.464   # Smaller position in high volatility
            )
        )
        
        # Risk-adjusted position sizing
        df['risk_adjusted_position'] = np.minimum(
            df['momentum_position_size'],
            df['vol_adjusted_position']
        )
        
        # Position timing indicators
        df['is_position_entry_optimal'] = (
            (df['is_high_momentum'] == 1) &
            (df['trend_alignment'] == 1) &
            (df['market_structure_score'] > 0) &
            (df['distance_to_resistance'] > 2.0)  # At least 2% below resistance
        ).astype(int)
        
        return df
    
    def _add_success_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that predict success probability based on patterns."""
        
        # Combine successful pattern indicators
        df['pattern_score'] = (
            df['is_high_momentum'] * 3 +  # High weight for momentum
            df['trend_alignment'] * 2 +   # Medium weight for trend
            df['is_optimal_hour'] * 1 +   # Low weight for timing
            df['bullish_structure'] * 1 + # Low weight for structure
            df['is_efficient_move'] * 1   # Low weight for efficiency
        )
        
        # Success probability based on historical patterns
        df['success_probability'] = np.clip(
            (df['pattern_score'] / 8.0) * 0.636,  # Scale by historical success rate
            0.0, 1.0
        )
        
        # Confidence level for trades
        df['confidence_level'] = pd.cut(
            df['success_probability'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # Entry signal strength
        df['entry_signal_strength'] = (
            df['is_position_entry_optimal'] * df['success_probability'] * 10
        ).round(1)
        
        # Exit signal based on optimal duration (0.6-1.2 hours)
        df['cycles_since_entry'] = range(len(df))  # Simplified - would track actual cycles
        df['is_exit_window'] = (
            (df['cycles_since_entry'] >= 2) &  # Min 8 hours (2 cycles of 4h)
            (df['cycles_since_entry'] <= 3)    # Max 12 hours (3 cycles of 4h)
        ).astype(int)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated target variables for Random Forest training."""
        
        # Multi-horizon price targets (based on successful 0.6-1.2h cycles)
        horizons = [1, 2, 3, 6]  # 4h, 8h, 12h, 24h
        
        for h in horizons:
            # Price movement prediction
            df[f'price_up_{h}h'] = (df['close'].shift(-h) > df['close']).astype(int)
            df[f'price_change_{h}h'] = (df['close'].shift(-h) / df['close'] - 1) * 100
            
            # Profit potential (based on successful 0.68% average return)
            df[f'profitable_{h}h'] = (df[f'price_change_{h}h'] > 0.68).astype(int)
            df[f'high_profit_{h}h'] = (df[f'price_change_{h}h'] > 1.5).astype(int)
            
            # Risk-adjusted returns
            future_vol = df['atr_pct'].shift(-h)
            df[f'risk_adj_return_{h}h'] = df[f'price_change_{h}h'] / future_vol
            df[f'good_risk_adj_{h}h'] = (df[f'risk_adj_return_{h}h'] > 0.5).astype(int)
        
        # Optimal cycle detection (based on 0.6-1.2h successful cycles)
        df['optimal_cycle_start'] = (
            (df['is_position_entry_optimal'] == 1) &
            (df['success_probability'] > 0.5)
        ).astype(int)
        
        # Momentum continuation prediction
        df['momentum_continues'] = (
            (df['momentum_2p'].shift(-1) > 0) & (df['momentum_2p'] > 0)
        ).astype(int)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get grouped features for Random Forest feature importance analysis."""
        return {
            'momentum_features': [
                'momentum_2p', 'momentum_4p', 'momentum_6p', 'momentum_12p',
                'ema_momentum_fast', 'ema_momentum_slow', 'momentum_acceleration',
                'is_high_momentum', 'momentum_strength_2p'
            ],
            'volume_features': [
                'volume_roc_2', 'volume_momentum', 'obv_momentum',
                'volume_trend', 'is_volume_increasing'
            ],
            'pattern_features': [
                'is_doji', 'is_hammer', 'bullish_engulfing', 'trend_continuation',
                'gap_up', 'body_size', 'pattern_score'
            ],
            'structure_features': [
                'market_structure_score', 'trend_alignment', 'bullish_structure',
                'distance_to_resistance', 'distance_to_support'
            ],
            'volatility_features': [
                'atr_pct', 'volatility_momentum', 'is_vol_expanding',
                'price_efficiency', 'is_efficient_move'
            ],
            'timing_features': [
                'is_optimal_hour', 'is_asian_session', 'is_weekend',
                'asian_session_momentum', 'european_session_momentum'
            ],
            'position_features': [
                'momentum_position_size', 'risk_adjusted_position',
                'is_position_entry_optimal', 'success_probability'
            ]
        }

def main():
    """Test the enhanced momentum features."""
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 120000 + np.random.randn(100) * 1000,
        'high': 120000 + np.random.randn(100) * 1000 + 500,
        'low': 120000 + np.random.randn(100) * 1000 - 500,
        'close': 120000 + np.random.randn(100) * 1000,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Fix high/low logic
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    # Test feature engineering
    engineer = MomentumFeatureEngineer()
    enhanced_data = engineer.create_enhanced_features(sample_data)
    enhanced_data = engineer.create_target_variables(enhanced_data)
    
    print(f"📊 Original features: {len(sample_data.columns)}")
    print(f"🚀 Enhanced features: {len(enhanced_data.columns)}")
    print(f"✅ Feature groups: {list(engineer.get_feature_importance_groups().keys())}")

if __name__ == "__main__":
    main()