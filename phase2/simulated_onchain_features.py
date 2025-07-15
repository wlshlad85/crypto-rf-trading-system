#!/usr/bin/env python3
"""
Phase 2A: Simulated On-Chain Features Engine
ULTRATHINK Implementation - Fundamental Blockchain Metrics

Implements simulated on-chain features as proxies for fundamental blockchain data:
- MVRV (Market Value to Realized Value) proxies
- NVT (Network Value to Transactions) proxies
- Exchange flow simulation
- Active address proxies
- Mining difficulty and hash rate proxies
- Whale activity simulation

Designed to provide fundamental blockchain insights without requiring real on-chain APIs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class OnChainConfig:
    """Configuration for on-chain feature simulation."""
    # MVRV proxy parameters
    mvrv_lookback_short: int = 30
    mvrv_lookback_long: int = 365
    mvrv_volume_weight: float = 0.3
    
    # NVT proxy parameters
    nvt_transaction_proxy_window: int = 14
    nvt_smoothing_window: int = 7
    
    # Exchange flow parameters
    exchange_flow_threshold: float = 1.5  # Volume spike threshold
    exchange_flow_window: int = 20
    
    # Active address proxy parameters
    active_address_vol_window: int = 14
    active_address_price_window: int = 30
    
    # Hash rate proxy parameters
    hashrate_stability_window: int = 168  # 1 week in hours
    hashrate_difficulty_window: int = 2016  # Bitcoin difficulty adjustment
    
    # Whale activity parameters
    whale_volume_threshold: float = 2.0  # Standard deviations above mean
    whale_price_impact_window: int = 5

class SimulatedOnChainFeatures:
    """
    Professional on-chain features simulation engine for cryptocurrency trading.
    
    Creates sophisticated proxies for fundamental blockchain metrics that would
    typically require direct blockchain data access.
    """
    
    def __init__(self, config: Optional[OnChainConfig] = None):
        """
        Initialize simulated on-chain features engine.
        
        Args:
            config: Configuration for on-chain feature parameters
        """
        self.config = config or OnChainConfig()
        self.feature_cache = {}
        
        print("🔗 Simulated On-Chain Features Engine Initialized")
        print(f"📊 MVRV Proxy: {self.config.mvrv_lookback_short}d / {self.config.mvrv_lookback_long}d lookback")
        print(f"🔄 NVT Proxy: {self.config.nvt_transaction_proxy_window}d transaction window")
        print(f"💱 Exchange Flow: {self.config.exchange_flow_threshold}x volume threshold")
        print(f"⛏️ Hash Rate: {self.config.hashrate_stability_window}h stability window")
    
    def calculate_all_onchain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all simulated on-chain features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all on-chain features added
        """
        print("\n🔗 Calculating Simulated On-Chain Features")
        print("=" * 50)
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = df.copy()
        
        # 1. MVRV Proxy Features
        print("📊 Calculating MVRV Proxies...")
        mvrv_features = self.calculate_mvrv_proxies(df)
        for name, values in mvrv_features.items():
            result_df[name] = values
        
        # 2. NVT Proxy Features
        print("🔄 Calculating NVT Proxies...")
        nvt_features = self.calculate_nvt_proxies(df)
        for name, values in nvt_features.items():
            result_df[name] = values
        
        # 3. Exchange Flow Simulation
        print("💱 Calculating Exchange Flow Simulation...")
        exchange_features = self.calculate_exchange_flow_simulation(df)
        for name, values in exchange_features.items():
            result_df[name] = values
        
        # 4. Active Address Proxies
        print("👥 Calculating Active Address Proxies...")
        address_features = self.calculate_active_address_proxies(df)
        for name, values in address_features.items():
            result_df[name] = values
        
        # 5. Hash Rate and Mining Proxies
        print("⛏️ Calculating Hash Rate Proxies...")
        mining_features = self.calculate_mining_proxies(df)
        for name, values in mining_features.items():
            result_df[name] = values
        
        # 6. Whale Activity Simulation
        print("🐋 Calculating Whale Activity Simulation...")
        whale_features = self.calculate_whale_activity_simulation(df)
        for name, values in whale_features.items():
            result_df[name] = values
        
        # 7. Network Health Proxies
        print("🏥 Calculating Network Health Proxies...")
        health_features = self.calculate_network_health_proxies(df)
        for name, values in health_features.items():
            result_df[name] = values
        
        # 8. Market Sentiment Proxies
        print("😊 Calculating Market Sentiment Proxies...")
        sentiment_features = self.calculate_market_sentiment_proxies(df)
        for name, values in sentiment_features.items():
            result_df[name] = values
        
        # Count new features
        original_columns = len(df.columns)
        new_columns = len(result_df.columns)
        features_added = new_columns - original_columns
        
        print(f"\n✅ Simulated On-Chain Features Complete")
        print(f"📊 Original Features: {original_columns}")
        print(f"🔗 On-Chain Features Added: {features_added}")
        print(f"📈 Total Features: {new_columns}")
        
        return result_df
    
    def calculate_mvrv_proxies(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MVRV (Market Value to Realized Value) proxy features."""
        close = df['Close']
        volume = df['Volume']
        
        # Volume-Weighted Average Price (VWAP) as realized value proxy
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap_short = (typical_price * volume).rolling(self.config.mvrv_lookback_short).sum() / volume.rolling(self.config.mvrv_lookback_short).sum()
        vwap_long = (typical_price * volume).rolling(self.config.mvrv_lookback_long).sum() / volume.rolling(self.config.mvrv_lookback_long).sum()
        
        # MVRV ratio proxies
        mvrv_short = close / vwap_short
        mvrv_long = close / vwap_long
        
        # MVRV percentiles (market top/bottom indicators)
        mvrv_short_percentile = mvrv_short.rolling(365).rank(pct=True)
        mvrv_long_percentile = mvrv_long.rolling(365).rank(pct=True)
        
        # MVRV Z-Score
        mvrv_short_zscore = (mvrv_short - mvrv_short.rolling(365).mean()) / mvrv_short.rolling(365).std()
        mvrv_long_zscore = (mvrv_long - mvrv_long.rolling(365).mean()) / mvrv_long.rolling(365).std()
        
        # MVRV oscillator
        mvrv_oscillator = (mvrv_short - mvrv_long) / mvrv_long
        
        # MVRV momentum
        mvrv_momentum = mvrv_short.pct_change(periods=7)
        
        # Market value distribution proxy
        market_cap_proxy = close * volume  # Simplified market cap proxy
        realized_cap_proxy = vwap_long * volume
        mvrv_ratio = market_cap_proxy / realized_cap_proxy
        
        return {
            'mvrv_short': mvrv_short,
            'mvrv_long': mvrv_long,
            'mvrv_short_percentile': mvrv_short_percentile,
            'mvrv_long_percentile': mvrv_long_percentile,
            'mvrv_short_zscore': mvrv_short_zscore,
            'mvrv_long_zscore': mvrv_long_zscore,
            'mvrv_oscillator': mvrv_oscillator,
            'mvrv_momentum': mvrv_momentum,
            'mvrv_ratio': mvrv_ratio
        }
    
    def calculate_nvt_proxies(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate NVT (Network Value to Transactions) proxy features."""
        close = df['Close']
        volume = df['Volume']
        
        # Network value proxy (market cap approximation)
        network_value = close * volume  # Simplified
        
        # Transaction volume proxy (using actual volume as transactions proxy)
        transaction_volume = volume.rolling(self.config.nvt_transaction_proxy_window).mean()
        
        # NVT ratio
        nvt_ratio = network_value / transaction_volume
        
        # NVT signal (smoothed version)
        nvt_signal = nvt_ratio.rolling(self.config.nvt_smoothing_window).mean()
        
        # NVT percentiles
        nvt_percentile = nvt_ratio.rolling(365).rank(pct=True)
        
        # NVT Z-Score
        nvt_zscore = (nvt_ratio - nvt_ratio.rolling(365).mean()) / nvt_ratio.rolling(365).std()
        
        # NVT momentum
        nvt_momentum = nvt_ratio.pct_change(periods=7)
        
        # Transaction efficiency proxy
        tx_efficiency = volume / close  # Volume per unit price
        
        # Network utilization proxy
        network_utilization = volume / volume.rolling(90).max()
        
        return {
            'nvt_ratio': nvt_ratio,
            'nvt_signal': nvt_signal,
            'nvt_percentile': nvt_percentile,
            'nvt_zscore': nvt_zscore,
            'nvt_momentum': nvt_momentum,
            'tx_efficiency': tx_efficiency,
            'network_utilization': network_utilization
        }
    
    def calculate_exchange_flow_simulation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Simulate exchange flow patterns based on volume analysis."""
        volume = df['Volume']
        close = df['Close']
        
        # Volume moving averages
        volume_ma = volume.rolling(self.config.exchange_flow_window).mean()
        volume_std = volume.rolling(self.config.exchange_flow_window).std()
        
        # Exchange inflow simulation (high volume spikes)
        exchange_inflow = (volume > volume_ma + volume_std * self.config.exchange_flow_threshold).astype(int)
        
        # Exchange outflow simulation (low volume with price stability)
        price_stability = close.rolling(5).std() / close.rolling(5).mean()
        exchange_outflow = ((volume < volume_ma * 0.7) & (price_stability < price_stability.median())).astype(int)
        
        # Net flow approximation
        net_flow = exchange_inflow - exchange_outflow
        
        # Flow momentum
        flow_momentum = net_flow.rolling(7).sum()
        
        # Exchange activity index
        volume_spike_count = exchange_inflow.rolling(30).sum()
        exchange_activity = volume_spike_count / 30
        
        # Liquidity proxy
        bid_ask_proxy = volume / (df['High'] - df['Low'])  # Volume per price range
        liquidity_index = bid_ask_proxy / bid_ask_proxy.rolling(30).mean()
        
        return {
            'exchange_inflow': exchange_inflow,
            'exchange_outflow': exchange_outflow,
            'net_flow': net_flow,
            'flow_momentum': flow_momentum,
            'exchange_activity': exchange_activity,
            'liquidity_index': liquidity_index
        }
    
    def calculate_active_address_proxies(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate proxies for active address metrics."""
        volume = df['Volume']
        close = df['Close']
        volatility = close.pct_change().rolling(self.config.active_address_vol_window).std()
        
        # Active addresses proxy (based on volume distribution)
        volume_distribution = volume / volume.rolling(90).sum()
        active_addresses_proxy = volume_distribution * volatility
        
        # New addresses proxy (volume spikes with price changes)
        price_change = abs(close.pct_change())
        volume_change = volume.pct_change()
        new_addresses_proxy = (price_change * abs(volume_change)).rolling(7).mean()
        
        # Address activity intensity
        activity_intensity = (volume / volume.rolling(30).mean()) * (volatility / volatility.rolling(30).mean())
        
        # Network growth proxy
        network_growth = new_addresses_proxy.rolling(30).mean() / new_addresses_proxy.rolling(90).mean()
        
        # User engagement proxy
        engagement_proxy = volume.rolling(7).sum() / close.rolling(7).mean()
        
        return {
            'active_addresses_proxy': active_addresses_proxy,
            'new_addresses_proxy': new_addresses_proxy,
            'activity_intensity': activity_intensity,
            'network_growth': network_growth,
            'user_engagement': engagement_proxy
        }
    
    def calculate_mining_proxies(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate hash rate and mining difficulty proxies."""
        close = df['Close']
        volume = df['Volume']
        
        # Hash rate proxy (price stability with consistent volume)
        price_stability = 1 / (close.pct_change().rolling(self.config.hashrate_stability_window).std() + 1e-6)
        volume_consistency = 1 / (volume.pct_change().rolling(self.config.hashrate_stability_window).std() + 1e-6)
        hash_rate_proxy = (price_stability * volume_consistency).rolling(24).mean()
        
        # Mining difficulty proxy (based on price trends and volatility)
        price_trend = close.rolling(self.config.hashrate_difficulty_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        difficulty_proxy = abs(price_trend) * close.rolling(self.config.hashrate_difficulty_window).std()
        
        # Miner capitulation proxy (high volume with price drops)
        price_drop = (close.pct_change() < -0.02).astype(int)
        volume_spike = (volume > volume.rolling(30).mean() * 1.5).astype(int)
        miner_capitulation = (price_drop * volume_spike).rolling(7).sum()
        
        # Mining profitability proxy
        profitability_proxy = close / close.rolling(180).mean()  # Price vs long-term average
        
        # Network security proxy
        security_proxy = hash_rate_proxy / hash_rate_proxy.rolling(365).mean()
        
        return {
            'hash_rate_proxy': hash_rate_proxy,
            'difficulty_proxy': difficulty_proxy,
            'miner_capitulation': miner_capitulation,
            'mining_profitability': profitability_proxy,
            'network_security': security_proxy
        }
    
    def calculate_whale_activity_simulation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Simulate whale activity based on volume and price patterns."""
        volume = df['Volume']
        close = df['Close']
        
        # Whale volume threshold
        volume_mean = volume.rolling(30).mean()
        volume_std = volume.rolling(30).std()
        whale_threshold = volume_mean + volume_std * self.config.whale_volume_threshold
        
        # Whale transactions (large volume spikes)
        whale_activity = (volume > whale_threshold).astype(int)
        
        # Whale accumulation (large volume with minimal price impact)
        price_impact = abs(close.pct_change())
        volume_to_impact_ratio = volume / (price_impact + 1e-6)
        whale_accumulation = (whale_activity & (volume_to_impact_ratio > volume_to_impact_ratio.rolling(30).quantile(0.8))).astype(int)
        
        # Whale distribution (large volume with significant price impact)
        whale_distribution = (whale_activity & (price_impact > price_impact.rolling(30).quantile(0.8))).astype(int)
        
        # Whale net position change
        whale_net_position = whale_accumulation - whale_distribution
        
        # Whale activity momentum
        whale_momentum = whale_net_position.rolling(7).sum()
        
        # Large holder concentration proxy
        large_holder_concentration = whale_activity.rolling(30).sum() / 30
        
        return {
            'whale_activity': whale_activity,
            'whale_accumulation': whale_accumulation,
            'whale_distribution': whale_distribution,
            'whale_net_position': whale_net_position,
            'whale_momentum': whale_momentum,
            'large_holder_concentration': large_holder_concentration
        }
    
    def calculate_network_health_proxies(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate network health and adoption proxies."""
        close = df['Close']
        volume = df['Volume']
        
        # Network value to GDP proxy (price vs transaction activity)
        gdp_proxy = volume.rolling(365).sum()  # Annual transaction volume
        nvt_to_gdp = (close * volume) / gdp_proxy
        
        # Adoption rate proxy (sustained volume growth)
        volume_growth = volume.rolling(30).mean() / volume.rolling(90).mean()
        adoption_rate = volume_growth.rolling(30).mean()
        
        # Network maturity proxy (price stability over time)
        volatility_decline = close.pct_change().rolling(90).std() / close.pct_change().rolling(365).std()
        network_maturity = 1 / (volatility_decline + 1)
        
        # Utility score proxy (consistent transaction activity)
        utility_score = volume.rolling(30).std() / volume.rolling(30).mean()  # Inverse coefficient of variation
        
        # Network effect proxy (volume correlation with price)
        correlation_window = 30
        network_effect = close.rolling(correlation_window).corr(volume)
        
        return {
            'nvt_to_gdp': nvt_to_gdp,
            'adoption_rate': adoption_rate,
            'network_maturity': network_maturity,
            'utility_score': utility_score,
            'network_effect': network_effect
        }
    
    def calculate_market_sentiment_proxies(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market sentiment proxies based on price and volume behavior."""
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        
        # Fear and greed proxy (volatility vs volume)
        volatility = close.pct_change().rolling(14).std()
        volume_normalized = volume / volume.rolling(50).mean()
        fear_greed_index = (1 - volatility) * volume_normalized
        
        # Market euphoria proxy (high prices with high volume)
        price_percentile = close.rolling(365).rank(pct=True)
        volume_percentile = volume.rolling(365).rank(pct=True)
        euphoria_index = price_percentile * volume_percentile
        
        # Capitulation proxy (low prices with high volume)
        price_decline = (1 - price_percentile)
        capitulation_index = price_decline * volume_percentile
        
        # HODL sentiment proxy (low volume with price stability)
        price_stability = 1 / (volatility + 1e-6)
        volume_decline = 1 / (volume_normalized + 1e-6)
        hodl_sentiment = price_stability * volume_decline
        
        # Momentum sentiment (price and volume alignment)
        price_momentum = close.pct_change(periods=7)
        volume_momentum = volume.pct_change(periods=7)
        momentum_alignment = np.sign(price_momentum) * np.sign(volume_momentum)
        
        return {
            'fear_greed_index': fear_greed_index,
            'euphoria_index': euphoria_index,
            'capitulation_index': capitulation_index,
            'hodl_sentiment': hodl_sentiment,
            'momentum_alignment': momentum_alignment
        }
    
    def get_onchain_feature_analysis(self, df_with_features: pd.DataFrame) -> Dict:
        """Analyze the quality and distribution of generated on-chain features."""
        onchain_columns = [col for col in df_with_features.columns 
                          if any(prefix in col for prefix in [
                              'mvrv', 'nvt', 'exchange', 'net_flow', 'active_addresses', 'hash_rate',
                              'whale', 'adoption', 'network', 'fear_greed', 'euphoria', 'capitulation',
                              'hodl', 'momentum_alignment', 'mining', 'difficulty'
                          ])]
        
        analysis = {
            'total_onchain_features': len(onchain_columns),
            'feature_categories': {
                'mvrv_features': len([c for c in onchain_columns if 'mvrv' in c]),
                'nvt_features': len([c for c in onchain_columns if 'nvt' in c]),
                'exchange_flow': len([c for c in onchain_columns if any(x in c for x in ['exchange', 'flow', 'liquidity'])]),
                'address_activity': len([c for c in onchain_columns if any(x in c for x in ['address', 'activity', 'engagement'])]),
                'mining_features': len([c for c in onchain_columns if any(x in c for x in ['hash', 'mining', 'difficulty', 'security'])]),
                'whale_activity': len([c for c in onchain_columns if 'whale' in c]),
                'network_health': len([c for c in onchain_columns if any(x in c for x in ['network', 'adoption', 'maturity', 'utility'])]),
                'sentiment_features': len([c for c in onchain_columns if any(x in c for x in ['fear', 'euphoria', 'capitulation', 'hodl', 'sentiment'])])
            },
            'data_quality': {
                'total_samples': len(df_with_features),
                'complete_samples': len(df_with_features.dropna()),
                'completeness_ratio': len(df_with_features.dropna()) / len(df_with_features)
            },
            'feature_statistics': {}
        }
        
        # Calculate basic statistics for key features
        key_features = ['mvrv_short', 'nvt_ratio', 'whale_activity', 'fear_greed_index']
        for feature in key_features:
            if feature in df_with_features.columns:
                series = df_with_features[feature].dropna()
                if len(series) > 0:
                    analysis['feature_statistics'][feature] = {
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'correlation_with_price': float(df_with_features['Close'].corr(series))
                    }
        
        return analysis

def main():
    """Demonstrate simulated on-chain features engine."""
    print("🔗 PHASE 2A: Simulated On-Chain Features Engine")
    print("ULTRATHINK Implementation - Fundamental Blockchain Metrics")
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
    
    # Initialize simulated on-chain features engine
    config = OnChainConfig()
    engine = SimulatedOnChainFeatures(config)
    
    # Calculate all on-chain features
    try:
        enhanced_df = engine.calculate_all_onchain_features(df)
        
        # Feature analysis
        analysis = engine.get_onchain_feature_analysis(enhanced_df)
        
        print(f"\n📊 ON-CHAIN FEATURE ENHANCEMENT SUMMARY:")
        print(f"   Total On-Chain Features: {analysis['total_onchain_features']}")
        print(f"   Data Completeness: {analysis['data_quality']['completeness_ratio']:.1%}")
        
        print(f"\n🔗 FEATURE CATEGORIES:")
        for category, count in analysis['feature_categories'].items():
            print(f"   {category.replace('_', ' ').title()}: {count}")
        
        # Show sample of new features
        onchain_cols = [col for col in enhanced_df.columns if col not in df.columns]
        print(f"\n📈 SAMPLE ON-CHAIN FEATURES (showing first 15):")
        for i, col in enumerate(onchain_cols[:15]):
            print(f"   {i+1}. {col}")
        
        if len(onchain_cols) > 15:
            print(f"   ... and {len(onchain_cols) - 15} more")
        
        # Feature correlations with price
        if analysis['feature_statistics']:
            print(f"\n📊 KEY FEATURE CORRELATIONS WITH PRICE:")
            for feature, stats in analysis['feature_statistics'].items():
                corr = stats['correlation_with_price']
                print(f"   {feature}: {corr:+.3f}")
        
        # Save enhanced dataset
        output_file = "phase2/enhanced_features_with_onchain.csv"
        enhanced_df.to_csv(output_file)
        print(f"\n💾 Enhanced dataset saved: {output_file}")
        
        print(f"\n🚀 Phase 2A Simulated On-Chain Features: COMPLETE")
        print(f"📊 Feature Enhancement: {len(df.columns)} → {len(enhanced_df.columns)} features")
        print(f"🎯 Ready for Phase 2A Next Step: Triple Barrier Labeling")
        
    except Exception as e:
        print(f"❌ Error calculating on-chain features: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()