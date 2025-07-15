#!/usr/bin/env python3
"""
Phase 2A: Fractional Differentiation Engine
ULTRATHINK Implementation - Advanced Feature Engineering

Implements fractional differentiation for handling non-stationary crypto time series:
- Optimal d parameter selection using ADF stationarity tests
- Memory preservation through fractional calculus
- Stationary feature generation while retaining predictive power
- Integration with institutional-grade validation framework

Based on ULTRATHINK research addressing 51.1% PBO overfitting risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.special import gamma
from scipy import stats
import warnings
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class FractionalDifferentiationEngine:
    """
    Professional fractional differentiation engine for cryptocurrency trading features.
    
    Implements optimal fractional differencing to create stationary features
    while preserving long-term memory structure in crypto time series.
    """
    
    def __init__(self,
                 d_range: Tuple[float, float] = (0.1, 0.9),
                 d_step: float = 0.1,
                 window_size: int = 252,
                 significance_level: float = 0.05,
                 min_memory_preservation: float = 0.7,
                 results_dir: str = "phase2/fractional_results"):
        """
        Initialize fractional differentiation engine.
        
        Args:
            d_range: Range for optimal d parameter search
            d_step: Step size for d parameter grid search
            window_size: Window size for weight calculations
            significance_level: ADF test significance level
            min_memory_preservation: Minimum autocorrelation preservation
            results_dir: Directory for results storage
        """
        self.d_range = d_range
        self.d_step = d_step
        self.window_size = window_size
        self.significance_level = significance_level
        self.min_memory_preservation = min_memory_preservation
        self.results_dir = Path(results_dir)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.optimization_results = {}
        self.feature_analysis = {}
        
        print("🔬 Fractional Differentiation Engine Initialized")
        print(f"📊 d Parameter Range: {d_range[0]:.1f} to {d_range[1]:.1f}")
        print(f"🎯 Target Memory Preservation: >{min_memory_preservation:.1%}")
        print(f"📈 Window Size: {window_size} periods")
        print(f"📁 Results Directory: {results_dir}")
    
    def find_optimal_d(self,
                      series: pd.Series,
                      series_name: str = "unnamed_series") -> Dict:
        """
        Find optimal fractional differentiation parameter for a time series.
        
        Args:
            series: Time series data
            series_name: Name for this series
            
        Returns:
            Optimization results with optimal d parameter
        """
        print(f"\n🔍 Finding Optimal d Parameter for: {series_name}")
        print("=" * 50)
        
        # Validate input
        if len(series) < 100:
            raise ValueError(f"Series too short: {len(series)} < 100")
        
        # Clean series
        clean_series = self._clean_series(series)
        
        # Test original series stationarity
        original_adf = self._adf_test(clean_series)
        print(f"📊 Original Series ADF: p-value = {original_adf['p_value']:.4f}")
        print(f"   {'✅ STATIONARY' if original_adf['is_stationary'] else '❌ NON-STATIONARY'}")
        
        if original_adf['is_stationary']:
            print("⚠️ Series already stationary. Fractional differencing may not be needed.")
            return {
                'series_name': series_name,
                'optimal_d': 0.0,
                'is_already_stationary': True,
                'original_adf': original_adf,
                'recommendation': 'Use original series - already stationary'
            }
        
        # Calculate original autocorrelation for memory preservation
        original_autocorr = self._calculate_autocorrelation(clean_series)
        
        # Grid search for optimal d
        d_candidates = np.arange(self.d_range[0], self.d_range[1] + self.d_step, self.d_step)
        results = []
        
        print(f"🔄 Testing {len(d_candidates)} d values...")
        
        for d in d_candidates:
            try:
                # Apply fractional differentiation
                diff_series = self.fractional_diff(clean_series, d)
                
                # Skip if insufficient data
                if len(diff_series.dropna()) < 50:
                    continue
                
                # Test stationarity
                adf_result = self._adf_test(diff_series.dropna())
                
                # Calculate memory preservation
                diff_autocorr = self._calculate_autocorrelation(diff_series.dropna())
                memory_preservation = diff_autocorr / original_autocorr if original_autocorr > 0 else 0
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    adf_result, memory_preservation, d
                )
                
                results.append({
                    'd': d,
                    'adf_statistic': adf_result['adf_statistic'],
                    'adf_pvalue': adf_result['p_value'],
                    'is_stationary': adf_result['is_stationary'],
                    'memory_preservation': memory_preservation,
                    'data_retention': len(diff_series.dropna()) / len(clean_series),
                    'quality_score': quality_score
                })
                
                print(f"   d={d:.1f}: ADF p={adf_result['p_value']:.4f}, "
                      f"Memory={memory_preservation:.3f}, Quality={quality_score:.3f}")
                
            except Exception as e:
                print(f"   d={d:.1f}: Error - {str(e)}")
                continue
        
        if not results:
            raise ValueError("No valid d parameters found")
        
        # Find optimal d
        optimal_result = self._select_optimal_d(results)
        
        # Generate final report
        optimization_report = {
            'series_name': series_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'original_series_stats': {
                'length': len(clean_series),
                'adf_test': original_adf,
                'autocorrelation': original_autocorr
            },
            'optimization_results': results,
            'optimal_d': optimal_result['d'],
            'optimal_quality_score': optimal_result['quality_score'],
            'optimal_memory_preservation': optimal_result['memory_preservation'],
            'recommendation': self._generate_recommendation(optimal_result)
        }
        
        # Store results
        self.optimization_results[series_name] = optimization_report
        
        # Print summary
        self._print_optimization_summary(optimization_report)
        
        return optimization_report
    
    def fractional_diff(self,
                       series: pd.Series,
                       d: float,
                       threshold: float = 0.01) -> pd.Series:
        """
        Apply fractional differentiation to a time series.
        
        Args:
            series: Input time series
            d: Fractional differentiation parameter
            threshold: Threshold for weight truncation
            
        Returns:
            Fractionally differenced series
        """
        # Calculate weights
        weights = self._get_weights(d, threshold, self.window_size)
        
        # Apply fractional differentiation
        width = len(weights) - 1
        output = []
        
        for i in range(width, len(series)):
            # Extract window
            window = series.iloc[i-width:i+1]
            
            # Apply weights (reverse order for convolution)
            if len(window) == len(weights):
                value = np.dot(window.values, weights[::-1])
                output.append(value)
            else:
                output.append(np.nan)
        
        # Create output series with proper index
        index = series.index[width:]
        result = pd.Series(output, index=index, name=f"{series.name}_fracdiff_{d:.2f}")
        
        return result
    
    def apply_to_features(self,
                         df: pd.DataFrame,
                         feature_columns: List[str],
                         target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Apply optimal fractional differentiation to multiple features.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to process
            target_column: Optional target column to preserve
            
        Returns:
            DataFrame with fractionally differenced features
        """
        print(f"\n🔧 Applying Fractional Differentiation to {len(feature_columns)} Features")
        print("=" * 60)
        
        result_df = df.copy()
        
        for feature in feature_columns:
            if feature not in df.columns:
                print(f"⚠️ Feature '{feature}' not found in DataFrame")
                continue
            
            print(f"🔄 Processing feature: {feature}")
            
            try:
                # Find optimal d for this feature
                optimization_result = self.find_optimal_d(df[feature], feature)
                optimal_d = optimization_result['optimal_d']
                
                if optimal_d > 0:
                    # Apply fractional differentiation
                    diff_series = self.fractional_diff(df[feature], optimal_d)
                    
                    # Add to result DataFrame
                    result_df[f"{feature}_fracdiff"] = diff_series
                    
                    print(f"   ✅ Added {feature}_fracdiff (d={optimal_d:.2f})")
                else:
                    print(f"   ℹ️ {feature} already stationary, keeping original")
                
            except Exception as e:
                print(f"   ❌ Error processing {feature}: {str(e)}")
                continue
        
        # Preserve target column if specified
        if target_column and target_column in df.columns:
            result_df[target_column] = df[target_column]
        
        # Drop rows with too many NaN values
        result_df = result_df.dropna()
        
        # Generate feature analysis report
        self.feature_analysis = self._analyze_feature_transformation(
            df[feature_columns], result_df, feature_columns
        )
        
        print(f"\n📊 Feature Transformation Summary:")
        print(f"   Original Features: {len(feature_columns)}")
        print(f"   Enhanced Features: {len([c for c in result_df.columns if 'fracdiff' in c])}")
        print(f"   Final Dataset Size: {len(result_df)} samples")
        
        return result_df
    
    def _clean_series(self, series: pd.Series) -> pd.Series:
        """Clean and validate time series data."""
        # Remove NaN values
        clean = series.dropna()
        
        # Remove infinite values
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check for constant series
        if clean.std() == 0:
            raise ValueError("Series is constant - cannot differentiate")
        
        return clean
    
    def _adf_test(self, series: pd.Series) -> Dict:
        """Perform Augmented Dickey-Fuller test for stationarity."""
        from statsmodels.tsa.stattools import adfuller
        
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] <= self.significance_level
            }
        except Exception as e:
            return {
                'adf_statistic': np.nan,
                'p_value': 1.0,
                'critical_values': {},
                'is_stationary': False,
                'error': str(e)
            }
    
    def _calculate_autocorrelation(self, series: pd.Series, max_lags: int = 10) -> float:
        """Calculate average autocorrelation for memory preservation assessment."""
        try:
            clean_series = series.dropna()
            if len(clean_series) < max_lags + 10:
                return 0.0
            
            autocorrs = []
            for lag in range(1, min(max_lags + 1, len(clean_series) // 4)):
                corr = clean_series.autocorr(lag)
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
            
            return np.mean(autocorrs) if autocorrs else 0.0
            
        except Exception:
            return 0.0
    
    def _get_weights(self, d: float, threshold: float, window_size: int) -> np.ndarray:
        """Calculate fractional differentiation weights."""
        weights = [1.0]
        
        for k in range(1, window_size):
            weight = weights[-1] * (d - k + 1) / k
            
            if abs(weight) < threshold:
                break
                
            weights.append(weight)
        
        return np.array(weights)
    
    def _calculate_quality_score(self,
                                adf_result: Dict,
                                memory_preservation: float,
                                d: float) -> float:
        """Calculate quality score for d parameter selection."""
        # Stationarity score (0-1)
        stationarity_score = 1.0 if adf_result['is_stationary'] else 0.0
        
        # Memory preservation score (0-1)
        memory_score = min(memory_preservation / self.min_memory_preservation, 1.0)
        
        # Parsimony score (prefer lower d values)
        parsimony_score = 1.0 - (d / self.d_range[1])
        
        # ADF p-value score (lower p-value is better for stationarity)
        pvalue_score = max(0, 1 - adf_result['p_value'] / self.significance_level)
        
        # Combined score with weights
        quality_score = (
            0.4 * stationarity_score +
            0.3 * memory_score +
            0.2 * pvalue_score +
            0.1 * parsimony_score
        )
        
        return quality_score
    
    def _select_optimal_d(self, results: List[Dict]) -> Dict:
        """Select optimal d parameter from results."""
        # Filter for stationary results
        stationary_results = [r for r in results if r['is_stationary']]
        
        if stationary_results:
            # Among stationary results, select highest quality score
            optimal = max(stationary_results, key=lambda x: x['quality_score'])
        else:
            # If no stationary results, select best quality score
            optimal = max(results, key=lambda x: x['quality_score'])
        
        return optimal
    
    def _generate_recommendation(self, optimal_result: Dict) -> str:
        """Generate recommendation based on optimization results."""
        d = optimal_result['d']
        memory = optimal_result['memory_preservation']
        is_stationary = optimal_result['is_stationary']
        
        if is_stationary and memory >= self.min_memory_preservation:
            return f"EXCELLENT: d={d:.2f} achieves stationarity with {memory:.1%} memory preservation"
        elif is_stationary:
            return f"GOOD: d={d:.2f} achieves stationarity but low memory preservation ({memory:.1%})"
        elif memory >= self.min_memory_preservation:
            return f"MODERATE: d={d:.2f} preserves memory but may not be fully stationary"
        else:
            return f"POOR: d={d:.2f} neither achieves stationarity nor preserves memory"
    
    def _analyze_feature_transformation(self,
                                      original_df: pd.DataFrame,
                                      transformed_df: pd.DataFrame,
                                      feature_columns: List[str]) -> Dict:
        """Analyze the impact of fractional differentiation on features."""
        analysis = {
            'transformation_summary': {},
            'stationarity_improvement': {},
            'memory_preservation': {},
            'data_quality': {}
        }
        
        for feature in feature_columns:
            if feature in original_df.columns:
                original_series = original_df[feature].dropna()
                
                # Check if transformed feature exists
                transformed_feature = f"{feature}_fracdiff"
                if transformed_feature in transformed_df.columns:
                    transformed_series = transformed_df[transformed_feature].dropna()
                    
                    # Stationarity tests
                    orig_adf = self._adf_test(original_series)
                    trans_adf = self._adf_test(transformed_series)
                    
                    # Memory preservation
                    orig_autocorr = self._calculate_autocorrelation(original_series)
                    trans_autocorr = self._calculate_autocorrelation(transformed_series)
                    memory_ratio = trans_autocorr / orig_autocorr if orig_autocorr > 0 else 0
                    
                    analysis['transformation_summary'][feature] = {
                        'original_stationary': orig_adf['is_stationary'],
                        'transformed_stationary': trans_adf['is_stationary'],
                        'stationarity_improved': trans_adf['is_stationary'] and not orig_adf['is_stationary'],
                        'memory_preservation_ratio': memory_ratio,
                        'optimal_d': self.optimization_results.get(feature, {}).get('optimal_d', 0)
                    }
        
        return analysis
    
    def _print_optimization_summary(self, report: Dict):
        """Print optimization summary."""
        print(f"\n🎯 OPTIMIZATION SUMMARY: {report['series_name']}")
        print("-" * 50)
        print(f"✅ Optimal d Parameter: {report['optimal_d']:.2f}")
        print(f"📊 Quality Score: {report['optimal_quality_score']:.3f}")
        print(f"🧠 Memory Preservation: {report['optimal_memory_preservation']:.1%}")
        print(f"💡 Recommendation: {report['recommendation']}")
    
    def save_results(self, filename_prefix: str = "fractional_analysis"):
        """Save optimization and analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save optimization results
        optimization_file = self.results_dir / f"{filename_prefix}_optimization_{timestamp}.json"
        with open(optimization_file, 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)
        
        # Save feature analysis
        if self.feature_analysis:
            analysis_file = self.results_dir / f"{filename_prefix}_features_{timestamp}.json"
            with open(analysis_file, 'w') as f:
                json.dump(self.feature_analysis, f, indent=2, default=str)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Optimization: {optimization_file}")
        if self.feature_analysis:
            print(f"   🔧 Feature Analysis: {analysis_file}")
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report."""
        if not self.optimization_results:
            return {"error": "No optimization results available"}
        
        # Aggregate statistics
        all_d_values = [r['optimal_d'] for r in self.optimization_results.values()]
        all_quality_scores = [r['optimal_quality_score'] for r in self.optimization_results.values()]
        all_memory_preservation = [r['optimal_memory_preservation'] for r in self.optimization_results.values()]
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_series_analyzed': len(self.optimization_results),
            'optimal_d_statistics': {
                'mean': np.mean(all_d_values),
                'std': np.std(all_d_values),
                'min': np.min(all_d_values),
                'max': np.max(all_d_values)
            },
            'quality_score_statistics': {
                'mean': np.mean(all_quality_scores),
                'std': np.std(all_quality_scores),
                'min': np.min(all_quality_scores),
                'max': np.max(all_quality_scores)
            },
            'memory_preservation_statistics': {
                'mean': np.mean(all_memory_preservation),
                'std': np.std(all_memory_preservation),
                'min': np.min(all_memory_preservation),
                'max': np.max(all_memory_preservation)
            },
            'stationarity_success_rate': sum(1 for r in self.optimization_results.values() 
                                           if any(res['is_stationary'] for res in r['optimization_results'])) / len(self.optimization_results),
            'individual_results': self.optimization_results
        }
        
        return summary

def main():
    """Demonstrate fractional differentiation engine."""
    print("🔬 PHASE 2A: Fractional Differentiation Engine")
    print("ULTRATHINK Implementation - Advanced Feature Engineering")
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
    
    # Basic feature engineering
    df['returns'] = df['Close'].pct_change()
    df['log_price'] = np.log(df['Close'])
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['price_volatility'] = df['returns'].rolling(20).std()
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 500:
        print("❌ Insufficient data for fractional differentiation analysis")
        return
    
    print(f"📊 Data loaded: {len(df)} samples")
    
    # Initialize fractional differentiation engine
    engine = FractionalDifferentiationEngine(
        d_range=(0.1, 0.8),
        d_step=0.1,
        window_size=50,  # Reduced for demo
        min_memory_preservation=0.6
    )
    
    # Test on individual series
    print("\n🔍 Testing individual series optimization:")
    features_to_test = ['Close', 'log_price', 'Volume', 'returns']
    
    for feature in features_to_test:
        if feature in df.columns:
            try:
                result = engine.find_optimal_d(df[feature], feature)
                print(f"✅ {feature}: d={result['optimal_d']:.2f}")
            except Exception as e:
                print(f"❌ {feature}: Error - {str(e)}")
    
    # Apply to multiple features
    print("\n🔧 Applying to multiple features:")
    feature_columns = ['log_price', 'volume_ratio', 'price_volatility']
    
    try:
        enhanced_df = engine.apply_to_features(
            df, feature_columns, target_column='returns'
        )
        
        print(f"\n📈 Enhanced dataset shape: {enhanced_df.shape}")
        print(f"📊 New features: {[c for c in enhanced_df.columns if 'fracdiff' in c]}")
        
        # Save results
        engine.save_results("phase2a_fractional_demo")
        
        # Generate summary
        summary = engine.generate_summary_report()
        print(f"\n🎯 SUMMARY STATISTICS:")
        print(f"   Series Analyzed: {summary['total_series_analyzed']}")
        print(f"   Mean Optimal d: {summary['optimal_d_statistics']['mean']:.3f}")
        print(f"   Stationarity Success: {summary['stationarity_success_rate']:.1%}")
        
        print("\n🚀 Phase 2A Fractional Differentiation: COMPLETE")
        print("Ready for Phase 2B: Advanced Technical Indicators")
        
    except Exception as e:
        print(f"❌ Feature application failed: {str(e)}")

if __name__ == "__main__":
    main()