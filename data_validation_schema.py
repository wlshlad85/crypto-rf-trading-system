#!/usr/bin/env python3
"""
Data Quality Validation Schema for Crypto Trading System

Implements comprehensive data validation, schema checks, and quality assurance
for the 19K+ training rows across 20+ cryptocurrencies.

Usage: from data_validation_schema import DataValidator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
import json

class DataValidator:
    """Comprehensive data validation for crypto trading datasets."""
    
    def __init__(self):
        self.schema = self._define_schema()
        self.quality_thresholds = self._define_quality_thresholds()
        self.validation_results = {}
        
    def _define_schema(self) -> Dict[str, Dict]:
        """Define expected data schema and constraints."""
        return {
            'required_columns': {
                'open': {'type': 'float64', 'min': 0, 'max': 1000000},
                'high': {'type': 'float64', 'min': 0, 'max': 1000000},
                'low': {'type': 'float64', 'min': 0, 'max': 1000000},
                'close': {'type': 'float64', 'min': 0, 'max': 1000000},
                'volume': {'type': 'float64', 'min': 0, 'max': 1e12},
                'symbol': {'type': 'object', 'values': None}
            },
            'technical_indicators': {
                'momentum_1': {'type': 'float64', 'min': -50, 'max': 50},
                'momentum_4': {'type': 'float64', 'min': -100, 'max': 100},
                'rsi': {'type': 'float64', 'min': 0, 'max': 100},
                'macd': {'type': 'float64', 'min': -10000, 'max': 10000},
                'bb_position': {'type': 'float64', 'min': 0, 'max': 1},
                'volume_ratio': {'type': 'float64', 'min': 0, 'max': 20},
                'volatility': {'type': 'float64', 'min': 0, 'max': 5},
                'atr': {'type': 'float64', 'min': 0, 'max': 50000}
            },
            'target_variables': {
                'price_up_1h': {'type': 'int64', 'values': [0, 1]},
                'price_up_2h': {'type': 'int64', 'values': [0, 1]},
                'price_change_1h': {'type': 'float64', 'min': -50, 'max': 50},
                'price_change_2h': {'type': 'float64', 'min': -50, 'max': 50}
            }
        }
    
    def _define_quality_thresholds(self) -> Dict[str, float]:
        """Define data quality thresholds."""
        return {
            'max_missing_rate': 0.05,  # Max 5% missing values per column
            'min_records_per_symbol': 100,  # Min 100 records per crypto
            'max_outlier_rate': 0.02,  # Max 2% outliers per column
            'min_date_coverage': 0.8,  # Min 80% date coverage
            'max_duplicate_rate': 0.01,  # Max 1% duplicates
            'momentum_correlation_threshold': 0.7  # Min correlation for momentum validation
        }
    
    def validate_dataset(self, data: pd.DataFrame, symbol_list: List[str] = None) -> Dict[str, Any]:
        """Comprehensive dataset validation."""
        print("🔍 Starting comprehensive data validation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'total_columns': len(data.columns),
            'validation_passed': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {},
            'schema_compliance': {},
            'data_integrity': {}
        }
        
        try:
            # 1. Schema validation
            schema_results = self._validate_schema(data)
            results['schema_compliance'] = schema_results
            if not schema_results['passed']:
                results['validation_passed'] = False
                results['errors'].extend(schema_results['errors'])
            
            # 2. Data quality checks
            quality_results = self._validate_quality(data)
            results['quality_metrics'] = quality_results
            if quality_results['quality_score'] < 0.8:
                results['validation_passed'] = False
                results['errors'].append(f"Data quality score {quality_results['quality_score']:.2f} below threshold 0.8")
            
            # 3. Symbol coverage validation
            if symbol_list:
                symbol_results = self._validate_symbol_coverage(data, symbol_list)
                results['symbol_coverage'] = symbol_results
                if not symbol_results['passed']:
                    results['warnings'].extend(symbol_results['warnings'])
            
            # 4. Temporal integrity
            temporal_results = self._validate_temporal_integrity(data)
            results['temporal_integrity'] = temporal_results
            if not temporal_results['passed']:
                results['errors'].extend(temporal_results['errors'])
            
            # 5. Momentum threshold validation (1.780%/hour)
            momentum_results = self._validate_momentum_thresholds(data)
            results['momentum_validation'] = momentum_results
            if not momentum_results['correlation_valid']:
                results['warnings'].append("Momentum threshold correlation below expected level")
            
            # 6. Financial data integrity
            financial_results = self._validate_financial_integrity(data)
            results['data_integrity'] = financial_results
            if not financial_results['passed']:
                results['errors'].extend(financial_results['errors'])
            
            print(f"✅ Validation complete: {'PASSED' if results['validation_passed'] else 'FAILED'}")
            print(f"📊 Quality score: {quality_results.get('quality_score', 0):.2f}")
            
        except Exception as e:
            results['validation_passed'] = False
            results['errors'].append(f"Validation error: {str(e)}")
            print(f"❌ Validation failed: {e}")
        
        self.validation_results = results
        return results
    
    def _validate_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema compliance."""
        results = {'passed': True, 'errors': [], 'missing_columns': [], 'type_mismatches': []}
        
        # Check required columns
        all_required = {**self.schema['required_columns'], **self.schema['technical_indicators']}
        
        for col, constraints in all_required.items():
            if col not in data.columns:
                results['missing_columns'].append(col)
                results['errors'].append(f"Missing required column: {col}")
                results['passed'] = False
            else:
                # Type validation
                expected_type = constraints['type']
                if not data[col].dtype.name.startswith(expected_type.split('64')[0]):
                    results['type_mismatches'].append(f"{col}: expected {expected_type}, got {data[col].dtype}")
                
                # Range validation
                if 'min' in constraints and 'max' in constraints:
                    valid_data = data[col].dropna()
                    if len(valid_data) > 0:
                        if valid_data.min() < constraints['min'] or valid_data.max() > constraints['max']:
                            results['errors'].append(f"{col} values outside range [{constraints['min']}, {constraints['max']}]")
        
        return results
    
    def _validate_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics."""
        results = {
            'missing_value_rate': {},
            'outlier_rate': {},
            'duplicate_rate': 0,
            'quality_score': 0
        }
        
        # Missing values
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        overall_missing_rate = missing_cells / total_cells
        
        for col in data.columns:
            missing_rate = data[col].isnull().sum() / len(data)
            results['missing_value_rate'][col] = missing_rate
        
        # Outliers (using IQR method for numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                results['outlier_rate'][col] = outliers / len(data) if len(data) > 0 else 0
        
        # Duplicates
        duplicates = data.duplicated().sum()
        results['duplicate_rate'] = duplicates / len(data) if len(data) > 0 else 0
        
        # Calculate quality score
        quality_factors = [
            1 - overall_missing_rate,  # Lower missing = higher quality
            1 - results['duplicate_rate'],  # Lower duplicates = higher quality
            1 - np.mean(list(results['outlier_rate'].values())) if results['outlier_rate'] else 1
        ]
        results['quality_score'] = np.mean(quality_factors)
        
        return results
    
    def _validate_symbol_coverage(self, data: pd.DataFrame, expected_symbols: List[str]) -> Dict[str, Any]:
        """Validate cryptocurrency symbol coverage."""
        results = {'passed': True, 'warnings': [], 'coverage_stats': {}}
        
        if 'symbol' not in data.columns:
            results['passed'] = False
            results['warnings'].append("No symbol column found for coverage validation")
            return results
        
        actual_symbols = set(data['symbol'].unique())
        expected_symbols_set = set(expected_symbols)
        
        missing_symbols = expected_symbols_set - actual_symbols
        extra_symbols = actual_symbols - expected_symbols_set
        
        if missing_symbols:
            results['warnings'].append(f"Missing symbols: {list(missing_symbols)}")
        
        if extra_symbols:
            results['warnings'].append(f"Unexpected symbols: {list(extra_symbols)}")
        
        # Check records per symbol
        symbol_counts = data['symbol'].value_counts()
        insufficient_symbols = symbol_counts[symbol_counts < self.quality_thresholds['min_records_per_symbol']]
        
        if len(insufficient_symbols) > 0:
            results['warnings'].append(f"Symbols with insufficient data: {insufficient_symbols.to_dict()}")
        
        results['coverage_stats'] = {
            'expected_symbols': len(expected_symbols),
            'actual_symbols': len(actual_symbols),
            'coverage_rate': len(actual_symbols & expected_symbols_set) / len(expected_symbols_set),
            'avg_records_per_symbol': symbol_counts.mean(),
            'min_records_per_symbol': symbol_counts.min(),
            'max_records_per_symbol': symbol_counts.max()
        }
        
        return results
    
    def _validate_temporal_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal data integrity."""
        results = {'passed': True, 'errors': [], 'temporal_stats': {}}
        
        if not isinstance(data.index, pd.DatetimeIndex):
            results['errors'].append("Index is not datetime - temporal validation failed")
            results['passed'] = False
            return results
        
        # Check for gaps in time series
        time_diff = data.index.to_series().diff()
        expected_interval = time_diff.mode().iloc[0] if not time_diff.mode().empty else pd.Timedelta(hours=4)
        
        # Find gaps larger than 2x expected interval
        large_gaps = time_diff[time_diff > expected_interval * 2]
        if len(large_gaps) > 0:
            results['errors'].append(f"Found {len(large_gaps)} large time gaps")
        
        # Check date range coverage
        date_range = data.index.max() - data.index.min()
        expected_range = timedelta(days=365)  # Expect ~1 year of data
        coverage = min(date_range.days / expected_range.days, 1.0)
        
        if coverage < self.quality_thresholds['min_date_coverage']:
            results['errors'].append(f"Date coverage {coverage:.2f} below threshold {self.quality_thresholds['min_date_coverage']}")
            results['passed'] = False
        
        results['temporal_stats'] = {
            'date_range_days': date_range.days,
            'coverage_ratio': coverage,
            'expected_interval': str(expected_interval),
            'large_gaps_count': len(large_gaps)
        }
        
        return results
    
    def _validate_momentum_thresholds(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate momentum threshold correlation with ROI (1.780%/hour validation)."""
        results = {'correlation_valid': False, 'metrics': {}}
        
        momentum_cols = [col for col in data.columns if 'momentum' in col.lower()]
        target_cols = [col for col in data.columns if 'price_change' in col.lower()]
        
        if not momentum_cols or not target_cols:
            results['metrics']['error'] = "Missing momentum or target columns for validation"
            return results
        
        # Calculate correlation between momentum and future returns
        correlations = {}
        for momentum_col in momentum_cols:
            for target_col in target_cols:
                if momentum_col in data.columns and target_col in data.columns:
                    corr = data[momentum_col].corr(data[target_col])
                    correlations[f"{momentum_col}_{target_col}"] = corr
        
        # Check if momentum strength correlates with profitability
        if 'momentum_1' in data.columns:
            # Calculate momentum strength per hour
            momentum_strength = data['momentum_1'] / 4  # Convert to hourly for 4h data
            high_momentum_threshold = 1.780
            
            # Validate threshold effectiveness
            if 'price_change_2h' in data.columns:
                high_momentum_mask = momentum_strength > high_momentum_threshold
                
                if high_momentum_mask.sum() > 10:  # Need sufficient samples
                    high_momentum_returns = data.loc[high_momentum_mask, 'price_change_2h'].mean()
                    low_momentum_returns = data.loc[~high_momentum_mask, 'price_change_2h'].mean()
                    
                    threshold_effectiveness = high_momentum_returns - low_momentum_returns
                    
                    results['metrics'] = {
                        'threshold_value': high_momentum_threshold,
                        'high_momentum_samples': high_momentum_mask.sum(),
                        'high_momentum_avg_return': high_momentum_returns,
                        'low_momentum_avg_return': low_momentum_returns,
                        'threshold_effectiveness': threshold_effectiveness,
                        'correlations': correlations
                    }
                    
                    # Validation passes if high momentum leads to better returns
                    results['correlation_valid'] = threshold_effectiveness > 0.1  # At least 0.1% better returns
        
        return results
    
    def _validate_financial_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate financial data integrity (OHLC relationships, etc.)."""
        results = {'passed': True, 'errors': [], 'integrity_checks': {}}
        
        required_ohlc = ['open', 'high', 'low', 'close']
        
        if not all(col in data.columns for col in required_ohlc):
            results['errors'].append("Missing OHLC columns for integrity validation")
            results['passed'] = False
            return results
        
        # Check OHLC relationships
        invalid_high = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
        invalid_low = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
        
        if invalid_high > 0:
            results['errors'].append(f"Found {invalid_high} records where high < max(open, close)")
            results['passed'] = False
        
        if invalid_low > 0:
            results['errors'].append(f"Found {invalid_low} records where low > min(open, close)")
            results['passed'] = False
        
        # Check for impossible price movements
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # >50% change in 4h
            
            if extreme_changes > len(data) * 0.001:  # More than 0.1% of data
                results['errors'].append(f"Found {extreme_changes} extreme price movements (>50%)")
        
        results['integrity_checks'] = {
            'invalid_high_count': invalid_high,
            'invalid_low_count': invalid_low,
            'extreme_movements': extreme_changes if 'extreme_changes' in locals() else 0
        }
        
        return results
    
    def save_validation_report(self, filepath: str = "validation_report.json"):
        """Save validation results to file."""
        if not self.validation_results:
            print("❌ No validation results to save")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"📄 Validation report saved to {filepath}")
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary."""
        if not self.validation_results:
            return "No validation results available"
        
        results = self.validation_results
        
        summary = f"""
🔍 DATA VALIDATION SUMMARY
========================
Status: {'✅ PASSED' if results['validation_passed'] else '❌ FAILED'}
Records: {results['total_records']:,}
Columns: {results['total_columns']}
Quality Score: {results.get('quality_metrics', {}).get('quality_score', 0):.2f}/1.0

📊 Schema Compliance: {'✅' if results.get('schema_compliance', {}).get('passed', False) else '❌'}
📈 Data Quality: {results.get('quality_metrics', {}).get('quality_score', 0):.2f}
⏰ Temporal Integrity: {'✅' if results.get('temporal_integrity', {}).get('passed', False) else '❌'}
🎯 Momentum Validation: {'✅' if results.get('momentum_validation', {}).get('correlation_valid', False) else '⚠️'}

Errors: {len(results['errors'])}
Warnings: {len(results['warnings'])}
"""
        
        if results['errors']:
            summary += "\n❌ ERRORS:\n" + "\n".join(f"  - {error}" for error in results['errors'])
        
        if results['warnings']:
            summary += "\n⚠️ WARNINGS:\n" + "\n".join(f"  - {warning}" for warning in results['warnings'])
        
        return summary

def main():
    """Test data validation."""
    # Load sample data for validation
    try:
        data = pd.read_csv('data/4h_training/crypto_4h_dataset_20250714_130201.csv', 
                          index_col=0, parse_dates=True, nrows=1000)
        
        validator = DataValidator()
        results = validator.validate_dataset(data)
        
        print(validator.get_validation_summary())
        validator.save_validation_report()
        
    except FileNotFoundError:
        print("❌ Sample dataset not found. Create synthetic data for testing.")
        
        # Create synthetic test data
        dates = pd.date_range('2024-01-01', periods=1000, freq='4H')
        synthetic_data = pd.DataFrame({
            'open': np.random.randn(1000) * 1000 + 50000,
            'high': np.random.randn(1000) * 1000 + 51000,
            'low': np.random.randn(1000) * 1000 + 49000,
            'close': np.random.randn(1000) * 1000 + 50000,
            'volume': np.random.randint(1000, 100000, 1000),
            'symbol': np.random.choice(['BTC', 'ETH', 'ADA'], 1000),
            'momentum_1': np.random.randn(1000) * 5,
            'rsi': np.random.uniform(20, 80, 1000),
            'price_change_2h': np.random.randn(1000) * 2
        }, index=dates)
        
        validator = DataValidator()
        results = validator.validate_dataset(synthetic_data, ['BTC', 'ETH', 'ADA'])
        
        print(validator.get_validation_summary())

if __name__ == "__main__":
    main()