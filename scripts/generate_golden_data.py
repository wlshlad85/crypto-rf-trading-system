#!/usr/bin/env python3
"""
Generate golden dataset for regression testing.
This creates a representative dataset with realistic finance/ML characteristics.
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def generate_golden_dataset(n_samples=10000, n_features=50, seed=42):
    """Generate a realistic dataset with financial characteristics."""
    np.random.seed(seed)
    
    # Feature matrix with different distributions
    features = {}
    
    # Market returns (normal with fat tails)
    for i in range(10):
        # Mix of normal and student-t for fat tails
        normal_part = np.random.normal(0, 0.02, int(n_samples * 0.8))
        t_part = np.random.standard_t(df=3, size=int(n_samples * 0.2)) * 0.03
        returns = np.concatenate([normal_part, t_part])
        np.random.shuffle(returns)
        features[f'return_{i}'] = returns[:n_samples]
    
    # Technical indicators (bounded)
    for i in range(10):
        # RSI-like (0-100)
        features[f'rsi_{i}'] = np.random.beta(2, 2, n_samples) * 100
        # MACD-like (unbounded but typically small)
        features[f'macd_{i}'] = np.random.normal(0, 0.5, n_samples)
    
    # Volume features (log-normal)
    for i in range(5):
        features[f'volume_{i}'] = np.random.lognormal(10, 2, n_samples)
    
    # Price ratios (around 1.0)
    for i in range(5):
        features[f'price_ratio_{i}'] = np.random.lognormal(0, 0.1, n_samples)
    
    # Categorical encoded as numeric
    for i in range(5):
        features[f'sector_{i}'] = np.random.randint(0, 10, n_samples)
    
    # Time features
    features['hour'] = np.random.randint(0, 24, n_samples)
    features['day_of_week'] = np.random.randint(0, 7, n_samples)
    features['month'] = np.random.randint(1, 13, n_samples)
    
    # Add some NaNs (2% missing data)
    for col in list(features.keys())[:10]:
        mask = np.random.random(n_samples) < 0.02
        features[col][mask] = np.nan
    
    # Create target variable (binary classification)
    # Weak signal: depends on first few features
    X = np.column_stack([features[f'return_{i}'] for i in range(3)])
    X_clean = np.nan_to_num(X, 0)
    signal = np.sum(X_clean, axis=1) + np.random.normal(0, 0.5, n_samples)
    y = (signal > np.percentile(signal, 60)).astype(int)
    
    # Add some outliers
    outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    for idx in outlier_idx:
        col = np.random.choice(list(features.keys()))
        if 'return' in col:
            features[col][idx] = np.random.choice([-0.2, 0.2])  # 20% moves
        elif 'volume' in col:
            features[col][idx] *= 100  # Volume spike
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['y'] = y
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate golden dataset')
    parser.add_argument('--output', default='tests/data/golden.parquet',
                        help='Output path for golden dataset')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples')
    parser.add_argument('--n-features', type=int, default=50,
                        help='Number of features')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate dataset
    df = generate_golden_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed
    )
    
    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression='snappy')
    
    # Print statistics
    print(f"Generated golden dataset: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution: {df['y'].value_counts().to_dict()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save basic stats for reference
    stats = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'target_mean': float(df['y'].mean()),
        'missing_pct': float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])),
        'columns': df.columns.tolist()
    }
    
    import json
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")

if __name__ == '__main__':
    main()