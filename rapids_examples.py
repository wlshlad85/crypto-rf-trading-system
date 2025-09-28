#!/usr/bin/env python3
"""
RAPIDS Examples - Quick demonstrations of key features
Run this script to see RAPIDS in action on Windows WSL2
"""

import time
import warnings
warnings.filterwarnings('ignore')

# Color formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")

def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print("-" * len(text))

def example_1_basic_cudf():
    """Basic cuDF operations"""
    print_header("Example 1: cuDF Basic Operations")
    
    try:
        import cudf
        import cupy as cp
        
        # Create a GPU DataFrame
        n = 1_000_000
        df = cudf.DataFrame({
            'id': cp.arange(n),
            'x': cp.random.normal(0, 1, n),
            'y': cp.random.normal(0, 1, n),
            'category': cp.random.choice(['A', 'B', 'C', 'D'], n)
        })
        
        print(f"Created DataFrame with {len(df):,} rows")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Perform operations
        df['distance'] = cp.sqrt(df['x']**2 + df['y']**2)
        
        # Group by and aggregate
        result = df.groupby('category').agg({
            'distance': ['mean', 'std', 'count']
        })
        
        print("\nGrouped statistics:")
        print(result)
        
        print(f"\n{Colors.GREEN}✓ cuDF operations completed successfully{Colors.END}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def example_2_performance_comparison():
    """Compare pandas vs cuDF performance"""
    print_header("Example 2: Performance Comparison")
    
    try:
        import pandas as pd
        import cudf
        import numpy as np
        
        # Create test data
        n = 5_000_000
        data = {
            'key': np.random.randint(0, 1000, n),
            'value1': np.random.randn(n),
            'value2': np.random.randn(n)
        }
        
        print(f"Testing with {n:,} rows...")
        
        # Pandas timing
        pdf = pd.DataFrame(data)
        start = time.time()
        pandas_result = pdf.groupby('key').agg({
            'value1': 'sum',
            'value2': 'mean'
        })
        pandas_time = time.time() - start
        
        # cuDF timing
        gdf = cudf.DataFrame(data)
        start = time.time()
        cudf_result = gdf.groupby('key').agg({
            'value1': 'sum',
            'value2': 'mean'
        })
        cudf_time = time.time() - start
        
        speedup = pandas_time / cudf_time
        
        print(f"\nResults:")
        print(f"Pandas time: {pandas_time:.3f} seconds")
        print(f"cuDF time: {cudf_time:.3f} seconds")
        print(f"{Colors.GREEN}Speedup: {speedup:.1f}x{Colors.END}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def example_3_machine_learning():
    """cuML machine learning example"""
    print_header("Example 3: GPU Machine Learning with cuML")
    
    try:
        import cuml
        from cuml.datasets import make_regression
        from cuml.linear_model import LinearRegression
        from cuml.metrics import r2_score
        
        # Generate synthetic data
        n_samples = 100_000
        n_features = 20
        
        print(f"Generating dataset: {n_samples:,} samples, {n_features} features")
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=10
        )
        
        # Split data
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train model
        print("Training Linear Regression model on GPU...")
        start = time.time()
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Predict
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        
        print(f"\nResults:")
        print(f"Training time: {train_time:.3f} seconds")
        print(f"R² score: {score:.4f}")
        print(f"{Colors.GREEN}✓ Model trained successfully{Colors.END}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def example_4_time_series():
    """Time series analysis example"""
    print_header("Example 4: Time Series Analysis")
    
    try:
        import cudf
        import cupy as cp
        import pandas as pd
        
        # Generate time series data
        print("Generating synthetic sensor data...")
        n_days = 30
        n_sensors = 50
        n_records = n_days * n_sensors * 24  # Hourly data
        
        # Create timestamps
        timestamps = pd.date_range('2025-01-01', periods=n_days*24, freq='H')
        
        # Generate sensor data
        sensor_data = []
        for sensor_id in range(n_sensors):
            for ts in timestamps:
                temp = 20 + 10 * cp.sin(2 * cp.pi * ts.hour / 24) + cp.random.normal(0, 2)
                sensor_data.append({
                    'timestamp': ts,
                    'sensor_id': sensor_id,
                    'temperature': float(temp),
                    'humidity': float(60 + cp.random.normal(0, 10))
                })
        
        # Create cuDF DataFrame
        df = cudf.DataFrame(sensor_data)
        print(f"Created {len(df):,} sensor readings")
        
        # Analyze data
        print("\nCalculating daily statistics...")
        df['date'] = df['timestamp'].dt.date
        
        daily_stats = df.groupby(['sensor_id', 'date']).agg({
            'temperature': ['mean', 'min', 'max'],
            'humidity': 'mean'
        })
        
        print("\nSample daily statistics (first 5 rows):")
        print(daily_stats.head())
        
        # Find anomalies
        sensor_means = df.groupby('sensor_id')['temperature'].mean()
        sensor_stds = df.groupby('sensor_id')['temperature'].std()
        
        print(f"\n{Colors.GREEN}✓ Time series analysis completed{Colors.END}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def example_5_data_pipeline():
    """End-to-end data pipeline example"""
    print_header("Example 5: GPU Data Pipeline")
    
    try:
        import cudf
        import cuml
        from cuml.preprocessing import StandardScaler
        from cuml.decomposition import PCA
        from cuml.cluster import KMeans
        
        print("Building GPU-accelerated data pipeline...")
        
        # Generate sample data
        n_samples = 50_000
        n_features = 50
        
        # Create random data
        data = cudf.DataFrame({
            f'feature_{i}': cudf.Series(cupy.random.randn(n_samples))
            for i in range(n_features)
        })
        
        print(f"Data shape: {data.shape}")
        
        # Pipeline steps
        print("\n1. Standardizing features...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        print("2. Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=10)
        reduced_data = pca.fit_transform(scaled_data)
        
        print("3. Clustering with K-Means...")
        kmeans = KMeans(n_clusters=5)
        clusters = kmeans.fit_predict(reduced_data)
        
        # Results
        cluster_counts = cudf.Series(clusters).value_counts().sort_index()
        print("\nCluster distribution:")
        for idx, count in cluster_counts.items():
            print(f"  Cluster {idx}: {count:,} samples")
        
        print(f"\n{Colors.GREEN}✓ Pipeline completed successfully{Colors.END}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:3]}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def main():
    """Run all examples"""
    print_header("RAPIDS Examples on Windows WSL2")
    print("This script demonstrates key RAPIDS features\n")
    
    examples = [
        ("Basic cuDF Operations", example_1_basic_cudf),
        ("Performance Comparison", example_2_performance_comparison),
        ("Machine Learning", example_3_machine_learning),
        ("Time Series Analysis", example_4_time_series),
        ("Data Pipeline", example_5_data_pipeline)
    ]
    
    # Check if RAPIDS is installed
    try:
        import cudf
        print(f"{Colors.GREEN}✓ RAPIDS detected: cuDF {cudf.__version__}{Colors.END}")
    except ImportError:
        print(f"{Colors.RED}✗ RAPIDS not found. Please install using one of the methods in the runbook.{Colors.END}")
        return
    
    # Menu
    while True:
        print(f"\n{Colors.BOLD}Available Examples:{Colors.END}")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("6. Run all examples")
        print("0. Exit")
        
        try:
            choice = input(f"\n{Colors.YELLOW}Select example (0-6): {Colors.END}")
            choice = int(choice)
            
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= 5:
                examples[choice-1][1]()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            elif choice == 6:
                for name, func in examples:
                    func()
                    time.sleep(1)
            else:
                print(f"{Colors.RED}Invalid choice{Colors.END}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")

if __name__ == "__main__":
    # Fix for cupy import
    import cupy
    main()