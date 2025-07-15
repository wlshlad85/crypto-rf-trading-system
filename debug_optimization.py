"""Debug optimization with comprehensive logging to identify issues."""

import pandas as pd
import numpy as np
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config, Config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('debug_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DebugOptimizer:
    """Debug version of optimizer with comprehensive logging."""
    
    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.data = None
        self.features = None
        self.targets = {}
        
    def run_debug_optimization(self):
        """Run debug optimization with detailed logging."""
        logger.info("="*80)
        logger.info("🔍 DEBUG OPTIMIZATION START")
        logger.info("="*80)
        
        try:
            # Step 1: Data preparation
            logger.info("Step 1: Preparing data...")
            self._debug_data_preparation()
            
            # Step 2: Test single model training
            logger.info("Step 2: Testing single model training...")
            self._debug_model_training()
            
            # Step 3: Test simple optimization trial
            logger.info("Step 3: Testing simple optimization trial...")
            self._debug_optimization_trial()
            
            # Step 4: Analyze data quality
            logger.info("Step 4: Analyzing data quality...")
            self._analyze_data_quality()
            
            logger.info("✅ Debug optimization completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Debug optimization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _debug_data_preparation(self):
        """Debug data preparation step."""
        try:
            # Use reduced data for debugging
            self.config.data.days = 30  # Only 1 month for debugging
            self.config.data.symbols = ['bitcoin', 'ethereum']  # Only 2 symbols
            
            logger.info(f"Fetching {self.config.data.days} days of data for {self.config.data.symbols}")
            
            # Fetch data
            fetcher = YFinanceCryptoFetcher(self.config.data)
            data_dict = fetcher.fetch_all_symbols(self.config.data.symbols)
            
            logger.info(f"Raw data fetched: {len(data_dict)} symbols")
            for symbol, df in data_dict.items():
                logger.info(f"  {symbol}: {df.shape} - columns: {list(df.columns)}")
            
            # Get latest prices
            prices = fetcher.get_latest_prices(self.config.data.symbols)
            logger.info(f"Latest prices: {prices}")
            
            # Combine and clean data
            self.raw_data = fetcher.combine_data(data_dict)
            logger.info(f"Combined raw data shape: {self.raw_data.shape}")
            logger.info(f"Combined raw data columns: {list(self.raw_data.columns)}")
            
            self.data = fetcher.get_clean_data(self.raw_data)
            logger.info(f"Clean data shape: {self.data.shape}")
            logger.info(f"Clean data columns: {list(self.data.columns)}")
            logger.info(f"Clean data dtypes:\n{self.data.dtypes}")
            
            # Check for NaN values
            nan_counts = self.data.isnull().sum()
            logger.info(f"NaN counts:\n{nan_counts[nan_counts > 0]}")
            
            # Generate features
            feature_engine = CryptoFeatureEngine(self.config.features)
            self.features = feature_engine.generate_features(self.data)
            logger.info(f"Features shape: {self.features.shape}")
            logger.info(f"Features dtypes: {self.features.dtypes.value_counts()}")
            
            # Check features for NaN/inf values
            nan_features = self.features.isnull().sum()
            inf_features = np.isinf(self.features.select_dtypes(include=[np.number])).sum()
            logger.info(f"Features with NaN: {nan_features[nan_features > 0].head()}")
            logger.info(f"Features with inf: {inf_features[inf_features > 0].head()}")
            
            # Prepare targets
            self._debug_target_preparation()
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _debug_target_preparation(self):
        """Debug target preparation."""
        try:
            logger.info("Preparing targets...")
            
            for symbol in self.config.data.symbols:
                logger.info(f"Processing targets for {symbol}")
                
                # Check if close column exists
                close_col = f'{symbol}_close'
                if close_col not in self.data.columns:
                    logger.warning(f"Close column {close_col} not found in data")
                    continue
                
                close_series = self.data[close_col]
                logger.info(f"Close series for {symbol}: shape={close_series.shape}, non-null={close_series.count()}")
                
                # Test different horizons
                for horizon in [6, 12]:  # Reduced horizons for debugging
                    target_key = f"{symbol}_{horizon}h"
                    logger.info(f"Creating target {target_key}")
                    
                    try:
                        # Calculate percentage change
                        target = close_series.pct_change(horizon).shift(-horizon)
                        logger.info(f"Target {target_key}: shape={target.shape}, non-null={target.count()}")
                        logger.info(f"Target {target_key} stats: mean={target.mean():.6f}, std={target.std():.6f}")
                        logger.info(f"Target {target_key} range: [{target.min():.6f}, {target.max():.6f}]")
                        
                        # Remove NaN values
                        target_clean = target.dropna()
                        logger.info(f"Target {target_key} after dropna: {target_clean.shape}")
                        
                        # Align with features
                        common_index = self.features.index.intersection(target_clean.index)
                        logger.info(f"Common index size for {target_key}: {len(common_index)}")
                        
                        if len(common_index) > 100:  # Minimum threshold
                            self.targets[target_key] = {
                                'X': self.features.loc[common_index],
                                'y': target_clean.loc[common_index],
                                'prices': self.data.loc[common_index, close_col]
                            }
                            logger.info(f"Target {target_key} prepared successfully")
                        else:
                            logger.warning(f"Insufficient data for {target_key}: {len(common_index)} samples")
                            
                    except Exception as e:
                        logger.error(f"Failed to create target {target_key}: {e}")
            
            logger.info(f"Total targets prepared: {len(self.targets)}")
            for key, target_data in self.targets.items():
                logger.info(f"  {key}: X={target_data['X'].shape}, y={target_data['y'].shape}")
            
        except Exception as e:
            logger.error(f"Target preparation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _debug_model_training(self):
        """Debug model training with simple parameters."""
        try:
            if not self.targets:
                logger.error("No targets available for model training")
                return
            
            # Use the first available target
            target_key = list(self.targets.keys())[0]
            target_data = self.targets[target_key]
            logger.info(f"Testing model training with target: {target_key}")
            
            X = target_data['X']
            y = target_data['y']
            
            logger.info(f"Training data: X={X.shape}, y={y.shape}")
            logger.info(f"X dtypes: {X.dtypes.value_counts()}")
            logger.info(f"y dtype: {y.dtype}")
            
            # Create model with simple parameters
            model_config = self.config.model
            model_config.n_estimators = 10  # Very small for debugging
            model_config.max_depth = 5
            
            model = CryptoRandomForestModel(model_config)
            logger.info(f"Model created with config: {model_config}")
            
            # Prepare data for model
            logger.info("Preparing data for model...")
            X_with_target = X.copy()
            X_with_target['target'] = y
            
            logger.info(f"Data with target shape: {X_with_target.shape}")
            logger.info(f"Target column stats: {X_with_target['target'].describe()}")
            
            # Call prepare_data
            try:
                X_clean, y_clean = model.prepare_data(X_with_target, 'target')
                logger.info(f"Clean data: X={X_clean.shape}, y={y_clean.shape}")
                logger.info(f"X_clean dtypes: {X_clean.dtypes.value_counts()}")
                logger.info(f"y_clean dtype: {y_clean.dtype}")
            except Exception as e:
                logger.error(f"prepare_data failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Test train/test split
            split_idx = int(len(X_clean) * 0.8)
            X_train = X_clean.iloc[:split_idx]
            y_train = y_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_test = y_clean.iloc[split_idx:]
            
            logger.info(f"Train set: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
            
            # Train model
            try:
                logger.info("Training model...")
                model.train(X_train, y_train, validation_split=0.2)
                logger.info("Model training completed successfully")
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Test prediction
            try:
                logger.info("Testing prediction...")
                predictions = model.predict(X_test)
                logger.info(f"Predictions: shape={predictions.shape}, dtype={type(predictions)}")
                logger.info(f"Prediction stats: mean={np.mean(predictions):.6f}, std={np.std(predictions):.6f}")
                logger.info("Prediction completed successfully")
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
        except Exception as e:
            logger.error(f"Model training debug failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _debug_optimization_trial(self):
        """Debug a single optimization trial."""
        try:
            if not self.targets:
                logger.error("No targets available for optimization trial")
                return
            
            logger.info("Testing single optimization trial...")
            
            # Use simple parameters
            test_params = {
                'n_estimators': 10,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
            
            target_key = list(self.targets.keys())[0]
            target_data = self.targets[target_key]
            
            logger.info(f"Testing with parameters: {test_params}")
            logger.info(f"Using target: {target_key}")
            
            try:
                # Simulate optimization objective
                score = self._test_objective_function(test_params, target_data)
                logger.info(f"Objective function returned: {score}")
                
                if score == -100:
                    logger.error("Objective function returned penalty value (-100)")
                else:
                    logger.info("Objective function completed successfully")
                    
            except Exception as e:
                logger.error(f"Objective function test failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
        except Exception as e:
            logger.error(f"Optimization trial debug failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _test_objective_function(self, params: dict, target_data: dict) -> float:
        """Test the objective function logic."""
        try:
            X = target_data['X']
            y = target_data['y']
            
            logger.info(f"Objective function input: X={X.shape}, y={y.shape}")
            
            # Feature selection (top 50 for speed)
            n_features = min(50, X.shape[1])
            feature_vars = X.var()
            top_features = feature_vars.nlargest(n_features).index
            X_selected = X[top_features]
            
            logger.info(f"Selected features: {X_selected.shape}")
            
            # Create model
            config = self.config
            config.model.__dict__.update(params)
            model = CryptoRandomForestModel(config.model)
            
            # Prepare data
            X_with_target = X_selected.copy()
            X_with_target['target'] = y
            X_clean, y_clean = model.prepare_data(X_with_target, 'target')
            
            logger.info(f"Prepared data: X={X_clean.shape}, y={y_clean.shape}")
            
            # Simple train/test split
            split_idx = int(len(X_clean) * 0.8)
            X_train = X_clean.iloc[:split_idx]
            y_train = y_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_test = y_clean.iloc[split_idx:]
            
            logger.info(f"Split: train={X_train.shape}, test={X_test.shape}")
            
            # Train
            model.train(X_train, y_train, validation_split=0.2)
            
            # Predict
            predictions = model.predict(X_test)
            logger.info(f"Predictions: {len(predictions)} values")
            
            # Calculate simple score
            if len(predictions) > 1:
                pred_series = pd.Series(predictions)
                score = pred_series.mean() * 100  # Simple score for testing
                logger.info(f"Calculated score: {score}")
                return score
            else:
                logger.warning("Insufficient predictions")
                return -50
                
        except Exception as e:
            logger.error(f"Objective function failed: {e}")
            logger.error(traceback.format_exc())
            return -100
    
    def _analyze_data_quality(self):
        """Analyze data quality issues."""
        try:
            logger.info("Analyzing data quality...")
            
            # Check data consistency
            logger.info(f"Raw data shape: {self.raw_data.shape if self.raw_data is not None else 'None'}")
            logger.info(f"Clean data shape: {self.data.shape if self.data is not None else 'None'}")
            logger.info(f"Features shape: {self.features.shape if self.features is not None else 'None'}")
            
            if self.data is not None:
                # Check for duplicate indices
                duplicate_indices = self.data.index.duplicated().sum()
                logger.info(f"Duplicate indices in data: {duplicate_indices}")
                
                # Check date range
                logger.info(f"Data date range: {self.data.index.min()} to {self.data.index.max()}")
                
                # Check for missing values
                missing_data = self.data.isnull().sum().sum()
                logger.info(f"Total missing values in data: {missing_data}")
            
            if self.features is not None:
                # Check feature statistics
                numeric_features = self.features.select_dtypes(include=[np.number])
                logger.info(f"Numeric features: {numeric_features.shape[1]}/{self.features.shape[1]}")
                
                # Check for constant features
                constant_features = (numeric_features.std() == 0).sum()
                logger.info(f"Constant features: {constant_features}")
                
                # Check for infinite values
                inf_values = np.isinf(numeric_features).sum().sum()
                logger.info(f"Infinite values in features: {inf_values}")
            
            # Check targets
            for target_key, target_data in self.targets.items():
                X = target_data['X']
                y = target_data['y']
                logger.info(f"Target {target_key}:")
                logger.info(f"  X shape: {X.shape}")
                logger.info(f"  y shape: {y.shape}")
                logger.info(f"  y stats: mean={y.mean():.6f}, std={y.std():.6f}")
                logger.info(f"  y range: [{y.min():.6f}, {y.max():.6f}]")
                
                # Check for extreme values in target
                extreme_threshold = 0.5  # 50% change
                extreme_values = (np.abs(y) > extreme_threshold).sum()
                logger.info(f"  Extreme values (>{extreme_threshold*100}%): {extreme_values}/{len(y)}")
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            logger.error(traceback.format_exc())


def main():
    """Run debug optimization."""
    try:
        logger.info("Starting debug optimization...")
        optimizer = DebugOptimizer()
        optimizer.run_debug_optimization()
        logger.info("Debug optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Debug optimization failed: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Debug completed - check debug_optimization.log for details")
    else:
        print("❌ Debug failed - check debug_optimization.log for errors")