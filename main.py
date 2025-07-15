"""Main execution script for the Crypto Random Forest Trading System."""

import asyncio
import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import load_config, get_default_config
from data.data_fetcher import CryptoDataFetcher
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel, EnsembleRandomForestModel
from backtesting.backtest_engine import CryptoBacktestEngine, run_walk_forward_backtest
from utils.visualization import create_full_report


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crypto_rf_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


class CryptoRFTradingSystem:
    """Main cryptocurrency Random Forest trading system."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path) if config_path else get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.data_fetcher = None
        self.feature_engine = None
        self.model = None
        self.backtest_engine = None
        
        # Data storage
        self.raw_data = None
        self.features = None
        self.targets = None
        
        # Results
        self.backtest_results = None
        
        # Ensure directories exist
        os.makedirs(self.config.system.results_dir, exist_ok=True)
        os.makedirs(self.config.system.models_dir, exist_ok=True)
    
    async def fetch_data(self) -> pd.DataFrame:
        """Fetch cryptocurrency data."""
        self.logger.info("Starting data fetching process")
        
        if self.config.data.data_source == "yfinance":
            # Use yfinance for free data
            fetcher = YFinanceCryptoFetcher(self.config.data)
            
            # Fetch data for all configured symbols
            data_dict = fetcher.fetch_all_symbols()
            
            if not data_dict:
                raise ValueError("No data fetched. Check network connection.")
            
            # Combine data
            combined_data = fetcher.combine_data(data_dict)
            
            # Clean data
            self.raw_data = fetcher.get_clean_data(combined_data)
        else:
            # Use CoinGecko (original implementation)
            async with CryptoDataFetcher(self.config.data) as fetcher:
                # Fetch data for all configured symbols
                data_dict = await fetcher.fetch_all_symbols()
                
                if not data_dict:
                    raise ValueError("No data fetched. Check API configuration and network connection.")
                
                # Combine data
                combined_data = fetcher.combine_data(data_dict)
                
                # Clean data
                self.raw_data = fetcher.get_clean_data(combined_data)
        
        self.logger.info(f"Fetched data shape: {self.raw_data.shape}")
        self.logger.info(f"Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        
        return self.raw_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features from raw data."""
        if self.raw_data is None:
            raise ValueError("Raw data not available. Run fetch_data() first.")
        
        self.logger.info("Starting feature engineering")
        
        self.feature_engine = CryptoFeatureEngine(self.config.features)
        
        # Generate features
        self.features = self.feature_engine.generate_features(self.raw_data)
        
        self.logger.info(f"Generated features shape: {self.features.shape}")
        
        return self.features
    
    def prepare_model(self, use_ensemble: bool = False) -> CryptoRandomForestModel:
        """Prepare the Random Forest model."""
        self.logger.info(f"Preparing {'ensemble' if use_ensemble else 'single'} Random Forest model")
        
        if use_ensemble:
            self.model = EnsembleRandomForestModel(self.config.model, n_models=5)
        else:
            self.model = CryptoRandomForestModel(self.config.model)
        
        return self.model
    
    def train_model(self, tune_hyperparameters: bool = True) -> Dict:
        """Train the Random Forest model."""
        if self.features is None:
            raise ValueError("Features not available. Run engineer_features() first.")
        
        if self.model is None:
            self.prepare_model()
        
        self.logger.info("Starting model training")
        
        # Create targets
        symbols = self.config.data.symbols
        self.targets = self.model.create_targets(self.raw_data, symbols)
        
        # Prepare training data for the first symbol (can be extended for multi-symbol)
        primary_symbol = symbols[0]
        target_col = f"{primary_symbol}_target"
        
        if target_col not in self.targets.columns:
            self.logger.warning(f"Target column {target_col} not found. Available columns: {list(self.targets.columns)}")
            # Create a simple target if none exists
            primary_close_col = f"{primary_symbol}_close"
            if primary_close_col in self.raw_data.columns:
                simple_target = self.raw_data[primary_close_col].pct_change(6).shift(-6)
                simple_target = simple_target.dropna()
                
                # Align with features
                common_index = self.features.index.intersection(simple_target.index)
                aligned_features = self.features.loc[common_index]
                aligned_features[target_col] = simple_target.loc[common_index]
                
                X, y = self.model.prepare_data(aligned_features, target_col)
            else:
                raise ValueError(f"Could not create target for {primary_symbol}")
        else:
            # Align features and targets
            common_index = self.features.index.intersection(self.targets.index)
            aligned_features = self.features.loc[common_index]
            aligned_targets = self.targets.loc[common_index]
            
            # Add target to features for prepare_data
            aligned_features[target_col] = aligned_targets[target_col]
            
            X, y = self.model.prepare_data(aligned_features, target_col)
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            self.logger.info("Performing hyperparameter tuning")
            tuning_results = self.model.hyperparameter_tuning(X, y, method='optuna')
            self.logger.info(f"Best parameters: {tuning_results['best_params']}")
        
        # Train model
        training_results = self.model.train(X, y)
        
        # Walk-forward validation
        validation_results = self.model.walk_forward_validation(X, y)
        
        self.logger.info(f"Training completed. Validation R2: {validation_results['avg_r2']:.4f}")
        
        return {
            'training_results': training_results,
            'validation_results': validation_results
        }
    
    def run_backtest(self, walk_forward: bool = True) -> Dict:
        """Run comprehensive backtest."""
        if self.model is None or not self.model.is_fitted:
            raise ValueError("Model must be trained before backtesting")
        
        self.logger.info("Starting backtest")
        
        symbols = self.config.data.symbols
        
        if walk_forward:
            # Run walk-forward backtest
            self.backtest_results = run_walk_forward_backtest(
                self.raw_data, self.features, self.config, symbols
            )
        else:
            # Run single period backtest
            self.backtest_engine = CryptoBacktestEngine(self.config)
            self.backtest_results = self.backtest_engine.run_backtest(
                self.raw_data, self.features, self.model, symbols
            )
        
        self.logger.info("Backtest completed")
        
        return self.backtest_results
    
    def generate_report(self, output_dir: str = None) -> Dict[str, str]:
        """Generate comprehensive trading report."""
        if self.backtest_results is None:
            raise ValueError("Backtest results not available. Run run_backtest() first.")
        
        output_dir = output_dir or f"{self.config.system.results_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Generating report in {output_dir}")
        
        # Get portfolio data (this would need to be extracted from backtest results)
        # For now, creating a placeholder
        portfolio_df = pd.DataFrame({
            'total_value': [100000],  # Placeholder
            'num_positions': [0]
        }, index=[datetime.now()])
        
        # Create visualizations
        saved_files = create_full_report(
            self.backtest_results,
            portfolio_df,
            self.features,
            self.config.data.symbols,
            output_dir
        )
        
        # Save backtest results
        results_file = f"{output_dir}/backtest_results.json"
        self.save_results(results_file)
        saved_files['backtest_results'] = results_file
        
        # Save model
        model_file = f"{output_dir}/trained_model.joblib"
        self.model.save_model(model_file)
        saved_files['model'] = model_file
        
        self.logger.info(f"Report generated successfully in {output_dir}")
        
        return saved_files
    
    def save_results(self, filepath: str):
        """Save results to file."""
        import json
        
        results_to_save = {
            'config': {
                'data': self.config.data.__dict__,
                'features': self.config.features.__dict__,
                'model': self.config.model.__dict__,
                'strategy': self.config.strategy.__dict__,
                'backtest': self.config.backtest.__dict__
            },
            'backtest_results': self.backtest_results,
            'data_summary': {
                'raw_data_shape': self.raw_data.shape if self.raw_data is not None else None,
                'features_shape': self.features.shape if self.features is not None else None,
                'date_range': {
                    'start': str(self.raw_data.index[0]) if self.raw_data is not None else None,
                    'end': str(self.raw_data.index[-1]) if self.raw_data is not None else None
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    async def run_full_pipeline(self, tune_hyperparameters: bool = True, 
                               use_ensemble: bool = False,
                               walk_forward_backtest: bool = True) -> Dict[str, str]:
        """Run the complete trading system pipeline."""
        self.logger.info("Starting full pipeline execution")
        
        try:
            # 1. Fetch data
            await self.fetch_data()
            
            # 2. Engineer features
            self.engineer_features()
            
            # 3. Prepare and train model
            self.prepare_model(use_ensemble=use_ensemble)
            self.train_model(tune_hyperparameters=tune_hyperparameters)
            
            # 4. Run backtest
            self.run_backtest(walk_forward=walk_forward_backtest)
            
            # 5. Generate report
            report_files = self.generate_report()
            
            self.logger.info("Full pipeline completed successfully")
            
            return report_files
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Random Forest Trading System')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['full', 'data', 'train', 'backtest', 'report'],
                       default='full', help='Execution mode')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--use-ensemble', action='store_true',
                       help='Use ensemble of Random Forest models')
    parser.add_argument('--walk-forward', action='store_true', default=True,
                       help='Use walk-forward backtesting')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Crypto Random Forest Trading System")
    logger.info(f"Execution mode: {args.mode}")
    
    # Initialize system
    system = CryptoRFTradingSystem(args.config)
    
    try:
        if args.mode == 'full':
            # Run complete pipeline
            report_files = await system.run_full_pipeline(
                tune_hyperparameters=args.tune_hyperparameters,
                use_ensemble=args.use_ensemble,
                walk_forward_backtest=args.walk_forward
            )
            
            logger.info("Pipeline completed successfully!")
            logger.info("Generated files:")
            for file_type, filepath in report_files.items():
                logger.info(f"  {file_type}: {filepath}")
        
        elif args.mode == 'data':
            # Only fetch data
            await system.fetch_data()
            logger.info(f"Data fetched successfully. Shape: {system.raw_data.shape}")
        
        elif args.mode == 'train':
            # Fetch data and train model
            await system.fetch_data()
            system.engineer_features()
            system.prepare_model(use_ensemble=args.use_ensemble)
            training_results = system.train_model(tune_hyperparameters=args.tune_hyperparameters)
            
            logger.info("Model training completed")
            logger.info(f"Validation results: {training_results['validation_results']}")
        
        elif args.mode == 'backtest':
            # Run full pipeline up to backtest
            await system.fetch_data()
            system.engineer_features()
            system.prepare_model(use_ensemble=args.use_ensemble)
            system.train_model(tune_hyperparameters=args.tune_hyperparameters)
            backtest_results = system.run_backtest(walk_forward=args.walk_forward)
            
            logger.info("Backtest completed")
            logger.info(f"Results: {backtest_results}")
        
        elif args.mode == 'report':
            # Generate report (assumes previous run saved results)
            logger.info("Report generation mode not fully implemented")
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())