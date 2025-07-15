"""Interactive training session for the Crypto RF Trading System."""

import asyncio
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_default_config, Config
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.feature_engineering import CryptoFeatureEngine
from models.random_forest_model import CryptoRandomForestModel

# Setup colorful logging
import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class InteractiveTrainer:
    """Interactive training session for crypto Random Forest models."""
    
    def __init__(self):
        self.config = get_default_config()
        self.model = None
        self.features = None
        self.data = None
        self.results = {}
        
    async def start_training_session(self):
        """Start an interactive training session."""
        print("\n" + "="*80)
        print("🚀 CRYPTOCURRENCY RANDOM FOREST - INTERACTIVE TRAINING SESSION")
        print("="*80 + "\n")
        
        # Step 1: Choose cryptocurrencies
        print("📋 Step 1: Select Cryptocurrencies for Training")
        print("\nAvailable options:")
        print("1. Top 3 (BTC, ETH, SOL) - Quick training")
        print("2. Top 5 (+ BNB, ADA) - Balanced")
        print("3. All 9 cryptocurrencies - Comprehensive")
        print("4. Custom selection")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            self.config.data.symbols = ['bitcoin', 'ethereum', 'solana']
        elif choice == '2':
            self.config.data.symbols = ['bitcoin', 'ethereum', 'solana', 'binancecoin', 'cardano']
        elif choice == '3':
            self.config.data.symbols = ['bitcoin', 'ethereum', 'tether', 'solana', 
                                       'binancecoin', 'usd-coin', 'ripple', 'dogecoin', 'cardano']
        else:
            print("\nEnter symbols (comma-separated, e.g., bitcoin,ethereum):")
            symbols = input().strip().split(',')
            self.config.data.symbols = [s.strip() for s in symbols]
        
        print(f"\n✅ Selected {len(self.config.data.symbols)} cryptocurrencies: {', '.join(self.config.data.symbols)}")
        
        # Step 2: Choose time period
        print("\n📅 Step 2: Select Training Data Period")
        print("1. 30 days - Quick test")
        print("2. 90 days - Standard")
        print("3. 180 days - Extended")
        print("4. 365 days - Full year")
        print("5. 730 days - Maximum (2 years)")
        
        period_choice = input("\nSelect period (1-5): ").strip()
        period_map = {'1': 30, '2': 90, '3': 180, '4': 365, '5': 730}
        self.config.data.days = period_map.get(period_choice, 90)
        
        print(f"\n✅ Selected {self.config.data.days} days of historical data")
        
        # Step 3: Model configuration
        print("\n🤖 Step 3: Configure Random Forest Model")
        print("1. Quick mode (100 trees, no tuning)")
        print("2. Standard mode (200 trees, no tuning)")
        print("3. Advanced mode (200 trees + hyperparameter tuning)")
        print("4. Custom configuration")
        
        model_choice = input("\nSelect mode (1-4): ").strip()
        
        tune_hyperparameters = False
        if model_choice == '1':
            self.config.model.n_estimators = 100
        elif model_choice == '2':
            self.config.model.n_estimators = 200
        elif model_choice == '3':
            self.config.model.n_estimators = 200
            tune_hyperparameters = True
        else:
            self.config.model.n_estimators = int(input("Number of trees: ") or 200)
            self.config.model.max_depth = int(input("Max depth (default 10): ") or 10)
            tune_hyperparameters = input("Enable hyperparameter tuning? (y/n): ").lower() == 'y'
        
        print(f"\n✅ Model configured with {self.config.model.n_estimators} trees")
        if tune_hyperparameters:
            print("   🔧 Hyperparameter tuning enabled")
        
        # Start training
        print("\n" + "-"*80)
        print("🏃 STARTING TRAINING PROCESS...")
        print("-"*80 + "\n")
        
        # Fetch data
        print("📊 Fetching cryptocurrency data...")
        self.data = await self._fetch_data()
        
        # Engineer features
        print("\n🔧 Engineering features...")
        self.features = self._engineer_features()
        
        # Train model
        print("\n🤖 Training Random Forest model...")
        self.model, train_results = await self._train_model(tune_hyperparameters)
        
        # Evaluate model
        print("\n📈 Evaluating model performance...")
        eval_results = self._evaluate_model()
        
        # Generate predictions
        print("\n🎯 Generating trading predictions...")
        predictions = self._generate_predictions()
        
        # Show results
        self._display_results(train_results, eval_results, predictions)
        
        # Save option
        if input("\n💾 Save trained model? (y/n): ").lower() == 'y':
            filename = f"models/crypto_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.model.save_model(filename)
            print(f"✅ Model saved to {filename}")
        
        return self.results
    
    async def _fetch_data(self):
        """Fetch cryptocurrency data."""
        fetcher = YFinanceCryptoFetcher(self.config.data)
        
        # Show progress for each symbol
        data_dict = {}
        for i, symbol in enumerate(self.config.data.symbols):
            print(f"   [{i+1}/{len(self.config.data.symbols)}] Fetching {symbol}...", end='', flush=True)
            df = fetcher.fetch_crypto_data(symbol)
            if not df.empty:
                data_dict[symbol] = df
                print(f" ✓ ({len(df)} records)")
            else:
                print(f" ✗ (failed)")
        
        # Get current prices
        print("\n💰 Current prices:")
        prices = fetcher.get_latest_prices(self.config.data.symbols)
        for symbol, price in prices.items():
            print(f"   {symbol.upper()}: ${price:,.2f}")
        
        # Combine data
        combined_data = fetcher.combine_data(data_dict)
        clean_data = fetcher.get_clean_data(combined_data)
        
        print(f"\n✅ Total data shape: {clean_data.shape}")
        print(f"   Date range: {clean_data.index[0]} to {clean_data.index[-1]}")
        
        return clean_data
    
    def _engineer_features(self):
        """Engineer features from data."""
        feature_engine = CryptoFeatureEngine(self.config.features)
        
        print("   Generating technical indicators...")
        features = feature_engine.generate_features(self.data)
        
        # Show feature breakdown
        feature_types = {
            'Price': len([c for c in features.columns if 'price' in c or 'close' in c]),
            'Volume': len([c for c in features.columns if 'volume' in c]),
            'Technical': len([c for c in features.columns if any(ind in c for ind in ['rsi', 'macd', 'bb', 'ema', 'sma'])]),
            'Volatility': len([c for c in features.columns if 'volatility' in c or 'atr' in c]),
            'Returns': len([c for c in features.columns if 'return' in c]),
        }
        
        print(f"\n✅ Generated {features.shape[1]} features:")
        for ftype, count in feature_types.items():
            print(f"   • {ftype}: {count} features")
        
        return features
    
    async def _train_model(self, tune_hyperparameters):
        """Train the Random Forest model."""
        model = CryptoRandomForestModel(self.config.model)
        
        # Create target
        primary_symbol = self.config.data.symbols[0]
        target_col = f"{primary_symbol}_close"
        
        if target_col in self.data.columns:
            print(f"   Creating target: {self.config.model.target_horizon}-hour ahead returns for {primary_symbol}")
            target = self.data[target_col].pct_change(self.config.model.target_horizon).shift(-self.config.model.target_horizon)
            target = target.dropna()
        else:
            raise ValueError(f"Could not find {target_col} in data")
        
        # Align features and target
        common_index = self.features.index.intersection(target.index)
        X = self.features.loc[common_index].copy()
        y = target.loc[common_index]
        
        # Prepare data
        X['target'] = y
        X_clean, y_clean = model.prepare_data(X, 'target')
        
        print(f"   Training data: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            print("\n   🔧 Running hyperparameter optimization (this may take a few minutes)...")
            tuning_results = model.hyperparameter_tuning(X_clean, y_clean, method='optuna')
            print(f"   ✅ Best parameters found: {tuning_results['best_params']}")
        
        # Train model
        print("\n   Training Random Forest...")
        train_results = model.train(X_clean, y_clean, validation_split=0.2)
        
        print(f"\n   📊 Training Results:")
        print(f"      • Train R²: {train_results['train']['r2']:.4f}")
        print(f"      • Validation R²: {train_results['validation']['r2']:.4f}")
        print(f"      • RMSE: {train_results['validation']['rmse']:.4f}")
        
        # Store data for evaluation
        self.X_clean = X_clean
        self.y_clean = y_clean
        
        return model, train_results
    
    def _evaluate_model(self):
        """Evaluate model with walk-forward validation."""
        print("   Running walk-forward validation...")
        
        # Quick walk-forward with fewer splits for demo
        self.config.model.walk_forward_splits = 5
        wf_results = self.model.walk_forward_validation(self.X_clean, self.y_clean)
        
        print(f"\n   📊 Walk-Forward Validation Results:")
        print(f"      • Average R²: {wf_results['avg_r2']:.4f} (±{wf_results['std_r2']:.4f})")
        print(f"      • Average RMSE: {wf_results['avg_rmse']:.4f}")
        print(f"      • Consistency: {sum(1 for r in wf_results['r2_scores'] if r > 0)}/{len(wf_results['r2_scores'])} positive periods")
        
        return wf_results
    
    def _generate_predictions(self):
        """Generate trading predictions on test set."""
        # Use last 20% as test set
        test_size = int(len(self.X_clean) * 0.2)
        X_test = self.X_clean.iloc[-test_size:]
        y_test = self.y_clean.iloc[-test_size:]
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Generate trading signals
        pred_series = pd.Series(predictions)
        signals = np.zeros(len(predictions))
        signals[pred_series > pred_series.quantile(0.7)] = 1  # Buy top 30%
        signals[pred_series < pred_series.quantile(0.3)] = -1  # Sell bottom 30%
        
        # Calculate simple performance
        actual_returns = y_test.values
        strategy_returns = signals[:-1] * actual_returns[1:]  # Lag signals
        
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6) * np.sqrt(365*24)
        win_rate = np.mean(strategy_returns > 0)
        
        print(f"\n   📊 Trading Strategy Performance (Test Set):")
        print(f"      • Total Return: {total_return:.2%}")
        print(f"      • Sharpe Ratio: {sharpe:.2f}")
        print(f"      • Win Rate: {win_rate:.2%}")
        print(f"      • Buy Signals: {np.sum(signals == 1)} ({np.mean(signals == 1):.1%})")
        print(f"      • Sell Signals: {np.sum(signals == -1)} ({np.mean(signals == -1):.1%})")
        
        return {
            'predictions': predictions,
            'signals': signals,
            'performance': {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate
            }
        }
    
    def _display_results(self, train_results, eval_results, predictions):
        """Display comprehensive training results."""
        print("\n" + "="*80)
        print("📊 TRAINING SESSION COMPLETE - RESULTS SUMMARY")
        print("="*80)
        
        # Model info
        print(f"\n🤖 Model Information:")
        print(f"   • Algorithm: Random Forest")
        print(f"   • Trees: {self.model.model.n_estimators}")
        print(f"   • Max Depth: {self.model.model.max_depth}")
        print(f"   • Features Used: {self.X_clean.shape[1]}")
        print(f"   • Training Samples: {self.X_clean.shape[0]}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   • Training R²: {train_results['train']['r2']:.4f}")
        print(f"   • Validation R²: {train_results['validation']['r2']:.4f}")
        print(f"   • Walk-Forward Avg R²: {eval_results['avg_r2']:.4f}")
        print(f"   • Strategy Sharpe Ratio: {predictions['performance']['sharpe_ratio']:.2f}")
        
        # Feature importance
        print(f"\n🏆 Top 5 Most Important Features:")
        importance_df = self.model.get_feature_importance(top_n=5)
        for idx, row in importance_df.iterrows():
            print(f"   {idx+1}. {row['feature'][:40]:40s} {row['importance']:.3f}")
        
        # Trading strategy
        print(f"\n💼 Trading Strategy Results:")
        print(f"   • Test Return: {predictions['performance']['total_return']:.2%}")
        print(f"   • Win Rate: {predictions['performance']['win_rate']:.2%}")
        print(f"   • Total Signals: {np.sum(predictions['signals'] != 0)}")
        
        # Store results
        self.results = {
            'model_config': {
                'n_estimators': self.model.model.n_estimators,
                'max_depth': self.model.model.max_depth,
                'features': self.X_clean.shape[1],
                'samples': self.X_clean.shape[0]
            },
            'training_results': train_results,
            'evaluation_results': eval_results,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create visualization
        self._create_visualization()
    
    def _create_visualization(self):
        """Create and save training visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Crypto Random Forest Training Results', fontsize=16)
        
        # Plot 1: Feature Importance
        ax1 = axes[0, 0]
        importance_df = self.model.get_feature_importance(top_n=15)
        ax1.barh(range(len(importance_df)), importance_df['importance'])
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels([f[:25] for f in importance_df['feature']])
        ax1.set_xlabel('Importance')
        ax1.set_title('Top 15 Feature Importance')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Walk-Forward R² Scores
        ax2 = axes[0, 1]
        if 'r2_scores' in self.results['evaluation_results']:
            r2_scores = self.results['evaluation_results']['r2_scores']
            ax2.plot(range(len(r2_scores)), r2_scores, 'o-', linewidth=2, markersize=8)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Validation Period')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Walk-Forward Validation Performance')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prediction Distribution
        ax3 = axes[1, 0]
        predictions = self.results['predictions']['predictions']
        ax3.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Predicted Returns')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Predictions')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Info
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = f"""Training Summary
        
Cryptocurrencies: {len(self.config.data.symbols)}
Training Period: {self.config.data.days} days
Total Features: {self.X_clean.shape[1]}
Training Samples: {self.X_clean.shape[0]}

Model Performance:
• Train R²: {self.results['training_results']['train']['r2']:.4f}
• Valid R²: {self.results['training_results']['validation']['r2']:.4f}
• Sharpe Ratio: {self.results['predictions']['performance']['sharpe_ratio']:.2f}
• Win Rate: {self.results['predictions']['performance']['win_rate']:.2%}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", 
                 facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to {filename}")


async def main():
    """Run interactive training session."""
    trainer = InteractiveTrainer()
    results = await trainer.start_training_session()
    
    # Save results
    with open(f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✨ Training session complete! Check the generated files for detailed results.")


if __name__ == "__main__":
    asyncio.run(main())