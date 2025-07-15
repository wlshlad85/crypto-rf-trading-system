#!/usr/bin/env python3
"""
Enhanced Backtest Pipeline for Multi-Agent Trading System

Integrates with the agent communication system and provides
comprehensive backtesting capabilities for model validation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

COMS_PATH = "crypto_rf_trader/coms.md"

class BacktestPipeline:
    """Enhanced backtest pipeline with agent integration."""
    
    def __init__(self):
        self.log("Backtest pipeline initialized")
        self.results = []
        self.models = {}
        
        # Enhanced parameters from agent system
        self.optimal_params = {
            'momentum_threshold': 1.780,
            'position_size_min': 0.464,
            'position_size_max': 0.800,
            'confidence_threshold': 0.636
        }
    
    def log(self, message):
        """Log message to communications file."""
        timestamp = datetime.utcnow().isoformat()
        with open(COMS_PATH, "a") as f:
            f.write(f"[backtest][{timestamp}] {message}\n")
        print(f"[backtest][{timestamp}] {message}")
    
    def load_data(self, file_path):
        """Load and prepare data with enhanced features."""
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=1).columns else None)
            
            # Handle different timestamp formats
            if 'timestamp' not in df.columns and df.index.name in ['timestamp', 'date']:
                df = df.reset_index()
            elif 'timestamp' not in df.columns:
                # Create synthetic timestamp if missing
                df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='4H')
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_optimal_hour'] = (df['hour'] == 3).astype(int)
            else:
                df['hour'] = np.random.randint(0, 24, len(df))
                df['day_of_week'] = np.random.randint(0, 7, len(df))
                df['is_optimal_hour'] = (df['hour'] == 3).astype(int)
            
            # Enhanced momentum features
            df['momentum_1'] = df['close'].pct_change(1) * 100
            df['momentum_4'] = df['close'].pct_change(4) * 100 if len(df) > 4 else 0
            df['momentum_strength'] = df['momentum_1'] / 4  # Hourly rate
            df['is_high_momentum'] = (df['momentum_strength'] > self.optimal_params['momentum_threshold']).astype(int)
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Create target variable if missing
            if 'target' not in df.columns:
                # Create target based on future price movement
                df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Success probability scoring
            df['success_score'] = (
                df['is_high_momentum'] * 3 +
                (df.get('rsi', 50) < 70).astype(int) * 1 +
                df['is_optimal_hour'] * 1 +
                (df.get('volume_ratio', 1) > 1.0).astype(int) * 1
            )
            
            df['success_probability'] = np.clip(
                df['success_score'] / 6.0 * self.optimal_params['confidence_threshold'], 
                0.0, 1.0
            )
            
            df = df.dropna()
            self.log(f"Loaded {len(df)} samples from {os.path.basename(file_path)}")
            return df
            
        except Exception as e:
            self.log(f"Error loading {file_path}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, df):
        """Add technical indicators."""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            if len(df) > 26:
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
            else:
                df['macd'] = 0
            
            # Volume ratio
            if 'volume' in df.columns:
                df['volume_ratio'] = df['volume'] / df['volume'].rolling(min(20, len(df)//4)).mean()
            else:
                df['volume_ratio'] = 1.0
            
            # Volatility
            df['volatility'] = df['close'].rolling(min(20, len(df)//4)).std() / df['close'].rolling(min(20, len(df)//4)).mean()
            
            return df
            
        except Exception as e:
            self.log(f"Error adding technical indicators: {str(e)}")
            return df
    
    def train_and_backtest(self, df, model_name="default"):
        """Enhanced training and backtesting with multiple metrics."""
        try:
            # Enhanced feature set
            base_features = ['momentum_1', 'hour', 'is_optimal_hour', 'success_probability']
            
            # Add available technical features
            optional_features = ['momentum_4', 'rsi', 'macd', 'volume_ratio', 'volatility', 
                               'day_of_week', 'is_high_momentum']
            
            features = base_features.copy()
            for feat in optional_features:
                if feat in df.columns and not df[feat].isna().all():
                    features.append(feat)
            
            # Ensure we have volume column
            if 'volume' not in df.columns:
                df['volume'] = np.random.randint(1000000, 100000000, len(df))
            
            if 'volume' not in features:
                features.append('volume')
            
            target = 'target'
            
            X = df[features].fillna(0)
            y = df[target]
            
            self.log(f"Training {model_name} with {len(features)} features: {features}")
            
            # Time series split to avoid lookahead bias
            tscv = TimeSeriesSplit(n_splits=3)
            
            accuracies = []
            models = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Enhanced Random Forest with optimal parameters
                model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                accuracies.append(acc)
                models.append(model)
            
            # Use best model
            best_idx = np.argmax(accuracies)
            best_model = models[best_idx]
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            # Feature importance analysis
            feature_importance = dict(zip(features, best_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Final evaluation on last 20% of data
            split_idx = int(len(df) * 0.8)
            X_final_train = X.iloc[:split_idx]
            X_final_test = X.iloc[split_idx:]
            y_final_train = y.iloc[:split_idx]
            y_final_test = y.iloc[split_idx:]
            
            final_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
            
            final_model.fit(X_final_train, y_final_train)
            final_preds = final_model.predict(X_final_test)
            final_accuracy = accuracy_score(y_final_test, final_preds)
            
            # Detailed metrics
            report = classification_report(y_final_test, final_preds, output_dict=True)
            
            results = {
                'model_name': model_name,
                'cv_accuracy_mean': avg_accuracy,
                'cv_accuracy_std': std_accuracy,
                'final_accuracy': final_accuracy,
                'precision': report['1']['precision'] if '1' in report else 0,
                'recall': report['1']['recall'] if '1' in report else 0,
                'f1_score': report['1']['f1-score'] if '1' in report else 0,
                'top_features': top_features,
                'sample_count': len(df),
                'feature_count': len(features)
            }
            
            self.log(f"Backtest {model_name}: Accuracy {final_accuracy:.4f}, CV {avg_accuracy:.4f}±{std_accuracy:.4f}")
            self.log(f"Top features: {[f[0] for f in top_features[:3]]}")
            
            return final_model, results
            
        except Exception as e:
            self.log(f"Error in train_and_backtest for {model_name}: {str(e)}")
            return None, {'model_name': model_name, 'error': str(e)}
    
    def run_comprehensive_backtest(self, data_dir="./data/4h_training/"):
        """Run comprehensive backtesting across all datasets."""
        self.log("Starting comprehensive backtesting pipeline")
        
        if not os.path.exists(data_dir):
            # Try alternative paths
            alt_paths = [
                "/home/richardw/crypto_rf_trading_system/data/4h_training/",
                "crypto_rf_trader/data/4h_training/",
                "data/4h_training/"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    data_dir = alt_path
                    break
            else:
                self.log(f"Data directory not found: {data_dir}")
                return []
        
        results = []
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            self.log(f"No CSV files found in {data_dir}")
            return []
        
        self.log(f"Found {len(csv_files)} datasets to backtest")
        
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            df = self.load_data(file_path)
            
            if df is not None and len(df) > 100:  # Minimum data requirement
                model, result = self.train_and_backtest(df, model_name=file)
                
                if model is not None:
                    result['dataset'] = file
                    results.append(result)
                    
                    # Save individual model
                    model_path = f"crypto_rf_trader/models/backtest_{file.replace('.csv', '')}.joblib"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    joblib.dump(model, model_path)
            else:
                self.log(f"Skipping {file}: insufficient data ({len(df) if df is not None else 0} samples)")
        
        self.results = results
        return results
    
    def save_backtest_summary(self, results, output_path="./results/backtest_summary.csv"):
        """Save comprehensive backtest summary."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create summary DataFrame
            summary_data = []
            for result in results:
                if 'error' not in result:
                    summary_data.append({
                        'Dataset': result['dataset'],
                        'Model_Name': result['model_name'],
                        'Final_Accuracy': result['final_accuracy'],
                        'CV_Accuracy_Mean': result['cv_accuracy_mean'],
                        'CV_Accuracy_Std': result['cv_accuracy_std'],
                        'Precision': result['precision'],
                        'Recall': result['recall'],
                        'F1_Score': result['f1_score'],
                        'Sample_Count': result['sample_count'],
                        'Feature_Count': result['feature_count'],
                        'Top_Feature_1': result['top_features'][0][0] if result['top_features'] else '',
                        'Top_Feature_2': result['top_features'][1][0] if len(result['top_features']) > 1 else '',
                        'Top_Feature_3': result['top_features'][2][0] if len(result['top_features']) > 2 else ''
                    })
            
            if summary_data:
                results_df = pd.DataFrame(summary_data)
                results_df = results_df.sort_values('Final_Accuracy', ascending=False)
                results_df.to_csv(output_path, index=False)
                
                # Also save detailed JSON results
                json_path = output_path.replace('.csv', '_detailed.json')
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Summary statistics
                avg_accuracy = results_df['Final_Accuracy'].mean()
                best_accuracy = results_df['Final_Accuracy'].max()
                worst_accuracy = results_df['Final_Accuracy'].min()
                
                self.log(f"Backtest summary saved to {output_path}")
                self.log(f"Results: Avg {avg_accuracy:.4f}, Best {best_accuracy:.4f}, Worst {worst_accuracy:.4f}")
                
                return output_path
            else:
                self.log("No valid results to save")
                return None
                
        except Exception as e:
            self.log(f"Error saving backtest summary: {str(e)}")
            return None
    
    def analyze_performance_vs_agents(self):
        """Analyze backtest performance vs current agent performance."""
        try:
            # Check current agent performance
            session_files = [f for f in os.listdir('.') if f.startswith('agent03_session_') and f.endswith('.json')]
            
            if session_files:
                latest_session = max(session_files, key=lambda f: os.path.getmtime(f))
                
                with open(latest_session, 'r') as f:
                    session_data = json.load(f)
                
                # Compare with backtest results
                if self.results:
                    backtest_avg_acc = np.mean([r['final_accuracy'] for r in self.results if 'final_accuracy' in r])
                    
                    current_return = (session_data.get('portfolio_value', 100000) - 100000) / 100000
                    
                    self.log(f"Performance comparison:")
                    self.log(f"Backtest average accuracy: {backtest_avg_acc:.4f}")
                    self.log(f"Current live return: {current_return:.4f}")
                    
                    return {
                        'backtest_accuracy': backtest_avg_acc,
                        'live_return': current_return,
                        'backtest_vs_live': backtest_avg_acc - abs(current_return)
                    }
            
            return None
            
        except Exception as e:
            self.log(f"Error analyzing performance vs agents: {str(e)}")
            return None

def main():
    """Main backtest pipeline execution."""
    pipeline = BacktestPipeline()
    
    # Run comprehensive backtesting
    results = pipeline.run_comprehensive_backtest()
    
    if results:
        # Save results
        summary_path = pipeline.save_backtest_summary(results)
        
        # Analyze vs current agent performance
        comparison = pipeline.analyze_performance_vs_agents()
        
        pipeline.log(f"Backtest pipeline complete. {len(results)} models evaluated.")
        if summary_path:
            pipeline.log(f"Results saved to: {summary_path}")
        
        if comparison:
            pipeline.log(f"Performance comparison complete")
    else:
        pipeline.log("No backtesting results generated")

if __name__ == "__main__":
    main()