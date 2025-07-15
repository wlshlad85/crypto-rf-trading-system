#!/usr/bin/env python3
"""
Phase 1C: Walk-Forward Testing Engine
ULTRATHINK Implementation - Robust Out-of-Sample Validation

Implements institutional-grade walk-forward analysis:
- 6-month rolling test windows 
- 36-month minimum training periods
- Statistical significance testing
- Performance attribution analysis
- Regime-aware validation

Based on ULTRATHINK research for preventing overfitting in crypto trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import warnings
from pathlib import Path
import json
import os

warnings.filterwarnings('ignore')

class WalkForwardEngine:
    """
    Professional walk-forward testing engine for cryptocurrency trading strategies.
    
    Implements rolling window validation with proper statistical testing
    and performance attribution analysis.
    """
    
    def __init__(self,
                 training_window_months: int = 36,
                 test_window_months: int = 6,
                 step_size_months: int = 1,
                 min_trades_threshold: int = 50,
                 results_dir: str = "phase1/walkforward_results"):
        """
        Initialize walk-forward testing engine.
        
        Args:
            training_window_months: Minimum training window size
            test_window_months: Test window size
            step_size_months: Step size for rolling windows
            min_trades_threshold: Minimum trades required for valid test
            results_dir: Directory for results storage
        """
        self.training_window = timedelta(days=training_window_months * 30)
        self.test_window = timedelta(days=test_window_months * 30)
        self.step_size = timedelta(days=step_size_months * 30)
        self.min_trades_threshold = min_trades_threshold
        self.results_dir = Path(results_dir)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.walk_forward_results = []
        self.performance_summary = {}
        
        print("🚀 Walk-Forward Testing Engine Initialized")
        print(f"📊 Training Window: {training_window_months} months")
        print(f"🔍 Test Window: {test_window_months} months")
        print(f"📈 Step Size: {step_size_months} month(s)")
        print(f"📁 Results Directory: {results_dir}")
    
    def run_walk_forward_analysis(self, 
                                 data: pd.DataFrame,
                                 model_factory: callable,
                                 feature_columns: List[str],
                                 target_column: str = 'target',
                                 strategy_name: str = "unnamed_strategy") -> Dict[str, Any]:
        """
        Execute comprehensive walk-forward analysis.
        
        Args:
            data: DataFrame with datetime index containing features and targets
            model_factory: Function that returns a new model instance
            feature_columns: List of feature column names
            target_column: Target column name
            strategy_name: Name for this strategy
            
        Returns:
            Comprehensive analysis results
        """
        print(f"\n🔬 Starting Walk-Forward Analysis: {strategy_name}")
        print("=" * 60)
        
        # Validate inputs
        self._validate_inputs(data, feature_columns, target_column)
        
        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(data)
        print(f"📊 Generated {len(windows)} walk-forward windows")
        
        if len(windows) == 0:
            raise ValueError("No valid walk-forward windows generated")
        
        # Execute walk-forward tests
        window_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n📈 Window {i+1}/{len(windows)}")
            print(f"   📚 Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"   🔍 Test:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            try:
                window_result = self._execute_window_test(
                    data, model_factory, feature_columns, target_column,
                    train_start, train_end, test_start, test_end, i+1
                )
                
                if window_result:
                    window_results.append(window_result)
                    print(f"   ✅ Window {i+1} completed: Accuracy {window_result['test_accuracy']:.4f}")
                else:
                    print(f"   ❌ Window {i+1} failed")
                    
            except Exception as e:
                print(f"   ❌ Window {i+1} error: {e}")
                continue
        
        # Aggregate results
        if not window_results:
            raise ValueError("No successful walk-forward windows")
        
        analysis_results = self._aggregate_walk_forward_results(
            window_results, strategy_name
        )
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(window_results)
        analysis_results['statistical_analysis'] = statistical_analysis
        
        # Performance attribution
        attribution_analysis = self._perform_attribution_analysis(window_results)
        analysis_results['attribution_analysis'] = attribution_analysis
        
        # Save results
        self._save_walk_forward_results(analysis_results, strategy_name)
        
        # Print comprehensive summary
        self._print_walk_forward_summary(analysis_results)
        
        return analysis_results
    
    def _validate_inputs(self, data: pd.DataFrame, feature_columns: List[str], 
                        target_column: str):
        """Validate input data and parameters."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if target_column not in data.columns:
            raise ValueError(f"Missing target column: {target_column}")
        
        # Check for sufficient data
        data_span = data.index.max() - data.index.min()
        required_span = self.training_window + self.test_window
        
        if data_span < required_span:
            raise ValueError(f"Insufficient data: {data_span} < {required_span}")
    
    def _generate_walk_forward_windows(self, data: pd.DataFrame) -> List[Tuple]:
        """Generate walk-forward testing windows."""
        windows = []
        
        data_start = data.index.min()
        data_end = data.index.max()
        
        # First window starts with minimum training period
        current_start = data_start
        
        while True:
            # Training period
            train_start = current_start
            train_end = train_start + self.training_window
            
            # Test period (immediately after training)
            test_start = train_end
            test_end = test_start + self.test_window
            
            # Check if we have enough data
            if test_end > data_end:
                break
            
            # Ensure we have sufficient data in both periods
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) > 100 and len(test_data) > 20:  # Minimum data requirements
                windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            current_start += self.step_size
        
        return windows
    
    def _execute_window_test(self, 
                           data: pd.DataFrame,
                           model_factory: callable,
                           feature_columns: List[str],
                           target_column: str,
                           train_start: pd.Timestamp,
                           train_end: pd.Timestamp,
                           test_start: pd.Timestamp,
                           test_end: pd.Timestamp,
                           window_id: int) -> Optional[Dict]:
        """Execute single walk-forward window test."""
        
        # Extract training data
        train_mask = (data.index >= train_start) & (data.index < train_end)
        train_data = data[train_mask]
        
        # Extract test data
        test_mask = (data.index >= test_start) & (data.index < test_end)
        test_data = data[test_mask]
        
        if len(train_data) == 0 or len(test_data) == 0:
            return None
        
        # Prepare features and targets
        X_train = train_data[feature_columns].dropna()
        y_train = train_data.loc[X_train.index, target_column]
        
        X_test = test_data[feature_columns].dropna()
        y_test = test_data.loc[X_test.index, target_column]
        
        if len(X_train) == 0 or len(X_test) == 0:
            return None
        
        # Create and train model
        model = model_factory()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        train_proba = None
        test_proba = None
        if hasattr(model, 'predict_proba'):
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        window_result = {
            'window_id': window_id,
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            
            # Training metrics
            'train_accuracy': self._calculate_accuracy(y_train, y_train_pred),
            'train_precision': self._calculate_precision(y_train, y_train_pred),
            'train_recall': self._calculate_recall(y_train, y_train_pred),
            'train_f1': self._calculate_f1(y_train, y_train_pred),
            
            # Test metrics (out-of-sample)
            'test_accuracy': self._calculate_accuracy(y_test, y_test_pred),
            'test_precision': self._calculate_precision(y_test, y_test_pred),
            'test_recall': self._calculate_recall(y_test, y_test_pred),
            'test_f1': self._calculate_f1(y_test, y_test_pred),
            
            # Trading simulation metrics
            'simulated_trades': self._simulate_trading_performance(
                test_data, y_test_pred, test_proba
            ),
            
            # Feature importance
            'feature_importance': self._extract_feature_importance(model, feature_columns),
            
            # Market conditions during test period
            'market_conditions': self._analyze_market_conditions(test_data)
        }
        
        # Add probability-based metrics if available
        if test_proba is not None and len(np.unique(y_test)) == 2:
            window_result.update({
                'test_auc': self._calculate_auc(y_test, test_proba[:, 1]),
                'test_log_loss': self._calculate_log_loss(y_test, test_proba)
            })
        
        return window_result
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    
    def _calculate_precision(self, y_true, y_pred):
        """Calculate precision score."""
        from sklearn.metrics import precision_score
        try:
            avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            return precision_score(y_true, y_pred, average=avg, zero_division=0)
        except:
            return 0.0
    
    def _calculate_recall(self, y_true, y_pred):
        """Calculate recall score."""
        from sklearn.metrics import recall_score
        try:
            avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            return recall_score(y_true, y_pred, average=avg, zero_division=0)
        except:
            return 0.0
    
    def _calculate_f1(self, y_true, y_pred):
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        try:
            avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            return f1_score(y_true, y_pred, average=avg, zero_division=0)
        except:
            return 0.0
    
    def _calculate_auc(self, y_true, y_scores):
        """Calculate AUC score."""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_scores)
        except:
            return 0.5
    
    def _calculate_log_loss(self, y_true, y_proba):
        """Calculate log loss."""
        from sklearn.metrics import log_loss
        try:
            return log_loss(y_true, y_proba)
        except:
            return 1.0
    
    def _simulate_trading_performance(self, test_data: pd.DataFrame, 
                                    predictions: np.ndarray,
                                    probabilities: Optional[np.ndarray] = None) -> Dict:
        """Simulate trading performance on test period."""
        
        # Simple trading simulation
        if 'Close' not in test_data.columns:
            return {'error': 'No price data for trading simulation'}
        
        prices = test_data['Close'].values
        signals = predictions
        
        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        
        # Align signals with returns (shift for next-period returns)
        if len(signals) > len(price_returns):
            signals = signals[:-1]
        elif len(signals) < len(price_returns):
            price_returns = price_returns[:len(signals)]
        
        # Strategy returns
        strategy_returns = signals * price_returns
        
        # Performance metrics
        total_return = np.prod(1 + strategy_returns) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'number_of_trades': len(strategy_returns)
        }
    
    def _extract_feature_importance(self, model, feature_columns: List[str]) -> Dict:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_columns, importances.tolist()))
        else:
            return {}
    
    def _analyze_market_conditions(self, test_data: pd.DataFrame) -> Dict:
        """Analyze market conditions during test period."""
        if 'Close' not in test_data.columns:
            return {}
        
        prices = test_data['Close']
        returns = prices.pct_change().dropna()
        
        conditions = {
            'period_return': (prices.iloc[-1] / prices.iloc[0]) - 1,
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'max_drawdown_period': self._calculate_period_drawdown(prices),
            'trending_strength': self._calculate_trend_strength(prices),
            'average_volume': test_data.get('Volume', pd.Series([0])).mean()
        }
        
        return conditions
    
    def _calculate_period_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for the period."""
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        return drawdown.min()
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength (R² of linear regression)."""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        y = prices.values
        
        # Linear regression
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        
        return r_squared
    
    def _aggregate_walk_forward_results(self, window_results: List[Dict], 
                                      strategy_name: str) -> Dict:
        """Aggregate results across all walk-forward windows."""
        
        # Extract metrics
        test_accuracies = [w['test_accuracy'] for w in window_results]
        test_sharpe_ratios = [w['simulated_trades'].get('sharpe_ratio', 0) for w in window_results]
        test_returns = [w['simulated_trades'].get('total_return', 0) for w in window_results]
        
        # Aggregate statistics
        aggregated = {
            'strategy_name': strategy_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_windows': len(window_results),
            'successful_windows': len(window_results),
            
            # Performance aggregation
            'mean_test_accuracy': np.mean(test_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'min_test_accuracy': np.min(test_accuracies),
            'max_test_accuracy': np.max(test_accuracies),
            
            'mean_sharpe_ratio': np.mean(test_sharpe_ratios),
            'std_sharpe_ratio': np.std(test_sharpe_ratios),
            
            'mean_total_return': np.mean(test_returns),
            'std_total_return': np.std(test_returns),
            
            # Consistency metrics
            'performance_consistency': 1 - (np.std(test_accuracies) / np.mean(test_accuracies)) if np.mean(test_accuracies) > 0 else 0,
            'positive_periods': sum(1 for r in test_returns if r > 0) / len(test_returns),
            
            # Detailed results
            'window_results': window_results
        }
        
        return aggregated
    
    def _perform_statistical_analysis(self, window_results: List[Dict]) -> Dict:
        """Perform statistical significance testing."""
        
        test_accuracies = [w['test_accuracy'] for w in window_results]
        test_returns = [w['simulated_trades'].get('total_return', 0) for w in window_results]
        
        # T-test against random chance
        from scipy import stats
        
        # Test accuracy against 0.5 (random chance)
        t_stat_acc, p_val_acc = stats.ttest_1samp(test_accuracies, 0.5)
        
        # Test returns against 0 (no profit)
        t_stat_ret, p_val_ret = stats.ttest_1samp(test_returns, 0)
        
        # Consistency test (low variance is good)
        accuracy_cv = np.std(test_accuracies) / np.mean(test_accuracies) if np.mean(test_accuracies) > 0 else float('inf')
        
        # Deflated Sharpe ratio calculation (simplified)
        sharpe_ratios = [w['simulated_trades'].get('sharpe_ratio', 0) for w in window_results]
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        # Deflation factor (simplified - full calculation requires more complex statistics)
        n_trials = len(window_results)
        deflation_factor = np.sqrt(1 + (n_trials - 1) * 0.1)  # Simplified approximation
        deflated_sharpe = mean_sharpe / deflation_factor if deflation_factor > 0 else 0
        
        return {
            'accuracy_t_test': {
                't_statistic': t_stat_acc,
                'p_value': p_val_acc,
                'significant': p_val_acc < 0.05
            },
            'returns_t_test': {
                't_statistic': t_stat_ret,
                'p_value': p_val_ret,
                'significant': p_val_ret < 0.05
            },
            'consistency_metrics': {
                'accuracy_coefficient_of_variation': accuracy_cv,
                'performance_stability': accuracy_cv < 0.2  # Good if CV < 20%
            },
            'deflated_sharpe_ratio': deflated_sharpe,
            'sample_size': len(window_results)
        }
    
    def _perform_attribution_analysis(self, window_results: List[Dict]) -> Dict:
        """Perform performance attribution analysis."""
        
        # Market condition analysis
        market_conditions = []
        performance_by_condition = {}
        
        for window in window_results:
            conditions = window.get('market_conditions', {})
            if 'period_return' in conditions:
                market_return = conditions['period_return']
                test_accuracy = window['test_accuracy']
                
                # Categorize market conditions
                if market_return > 0.1:
                    condition = 'bull_market'
                elif market_return < -0.1:
                    condition = 'bear_market'
                else:
                    condition = 'sideways_market'
                
                if condition not in performance_by_condition:
                    performance_by_condition[condition] = []
                performance_by_condition[condition].append(test_accuracy)
        
        # Aggregate by condition
        condition_analysis = {}
        for condition, accuracies in performance_by_condition.items():
            condition_analysis[condition] = {
                'count': len(accuracies),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        # Feature importance analysis
        all_feature_importances = {}
        for window in window_results:
            importances = window.get('feature_importance', {})
            for feature, importance in importances.items():
                if feature not in all_feature_importances:
                    all_feature_importances[feature] = []
                all_feature_importances[feature].append(importance)
        
        # Average feature importance
        avg_feature_importance = {}
        for feature, importances in all_feature_importances.items():
            avg_feature_importance[feature] = {
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances)
            }
        
        return {
            'market_condition_analysis': condition_analysis,
            'feature_importance_analysis': avg_feature_importance,
            'temporal_analysis': self._analyze_temporal_performance(window_results)
        }
    
    def _analyze_temporal_performance(self, window_results: List[Dict]) -> Dict:
        """Analyze performance over time."""
        
        # Sort by test period start time
        sorted_windows = sorted(window_results, 
                              key=lambda w: w['test_period'][0])
        
        accuracies = [w['test_accuracy'] for w in sorted_windows]
        
        # Performance trend
        x = np.arange(len(accuracies))
        if len(x) > 1:
            trend_slope = np.polyfit(x, accuracies, 1)[0]
        else:
            trend_slope = 0
        
        # Performance degradation
        if len(accuracies) >= 4:
            first_quarter = accuracies[:len(accuracies)//4]
            last_quarter = accuracies[-len(accuracies)//4:]
            degradation = np.mean(first_quarter) - np.mean(last_quarter)
        else:
            degradation = 0
        
        return {
            'performance_trend_slope': trend_slope,
            'performance_degradation': degradation,
            'temporal_variance': np.var(accuracies)
        }
    
    def _save_walk_forward_results(self, results: Dict, strategy_name: str):
        """Save walk-forward results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"walkforward_results_{strategy_name}_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"💾 Walk-forward results saved: {filename}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def _print_walk_forward_summary(self, results: Dict):
        """Print comprehensive walk-forward analysis summary."""
        print("\n" + "=" * 70)
        print("🎯 WALK-FORWARD ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"📊 Strategy: {results.get('strategy_name', 'Unknown')}")
        print(f"🔍 Windows Analyzed: {results.get('successful_windows', 0)}")
        print(f"📈 Mean Test Accuracy: {results.get('mean_test_accuracy', 0):.4f} ± {results.get('std_test_accuracy', 0):.4f}")
        print(f"💰 Mean Return: {results.get('mean_total_return', 0):.4f} ± {results.get('std_total_return', 0):.4f}")
        print(f"📊 Mean Sharpe Ratio: {results.get('mean_sharpe_ratio', 0):.4f} ± {results.get('std_sharpe_ratio', 0):.4f}")
        print(f"🎯 Performance Consistency: {results.get('performance_consistency', 0):.4f}")
        print(f"📈 Positive Periods: {results.get('positive_periods', 0):.1%}")
        
        # Statistical significance
        stats = results.get('statistical_analysis', {})
        if stats:
            acc_test = stats.get('accuracy_t_test', {})
            ret_test = stats.get('returns_t_test', {})
            
            print(f"\n📊 STATISTICAL SIGNIFICANCE:")
            print(f"   Accuracy vs Random: {'✅ SIGNIFICANT' if acc_test.get('significant') else '❌ NOT SIGNIFICANT'} (p={acc_test.get('p_value', 1):.4f})")
            print(f"   Returns vs Zero: {'✅ SIGNIFICANT' if ret_test.get('significant') else '❌ NOT SIGNIFICANT'} (p={ret_test.get('p_value', 1):.4f})")
            print(f"   Deflated Sharpe: {stats.get('deflated_sharpe_ratio', 0):.4f}")
        
        # Attribution analysis
        attribution = results.get('attribution_analysis', {})
        if attribution:
            market_conditions = attribution.get('market_condition_analysis', {})
            if market_conditions:
                print(f"\n📊 MARKET CONDITION PERFORMANCE:")
                for condition, metrics in market_conditions.items():
                    print(f"   {condition.replace('_', ' ').title()}: {metrics.get('mean_accuracy', 0):.4f} ({metrics.get('count', 0)} windows)")
        
        print("=" * 70)

def main():
    """Demonstrate walk-forward testing engine."""
    print("🏛️ PHASE 1C: Walk-Forward Testing Engine")
    print("ULTRATHINK Implementation - Robust Validation")
    print("=" * 60)
    
    # Load sample data
    data_dir = "phase1/data/processed"
    import glob
    
    data_files = glob.glob(f"{data_dir}/BTC-USD_*.csv")
    if not data_files:
        print("❌ No data files found. Run Phase 1A first.")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"📂 Loading data from: {latest_file}")
    
    # Load and prepare data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    # Feature engineering
    df['returns'] = df['Close'].pct_change()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['Close'])
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['target'] = (df['returns'].shift(-1) > 0.005).astype(int)  # 0.5% threshold
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 1000:
        print("❌ Insufficient data for walk-forward analysis")
        return
    
    # Define features and model factory
    feature_columns = ['returns', 'sma_10', 'sma_50', 'rsi', 'volume_ma']
    
    def model_factory():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    
    # Initialize walk-forward engine
    engine = WalkForwardEngine(
        training_window_months=12,  # Reduced for demo
        test_window_months=3,       # Reduced for demo
        step_size_months=1
    )
    
    # Run analysis
    results = engine.run_walk_forward_analysis(
        data=df,
        model_factory=model_factory,
        feature_columns=feature_columns,
        target_column='target',
        strategy_name='RandomForest_WalkForward_Demo'
    )
    
    # Success criteria
    mean_accuracy = results.get('mean_test_accuracy', 0)
    consistency = results.get('performance_consistency', 0)
    statistical_sig = results.get('statistical_analysis', {}).get('accuracy_t_test', {}).get('significant', False)
    
    print("\n🏁 PHASE 1C SUCCESS CRITERIA:")
    print(f"✅ Walk-Forward Implementation: Functional")
    print(f"📊 Mean Accuracy: {mean_accuracy:.4f} ({'✅ GOOD' if mean_accuracy > 0.55 else '❌ NEEDS IMPROVEMENT'})")
    print(f"🎯 Consistency: {consistency:.4f} ({'✅ GOOD' if consistency > 0.8 else '❌ NEEDS IMPROVEMENT'})")
    print(f"📈 Statistical Significance: {'✅ PASSED' if statistical_sig else '❌ FAILED'}")
    
    if mean_accuracy > 0.55 and consistency > 0.8:
        print("\n🚀 Phase 1 COMPLETE! Ready for Phase 2 Implementation")
    else:
        print("\n⚠️ Model requires optimization before Phase 2")

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    main()