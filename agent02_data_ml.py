#!/usr/bin/env python3
"""
Agent02: Data Loader, Feature Engineer & Random Forest Retrainer

Handles data loading, feature engineering, Random Forest retraining,
and genetic algorithm meta-optimizer (uses DEAP if available, else heuristics).
"""

import time
import os
import json
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

COMS_PATH = "coms.md"

# Try to import DEAP for genetic algorithm, fallback to heuristics
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

class Agent02DataML:
    """Data processing, feature engineering, and ML model management."""
    
    def __init__(self):
        """Initialize Agent02."""
        self.running = False
        
        # Data and model storage
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Genetic algorithm setup
        self.ga_population_size = 50
        self.ga_generations = 20
        self.ga_crossover_prob = 0.7
        self.ga_mutation_prob = 0.2
        
        # Current parameters
        self.current_params = {
            'momentum_threshold': 1.780,
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_split': 8,
            'min_samples_leaf': 4,
            'confidence_threshold': 0.65
        }
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.log("Agent02 starting data processing and ML operations")
    
    def log(self, message: str):
        """Log message to communications file."""
        timestamp = datetime.utcnow().isoformat()
        try:
            with open(COMS_PATH, "a") as f:
                f.write(f"[agent02][{timestamp}] {message}\n")
            print(f"[agent02][{timestamp}] {message}")
        except Exception as e:
            print(f"[agent02] Logging error: {e}")
    
    def load_training_data(self) -> bool:
        """Load and prepare training data."""
        try:
            self.log("Loading training data for model retraining")
            
            # Try to load existing 4h dataset
            data_file = "data/4h_training/crypto_4h_dataset_20250714_130201.csv"
            if os.path.exists(data_file):
                self.data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                self.log(f"Loaded {len(self.data)} records from {data_file}")
            else:
                # Create synthetic data for testing
                self.log("Training data not found, creating synthetic dataset")
                self.data = self._create_synthetic_data()
            
            # Prepare enhanced features
            self.data = self._prepare_enhanced_features(self.data)
            
            # Clean data
            self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.log(f"Data preparation complete: {len(self.data)} samples, {len(self.data.columns)} features")
            return True
            
        except Exception as e:
            self.log(f"Error loading training data: {e}")
            return False
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic crypto data for testing."""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate time series
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='4H')
        
        # Realistic crypto price simulation
        price_base = 50000
        returns = np.random.randn(n_samples) * 0.02  # 2% volatility
        price_trend = np.cumsum(returns)
        
        data = pd.DataFrame({
            'open': price_base * (1 + price_trend),
            'high': price_base * (1 + price_trend + np.abs(np.random.randn(n_samples) * 0.01)),
            'low': price_base * (1 + price_trend - np.abs(np.random.randn(n_samples) * 0.01)),
            'close': price_base * (1 + price_trend),
            'volume': np.random.randint(10000, 1000000, n_samples),
            'symbol': 'BTC'
        }, index=dates)
        
        # Fix OHLC relationships
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return data
    
    def _prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features with pattern recognition."""
        
        # Basic momentum features
        data['momentum_1'] = data['close'].pct_change(1) * 100
        data['momentum_4'] = data['close'].pct_change(4) * 100
        data['momentum_12'] = data['close'].pct_change(12) * 100
        data['momentum_strength'] = data['momentum_1'] / 4  # Hourly rate
        
        # Enhanced momentum patterns
        data['is_high_momentum'] = (data['momentum_strength'] > self.current_params['momentum_threshold']).astype(int)
        data['momentum_acceleration'] = data['momentum_strength'].diff()
        data['momentum_consistency'] = data['momentum_strength'].rolling(6).std()
        
        # Technical indicators
        data = self._add_technical_indicators(data)
        
        # Market structure features
        data = self._add_market_structure_features(data)
        
        # Time-based features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_optimal_hour'] = (data['hour'] == 3).astype(int)
        
        # Success pattern scoring
        data['success_score'] = (
            data['is_high_momentum'] * 3 +
            (data['momentum_acceleration'] > 0).astype(int) * 2 +
            data['is_optimal_hour'] * 1 +
            (data.get('volume_ratio', 1) > 1.2).astype(int) * 1 +
            (data.get('rsi', 50) < 70).astype(int) * 1
        )
        
        data['success_probability'] = np.clip(data['success_score'] / 8.0 * 0.636, 0.0, 1.0)
        
        # Target variables for training
        for horizon in [1, 2, 4, 8]:
            data[f'price_up_{horizon}h'] = (data['close'].shift(-horizon) > data['close']).astype(int)
            data[f'price_change_{horizon}h'] = (data['close'].shift(-horizon) / data['close'] - 1) * 100
            data[f'profitable_{horizon}h'] = (data[f'price_change_{horizon}h'] > 0.68).astype(int)
        
        # Position sizing targets
        data['optimal_position_size'] = np.where(
            data['is_high_momentum'] == 1,
            0.800,  # High momentum: larger position
            np.where(data['momentum_strength'] > 0.5, 0.620, 0.464)  # Medium/low momentum
        )
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        bb_std = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Volume indicators
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['volume_trend'] = data['volume'].pct_change(5) * 100
        data['price_volume_trend'] = data['close'].pct_change() * data['volume_ratio']
        
        # Volatility and ATR
        data['volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        data['atr'] = true_range.rolling(14).mean()
        data['atr_ratio'] = data['atr'] / data['close']
        
        return data
    
    def _add_market_structure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and trend features."""
        
        # Higher highs and higher lows
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['higher_low'] = (data['low'] > data['low'].shift(1)).astype(int)
        data['market_structure'] = data['higher_high'] + data['higher_low'] - 1
        
        # Trend strength
        sma_10 = data['close'].rolling(10).mean()
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        data['trend_strength'] = np.abs(sma_10 - sma_20) / data['close']
        data['price_vs_sma10'] = (data['close'] - sma_10) / sma_10
        data['price_vs_sma20'] = (data['close'] - sma_20) / sma_20
        data['price_vs_sma50'] = (data['close'] - sma_50) / sma_50
        
        # Trend alignment
        data['trend_alignment'] = (
            (data['close'] > sma_10).astype(int) +
            (sma_10 > sma_20).astype(int) +
            (sma_20 > sma_50).astype(int)
        )
        
        # Support and resistance levels
        data['resistance_level'] = data['high'].rolling(20).max()
        data['support_level'] = data['low'].rolling(20).min()
        data['distance_to_resistance'] = (data['resistance_level'] - data['close']) / data['close'] * 100
        data['distance_to_support'] = (data['close'] - data['support_level']) / data['close'] * 100
        
        return data
    
    def train_random_forest_models(self) -> bool:
        """Train specialized Random Forest models."""
        try:
            self.log("Training Random Forest models with current parameters")
            
            if self.data is None or len(self.data) < 1000:
                self.log("Insufficient data for training")
                return False
            
            # Split data
            train_size = int(len(self.data) * 0.8)
            train_data = self.data.iloc[:train_size]
            val_data = self.data.iloc[train_size:]
            
            # Train individual models
            models_trained = 0
            
            # Entry model
            if self._train_entry_model(train_data, val_data):
                models_trained += 1
            
            # Position sizing model
            if self._train_position_model(train_data, val_data):
                models_trained += 1
            
            # Exit model
            if self._train_exit_model(train_data, val_data):
                models_trained += 1
            
            # Profit prediction model
            if self._train_profit_model(train_data, val_data):
                models_trained += 1
            
            self.log(f"Training complete: {models_trained}/4 models trained successfully")
            
            # Save models
            self._save_models()
            
            return models_trained >= 3  # Need at least 3 models for operation
            
        except Exception as e:
            self.log(f"Error training Random Forest models: {e}")
            return False
    
    def _train_entry_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> bool:
        """Train entry signal model."""
        try:
            entry_features = [
                'momentum_1', 'momentum_4', 'momentum_strength', 'momentum_acceleration',
                'rsi', 'macd', 'macd_histogram', 'bb_position', 'bb_width',
                'volume_ratio', 'volume_trend', 'market_structure', 'trend_alignment',
                'is_optimal_hour', 'hour', 'day_of_week', 'volatility', 'atr_ratio',
                'trend_strength', 'success_probability', 'distance_to_support'
            ]
            
            # Prepare features and targets
            X_train = train_data[entry_features].dropna()
            y_train = train_data.loc[X_train.index, 'price_up_2h']
            
            X_val = val_data[entry_features].dropna()
            y_val = val_data.loc[X_val.index, 'price_up_2h']
            
            if len(X_train) < 100:
                return False
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=self.current_params['n_estimators'],
                max_depth=self.current_params['max_depth'],
                min_samples_split=self.current_params['min_samples_split'],
                min_samples_leaf=self.current_params['min_samples_leaf'],
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Validate
            train_accuracy = model.score(X_train_scaled, y_train)
            val_accuracy = model.score(X_val_scaled, y_val) if len(X_val) > 0 else train_accuracy
            
            self.models['entry'] = model
            self.scalers['entry'] = scaler
            self.feature_importance['entry'] = dict(zip(entry_features, model.feature_importances_))
            
            self.log(f"Entry model trained - Train accuracy: {train_accuracy:.1%}, Val accuracy: {val_accuracy:.1%}")
            return True
            
        except Exception as e:
            self.log(f"Error training entry model: {e}")
            return False
    
    def _train_position_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> bool:
        """Train position sizing model."""
        try:
            position_features = [
                'momentum_strength', 'momentum_acceleration', 'momentum_consistency',
                'is_high_momentum', 'volatility', 'atr_ratio', 'rsi', 'market_structure',
                'success_probability', 'volume_ratio', 'trend_alignment', 'bb_position'
            ]
            
            X_train = train_data[position_features].dropna()
            y_train = train_data.loc[X_train.index, 'optimal_position_size']
            
            X_val = val_data[position_features].dropna()
            y_val = val_data.loc[X_val.index, 'optimal_position_size']
            
            if len(X_train) < 100:
                return False
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = RandomForestRegressor(
                n_estimators=self.current_params['n_estimators'],
                max_depth=self.current_params['max_depth'] - 2,
                min_samples_split=self.current_params['min_samples_split'],
                min_samples_leaf=self.current_params['min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_r2 = model.score(X_train_scaled, y_train)
            val_r2 = model.score(X_val_scaled, y_val) if len(X_val) > 0 else train_r2
            
            self.models['position'] = model
            self.scalers['position'] = scaler
            self.feature_importance['position'] = dict(zip(position_features, model.feature_importances_))
            
            self.log(f"Position model trained - Train R²: {train_r2:.3f}, Val R²: {val_r2:.3f}")
            return True
            
        except Exception as e:
            self.log(f"Error training position model: {e}")
            return False
    
    def _train_exit_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> bool:
        """Train exit signal model."""
        try:
            exit_features = [
                'momentum_1', 'momentum_acceleration', 'rsi', 'bb_position', 'bb_width',
                'market_structure', 'volatility', 'volume_ratio', 'price_vs_sma10',
                'macd', 'macd_histogram', 'distance_to_resistance'
            ]
            
            # Create exit signals
            exit_signals = (
                (train_data['rsi'] > 70) |
                (train_data['momentum_strength'] < -0.5) |
                (train_data['bb_position'] > 0.8) |
                (train_data['distance_to_resistance'] < 2.0)
            ).astype(int)
            
            X_train = train_data[exit_features].dropna()
            y_train = exit_signals.loc[X_train.index]
            
            X_val = val_data[exit_features].dropna()
            val_exit_signals = (
                (val_data['rsi'] > 70) |
                (val_data['momentum_strength'] < -0.5) |
                (val_data['bb_position'] > 0.8) |
                (val_data['distance_to_resistance'] < 2.0)
            ).astype(int)
            y_val = val_exit_signals.loc[X_val.index]
            
            if len(X_train) < 100:
                return False
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = RandomForestClassifier(
                n_estimators=self.current_params['n_estimators'],
                max_depth=self.current_params['max_depth'] - 3,
                min_samples_split=self.current_params['min_samples_split'] + 2,
                min_samples_leaf=self.current_params['min_samples_leaf'] + 1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_accuracy = model.score(X_train_scaled, y_train)
            val_accuracy = model.score(X_val_scaled, y_val) if len(X_val) > 0 else train_accuracy
            
            self.models['exit'] = model
            self.scalers['exit'] = scaler
            self.feature_importance['exit'] = dict(zip(exit_features, model.feature_importances_))
            
            self.log(f"Exit model trained - Train accuracy: {train_accuracy:.1%}, Val accuracy: {val_accuracy:.1%}")
            return True
            
        except Exception as e:
            self.log(f"Error training exit model: {e}")
            return False
    
    def _train_profit_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> bool:
        """Train profit prediction model."""
        try:
            profit_features = [
                'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
                'bb_position', 'market_structure', 'volume_ratio', 'volatility',
                'success_probability', 'trend_alignment', 'price_vs_sma20'
            ]
            
            X_train = train_data[profit_features].dropna()
            y_train = train_data.loc[X_train.index, 'profitable_2h']
            
            X_val = val_data[profit_features].dropna()
            y_val = val_data.loc[X_val.index, 'profitable_2h']
            
            if len(X_train) < 100:
                return False
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = RandomForestClassifier(
                n_estimators=self.current_params['n_estimators'] + 50,
                max_depth=self.current_params['max_depth'] + 3,
                min_samples_split=self.current_params['min_samples_split'] - 2,
                min_samples_leaf=self.current_params['min_samples_leaf'] - 1,
                max_features='log2',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_accuracy = model.score(X_train_scaled, y_train)
            val_accuracy = model.score(X_val_scaled, y_val) if len(X_val) > 0 else train_accuracy
            
            self.models['profit'] = model
            self.scalers['profit'] = scaler
            self.feature_importance['profit'] = dict(zip(profit_features, model.feature_importances_))
            
            self.log(f"Profit model trained - Train accuracy: {train_accuracy:.1%}, Val accuracy: {val_accuracy:.1%}")
            return True
            
        except Exception as e:
            self.log(f"Error training profit model: {e}")
            return False
    
    def _save_models(self):
        """Save trained models and scalers."""
        try:
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save each model and scaler
            for model_name, model in self.models.items():
                model_file = os.path.join(models_dir, f'{model_name}_model_{timestamp}.joblib')
                scaler_file = os.path.join(models_dir, f'{model_name}_scaler_{timestamp}.joblib')
                
                joblib.dump(model, model_file)
                if model_name in self.scalers:
                    joblib.dump(self.scalers[model_name], scaler_file)
            
            # Save feature importance
            importance_file = os.path.join(models_dir, f'feature_importance_{timestamp}.json')
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            self.log(f"Models saved to {models_dir}")
            
        except Exception as e:
            self.log(f"Error saving models: {e}")
    
    def run_genetic_algorithm_optimization(self) -> Dict[str, Any]:
        """Run genetic algorithm for parameter optimization."""
        self.log("Starting genetic algorithm parameter optimization")
        
        if DEAP_AVAILABLE:
            return self._run_deap_optimization()
        else:
            return self._run_heuristic_optimization()
    
    def _run_deap_optimization(self) -> Dict[str, Any]:
        """Run optimization using DEAP genetic algorithm."""
        try:
            # Define fitness and individual
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Parameter ranges
            toolbox.register("momentum_threshold", np.random.uniform, 1.0, 3.0)
            toolbox.register("n_estimators", np.random.randint, 50, 300)
            toolbox.register("max_depth", np.random.randint, 5, 20)
            toolbox.register("min_samples_split", np.random.randint, 2, 15)
            toolbox.register("min_samples_leaf", np.random.randint, 1, 10)
            toolbox.register("confidence_threshold", np.random.uniform, 0.5, 0.8)
            
            # Individual and population
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (toolbox.momentum_threshold, toolbox.n_estimators, toolbox.max_depth,
                            toolbox.min_samples_split, toolbox.min_samples_leaf, toolbox.confidence_threshold), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Genetic operators
            toolbox.register("evaluate", self._evaluate_individual)
            toolbox.register("mate", tools.cxBlend, alpha=0.1)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Run algorithm
            population = toolbox.population(n=self.ga_population_size)
            
            for generation in range(self.ga_generations):
                # Evaluate population
                fitnesses = list(map(toolbox.evaluate, population))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit
                
                # Select next generation
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                
                # Apply crossover and mutation
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.random() < self.ga_crossover_prob:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                for mutant in offspring:
                    if np.random.random() < self.ga_mutation_prob:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                population[:] = offspring
                
                if generation % 5 == 0:
                    best_fitness = max(ind.fitness.values[0] for ind in population)
                    self.log(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Get best individual
            best_individual = max(population, key=lambda x: x.fitness.values[0])
            best_params = {
                'momentum_threshold': best_individual[0],
                'n_estimators': int(best_individual[1]),
                'max_depth': int(best_individual[2]),
                'min_samples_split': int(best_individual[3]),
                'min_samples_leaf': int(best_individual[4]),
                'confidence_threshold': best_individual[5]
            }
            
            self.log(f"GA optimization complete. Best fitness: {best_individual.fitness.values[0]:.4f}")
            return {'success': True, 'best_params': best_params, 'method': 'DEAP'}
            
        except Exception as e:
            self.log(f"Error in DEAP optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_heuristic_optimization(self) -> Dict[str, Any]:
        """Run heuristic optimization without DEAP."""
        try:
            self.log("Running heuristic parameter optimization")
            
            best_score = -999
            best_params = self.current_params.copy()
            
            # Define parameter ranges
            param_ranges = {
                'momentum_threshold': np.linspace(1.2, 2.5, 8),
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [6, 8, 10, 12, 15, 18],
                'min_samples_split': [2, 5, 8, 10, 12],
                'min_samples_leaf': [1, 3, 5, 7],
                'confidence_threshold': np.linspace(0.55, 0.75, 6)
            }
            
            # Random search with local optimization
            for iteration in range(50):  # 50 random trials
                # Generate random parameters
                test_params = {}
                for param, values in param_ranges.items():
                    test_params[param] = np.random.choice(values)
                    if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                        test_params[param] = int(test_params[param])
                
                # Evaluate parameters
                score = self._evaluate_parameters(test_params)
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    self.log(f"Iteration {iteration}: New best score = {score:.4f}")
            
            self.log(f"Heuristic optimization complete. Best score: {best_score:.4f}")
            return {'success': True, 'best_params': best_params, 'method': 'Heuristic'}
            
        except Exception as e:
            self.log(f"Error in heuristic optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_individual(self, individual: List) -> Tuple[float]:
        """Evaluate individual for genetic algorithm."""
        params = {
            'momentum_threshold': individual[0],
            'n_estimators': int(individual[1]),
            'max_depth': int(individual[2]),
            'min_samples_split': int(individual[3]),
            'min_samples_leaf': int(individual[4]),
            'confidence_threshold': individual[5]
        }
        
        score = self._evaluate_parameters(params)
        return (score,)
    
    def _evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """Evaluate parameter set using cross-validation."""
        try:
            if self.data is None or len(self.data) < 500:
                return -999
            
            # Quick model training with parameters
            train_size = int(len(self.data) * 0.7)
            train_data = self.data.iloc[:train_size]
            test_data = self.data.iloc[train_size:]
            
            # Simple evaluation: train entry model and measure accuracy
            entry_features = [
                'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
                'volume_ratio', 'market_structure', 'bb_position', 'volatility'
            ]
            
            X_train = train_data[entry_features].dropna()
            y_train = train_data.loc[X_train.index, 'price_up_2h']
            
            X_test = test_data[entry_features].dropna()
            y_test = test_data.loc[X_test.index, 'price_up_2h']
            
            if len(X_train) < 100 or len(X_test) < 50:
                return -999
            
            # Train quick model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(
                n_estimators=min(params['n_estimators'], 100),  # Limit for speed
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                n_jobs=2
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Calculate composite score
            train_accuracy = model.score(X_train_scaled, y_train)
            test_accuracy = model.score(X_test_scaled, y_test)
            
            # Penalty for overfitting
            overfitting_penalty = max(0, train_accuracy - test_accuracy - 0.1)
            
            # Combined score (accuracy - overfitting penalty)
            score = test_accuracy - overfitting_penalty * 2
            
            return score
            
        except Exception as e:
            return -999
    
    def check_for_optimization_signals(self) -> bool:
        """Check for optimization signals from Agent01."""
        try:
            signal_file = "optimization_suggestions.json"
            if os.path.exists(signal_file):
                with open(signal_file, 'r') as f:
                    signal_data = json.load(f)
                
                # Check if this is a new signal
                timestamp = signal_data.get('timestamp', '')
                if 'agent01_signal' in signal_data:
                    self.log(f"Optimization signal received from Agent01: {timestamp}")
                    
                    # Update parameters
                    suggestions = signal_data.get('suggestions', {})
                    self._update_parameters(suggestions)
                    
                    # Remove signal file
                    os.remove(signal_file)
                    
                    # Send response
                    self._send_optimization_response({'status': 'parameters_updated'})
                    
                    return True
            
            return False
            
        except Exception as e:
            self.log(f"Error checking optimization signals: {e}")
            return False
    
    def _update_parameters(self, suggestions: Dict[str, Any]):
        """Update current parameters based on suggestions."""
        try:
            for param, value in suggestions.items():
                if param in self.current_params:
                    old_value = self.current_params[param]
                    self.current_params[param] = value
                    self.log(f"Parameter updated: {param} = {old_value} → {value}")
            
            self.log("Parameters updated successfully")
            
        except Exception as e:
            self.log(f"Error updating parameters: {e}")
    
    def _send_optimization_response(self, response: Dict[str, Any]):
        """Send optimization response to Agent01."""
        try:
            response_file = "optimization_response.json"
            with open(response_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'agent': 'agent02',
                    **response
                }, f, indent=2)
            
            self.log(f"Optimization response sent: {response.get('status', 'unknown')}")
            
        except Exception as e:
            self.log(f"Error sending optimization response: {e}")
    
    def run_main_loop(self):
        """Run main Agent02 loop."""
        self.log("Starting Agent02 main processing loop")
        self.running = True
        
        cycle_count = 0
        last_training = datetime.now() - timedelta(hours=2)  # Initial training
        
        try:
            while self.running:
                cycle_count += 1
                self.log(f"Agent02 cycle #{cycle_count}")
                
                # 1. Load/refresh data
                if not self.load_training_data():
                    self.log("Failed to load training data, retrying in 5 minutes")
                    time.sleep(300)
                    continue
                
                # 2. Check for optimization signals
                if self.check_for_optimization_signals():
                    self.log("Optimization signal processed, retraining models")
                    last_training = datetime.now() - timedelta(hours=2)  # Force retrain
                
                # 3. Retrain models if needed (every 2 hours)
                time_since_training = datetime.now() - last_training
                if time_since_training.total_seconds() >= 7200:  # 2 hours
                    self.log("Periodic model retraining initiated")
                    if self.train_random_forest_models():
                        last_training = datetime.now()
                        self.log("Model retraining completed successfully")
                    else:
                        self.log("Model retraining failed")
                
                # 4. Periodic genetic algorithm optimization (every 6 hours)
                if cycle_count % 72 == 1:  # Every 72 cycles (6 hours at 5-min intervals)
                    self.log("Periodic genetic algorithm optimization initiated")
                    ga_results = self.run_genetic_algorithm_optimization()
                    
                    if ga_results.get('success', False):
                        best_params = ga_results['best_params']
                        self.current_params.update(best_params)
                        self.log("GA optimization completed, parameters updated")
                        
                        # Retrain with new parameters
                        if self.train_random_forest_models():
                            last_training = datetime.now()
                
                # 5. Log status
                self.log(f"Data samples: {len(self.data) if self.data is not None else 0}")
                self.log(f"Models trained: {len(self.models)}")
                self.log(f"Next training in: {7200 - time_since_training.total_seconds():.0f} seconds")
                
                # 6. Sleep between cycles
                time.sleep(300)  # 5 minutes
                
        except Exception as e:
            self.log(f"Error in Agent02 main loop: {e}")
            time.sleep(60)  # Cooldown
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log(f"Agent02 received signal {signum} - shutting down")
        self.running = False

def main():
    """Entry point for Agent02."""
    agent = Agent02DataML()
    agent.run_main_loop()

if __name__ == "__main__":
    main()