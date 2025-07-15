#!/usr/bin/env python3
"""
Agent02: Data Loader, Feature Engineer & Random Forest Retrainer

Handles data loading, feature engineering, and model retraining
with genetic algorithm optimization (DEAP if available, else heuristics).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import DEAP for genetic algorithm
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

COMS_PATH = "crypto_rf_trader/coms.md"
DATA_DIR = "crypto_rf_trader/data/4h_training/"
MODEL_PATH = "crypto_rf_trader/models/enhanced_rf_models.pkl"
FEATURES_PATH = "crypto_rf_trader/models/feature_config.json"

class Agent02DataML:
    """Data processing and ML model management agent."""
    
    def __init__(self):
        self.log("Agent02 initializing data and ML operations")
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_config = {}
        
        # GA parameters
        self.ga_population_size = 50
        self.ga_generations = 10
        self.ga_mutation_prob = 0.2
        self.ga_crossover_prob = 0.7
        
        # Optimal parameters from analysis
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
            f.write(f"[agent02][{timestamp}] {message}\n")
        print(f"[agent02][{timestamp}] {message}")
    
    def load_data(self):
        """Load and combine all training data."""
        try:
            # First check if we have existing 4h data
            existing_data = "/home/richardw/crypto_rf_trading_system/data/4h_training/crypto_4h_dataset_20250714_130201.csv"
            if os.path.exists(existing_data):
                self.log(f"Loading existing dataset: {existing_data}")
                self.data = pd.read_csv(existing_data, index_col=0, parse_dates=True)
                self.log(f"Loaded {len(self.data)} rows from existing dataset")
                return self.data
            
            # Otherwise look for CSV files in DATA_DIR
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR, exist_ok=True)
                self.log(f"Created data directory: {DATA_DIR}")
                
            all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            
            if not all_files:
                self.log("No training data found, generating synthetic data")
                self.data = self._generate_synthetic_data()
                return self.data
            
            df_list = []
            for file in all_files:
                df = pd.read_csv(file)
                df_list.append(df)
                self.log(f"Loaded {len(df)} rows from {file}")
            
            combined = pd.concat(df_list, ignore_index=True)
            self.data = combined
            self.log(f"Combined data: {len(self.data)} total rows")
            return self.data
            
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            self.data = self._generate_synthetic_data()
            return self.data
    
    def _generate_synthetic_data(self):
        """Generate synthetic crypto data for testing."""
        self.log("Generating synthetic training data")
        
        # Generate realistic crypto price data
        n_samples = 5000
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='4H')
        
        # Simulate BTC-like price movements
        np.random.seed(42)
        returns = np.random.randn(n_samples) * 0.02  # 2% volatility
        price_base = 50000
        prices = price_base * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(n_samples) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
            'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
            'close': prices,
            'volume': np.random.randint(1000000, 100000000, n_samples),
            'symbol': 'BTC'
        })
        
        # Fix OHLC relationships
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return data
    
    def engineer_features(self, df):
        """Engineer features including momentum patterns."""
        self.log("Engineering features with momentum patterns")
        
        # Basic momentum features
        df['momentum_1'] = df['close'].pct_change(1) * 100
        df['momentum_4'] = df['close'].pct_change(4) * 100
        df['momentum_12'] = df['close'].pct_change(12) * 100
        
        # Enhanced momentum features
        df['momentum_strength'] = df['momentum_1'] / 4  # Convert to hourly
        df['is_high_momentum'] = (df['momentum_strength'] > self.optimal_params['momentum_threshold']).astype(int)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Time features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_optimal_hour'] = (df['hour'] == 3).astype(int)  # 3 AM optimal
        elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_optimal_hour'] = (df['hour'] == 3).astype(int)
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['market_structure'] = df['higher_high'] + df['higher_low'] - 1
        
        # Success pattern scoring
        df['success_score'] = (
            df['is_high_momentum'] * 3 +
            (df['market_structure'] >= 0).astype(int) * 2 +
            df.get('is_optimal_hour', 0) * 1 +
            (df['volume_ratio'] > 1.0).astype(int) * 1 +
            (df['rsi'] < 70).astype(int) * 1
        )
        
        df['success_probability'] = np.clip(
            df['success_score'] / 8.0 * self.optimal_params['confidence_threshold'], 
            0.0, 1.0
        )
        
        # Create target variables
        df['price_up_1h'] = (df['close'].shift(-1) > df['close']).astype(int)
        df['price_up_4h'] = (df['close'].shift(-4) > df['close']).astype(int)
        df['returns_4h'] = (df['close'].shift(-4) / df['close'] - 1) * 100
        df['profitable_4h'] = (df['returns_4h'] > 0.68).astype(int)  # 0.68% profit threshold
        
        # Position sizing targets
        df['optimal_position'] = np.where(
            df['is_high_momentum'] == 1,
            self.optimal_params['position_size_max'],
            np.where(df['momentum_strength'] > 0.5, 0.620, self.optimal_params['position_size_min'])
        )
        
        # Store feature configuration
        self.feature_config = {
            'entry_features': ['momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
                              'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
                              'hour', 'day_of_week', 'volatility', 'success_probability'],
            'position_features': ['momentum_strength', 'is_high_momentum', 'volatility', 'rsi',
                                 'market_structure', 'success_probability', 'volume_ratio'],
            'exit_features': ['momentum_1', 'rsi', 'bb_position', 'market_structure',
                             'volatility', 'volume_ratio', 'macd']
        }
        
        return df.dropna()
    
    def train_models(self, df):
        """Train ensemble of specialized Random Forest models."""
        self.log("Training Random Forest ensemble models")
        
        # Ensure we have features
        if 'momentum_1' not in df.columns:
            df = self.engineer_features(df)
        
        # Split data
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        results = {}
        
        # 1. Entry Model
        X_entry = train_df[self.feature_config['entry_features']]
        y_entry = train_df['price_up_4h']
        
        self.scalers['entry'] = StandardScaler()
        X_entry_scaled = self.scalers['entry'].fit_transform(X_entry)
        
        self.models['entry'] = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=8,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        self.models['entry'].fit(X_entry_scaled, y_entry)
        
        # Test accuracy
        X_test_entry = test_df[self.feature_config['entry_features']]
        X_test_entry_scaled = self.scalers['entry'].transform(X_test_entry)
        entry_acc = self.models['entry'].score(X_test_entry_scaled, test_df['price_up_4h'])
        results['entry_accuracy'] = entry_acc
        self.log(f"Entry model accuracy: {entry_acc:.4f}")
        
        # 2. Position Model
        X_position = train_df[self.feature_config['position_features']]
        y_position = train_df['optimal_position']
        
        self.scalers['position'] = StandardScaler()
        X_position_scaled = self.scalers['position'].fit_transform(X_position)
        
        self.models['position'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        self.models['position'].fit(X_position_scaled, y_position)
        
        # Test R²
        X_test_position = test_df[self.feature_config['position_features']]
        X_test_position_scaled = self.scalers['position'].transform(X_test_position)
        position_r2 = self.models['position'].score(X_test_position_scaled, test_df['optimal_position'])
        results['position_r2'] = position_r2
        self.log(f"Position model R²: {position_r2:.4f}")
        
        # 3. Exit Model
        # Create exit signals
        train_df['exit_signal'] = (
            (train_df['rsi'] > 70) |
            (train_df['momentum_strength'] < 0) |
            (train_df['bb_position'] > 0.8)
        ).astype(int)
        
        X_exit = train_df[self.feature_config['exit_features']]
        y_exit = train_df['exit_signal']
        
        self.scalers['exit'] = StandardScaler()
        X_exit_scaled = self.scalers['exit'].fit_transform(X_exit)
        
        self.models['exit'] = RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_split=12,
            min_samples_leaf=6, random_state=42, n_jobs=-1
        )
        self.models['exit'].fit(X_exit_scaled, y_exit)
        
        # Test accuracy
        test_df['exit_signal'] = (
            (test_df['rsi'] > 70) |
            (test_df['momentum_strength'] < 0) |
            (test_df['bb_position'] > 0.8)
        ).astype(int)
        
        X_test_exit = test_df[self.feature_config['exit_features']]
        X_test_exit_scaled = self.scalers['exit'].transform(X_test_exit)
        exit_acc = self.models['exit'].score(X_test_exit_scaled, test_df['exit_signal'])
        results['exit_accuracy'] = exit_acc
        self.log(f"Exit model accuracy: {exit_acc:.4f}")
        
        # Save models and configuration
        self._save_models()
        
        return results
    
    def _save_models(self):
        """Save trained models and configuration."""
        try:
            # Save models
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({
                    'models': self.models,
                    'scalers': self.scalers,
                    'timestamp': datetime.now().isoformat(),
                    'optimal_params': self.optimal_params
                }, f)
            self.log(f"Models saved to {MODEL_PATH}")
            
            # Save feature configuration
            with open(FEATURES_PATH, "w") as f:
                json.dump(self.feature_config, f, indent=2)
            self.log(f"Feature config saved to {FEATURES_PATH}")
            
        except Exception as e:
            self.log(f"Error saving models: {str(e)}")
    
    def run_ga_optimization(self):
        """Run genetic algorithm for hyperparameter optimization."""
        if DEAP_AVAILABLE:
            self.log("Running DEAP genetic algorithm optimization")
            return self._run_deap_ga()
        else:
            self.log("DEAP not available, running heuristic optimization")
            return self._run_heuristic_optimization()
    
    def _run_deap_ga(self):
        """Run genetic algorithm using DEAP."""
        # Define fitness and individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Parameter ranges
        toolbox.register("momentum_threshold", np.random.uniform, 1.0, 3.0)
        toolbox.register("n_estimators", np.random.randint, 50, 300)
        toolbox.register("max_depth", np.random.randint, 5, 20)
        toolbox.register("min_samples_split", np.random.randint, 2, 20)
        
        # Create individuals
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.momentum_threshold, toolbox.n_estimators,
                         toolbox.max_depth, toolbox.min_samples_split), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("evaluate", self._evaluate_params)
        toolbox.register("mate", tools.cxBlend, alpha=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run algorithm
        population = toolbox.population(n=self.ga_population_size)
        
        for generation in range(self.ga_generations):
            # Evaluate population
            fitnesses = map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Select and reproduce
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
            
            # Log progress
            fits = [ind.fitness.values[0] for ind in population if ind.fitness.values]
            if fits:
                self.log(f"Generation {generation}: Max fitness = {max(fits):.4f}")
        
        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        best_params = {
            'momentum_threshold': best_ind[0],
            'n_estimators': int(best_ind[1]),
            'max_depth': int(best_ind[2]),
            'min_samples_split': int(best_ind[3])
        }
        
        self.log(f"GA optimization complete. Best params: {best_params}")
        return best_params
    
    def _run_heuristic_optimization(self):
        """Run heuristic optimization without DEAP."""
        best_score = -float('inf')
        best_params = {}
        
        # Parameter ranges
        param_ranges = {
            'momentum_threshold': np.linspace(1.0, 3.0, 10),
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [6, 8, 10, 12, 15],
            'min_samples_split': [5, 10, 15, 20]
        }
        
        # Random search
        for i in range(20):  # 20 random combinations
            params = {
                'momentum_threshold': np.random.choice(param_ranges['momentum_threshold']),
                'n_estimators': np.random.choice(param_ranges['n_estimators']),
                'max_depth': np.random.choice(param_ranges['max_depth']),
                'min_samples_split': np.random.choice(param_ranges['min_samples_split'])
            }
            
            score = self._evaluate_params_dict(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                self.log(f"New best score: {best_score:.4f} with params: {params}")
        
        return best_params
    
    def _evaluate_params(self, individual):
        """Evaluate parameter set (DEAP format)."""
        params = {
            'momentum_threshold': individual[0],
            'n_estimators': int(individual[1]),
            'max_depth': int(individual[2]),
            'min_samples_split': int(individual[3])
        }
        return (self._evaluate_params_dict(params),)
    
    def _evaluate_params_dict(self, params):
        """Evaluate parameter set and return fitness score."""
        try:
            # Use a subset of data for faster evaluation
            if self.data is None or len(self.data) < 1000:
                return -1.0
            
            eval_data = self.data.sample(min(1000, len(self.data)))
            
            # Update momentum threshold
            old_threshold = self.optimal_params['momentum_threshold']
            self.optimal_params['momentum_threshold'] = params['momentum_threshold']
            
            # Re-engineer features with new threshold
            eval_data = self.engineer_features(eval_data)
            
            # Quick model training
            X = eval_data[self.feature_config['entry_features']]
            y = eval_data['price_up_4h']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(
                n_estimators=min(params['n_estimators'], 50),  # Limit for speed
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                random_state=42,
                n_jobs=2
            )
            
            model.fit(X_train_scaled, y_train)
            accuracy = model.score(X_test_scaled, y_test)
            
            # Restore original threshold
            self.optimal_params['momentum_threshold'] = old_threshold
            
            return accuracy
            
        except Exception as e:
            self.log(f"Error evaluating params: {str(e)}")
            return -1.0
    
    def continuous_retraining_loop(self):
        """Run continuous retraining loop."""
        self.log("Starting continuous retraining loop")
        
        while True:
            try:
                # Load fresh data
                self.load_data()
                
                # Engineer features
                if self.data is not None:
                    self.data = self.engineer_features(self.data)
                    
                    # Train models
                    results = self.train_models(self.data)
                    self.log(f"Model training complete: {results}")
                    
                    # Run optimization every 6 hours
                    if np.random.random() < 0.25:  # 25% chance each cycle
                        self.log("Running hyperparameter optimization")
                        best_params = self.run_ga_optimization()
                        self.log(f"Optimization complete: {best_params}")
                
                # Sleep for 30 minutes before next cycle
                time.sleep(1800)
                
            except Exception as e:
                self.log(f"Error in retraining loop: {str(e)}")
                time.sleep(300)  # Sleep 5 minutes on error

def main():
    agent = Agent02DataML()
    
    # Initial data load and training
    agent.log("Agent02 starting initial data load and model training")
    agent.load_data()
    
    if agent.data is not None:
        agent.data = agent.engineer_features(agent.data)
        results = agent.train_models(agent.data)
        agent.log(f"Initial training complete: {results}")
    
    # Start continuous loop
    agent.continuous_retraining_loop()

if __name__ == "__main__":
    main()