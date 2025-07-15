#!/usr/bin/env python3
"""
Phase 2A: Triple Barrier Labeling with Genetic Algorithm Optimization
ULTRATHINK Implementation - Advanced Target Generation

Implements sophisticated labeling technique used by institutional trading firms:
- Triple barrier method (profit, loss, time barriers)
- Genetic algorithm optimization for barrier parameters
- Adaptive barrier sizing based on volatility
- Multi-objective optimization (return vs risk)
- Meta-labeling for ensemble strategies

Designed to create higher quality targets and reduce overfitting risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    # Barrier parameters
    profit_threshold: float = 0.02  # 2% profit target
    loss_threshold: float = 0.01    # 1% stop loss
    max_holding_period: int = 10    # Maximum days to hold
    
    # Volatility adaptation
    use_dynamic_barriers: bool = True
    volatility_window: int = 20
    volatility_multiplier: float = 2.0
    
    # Genetic algorithm parameters
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 5
    
    # Optimization bounds
    profit_bounds: Tuple[float, float] = (0.005, 0.05)  # 0.5% to 5%
    loss_bounds: Tuple[float, float] = (0.002, 0.03)    # 0.2% to 3%
    time_bounds: Tuple[int, int] = (1, 30)              # 1 to 30 days
    
    # Fitness function weights
    return_weight: float = 0.4
    sharpe_weight: float = 0.3
    win_rate_weight: float = 0.2
    trade_count_weight: float = 0.1

class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    def __init__(self, profit_threshold: float, loss_threshold: float, 
                 max_holding_period: int):
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.max_holding_period = max_holding_period
        self.fitness = 0.0
        self.metrics = {}
    
    def to_dict(self) -> Dict:
        return {
            'profit_threshold': self.profit_threshold,
            'loss_threshold': self.loss_threshold,
            'max_holding_period': self.max_holding_period,
            'fitness': self.fitness,
            'metrics': self.metrics
        }

class TripleBarrierLabeling:
    """
    Professional triple barrier labeling with genetic algorithm optimization.
    
    Implements sophisticated target generation techniques used by institutional
    trading firms for creating high-quality machine learning labels.
    """
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """
        Initialize triple barrier labeling engine.
        
        Args:
            config: Configuration for labeling parameters
        """
        self.config = config or TripleBarrierConfig()
        self.optimization_results = {}
        self.labeling_results = {}
        self.best_individual = None
        
        print("🎯 Triple Barrier Labeling Engine Initialized")
        print(f"📊 Default Barriers: Profit {self.config.profit_threshold:.1%}, Loss {self.config.loss_threshold:.1%}")
        print(f"⏰ Max Holding: {self.config.max_holding_period} periods")
        print(f"🧬 GA Parameters: {self.config.population_size} pop, {self.config.generations} gen")
        print(f"🎛️ Dynamic Barriers: {'Enabled' if self.config.use_dynamic_barriers else 'Disabled'}")
    
    def generate_labels(self, df: pd.DataFrame, 
                       price_column: str = 'Close',
                       optimize_barriers: bool = True) -> pd.DataFrame:
        """
        Generate triple barrier labels for the dataset.
        
        Args:
            df: DataFrame with price data
            price_column: Column name for price data
            optimize_barriers: Whether to optimize barriers using GA
            
        Returns:
            DataFrame with barrier labels added
        """
        print(f"\n🎯 Generating Triple Barrier Labels")
        print("=" * 50)
        
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found")
        
        result_df = df.copy()
        prices = df[price_column]
        
        # Optimize barriers if requested
        if optimize_barriers:
            print("🧬 Optimizing barrier parameters with genetic algorithm...")
            self.best_individual = self.optimize_barriers_with_ga(df, price_column)
            
            optimal_config = TripleBarrierConfig(
                profit_threshold=self.best_individual.profit_threshold,
                loss_threshold=self.best_individual.loss_threshold,
                max_holding_period=self.best_individual.max_holding_period,
                use_dynamic_barriers=self.config.use_dynamic_barriers,
                volatility_window=self.config.volatility_window,
                volatility_multiplier=self.config.volatility_multiplier
            )
        else:
            optimal_config = self.config
        
        print(f"📊 Using barriers: Profit {optimal_config.profit_threshold:.1%}, Loss {optimal_config.loss_threshold:.1%}, Time {optimal_config.max_holding_period}")
        
        # Calculate dynamic barriers if enabled
        if optimal_config.use_dynamic_barriers:
            print("📈 Calculating dynamic volatility-adjusted barriers...")
            volatility = self.calculate_volatility(prices, optimal_config.volatility_window)
            profit_barriers = optimal_config.profit_threshold * (1 + volatility * optimal_config.volatility_multiplier)
            loss_barriers = optimal_config.loss_threshold * (1 + volatility * optimal_config.volatility_multiplier)
        else:
            profit_barriers = pd.Series(optimal_config.profit_threshold, index=prices.index)
            loss_barriers = pd.Series(optimal_config.loss_threshold, index=prices.index)
        
        # Apply triple barrier method
        print("🎯 Applying triple barrier method...")
        labels, barrier_info = self.apply_triple_barriers(
            prices, profit_barriers, loss_barriers, optimal_config.max_holding_period
        )
        
        # Add results to DataFrame
        result_df['barrier_label'] = labels
        result_df['barrier_return'] = barrier_info['returns']
        result_df['barrier_hit'] = barrier_info['barrier_hit']
        result_df['holding_period'] = barrier_info['holding_periods']
        result_df['profit_barrier'] = profit_barriers
        result_df['loss_barrier'] = loss_barriers
        
        # Calculate additional metrics
        result_df = self.add_meta_labeling_features(result_df, prices)
        
        # Store results
        self.labeling_results = self.analyze_labeling_performance(result_df)
        
        # Print summary
        self.print_labeling_summary(self.labeling_results)
        
        return result_df
    
    def optimize_barriers_with_ga(self, df: pd.DataFrame, 
                                 price_column: str) -> Individual:
        """Optimize barrier parameters using genetic algorithm."""
        prices = df[price_column]
        
        # Initialize population
        population = self.initialize_population()
        
        print(f"🧬 Starting genetic algorithm optimization...")
        print(f"   Population size: {len(population)}")
        print(f"   Generations: {self.config.generations}")
        
        best_fitness_history = []
        
        for generation in range(self.config.generations):
            # Evaluate fitness for all individuals
            for individual in population:
                individual.fitness, individual.metrics = self.calculate_fitness(
                    individual, prices
                )
            
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness = population[0].fitness
            best_fitness_history.append(best_fitness)
            
            print(f"   Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
            
            # Create next generation
            if generation < self.config.generations - 1:
                population = self.create_next_generation(population)
        
        best_individual = population[0]
        print(f"\n🏆 Optimization complete!")
        print(f"   Best fitness: {best_individual.fitness:.4f}")
        print(f"   Optimal barriers: P={best_individual.profit_threshold:.1%}, L={best_individual.loss_threshold:.1%}, T={best_individual.max_holding_period}")
        
        # Store optimization results
        self.optimization_results = {
            'best_individual': best_individual.to_dict(),
            'fitness_history': best_fitness_history,
            'final_population': [ind.to_dict() for ind in population[:10]]  # Top 10
        }
        
        return best_individual
    
    def initialize_population(self) -> List[Individual]:
        """Initialize random population for genetic algorithm."""
        population = []
        
        for _ in range(self.config.population_size):
            profit_threshold = np.random.uniform(*self.config.profit_bounds)
            loss_threshold = np.random.uniform(*self.config.loss_bounds)
            max_holding_period = np.random.randint(*self.config.time_bounds)
            
            individual = Individual(profit_threshold, loss_threshold, max_holding_period)
            population.append(individual)
        
        return population
    
    def calculate_fitness(self, individual: Individual, 
                         prices: pd.Series) -> Tuple[float, Dict]:
        """Calculate fitness score for an individual."""
        # Apply triple barriers with individual's parameters
        if self.config.use_dynamic_barriers:
            volatility = self.calculate_volatility(prices, self.config.volatility_window)
            profit_barriers = individual.profit_threshold * (1 + volatility * self.config.volatility_multiplier)
            loss_barriers = individual.loss_threshold * (1 + volatility * self.config.volatility_multiplier)
        else:
            profit_barriers = pd.Series(individual.profit_threshold, index=prices.index)
            loss_barriers = pd.Series(individual.loss_threshold, index=prices.index)
        
        labels, barrier_info = self.apply_triple_barriers(
            prices, profit_barriers, loss_barriers, individual.max_holding_period
        )
        
        # Calculate performance metrics
        returns = barrier_info['returns'].dropna()
        
        if len(returns) < 10:  # Minimum trades for meaningful evaluation
            return 0.0, {}
        
        # Performance metrics
        total_return = returns.sum()
        avg_return = returns.mean()
        return_std = returns.std()
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        win_rate = (returns > 0).mean()
        trade_count = len(returns)
        
        # Multi-objective fitness function
        fitness = (
            self.config.return_weight * avg_return +
            self.config.sharpe_weight * sharpe_ratio +
            self.config.win_rate_weight * win_rate +
            self.config.trade_count_weight * min(trade_count / 100, 1.0)  # Normalize trade count
        )
        
        metrics = {
            'total_return': total_return,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'trade_count': trade_count,
            'return_std': return_std
        }
        
        return fitness, metrics
    
    def create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Create next generation using selection, crossover, and mutation."""
        next_generation = []
        
        # Elitism: keep best individuals
        elite_count = min(self.config.elite_size, len(population))
        next_generation.extend(population[:elite_count])
        
        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Tournament selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child1 = self.mutate(child1)
            if np.random.random() < self.config.mutation_rate:
                child2 = self.mutate(child2)
            
            next_generation.extend([child1, child2])
        
        # Trim to exact population size
        return next_generation[:self.config.population_size]
    
    def tournament_selection(self, population: List[Individual], 
                           tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, 
                 parent2: Individual) -> Tuple[Individual, Individual]:
        """Create offspring using crossover."""
        # Arithmetic crossover
        alpha = np.random.random()
        
        child1_profit = alpha * parent1.profit_threshold + (1 - alpha) * parent2.profit_threshold
        child1_loss = alpha * parent1.loss_threshold + (1 - alpha) * parent2.loss_threshold
        child1_time = int(alpha * parent1.max_holding_period + (1 - alpha) * parent2.max_holding_period)
        
        child2_profit = (1 - alpha) * parent1.profit_threshold + alpha * parent2.profit_threshold
        child2_loss = (1 - alpha) * parent1.loss_threshold + alpha * parent2.loss_threshold
        child2_time = int((1 - alpha) * parent1.max_holding_period + alpha * parent2.max_holding_period)
        
        # Ensure bounds
        child1_profit = np.clip(child1_profit, *self.config.profit_bounds)
        child1_loss = np.clip(child1_loss, *self.config.loss_bounds)
        child1_time = np.clip(child1_time, *self.config.time_bounds)
        
        child2_profit = np.clip(child2_profit, *self.config.profit_bounds)
        child2_loss = np.clip(child2_loss, *self.config.loss_bounds)
        child2_time = np.clip(child2_time, *self.config.time_bounds)
        
        child1 = Individual(child1_profit, child1_loss, child1_time)
        child2 = Individual(child2_profit, child2_loss, child2_time)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        # Gaussian mutation
        mutation_strength = 0.1
        
        profit_mutation = np.random.normal(0, mutation_strength * individual.profit_threshold)
        loss_mutation = np.random.normal(0, mutation_strength * individual.loss_threshold)
        time_mutation = np.random.randint(-2, 3)  # Small integer change
        
        new_profit = np.clip(
            individual.profit_threshold + profit_mutation,
            *self.config.profit_bounds
        )
        new_loss = np.clip(
            individual.loss_threshold + loss_mutation,
            *self.config.loss_bounds
        )
        new_time = np.clip(
            individual.max_holding_period + time_mutation,
            *self.config.time_bounds
        )
        
        return Individual(new_profit, new_loss, new_time)
    
    def apply_triple_barriers(self, prices: pd.Series,
                            profit_barriers: pd.Series,
                            loss_barriers: pd.Series,
                            max_holding_period: int) -> Tuple[pd.Series, Dict]:
        """Apply triple barrier method to generate labels."""
        labels = pd.Series(index=prices.index, dtype=int)
        returns = pd.Series(index=prices.index, dtype=float)
        barrier_hit = pd.Series(index=prices.index, dtype=str)
        holding_periods = pd.Series(index=prices.index, dtype=int)
        
        for i in range(len(prices) - max_holding_period):
            entry_price = prices.iloc[i]
            entry_time = prices.index[i]
            
            profit_threshold = profit_barriers.iloc[i]
            loss_threshold = loss_barriers.iloc[i]
            
            # Look ahead for barrier hits
            for j in range(1, max_holding_period + 1):
                if i + j >= len(prices):
                    break
                
                current_price = prices.iloc[i + j]
                price_change = (current_price - entry_price) / entry_price
                
                # Check barriers
                if price_change >= profit_threshold:
                    # Profit barrier hit
                    labels.iloc[i] = 1
                    returns.iloc[i] = price_change
                    barrier_hit.iloc[i] = 'profit'
                    holding_periods.iloc[i] = j
                    break
                elif price_change <= -loss_threshold:
                    # Loss barrier hit
                    labels.iloc[i] = -1
                    returns.iloc[i] = price_change
                    barrier_hit.iloc[i] = 'loss'
                    holding_periods.iloc[i] = j
                    break
            else:
                # Time barrier hit (max holding period reached)
                if i + max_holding_period < len(prices):
                    final_price = prices.iloc[i + max_holding_period]
                    price_change = (final_price - entry_price) / entry_price
                    labels.iloc[i] = np.sign(price_change)
                    returns.iloc[i] = price_change
                    barrier_hit.iloc[i] = 'time'
                    holding_periods.iloc[i] = max_holding_period
        
        barrier_info = {
            'returns': returns,
            'barrier_hit': barrier_hit,
            'holding_periods': holding_periods
        }
        
        return labels, barrier_info
    
    def calculate_volatility(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility for dynamic barrier sizing."""
        returns = prices.pct_change()
        volatility = returns.rolling(window).std()
        return volatility.fillna(volatility.mean())
    
    def add_meta_labeling_features(self, df: pd.DataFrame, 
                                  prices: pd.Series) -> pd.DataFrame:
        """Add meta-labeling features for ensemble strategies."""
        result_df = df.copy()
        
        # Label quality indicators
        result_df['label_confidence'] = abs(result_df['barrier_return'])
        
        # Regime indicators
        volatility = self.calculate_volatility(prices, 20)
        result_df['volatility_regime'] = pd.cut(volatility, bins=3, labels=['low', 'medium', 'high'])
        
        # Trend indicators
        trend = prices.rolling(20).mean() / prices.rolling(50).mean()
        result_df['trend_regime'] = pd.cut(trend, bins=3, labels=['down', 'sideways', 'up'])
        
        # Volume regime (if available)
        if 'Volume' in df.columns:
            volume_ma = df['Volume'].rolling(20).mean()
            volume_regime = df['Volume'] / volume_ma
            result_df['volume_regime'] = pd.cut(volume_regime, bins=3, labels=['low', 'normal', 'high'])
        
        return result_df
    
    def analyze_labeling_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze the performance of generated labels."""
        labels = df['barrier_label'].dropna()
        returns = df['barrier_return'].dropna()
        barrier_hits = df['barrier_hit'].dropna()
        holding_periods = df['holding_period'].dropna()
        
        if len(labels) == 0:
            return {'error': 'No labels generated'}
        
        analysis = {
            'total_labels': len(labels),
            'label_distribution': {
                'buy_signals': (labels == 1).sum(),
                'sell_signals': (labels == -1).sum(),
                'neutral_signals': (labels == 0).sum()
            },
            'barrier_hit_distribution': barrier_hits.value_counts().to_dict(),
            'performance_metrics': {
                'avg_return': returns.mean(),
                'total_return': returns.sum(),
                'return_std': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'win_rate': (returns > 0).mean(),
                'avg_holding_period': holding_periods.mean(),
                'max_holding_period': holding_periods.max()
            },
            'regime_analysis': {}
        }
        
        # Regime-specific analysis
        if 'volatility_regime' in df.columns:
            volatility_performance = df.groupby('volatility_regime')['barrier_return'].agg(['mean', 'std', 'count'])
            analysis['regime_analysis']['volatility'] = volatility_performance.to_dict()
        
        if 'trend_regime' in df.columns:
            trend_performance = df.groupby('trend_regime')['barrier_return'].agg(['mean', 'std', 'count'])
            analysis['regime_analysis']['trend'] = trend_performance.to_dict()
        
        return analysis
    
    def print_labeling_summary(self, analysis: Dict):
        """Print summary of labeling performance."""
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        print(f"\n📊 TRIPLE BARRIER LABELING SUMMARY")
        print("-" * 50)
        print(f"📈 Total Labels Generated: {analysis['total_labels']}")
        
        dist = analysis['label_distribution']
        print(f"📊 Label Distribution:")
        print(f"   Buy Signals: {dist.get('buy_signals', 0)} ({dist.get('buy_signals', 0)/analysis['total_labels']:.1%})")
        print(f"   Sell Signals: {dist.get('sell_signals', 0)} ({dist.get('sell_signals', 0)/analysis['total_labels']:.1%})")
        print(f"   Neutral: {dist.get('neutral_signals', 0)} ({dist.get('neutral_signals', 0)/analysis['total_labels']:.1%})")
        
        barrier_dist = analysis['barrier_hit_distribution']
        print(f"\n🎯 Barrier Hit Distribution:")
        for barrier, count in barrier_dist.items():
            print(f"   {barrier.title()}: {count} ({count/analysis['total_labels']:.1%})")
        
        metrics = analysis['performance_metrics']
        print(f"\n📈 Performance Metrics:")
        print(f"   Average Return: {metrics.get('avg_return', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Avg Holding Period: {metrics.get('avg_holding_period', 0):.1f} periods")

def main():
    """Demonstrate triple barrier labeling with genetic algorithm optimization."""
    print("🎯 PHASE 2A: Triple Barrier Labeling with Genetic Algorithm")
    print("ULTRATHINK Implementation - Advanced Target Generation")
    print("=" * 60)
    
    # Load sample data from Phase 1
    data_dir = "phase1/data/processed"
    import glob
    import os
    
    data_files = glob.glob(f"{data_dir}/BTC-USD_*.csv")
    if not data_files:
        print("❌ No data files found. Run Phase 1A first.")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"📂 Loading data from: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    print(f"📊 Data loaded: {len(df)} samples")
    
    # Initialize triple barrier labeling engine
    config = TripleBarrierConfig(
        profit_threshold=0.02,
        loss_threshold=0.01,
        max_holding_period=7,
        population_size=20,  # Reduced for demo
        generations=10,      # Reduced for demo
        use_dynamic_barriers=True
    )
    
    engine = TripleBarrierLabeling(config)
    
    # Generate labels with optimization
    try:
        # Use a subset for faster demo
        sample_df = df.iloc[:1000].copy()  # First 1000 samples
        
        print(f"\n🔧 Processing {len(sample_df)} samples for demonstration...")
        
        labeled_df = engine.generate_labels(
            sample_df, 
            price_column='Close',
            optimize_barriers=True
        )
        
        # Show results
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Original samples: {len(sample_df)}")
        print(f"   Labeled samples: {labeled_df['barrier_label'].notna().sum()}")
        print(f"   Feature columns added: {len(labeled_df.columns) - len(sample_df.columns)}")
        
        # Save results
        output_file = "phase2/enhanced_features_with_barriers.csv"
        labeled_df.to_csv(output_file)
        print(f"\n💾 Enhanced dataset saved: {output_file}")
        
        # Save optimization results
        if engine.optimization_results:
            import json
            opt_file = "phase2/barrier_optimization_results.json"
            with open(opt_file, 'w') as f:
                json.dump(engine.optimization_results, f, indent=2, default=str)
            print(f"🧬 Optimization results saved: {opt_file}")
        
        print(f"\n🚀 Phase 2A Triple Barrier Labeling: COMPLETE")
        print(f"🎯 Ready for Phase 2A Next Step: Multi-Timeframe Fusion")
        
    except Exception as e:
        print(f"❌ Error in triple barrier labeling: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()