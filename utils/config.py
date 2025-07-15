"""Configuration management for the crypto RF trading system."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


@dataclass
class DataConfig:
    """Data fetching and processing configuration."""
    # Top 9 cryptocurrencies by market cap
    symbols: List[str] = field(default_factory=lambda: [
        'bitcoin', 'ethereum', 'tether', 'solana', 'binancecoin', 
        'usd-coin', 'ripple', 'dogecoin', 'cardano'
    ])
    
    # Data source configuration
    data_source: str = "yfinance"  # "yfinance", "coingecko", "multi_timeframe"
    coingecko_api_key: str = ""  # Optional pro API key
    coinmarketcap_api_key: str = ""  # Optional pro API key
    
    # Data parameters
    vs_currency: str = "usd"
    days: int = 730  # 2 years of data
    interval: str = "hourly"  # hourly, daily, 5m, etc.
    
    # Caching
    cache_dir: str = "data/cached_data"
    cache_expiry_hours: int = 1  # Cache expiry in hours
    
    # Rate limiting
    requests_per_minute: int = 25  # Conservative rate limit
    
    # Ultra optimization features
    use_ultra_features: bool = False
    use_ultra_targets: bool = False
    

@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    
    # Volume indicators
    volume_sma_period: int = 20
    obv_period: int = 10
    
    # J/K returns and volatility
    return_periods: List[int] = field(default_factory=lambda: [1, 6, 12, 24])  # hours
    volatility_periods: List[int] = field(default_factory=lambda: [24, 168, 720])  # hours
    
    # Cross-asset features
    correlation_period: int = 168  # 1 week in hours
    
    # Feature selection
    max_features: int = 50
    feature_importance_threshold: float = 0.01
    
    # Ultra optimization features
    use_ultra_engineering: bool = False
    feature_selection_method: str = "importance"
    interaction_features: bool = False
    polynomial_features: bool = False


@dataclass
class ModelConfig:
    """Random Forest model configuration."""
    # Random Forest parameters
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 42
    
    # Hyperparameter tuning
    use_grid_search: bool = True
    cv_folds: int = 5
    scoring: str = "neg_mean_squared_error"
    
    # Model validation
    test_size: float = 0.2
    validation_size: float = 0.2
    walk_forward_splits: int = 10
    
    # Target variable
    target_horizon: int = 6  # hours ahead to predict
    target_type: str = "returns"  # returns, log_returns, price_change, meta_sharpe
    
    # Ultra optimization fields
    model_type: str = "random_forest"  # random_forest, ensemble
    ensemble_models: List[str] = field(default_factory=lambda: ["random_forest"])
    ensemble_method: str = "voting"  # voting, weighted_voting, stacking
    use_optuna_optimization: bool = False
    optuna_trials: int = 100
    optuna_timeout: int = 3600
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["accuracy"])
    purged_cv: bool = False
    embargo_periods: int = 0
    target_preprocessing: str = "none"  # none, normalized, quantile
    
    # Additional ultra optimization fields
    multi_timeframe_analysis: bool = False
    primary_timeframe: str = "1h"
    secondary_timeframes: List[str] = field(default_factory=lambda: ["4h", "1d"])
    advanced_feature_engineering: bool = False
    risk_adjusted_targets: bool = False
    regime_aware_modeling: bool = False
    ensemble_stacking: bool = False
    hyperparameter_search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Target engineering config as fields
    sharpe_targets: bool = False
    sortino_targets: bool = False
    regime_conditional_targets: bool = False
    tail_risk_targets: bool = False
    momentum_targets: bool = False
    cross_sectional_targets: bool = False
    meta_targets: bool = False
    
    # Feature engineering config as fields
    fractal_features: bool = False
    microstructure_features: bool = False
    regime_features: bool = False
    volatility_clustering: bool = False
    order_flow_features: bool = False
    time_decay_features: bool = False
    cross_timeframe_features: bool = False


@dataclass
class StrategyConfig:
    """Trading strategy configuration."""
    # Long/short strategy
    long_threshold: float = 0.6  # Top 60% percentile for long positions
    short_threshold: float = 0.4  # Bottom 40% percentile for short positions
    
    # Position sizing
    max_position_size: float = 0.15  # Max 15% per asset
    min_position_size: float = 0.05  # Min 5% per asset
    position_sizing_method: str = "equal_weight"  # equal_weight, volatility_adjusted
    
    # Rebalancing
    rebalancing_frequency: str = "monthly"  # monthly, weekly, daily
    rebalancing_threshold: float = 0.05  # 5% deviation threshold
    
    # Risk management
    max_portfolio_leverage: float = 2.0
    stop_loss_pct: float = 0.10  # 10% stop loss
    take_profit_pct: float = 0.20  # 20% take profit
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    
    # Transaction costs
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    
    # Ultra optimization fields
    risk_adjusted_sizing: bool = False
    dynamic_thresholds: bool = False
    regime_aware_strategy: bool = False


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Time periods
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    
    # Initial capital
    initial_capital: float = 100000.0
    
    # Performance metrics
    benchmark_symbol: str = "bitcoin"  # Benchmark for comparison
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Reporting
    save_results: bool = True
    results_dir: str = "results"
    plot_results: bool = True
    
    # Ultra optimization fields
    detailed_analysis: bool = False
    regime_analysis: bool = False
    risk_metrics: bool = False


@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Logging
    log_level: str = "INFO"
    log_file: str = "crypto_rf_trading.log"
    
    # Performance
    n_jobs: int = -1
    memory_limit_gb: float = 8.0
    
    # Directories
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    
    # Alerts
    enable_alerts: bool = False
    alert_email: str = ""
    
    # Random seed
    random_seed: int = 42
    
    # Ultra optimization fields
    enable_profiling: bool = False
    save_intermediate_results: bool = False
    checkpoint_frequency: int = 100


@dataclass
class Config:
    """Main configuration class combining all config sections."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'model': self.model.__dict__,
            'strategy': self.strategy.__dict__,
            'backtest': self.backtest.__dict__,
            'system': self.system.__dict__
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Handle ultra_optimization section by merging into appropriate sections
        if 'ultra_optimization' in config_dict:
            ultra_config = config_dict['ultra_optimization']
            
            # Merge feature engineering config
            if 'feature_engineering_config' in ultra_config:
                config_dict['features'].update(ultra_config['feature_engineering_config'])
            
            # Merge target engineering config
            if 'target_engineering_config' in ultra_config:
                config_dict['model'].update(ultra_config['target_engineering_config'])
            
            # Merge hyperparameter search space
            if 'hyperparameter_search_space' in ultra_config:
                config_dict['model']['hyperparameter_search_space'] = ultra_config['hyperparameter_search_space']
            
            # Merge other ultra optimization settings into model config
            for key, value in ultra_config.items():
                if key not in ['feature_engineering_config', 'target_engineering_config', 'hyperparameter_search_space']:
                    config_dict['model'][key] = value
        
        # Filter config dicts to only include supported fields
        def filter_config(config_dict, config_class):
            """Filter config dict to only include fields supported by the dataclass."""
            from dataclasses import fields
            supported_fields = {f.name for f in fields(config_class)}
            return {k: v for k, v in config_dict.items() if k in supported_fields}
        
        return cls(
            data=DataConfig(**filter_config(config_dict['data'], DataConfig)),
            features=FeatureConfig(**filter_config(config_dict['features'], FeatureConfig)),
            model=ModelConfig(**filter_config(config_dict['model'], ModelConfig)),
            strategy=StrategyConfig(**filter_config(config_dict['strategy'], StrategyConfig)),
            backtest=BacktestConfig(**filter_config(config_dict['backtest'], BacktestConfig)),
            system=SystemConfig(**filter_config(config_dict['system'], SystemConfig))
        )
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # API keys
        if os.getenv('COINGECKO_API_KEY'):
            self.data.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        if os.getenv('COINMARKETCAP_API_KEY'):
            self.data.coinmarketcap_api_key = os.getenv('COINMARKETCAP_API_KEY')
        
        # System settings
        if os.getenv('LOG_LEVEL'):
            self.system.log_level = os.getenv('LOG_LEVEL')
        if os.getenv('N_JOBS'):
            self.system.n_jobs = int(os.getenv('N_JOBS'))


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(filepath: str = "configs/config.json") -> Config:
    """Load configuration from file with fallback to defaults."""
    try:
        config = Config.load(filepath)
        config.update_from_env()
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load config from {filepath}: {e}")
        print("Using default configuration.")
        config = get_default_config()
        config.update_from_env()
        return config


if __name__ == "__main__":
    # Create and save default configuration
    config = get_default_config()
    config.save("configs/config.json")
    print("Default configuration saved to configs/config.json")