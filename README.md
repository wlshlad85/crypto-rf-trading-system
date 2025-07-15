# Cryptocurrency Random Forest Trading System

A sophisticated cryptocurrency trading system using Random Forest machine learning for multi-asset portfolio management with long/short strategies and monthly rebalancing.

## 🚀 Features

- **Multi-Source Data Fetching**: CoinGecko and CoinMarketCap APIs
- **Top 9 Cryptocurrencies**: BTC, ETH, USDT, SOL, BNB, USDC, XRP, DOGE, ADA
- **Advanced Feature Engineering**: 50+ technical indicators and cross-asset features
- **Random Forest Models**: Single model and ensemble approaches with hyperparameter tuning
- **Long/Short Strategy**: Dynamic long/short positioning based on model predictions
- **Monthly Rebalancing**: Systematic portfolio rebalancing to maintain optimal allocations
- **Comprehensive Backtesting**: Walk-forward validation with empyrical performance metrics
- **Risk Management**: Position sizing, stop-loss, drawdown limits, and volatility controls
- **Professional Visualization**: Interactive charts and performance dashboards

## 📊 Performance Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| Sharpe Ratio | > 1.5 | > 2.0 |
| Max Drawdown | < 20% | < 15% |
| Win Rate | > 55% | > 65% |
| Annualized Return | > 15% | > 25% |

## 🏗️ Architecture

```
crypto_rf_trading_system/
├── data/                    # Multi-source data fetching
│   ├── data_fetcher.py     # CoinGecko/CMC API integration
│   └── cached_data/        # Historical data cache
├── features/               # Feature engineering
│   └── feature_engineering.py  # Technical indicators & cross-asset features
├── models/                 # Machine learning models
│   └── random_forest_model.py  # RF implementation with tuning
├── strategies/             # Trading strategies
│   └── long_short_strategy.py  # Long/short portfolio management
├── backtesting/            # Backtesting engine
│   └── backtest_engine.py  # Walk-forward validation
├── utils/                  # Utilities
│   ├── config.py          # Configuration management
│   └── visualization.py   # Charts and dashboards
└── main.py                 # Main execution script
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for feature engineering)
- Optional: API keys for premium data access

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd crypto_rf_trading_system

# Install dependencies
pip install -r requirements.txt

# Create default configuration
python -c "from utils.config import get_default_config; get_default_config().save('configs/config.json')"
```

### Optional API Configuration
```bash
# Set environment variables for premium data access
export COINGECKO_API_KEY="your_coingecko_pro_key"
export COINMARKETCAP_API_KEY="your_coinmarketcap_pro_key"
```

## 🚀 Quick Start

### Run Complete Pipeline
```bash
# Full pipeline with hyperparameter tuning
python main.py --mode full --tune-hyperparameters --log-level INFO

# Quick run without tuning
python main.py --mode full --log-level INFO

# Use ensemble model for better robustness
python main.py --mode full --use-ensemble --tune-hyperparameters
```

### Individual Components
```bash
# Data fetching only
python main.py --mode data

# Train model only
python main.py --mode train --tune-hyperparameters

# Run backtest only
python main.py --mode backtest --walk-forward
```

## 📈 Usage Examples

### Basic Usage
```python
import asyncio
from utils.config import get_default_config
from main import CryptoRFTradingSystem

async def run_trading_system():
    # Initialize system
    system = CryptoRFTradingSystem()
    
    # Run complete pipeline
    report_files = await system.run_full_pipeline(
        tune_hyperparameters=True,
        use_ensemble=False,
        walk_forward_backtest=True
    )
    
    print("Generated files:", report_files)

# Run the system
asyncio.run(run_trading_system())
```

### Custom Configuration
```python
from utils.config import Config, DataConfig, ModelConfig, StrategyConfig

# Create custom configuration
config = Config()

# Modify data settings
config.data.symbols = ['bitcoin', 'ethereum', 'solana']  # Subset of cryptos
config.data.days = 365  # 1 year of data

# Adjust model parameters
config.model.n_estimators = 500
config.model.max_depth = 15

# Configure strategy
config.strategy.long_threshold = 0.7  # Top 30% for long positions
config.strategy.short_threshold = 0.3  # Bottom 30% for short positions
config.strategy.max_position_size = 0.20  # Max 20% per position

# Save custom config
config.save('configs/custom_config.json')

# Use custom config
system = CryptoRFTradingSystem('configs/custom_config.json')
```

## 🔧 Configuration

The system uses a comprehensive configuration system. Key parameters:

### Data Configuration
- **symbols**: List of cryptocurrencies to trade
- **vs_currency**: Base currency (default: "usd")
- **days**: Historical data period (default: 730 days)
- **interval**: Data frequency ("hourly" or "daily")

### Model Configuration
- **n_estimators**: Number of trees (default: 200)
- **max_depth**: Maximum tree depth (default: 10)
- **target_horizon**: Prediction horizon in hours (default: 6)

### Strategy Configuration
- **long_threshold**: Percentile for long positions (default: 0.6)
- **short_threshold**: Percentile for short positions (default: 0.4)
- **max_position_size**: Maximum position size (default: 0.15)
- **rebalancing_frequency**: "monthly", "weekly", or "daily"

### Risk Management
- **stop_loss_pct**: Stop loss percentage (default: 0.10)
- **max_drawdown_pct**: Maximum portfolio drawdown (default: 0.20)
- **transaction_cost_pct**: Transaction costs (default: 0.001)

## 📊 Features Generated

The system creates 50+ features including:

### Technical Indicators
- RSI, MACD, Bollinger Bands, ATR
- Stochastic Oscillator, Williams %R, CCI
- Multiple EMA/SMA periods

### Price Features
- Multi-period returns (1h, 6h, 12h, 24h)
- Price momentum and acceleration
- Support/resistance levels

### Volume Features
- Volume-weighted average price (VWAP)
- On-balance volume (OBV)
- Volume rate of change

### Volatility Features
- Rolling volatility (24h, 168h, 720h)
- Volatility ratios and momentum
- Garman-Klass volatility

### Cross-Asset Features
- Correlation matrices between cryptocurrencies
- Market dominance ratios
- Relative strength vs Bitcoin

### Market Regime Features
- Bull/bear market indicators
- Volatility regime classification
- Trend strength measures

## 🎯 Strategy Details

### Long/Short Approach
1. **Prediction**: Random Forest predicts future returns for each cryptocurrency
2. **Ranking**: Assets ranked by predicted returns
3. **Signal Generation**: 
   - Long positions: Top 60th percentile predictions
   - Short positions: Bottom 40th percentile predictions
4. **Position Sizing**: Equal-weight or volatility-adjusted
5. **Rebalancing**: Monthly portfolio rebalancing

### Risk Management
- **Position Limits**: Maximum 15% per asset by default
- **Portfolio Leverage**: Maximum 2x leverage
- **Stop Loss**: Automatic 10% stop loss per position
- **Drawdown Protection**: Emergency liquidation at 20% drawdown

## 📈 Performance Analysis

The system provides comprehensive performance metrics:

### Return Metrics
- Total return, annualized return
- Sharpe ratio, Sortino ratio
- Calmar ratio, Omega ratio

### Risk Metrics
- Maximum drawdown
- Value at Risk (VaR) 95% and 99%
- Conditional VaR
- Volatility analysis

### Trade Analysis
- Win rate, profit factor
- Average trade duration
- Transaction cost impact

## 📋 Results and Reporting

Generated outputs include:

### Visualizations
- Portfolio performance charts
- Drawdown analysis
- Returns distribution
- Feature importance plots
- Correlation matrices
- Interactive dashboards (if Plotly installed)

### Data Files
- Backtest results (JSON)
- Trained model (joblib)
- Portfolio history (CSV)
- Trade history (CSV)

## 🔄 Backtesting

### Walk-Forward Validation
- Expanding training window
- Out-of-sample testing
- Multiple validation periods
- Robust performance estimation

### Performance Benchmarks
- Bitcoin buy-and-hold
- Equal-weight crypto portfolio
- Risk-free rate comparison

## ⚠️ Important Warnings

- **Educational Purpose Only**: This system is for research and educational purposes
- **No Trading Guarantees**: Past performance does not guarantee future results
- **Risk Management**: Always use proper risk management in live trading
- **Paper Trading First**: Extensively test with paper trading before real money
- **Market Volatility**: Cryptocurrency markets are highly volatile and unpredictable

## 🛠️ Development

### Adding New Features
```python
# Extend feature_engineering.py
def _add_custom_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # Add your custom technical indicators
    df[f"{symbol}_custom_indicator"] = calculate_custom_indicator(df[f"{symbol}_close"])
    return df
```

### Custom Strategy
```python
# Extend long_short_strategy.py
class CustomStrategy(LongShortStrategy):
    def generate_signals(self, predictions, prices):
        # Implement your custom signal generation logic
        pass
```

### Additional Data Sources
```python
# Extend data_fetcher.py
async def fetch_custom_data_source(self, symbol: str):
    # Add integration with new data provider
    pass
```

## 📚 Dependencies

Core dependencies:
- pandas, numpy, scikit-learn
- aiohttp (async data fetching)
- optuna (hyperparameter tuning)
- empyrical (performance metrics)
- matplotlib, seaborn (visualization)
- plotly (interactive charts, optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Projects

This system complements existing reinforcement learning trading infrastructure and can be integrated with:
- Deep Q-Learning (DQN) trading agents
- OANDA API for forex trading
- Real-time streaming data systems

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check existing documentation
- Review configuration examples

---

**Disclaimer**: This software is provided for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always conduct thorough testing before considering live trading.