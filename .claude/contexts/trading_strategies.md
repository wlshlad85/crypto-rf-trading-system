# Trading Strategies Context

## Overview
Implementation of 15+ algorithmic trading strategies with ML optimization for cryptocurrency markets. Focuses on momentum, mean reversion, and ensemble-based approaches with institutional-grade risk management integration.

## Critical Files
- `strategies/long_short_strategy.py` - Core long/short logic with position management
- `strategies/minute_trading_strategies.py` - High-frequency trading strategies
- `enhanced_rf_ensemble.py` - ML ensemble prediction and signal generation
- `ultrathink/strategy_selector.py` - Adaptive strategy selection framework
- `ultrathink/decision_framework.py` - Real-time decision making engine

## Strategy Categories

### 1. Momentum-Based Strategies
- **Trend Following**: Captures sustained price movements
- **Breakout Detection**: Identifies key resistance/support breaks
- **Momentum Threshold**: 1.78%/hr for trade execution (currently active)
- **Risk Controls**: Dynamic stop-loss based on volatility

### 2. Mean Reversion Strategies  
- **Statistical Arbitrage**: Exploits price deviations from mean
- **Pairs Trading**: Relative value between correlated assets
- **Bollinger Band Reversals**: Overbought/oversold conditions
- **RSI-based Entries**: Contrarian signals at extremes

### 3. ML-Enhanced Strategies
- **Random Forest Ensemble**: 4-model voting system (entry/position/exit/profit)
- **HMM Regime Detection**: Strategy adaptation based on market regimes
- **Feature-based Signals**: 78+ engineered features for decision making
- **Confidence Scoring**: Probabilistic signal strength assessment

### 4. Market Making Strategies
- **Bid-Ask Spread Capture**: Liquidity provision strategies
- **Order Flow Analysis**: Volume profile and imbalance detection
- **Microstructure Exploitation**: Sub-second trading opportunities

## Critical Strategy Modules

### LongShortStrategy
- **Purpose**: Core directional trading with risk management
- **Key Methods**:
  - `generate_signals()` - Return buy/sell/hold signals with confidence
  - `calculate_position_size()` - Kelly criterion-based sizing
  - `manage_risk()` - Real-time risk monitoring and adjustments
  - `execute_trade()` - Order routing with slippage management
- **Performance**: Currently 52% accuracy with ensemble optimization

### EnhancedRFEnsemble
- **Purpose**: ML-based signal generation and prediction
- **Key Methods**:
  - `predict_entry_signals()` - Entry timing optimization
  - `predict_position_size()` - Dynamic position sizing
  - `predict_exit_signals()` - Profit-taking and stop-loss optimization
  - `get_ensemble_confidence()` - Aggregate confidence scoring
- **Models**: Random Forest + XGBoost + LightGBM + Neural Networks

### UltraThinkStrategySelector
- **Purpose**: Adaptive strategy selection based on market conditions
- **Key Methods**:
  - `analyze_market_regime()` - Classify current market state
  - `select_optimal_strategy()` - Choose best strategy for conditions
  - `blend_strategies()` - Weighted ensemble of multiple strategies
  - `monitor_performance()` - Real-time strategy performance tracking

## Strategy Parameters
- **Momentum Threshold**: 1.78% per hour (live trading parameter)
- **Position Limits**: Maximum 50% portfolio per position
- **Stop Loss**: Dynamic based on volatility (default -2%)
- **Take Profit**: Adaptive based on market conditions (default +5%)
- **Confidence Threshold**: Minimum 0.6 for signal execution
- **Risk Per Trade**: Maximum 2% of portfolio value

## Integration Points
- **Signal Generation**: Feeds into `execution/enhanced_paper_trader_24h.py`
- **Risk Validation**: Checked by `phase2b/advanced_risk_management.py`
- **Performance Tracking**: Monitored by `analytics/trading_pattern_analyzer.py`
- **Feature Input**: Receives data from `features/ultra_feature_engineering.py`
- **Regime Detection**: Adapted by `phase2b/hmm_regime_detection.py`

## Current Live Implementation
- **Active Strategy**: Enhanced RF Ensemble with momentum filtering
- **Signal Confidence**: 0.99 (SELL signals with high confidence)
- **Position Status**: 0 BTC (conservative thresholds preventing overtrading)
- **Momentum Filter**: -0.11%/hr (below 1.78%/hr threshold)
- **Risk Status**: All limits within acceptable ranges

## Strategy Framework Architecture

### Base Strategy Class
```python
class BaseStrategy:
    def generate_signals(self, market_data: pd.DataFrame) -> Dict:
        """Return standardized signal format"""
        pass
    
    def calculate_position_size(self, signal: Dict, portfolio: Dict) -> float:
        """Risk-adjusted position sizing"""
        pass
    
    def get_strategy_params(self) -> Dict:
        """Return current hyperparameters"""
        pass
```

### Signal Format Standard
```python
signal = {
    'action': 'BUY'|'SELL'|'HOLD',
    'confidence': 0.0-1.0,
    'strength': -1.0-1.0,
    'expected_return': float,
    'risk_score': 0.0-1.0,
    'holding_period': int,  # expected bars
    'stop_loss': float,
    'take_profit': float
}
```

## Available Indicators
- **Technical**: RSI, MACD, Bollinger Bands, Ichimoku Cloud
- **Volume**: Volume Profile, On-Balance Volume, Volume Weighted Average Price
- **Advanced**: Fractional differentiation, Multi-timeframe confluence
- **On-chain Proxies**: MVRV, NVT, Exchange flow simulation
- **Regime**: HMM state probabilities, Volatility clustering

## Performance Optimization
- **Vectorization**: All calculations use numpy/pandas vectorized operations
- **Caching**: Strategy parameters cached for repeated calculations
- **Parallel Processing**: Multiple strategies evaluated concurrently
- **Memory Management**: Rolling windows for real-time calculations

## Backtesting Pipeline
1. **Data Validation**: Enhanced data quality checks (99.5% threshold)
2. **Feature Engineering**: 78+ features across multiple timeframes
3. **Walk-Forward Testing**: 51 validation windows with purged CV
4. **Performance Analysis**: Sharpe, Sortino, Calmar ratios
5. **Risk Attribution**: Contribution analysis by strategy component

## Real-time Decision Flow
```
Market Data → Feature Engineering → Strategy Signals → 
Ensemble Voting → Risk Validation → Position Sizing → 
Order Execution → Performance Monitoring
```

## Testing Framework
- **Unit Tests**: Individual strategy component testing
- **Integration Tests**: End-to-end signal generation
- **Backtesting**: Historical performance validation
- **Paper Trading**: Live market testing with virtual capital
- **Stress Testing**: Extreme market condition scenarios

## Strategy Performance Metrics
- **Current Ensemble Accuracy**: 52% (target: 55%+)
- **Sharpe Ratio**: 1.8+ (risk-adjusted returns)
- **Maximum Drawdown**: < 15% (Kelly criterion optimization)
- **Win Rate**: 63.6% (from historical analysis)
- **Average Trade Duration**: 0.6-1.2 hours optimal

## Risk Integration
- **Pre-trade Validation**: All signals validated against risk limits
- **Real-time Monitoring**: Continuous position and exposure tracking
- **Automatic Stops**: Integrated with Kelly criterion position sizing
- **Regime Adaptation**: Strategy weights adjusted based on HMM states
- **Emergency Protocols**: Automatic strategy shutdown on severe losses

## Known Strategy Issues & Solutions
- **Momentum Lag**: Implement faster signal generation (< 1 minute)
- **False Breakouts**: Add volume confirmation filters
- **Regime Changes**: Faster adaptation to new market conditions
- **Correlation Risk**: Monitor cross-strategy correlation matrix
- **Overfitting**: Regular out-of-sample validation

## Future Strategy Development
- [ ] Implement reinforcement learning strategies
- [ ] Add news sentiment integration
- [ ] Develop multi-asset correlation strategies
- [ ] Create adaptive position sizing algorithms
- [ ] Build strategy tournament selection system