# Risk Management Context

## Overview
Advanced institutional-grade risk management system implementing Kelly criterion optimization, CVaR tail risk management, and real-time position monitoring. Designed to maintain < 15% maximum drawdown while optimizing risk-adjusted returns in volatile cryptocurrency markets.

## Critical Files
- `phase2b/advanced_risk_management.py` - Kelly criterion + CVaR optimization engine
- `risk/position_manager.py` - Real-time position tracking and limits
- `risk/risk_monitor.py` - Continuous risk assessment and alerts
- `risk/drawdown_control.py` - Maximum drawdown prevention system
- `utils/risk_metrics.py` - Sharpe, Sortino, Calmar ratio calculations

## Key Classes and Functions

### AdvancedRiskManager
- **Purpose**: Central risk management orchestration with Kelly criterion optimization
- **Key Methods**:
  - `calculate_kelly_position_size()` - Optimal position sizing based on win rate and payoff
  - `assess_portfolio_risk()` - Real-time portfolio risk evaluation
  - `apply_cvar_optimization()` - 5% tail risk constraint optimization
  - `monitor_drawdown()` - Track and control maximum drawdown limits
- **Performance**: < 5ms risk calculations for real-time trading decisions

### PositionManager
- **Purpose**: Real-time position tracking and enforcement of position limits
- **Key Methods**:
  - `validate_position_size()` - Pre-trade position size validation
  - `monitor_position_exposure()` - Continuous exposure monitoring
  - `enforce_position_limits()` - Hard limits enforcement (50% max per position)
  - `calculate_margin_requirements()` - Dynamic margin calculation
- **Integration**: Connected to enhanced_paper_trader_24h.py for live enforcement

### CVaROptimizer
- **Purpose**: Tail risk management using Conditional Value at Risk optimization
- **Key Methods**:
  - `calculate_cvar()` - 5% tail risk measurement
  - `optimize_portfolio_weights()` - Risk-adjusted portfolio optimization
  - `stress_test_scenarios()` - Extreme market condition testing
  - `adjust_risk_targets()` - Dynamic risk target adjustment

## Risk Parameters & Thresholds

### Position Limits
- **Maximum Position Size**: 50% of total portfolio value
- **Maximum Single Asset Exposure**: 95% (BTC focus for current system)
- **Minimum Cash Reserve**: 5% for operational flexibility
- **Leverage Limit**: 1x (no leverage for paper trading phase)

### Stop Loss & Take Profit
- **Dynamic Stop Loss**: 2% base + volatility adjustment (ATR-based)
- **Take Profit Target**: 5% base + momentum adjustment
- **Trailing Stop**: 1.5% from peak portfolio value
- **Maximum Hold Period**: 24 hours (prevent overnight risk)

### Kelly Criterion Implementation
- **Fractional Kelly**: 25% of full Kelly (reduces volatility)
- **Win Rate Calculation**: Rolling 100-trade window
- **Payoff Ratio**: Average win / Average loss
- **Minimum Sample Size**: 50 trades before Kelly activation
- **Kelly Floor/Ceiling**: 1% minimum, 50% maximum position size

### CVaR Risk Management
- **Confidence Level**: 95% (5% tail risk focus)
- **Lookback Period**: 252 trading days (1 year)
- **Risk Budget**: 2% VaR per trade
- **Portfolio CVaR Limit**: 15% maximum expected tail loss
- **Stress Test Scenarios**: 2008, 2020, May 2022 crypto crashes

## Current Live Risk Status
- **Active Capital**: $100,000 paper trading
- **Current Position**: 0 BTC (no active positions)
- **Portfolio Drawdown**: 0% (conservative thresholds working)
- **Risk Utilization**: 0% (waiting for momentum > 1.78%/hr)
- **Kelly Signal**: Conservative (insufficient sample size)
- **CVaR Status**: Well within limits

## Integration Points
- **Signal Validation**: All trading signals validated against risk limits
- **Position Sizing**: Kelly criterion feeds into order execution
- **Real-time Monitoring**: Connected to enhanced_paper_trader_24h.py
- **Performance Tracking**: Risk metrics logged in analytics/
- **Emergency Controls**: Automatic position liquidation on breach

## Risk Metrics & Monitoring

### Real-time Metrics
- **Portfolio Beta**: Correlation with market movements
- **Sharpe Ratio**: Risk-adjusted return measurement (target: > 1.5)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline (limit: 15%)
- **Value at Risk (VaR)**: 95% confidence daily loss estimate
- **Conditional VaR**: Expected loss in worst 5% scenarios

### Performance Benchmarks
- **Current Sharpe**: 1.8+ target (risk-adjusted excellence)
- **Win Rate**: 63.6% historical (used in Kelly calculation)
- **Average Win**: $X vs Average Loss: $Y (payoff ratio)
- **Profit Factor**: 1.5+ target (gross profit / gross loss)
- **Calmar Ratio**: Return / Maximum Drawdown (target: > 3)

## Risk Control Mechanisms

### Pre-Trade Controls
```python
def validate_trade_risk(signal, portfolio_state):
    # 1. Position size validation
    max_position = min(kelly_size, position_limit, cvar_limit)
    
    # 2. Correlation check
    if portfolio_correlation > 0.8:
        reduce_position_size()
    
    # 3. Volatility adjustment
    if current_volatility > historical_avg * 2:
        apply_volatility_discount()
    
    return approved_position_size
```

### Post-Trade Monitoring
- **Real-time P&L tracking**: Every 5 minutes during market hours
- **Stop loss monitoring**: Automatic execution on breach
- **Correlation monitoring**: Portfolio concentration risk
- **Liquidity monitoring**: Ensure position exit capability

### Emergency Protocols
- **Circuit Breaker**: Halt trading on 5% portfolio loss in 1 hour
- **Force Liquidation**: Automatic position closure on 10% drawdown
- **Manual Override**: Human intervention capability for all controls
- **Risk Alert System**: Immediate notifications on threshold breaches

## Testing & Validation

### Backtesting Risk Controls
- **Historical Stress Testing**: 2017-2024 crypto market data
- **Monte Carlo Simulation**: 10,000 scenarios for portfolio outcomes
- **Walk-Forward Validation**: 51 windows with risk parameter stability
- **Extreme Event Testing**: Black swan event simulation

### Live Validation
- **Paper Trading**: Current 24-hour live validation
- **Risk Metric Validation**: Real-time vs expected risk measurements
- **Model Performance**: Kelly criterion accuracy tracking
- **Alert System Testing**: Verify all risk alerts trigger correctly

## Configuration & Parameters

### Environment Variables
- `RISK_MAX_POSITION_PCT` = 0.50 (50% maximum position)
- `RISK_STOP_LOSS_PCT` = 0.02 (2% stop loss)
- `RISK_KELLY_FRACTION` = 0.25 (25% fractional Kelly)
- `RISK_CVAR_CONFIDENCE` = 0.95 (95% confidence level)
- `RISK_MAX_DRAWDOWN_PCT` = 0.15 (15% maximum drawdown)

### Risk Model Updates
- **Parameter Recalibration**: Weekly updates for Kelly parameters
- **Volatility Model Updates**: Daily ATR recalculation
- **Correlation Updates**: Real-time correlation matrix updates
- **Risk Limit Reviews**: Monthly limit effectiveness review

## Known Issues & Solutions
- **Kelly Sensitivity**: Small sample bias in win rate calculation
  - Solution: 100-trade minimum before full Kelly activation
- **Volatility Clustering**: Risk underestimation during calm periods
  - Solution: GARCH model integration (Phase 2C)
- **Tail Risk Events**: CVaR underestimation for crypto markets
  - Solution: Stress testing with crypto-specific scenarios
- **Regime Changes**: Risk parameters lag market regime shifts
  - Solution: HMM regime detection integration

## Future Risk Enhancements
- [ ] GARCH volatility modeling for better risk estimation
- [ ] Multi-asset correlation risk management
- [ ] Options-based portfolio hedging strategies
- [ ] Machine learning-based risk factor identification
- [ ] Real-time market microstructure risk assessment

## Emergency Contacts & Procedures
- **Risk Breach Alert**: Check logs/enhanced_24hr_trading/ for details
- **System Override**: Manual intervention via config parameter updates
- **Emergency Stop**: Ctrl+C on enhanced_paper_trader_24h.py
- **Risk Review**: Weekly risk metric analysis and parameter adjustment

## Risk Philosophy
The system prioritizes capital preservation over profit maximization. Conservative risk parameters ensure long-term sustainability while Kelly criterion optimization provides mathematical foundation for position sizing. CVaR constraints protect against tail events that could cause catastrophic losses in volatile cryptocurrency markets.