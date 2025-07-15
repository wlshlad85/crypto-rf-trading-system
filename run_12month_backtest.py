#!/usr/bin/env python3
"""
12-Month Backtest with Current Trading Strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_backtest():
    print('🚀 Running 12-Month Backtest with Current Strategy')
    print('=' * 60)

    # Fetch BTC data for last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f'📅 Backtesting period: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')

    try:
        # Get BTC data
        btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='4h')
        print(f'✅ Downloaded {len(btc)} 4-hour candles')
        
        # Prepare features (simplified version of our strategy)
        btc['returns'] = btc['Close'].pct_change()
        btc['momentum_1'] = btc['returns'].rolling(1).mean() * 100
        btc['momentum_4'] = btc['returns'].rolling(4).mean() * 100
        btc['rsi'] = calculate_rsi(btc['Close'])
        btc['volume_ratio'] = btc['Volume'] / btc['Volume'].rolling(24).mean()
        btc['volatility'] = btc['returns'].rolling(24).std() * 100
        btc['hour'] = btc.index.hour
        btc['day_of_week'] = btc.index.dayofweek
        
        # Create target based on our strategy rules
        btc['price_change_4h'] = btc['Close'].pct_change(4) * 100
        btc['target'] = (btc['price_change_4h'] > 1.78).astype(int)  # 1.78% threshold from pattern analysis
        
        # Feature engineering matching our current strategy
        btc['is_optimal_hour'] = (btc['hour'] == 3).astype(int)  # 3 AM preference
        btc['is_high_momentum'] = (btc['momentum_1'] > 1.78).astype(int)
        btc['macd'] = btc['Close'].ewm(span=12).mean() - btc['Close'].ewm(span=26).mean()
        btc['success_probability'] = btc['target'].rolling(50).mean()
        
        # Clean data
        btc = btc.dropna()
        
        if len(btc) < 100:
            print('❌ Insufficient data for backtesting')
            return
            
        features = ['momentum_1', 'momentum_4', 'rsi', 'volume_ratio', 'volatility', 
                    'hour', 'day_of_week', 'is_optimal_hour', 'is_high_momentum', 
                    'macd', 'success_probability']
        
        # Split data (first 80% for training, last 20% for testing)
        split_idx = int(len(btc) * 0.8)
        train_data = btc.iloc[:split_idx]
        test_data = btc.iloc[split_idx:]
        
        print(f'📊 Training samples: {len(train_data)}')
        print(f'📊 Testing samples: {len(test_data)}')
        
        # Train Random Forest (simplified version)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(train_data[features])
        y_train = train_data['target']
        
        rf_model.fit(X_train, y_train)
        
        # Backtest simulation
        initial_capital = 100000
        capital = initial_capital
        position = 0
        trades = []
        portfolio_values = []
        
        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            current_price = row['Close']
            
            # Get prediction
            X_current = scaler.transform([row[features]])
            signal_prob = rf_model.predict_proba(X_current)[0][1]
            
            # Trading logic (matching our current strategy)
            momentum_threshold = 1.78
            confidence_threshold = 0.7
            
            if signal_prob > confidence_threshold and row['momentum_1'] > momentum_threshold and position == 0:
                # BUY signal
                position = capital * 0.8 / current_price  # 80% position size
                capital = capital * 0.2  # Keep 20% cash
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': position,
                    'value': position * current_price
                })
            elif signal_prob < 0.3 and position > 0:
                # SELL signal
                sell_value = position * current_price
                capital += sell_value
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': position,
                    'value': sell_value
                })
                position = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (position * current_price)
            portfolio_values.append(portfolio_value)
        
        # Final portfolio value
        final_price = test_data['Close'].iloc[-1]
        final_portfolio_value = capital + (position * final_price)
        
        # Calculate performance metrics
        total_return = (final_portfolio_value - initial_capital) / initial_capital * 100
        
        # Calculate maximum drawdown
        peak = initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trading statistics
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        profitable_trades = 0
        total_profit = 0
        
        for buy, sell in zip(buy_trades, sell_trades):
            profit = (sell['price'] - buy['price']) * buy['quantity']
            total_profit += profit
            if profit > 0:
                profitable_trades += 1
        
        win_rate = (profitable_trades / len(buy_trades) * 100) if buy_trades else 0
        
        # Annualized metrics
        days_traded = (test_data.index[-1] - test_data.index[0]).days
        annualized_return = (final_portfolio_value / initial_capital) ** (365 / days_traded) - 1
        
        print('')
        print('📈 12-MONTH BACKTEST RESULTS')
        print('=' * 60)
        print(f'💰 Initial Capital: ${initial_capital:,.2f}')
        print(f'💰 Final Portfolio Value: ${final_portfolio_value:,.2f}')
        print(f'📊 Total Return: {total_return:.2f}%')
        print(f'📊 Annualized Return: {annualized_return * 100:.2f}%')
        print(f'📉 Maximum Drawdown: {max_drawdown:.2f}%')
        print(f'📝 Total Trades: {len(trades)}')
        print(f'📈 Win Rate: {win_rate:.1f}%')
        print(f'🎯 Strategy Accuracy: {rf_model.score(scaler.transform(test_data[features]), test_data["target"]) * 100:.1f}%')
        print('')
        
        # Compare to buy-and-hold
        buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
        print(f'📊 Buy & Hold Return: {buy_hold_return:.2f}%')
        print(f'🚀 Strategy Outperformance: {total_return - buy_hold_return:.2f}%')
        
        # Monthly breakdown
        print('')
        print('📅 MONTHLY PERFORMANCE BREAKDOWN')
        print('-' * 40)
        
        # Group by month and calculate returns
        test_data['portfolio_value'] = portfolio_values
        monthly_returns = test_data.groupby(test_data.index.to_period('M'))['portfolio_value'].agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100
        
        for month, row in monthly_returns.iterrows():
            print(f'{month}: {row["return"]:+.2f}%')
        
        print('')
        print('🎯 STRATEGY EFFECTIVENESS')
        print('-' * 40)
        if total_return > buy_hold_return:
            print('✅ Strategy OUTPERFORMED buy-and-hold')
        else:
            print('❌ Strategy UNDERPERFORMED buy-and-hold')
        
        if win_rate > 50:
            print(f'✅ Positive win rate: {win_rate:.1f}%')
        else:
            print(f'❌ Low win rate: {win_rate:.1f}%')
            
        if max_drawdown < 20:
            print(f'✅ Acceptable drawdown: {max_drawdown:.1f}%')
        else:
            print(f'⚠️ High drawdown: {max_drawdown:.1f}%')

    except Exception as e:
        print(f'❌ Error in backtest: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_backtest()