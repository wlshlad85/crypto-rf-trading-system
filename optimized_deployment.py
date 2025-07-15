#!/usr/bin/env python3
"""
Optimized Model Deployment for 24-Hour Trading

Deploys the enhanced Random Forest models with optimized parameters
based on the successful pattern analysis from previous sessions.
"""

import os
import sys
import json
import time
import signal
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class OptimizedTradingDeployment:
    """Deploy optimized trading models for 24-hour session."""
    
    def __init__(self):
        """Initialize with optimized configuration."""
        self.initial_capital = 100000.0
        self.cash = 100000.0
        self.btc_position = 0.0
        self.trades = []
        self.portfolio_values = [100000.0]
        
        # Optimized parameters based on pattern analysis
        self.config = {
            'momentum_threshold': 1.780,  # Optimal from 79-trade analysis
            'position_size_min': 0.464,   # Proven range
            'position_size_max': 0.800,   # Maximum successful position
            'success_probability_threshold': 0.636,  # 63.6% historical success
            'confidence_threshold': 0.65,  # Enhanced confidence
            'exit_threshold': 0.55,        # Improved exit timing
            'stop_loss_pct': 0.02,         # 2% stop loss
            'take_profit_pct': 0.05,       # 5% take profit
            'trailing_stop_pct': 0.015,    # 1.5% trailing stop
        }
        
        # Enhanced model parameters
        self.model_config = {
            'entry_model': {
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'random_state': 42
            },
            'position_model': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            },
            'exit_model': {
                'n_estimators': 120,
                'max_depth': 8,
                'min_samples_split': 12,
                'min_samples_leaf': 6,
                'max_features': 'sqrt',
                'random_state': 42
            },
            'profit_model': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 6,
                'min_samples_leaf': 3,
                'max_features': 'log2',
                'random_state': 42
            }
        }
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.running = False
        
        # Setup logging
        self.log_dir = "logs/optimized_24hr_trading"
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'optimized_btc_24hr_{timestamp}.log')
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🚀 Optimized 24-Hour BTC Trading System Initialized")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Target Return: 4-6% (Enhanced from 2.82% baseline)")
        print(f"📝 Log file: {self.log_file}")
    
    def _log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        log_entry = f"{timestamp} {message}"
        print(log_entry)
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
                f.flush()
        except Exception as e:
            print(f"❌ Logging error: {e}")
    
    def train_optimized_models(self):
        """Train enhanced models with optimized parameters."""
        self._log("🔄 Training optimized Random Forest models...")
        
        try:
            # Load training data
            data_file = "data/4h_training/crypto_4h_dataset_20250714_130201.csv"
            if not os.path.exists(data_file):
                self._log("⚠️ Training data not found, using simplified models")
                return self._create_simplified_models()
            
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            self._log(f"📊 Loaded {len(data)} training records")
            
            # Prepare features
            data = self._prepare_enhanced_features(data)
            
            # Clean data
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split training data
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            
            # Train models
            self._train_entry_model(train_data)
            self._train_position_model(train_data)
            self._train_exit_model(train_data)
            self._train_profit_model(train_data)
            
            self._log("✅ All optimized models trained successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Model training error: {e}")
            return self._create_simplified_models()
    
    def _prepare_enhanced_features(self, data):
        """Prepare enhanced features for model training."""
        
        # Basic momentum features
        data['momentum_1'] = data['close'].pct_change(1) * 100
        data['momentum_4'] = data['close'].pct_change(4) * 100
        data['momentum_12'] = data['close'].pct_change(12) * 100
        
        # Enhanced momentum strength (hourly rate)
        data['momentum_strength'] = data['momentum_1'] / 4  # Convert to hourly
        data['is_high_momentum'] = (data['momentum_strength'] > self.config['momentum_threshold']).astype(int)
        
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
        
        # Volume features
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['volume_trend'] = data['volume'].pct_change(5) * 100
        
        # Market structure
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['higher_low'] = (data['low'] > data['low'].shift(1)).astype(int)
        data['market_structure'] = data['higher_high'] + data['higher_low'] - 1
        
        # Volatility and ATR
        data['volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        data['atr'] = true_range.rolling(14).mean()
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_optimal_hour'] = (data['hour'] == 3).astype(int)  # 3 AM optimal time
        
        # Trend features
        sma_10 = data['close'].rolling(10).mean()
        sma_20 = data['close'].rolling(20).mean()
        data['trend_strength'] = np.abs(sma_10 - sma_20) / data['close']
        data['price_vs_sma10'] = (data['close'] - sma_10) / sma_10
        data['price_vs_sma20'] = (data['close'] - sma_20) / sma_20
        
        # Pattern recognition features
        data['success_score'] = (
            data['is_high_momentum'] * 3 +
            (data['market_structure'] >= 0).astype(int) * 2 +
            data['is_optimal_hour'] * 1 +
            (data['volume_ratio'] > 1.0).astype(int) * 1 +
            (data['rsi'] < 70).astype(int) * 1
        )
        
        data['success_probability'] = np.clip(data['success_score'] / 8.0 * self.config['success_probability_threshold'], 0.0, 1.0)
        
        # Target variables for training
        for horizon in [1, 2, 4]:
            data[f'price_up_{horizon}h'] = (data['close'].shift(-horizon) > data['close']).astype(int)
            data[f'price_change_{horizon}h'] = (data['close'].shift(-horizon) / data['close'] - 1) * 100
            data[f'profitable_{horizon}h'] = (data[f'price_change_{horizon}h'] > 0.68).astype(int)
        
        # Position sizing targets
        data['position_size_signal'] = np.where(
            data['is_high_momentum'] == 1,
            self.config['position_size_max'],
            np.where(data['momentum_strength'] > 0.5, 
                    (self.config['position_size_min'] + self.config['position_size_max']) / 2, 
                    self.config['position_size_min'])
        )
        
        return data
    
    def _train_entry_model(self, train_data):
        """Train entry signal model."""
        entry_features = [
            'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
            'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
            'hour', 'day_of_week', 'volatility', 'trend_strength', 'success_probability'
        ]
        
        X_entry = train_data[entry_features].dropna()
        y_entry = train_data.loc[X_entry.index, 'price_up_2h']
        
        if len(X_entry) > 100:
            scaler = StandardScaler()
            X_entry_scaled = scaler.fit_transform(X_entry)
            
            model = RandomForestClassifier(**self.model_config['entry_model'])
            model.fit(X_entry_scaled, y_entry)
            
            self.models['entry'] = model
            self.scalers['entry'] = scaler
            
            accuracy = model.score(X_entry_scaled, y_entry)
            self._log(f"✅ Entry model trained - Accuracy: {accuracy:.1%}")
    
    def _train_position_model(self, train_data):
        """Train position sizing model."""
        position_features = [
            'momentum_strength', 'is_high_momentum', 'volatility', 'atr',
            'rsi', 'market_structure', 'success_probability', 'volume_ratio'
        ]
        
        X_position = train_data[position_features].dropna()
        y_position = train_data.loc[X_position.index, 'position_size_signal']
        
        if len(X_position) > 100:
            scaler = StandardScaler()
            X_position_scaled = scaler.fit_transform(X_position)
            
            model = RandomForestRegressor(**self.model_config['position_model'])
            model.fit(X_position_scaled, y_position)
            
            self.models['position'] = model
            self.scalers['position'] = scaler
            
            score = model.score(X_position_scaled, y_position)
            self._log(f"✅ Position model trained - R²: {score:.3f}")
    
    def _train_exit_model(self, train_data):
        """Train exit signal model."""
        exit_features = [
            'momentum_1', 'rsi', 'bb_position', 'market_structure',
            'volatility', 'volume_ratio', 'price_vs_sma10', 'macd'
        ]
        
        # Create exit signals
        exit_signals = (
            (train_data['rsi'] > 70) |
            (train_data['momentum_strength'] < 0) |
            (train_data['bb_position'] > 0.8)
        ).astype(int)
        
        X_exit = train_data[exit_features].dropna()
        y_exit = exit_signals.loc[X_exit.index]
        
        if len(X_exit) > 100:
            scaler = StandardScaler()
            X_exit_scaled = scaler.fit_transform(X_exit)
            
            model = RandomForestClassifier(**self.model_config['exit_model'])
            model.fit(X_exit_scaled, y_exit)
            
            self.models['exit'] = model
            self.scalers['exit'] = scaler
            
            accuracy = model.score(X_exit_scaled, y_exit)
            self._log(f"✅ Exit model trained - Accuracy: {accuracy:.1%}")
    
    def _train_profit_model(self, train_data):
        """Train profit prediction model."""
        profit_features = [
            'momentum_1', 'momentum_4', 'rsi', 'macd', 'bb_position',
            'market_structure', 'volume_ratio', 'volatility', 'success_probability'
        ]
        
        X_profit = train_data[profit_features].dropna()
        y_profit = train_data.loc[X_profit.index, 'profitable_2h']
        
        if len(X_profit) > 100:
            scaler = StandardScaler()
            X_profit_scaled = scaler.fit_transform(X_profit)
            
            model = RandomForestClassifier(**self.model_config['profit_model'])
            model.fit(X_profit_scaled, y_profit)
            
            self.models['profit'] = model
            self.scalers['profit'] = scaler
            
            accuracy = model.score(X_profit_scaled, y_profit)
            self._log(f"✅ Profit model trained - Accuracy: {accuracy:.1%}")
    
    def _create_simplified_models(self):
        """Create simplified models if training data unavailable."""
        self._log("🔄 Creating simplified rule-based models...")
        
        # Simple rule-based models
        self.models['simplified'] = True
        
        self._log("✅ Simplified models ready")
        return True
    
    def get_current_data(self):
        """Get current market data with indicators."""
        try:
            # Fetch recent data
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period="5d", interval="1h")
            
            if data.empty:
                return None
            
            # Prepare features
            data = data.rename(columns={col.lower(): col.lower() for col in data.columns})
            data = self._prepare_enhanced_features(data)
            
            return data.iloc[-1]  # Latest row
            
        except Exception as e:
            self._log(f"❌ Data fetch error: {e}")
            return None
    
    def generate_signals(self, current_data):
        """Generate trading signals using optimized models."""
        
        if current_data is None:
            return {'should_enter': False, 'should_exit': False, 'position_size': 0.5}
        
        signals = {}
        
        try:
            # Entry signal
            if 'entry' in self.models and not self.models.get('simplified'):
                entry_features = [
                    'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
                    'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
                    'hour', 'day_of_week', 'volatility', 'trend_strength', 'success_probability'
                ]
                
                entry_data = [[current_data[f] for f in entry_features]]
                entry_scaled = self.scalers['entry'].transform(entry_data)
                entry_prob = self.models['entry'].predict_proba(entry_scaled)[0]
                signals['entry_prob'] = entry_prob[1] if len(entry_prob) > 1 else entry_prob[0]
                signals['should_enter'] = signals['entry_prob'] > self.config['confidence_threshold']
            else:
                # Simplified entry logic
                momentum_signal = current_data.get('momentum_strength', 0) > self.config['momentum_threshold']
                rsi_signal = current_data.get('rsi', 50) < 70
                volume_signal = current_data.get('volume_ratio', 1) > 1.0
                
                signals['should_enter'] = momentum_signal and rsi_signal and volume_signal
                signals['entry_prob'] = 0.7 if signals['should_enter'] else 0.3
            
            # Position sizing
            if 'position' in self.models and not self.models.get('simplified'):
                position_features = [
                    'momentum_strength', 'is_high_momentum', 'volatility', 'atr',
                    'rsi', 'market_structure', 'success_probability', 'volume_ratio'
                ]
                
                position_data = [[current_data[f] for f in position_features]]
                position_scaled = self.scalers['position'].transform(position_data)
                position_size = self.models['position'].predict(position_scaled)[0]
                signals['position_size'] = np.clip(position_size, 
                                                  self.config['position_size_min'], 
                                                  self.config['position_size_max'])
            else:
                # Simplified position sizing
                if current_data.get('is_high_momentum', 0):
                    signals['position_size'] = self.config['position_size_max']
                elif current_data.get('momentum_strength', 0) > 0.5:
                    signals['position_size'] = (self.config['position_size_min'] + self.config['position_size_max']) / 2
                else:
                    signals['position_size'] = self.config['position_size_min']
            
            # Exit signal
            if 'exit' in self.models and not self.models.get('simplified'):
                exit_features = [
                    'momentum_1', 'rsi', 'bb_position', 'market_structure',
                    'volatility', 'volume_ratio', 'price_vs_sma10', 'macd'
                ]
                
                exit_data = [[current_data[f] for f in exit_features]]
                exit_scaled = self.scalers['exit'].transform(exit_data)
                exit_prob = self.models['exit'].predict_proba(exit_scaled)[0]
                signals['exit_prob'] = exit_prob[1] if len(exit_prob) > 1 else exit_prob[0]
                signals['should_exit'] = signals['exit_prob'] > self.config['exit_threshold']
            else:
                # Simplified exit logic
                rsi_exit = current_data.get('rsi', 50) > 70
                momentum_exit = current_data.get('momentum_strength', 0) < 0
                bb_exit = current_data.get('bb_position', 0.5) > 0.8
                
                signals['should_exit'] = rsi_exit or momentum_exit or bb_exit
                signals['exit_prob'] = 0.6 if signals['should_exit'] else 0.4
            
            return signals
            
        except Exception as e:
            self._log(f"❌ Signal generation error: {e}")
            return {'should_enter': False, 'should_exit': False, 'position_size': 0.5}
    
    def execute_trade(self, action, current_price, quantity=None):
        """Execute a trade."""
        
        if action == 'BUY' and self.btc_position == 0 and self.cash > 1000:
            if quantity is None:
                # Calculate position size
                signals = self.generate_signals(self.get_current_data())
                position_fraction = signals.get('position_size', 0.5)
                target_value = self.get_portfolio_value() * position_fraction
                quantity = min(target_value, self.cash * 0.99) / current_price
            
            if quantity > 0.001:
                cost = quantity * current_price
                self.cash -= cost
                self.btc_position += quantity
                
                trade = {
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'value': cost,
                    'pnl': 0
                }
                self.trades.append(trade)
                
                self._log(f"🟢 BUY {quantity:.6f} BTC @ ${current_price:,.2f} (${cost:,.2f}) - OPTIMIZED")
                return trade
        
        elif action == 'SELL' and self.btc_position > 0.001:
            proceeds = self.btc_position * current_price
            
            # Calculate P&L
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            total_cost = sum(t['value'] for t in buy_trades)
            pnl = proceeds - total_cost
            
            self.cash += proceeds
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'quantity': self.btc_position,
                'price': current_price,
                'value': proceeds,
                'pnl': pnl
            }
            self.trades.append(trade)
            
            self._log(f"🔴 SELL {self.btc_position:.6f} BTC @ ${current_price:,.2f} (${proceeds:,.2f}) - P&L: ${pnl:+,.2f}")
            
            self.btc_position = 0
            return trade
        
        return None
    
    def get_portfolio_value(self):
        """Get current portfolio value."""
        try:
            if self.btc_position > 0:
                btc = yf.Ticker("BTC-USD")
                current_price = btc.history(period="1d", interval="1m")['Close'].iloc[-1]
                btc_value = self.btc_position * current_price
            else:
                btc_value = 0
                
            return self.cash + btc_value
        except:
            return self.cash
    
    def run_24h_session(self):
        """Run 24-hour optimized trading session."""
        
        # Train models first
        if not self.train_optimized_models():
            self._log("❌ Model training failed, exiting")
            return
        
        session_start = datetime.now()
        session_end = session_start + timedelta(hours=24)
        
        self._log(f"🚀 24-Hour Optimized Trading Session Started")
        self._log(f"🕐 Session will run until: {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"⏱️ Update interval: 300 seconds")
        self._log(f"🎯 Target Return: 4-6% (Enhanced models)")
        self._log("Press Ctrl+C to stop early and generate report")
        
        self.running = True
        update_interval = 300  # 5 minutes
        last_update = datetime.now()
        hour_counter = 0
        
        try:
            while self.running and datetime.now() < session_end:
                current_time = datetime.now()
                
                # Update every 5 minutes
                if (current_time - last_update).total_seconds() >= update_interval:
                    
                    # Get current market data
                    current_data = self.get_current_data()
                    
                    if current_data is not None:
                        current_price = current_data.get('close', 0)
                        
                        if current_price > 0:
                            # Generate signals
                            signals = self.generate_signals(current_data)
                            
                            # Execute trades based on signals
                            if signals['should_enter'] and self.btc_position == 0:
                                self.execute_trade('BUY', current_price)
                            elif signals['should_exit'] and self.btc_position > 0:
                                self.execute_trade('SELL', current_price)
                            
                            # Update portfolio value
                            portfolio_value = self.get_portfolio_value()
                            self.portfolio_values.append(portfolio_value)
                    
                    last_update = current_time
                    
                    # Hourly status report
                    hours_elapsed = (current_time - session_start).total_seconds() / 3600
                    if hours_elapsed >= hour_counter + 1:
                        hour_counter = int(hours_elapsed)
                        self._generate_hourly_report(hour_counter, session_start, session_end)
                    
                    # Log next update time
                    hours_remaining = (session_end - current_time).total_seconds() / 3600
                    self._log(f"⏳ Next update in {update_interval}s... ({hours_remaining:.1f} hours remaining)")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self._log("🛑 Manual stop requested")
            self.running = False
        
        except Exception as e:
            self._log(f"❌ Trading session error: {e}")
        
        finally:
            self._generate_final_report()
    
    def _generate_hourly_report(self, hour, session_start, session_end):
        """Generate hourly status report."""
        current_time = datetime.now()
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        self._log("")
        self._log(f"📊 PORTFOLIO STATUS - Hour {hour:.1f}/24")
        self._log(f"💰 Total Value: ${portfolio_value:,.2f}")
        self._log(f"📈 Return: {total_return:+.2f}%")
        self._log(f"💵 Cash: ${self.cash:,.2f}")
        
        if self.btc_position > 0:
            try:
                btc = yf.Ticker("BTC-USD")
                current_price = btc.history(period="1d", interval="1m")['Close'].iloc[-1]
                btc_value = self.btc_position * current_price
                
                # Calculate position P&L
                buy_trades = [t for t in self.trades if t['action'] == 'BUY']
                if buy_trades:
                    avg_buy_price = sum(t['value'] for t in buy_trades) / sum(t['quantity'] for t in buy_trades)
                    position_pnl = (current_price - avg_buy_price) / avg_buy_price * 100
                else:
                    position_pnl = 0
                
                self._log(f"₿ BTC Position: {self.btc_position:.6f} BTC (${btc_value:,.2f})")
                self._log(f"💲 BTC Price: ${current_price:,.2f}")
                self._log(f"📊 Position P&L: {position_pnl:+.2f}%")
            except:
                self._log(f"₿ BTC Position: {self.btc_position:.6f} BTC")
        else:
            self._log("₿ BTC Position: 0.000000 BTC ($0.00)")
        
        self._log(f"📝 Total Trades: {len(self.trades)}")
        self._log("-" * 60)
        
        # Save session data
        self._save_session_data()
    
    def _save_session_data(self):
        """Save current session data."""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'btc_position': self.btc_position,
            'portfolio_value': self.get_portfolio_value(),
            'total_trades': len(self.trades),
            'trades': [
                {
                    'timestamp': t['timestamp'].isoformat(),
                    'action': t['action'],
                    'quantity': t['quantity'],
                    'price': t['price'],
                    'value': t['value'],
                    'pnl': t.get('pnl', 0)
                } for t in self.trades
            ],
            'portfolio_values': self.portfolio_values
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_file = f"optimized_session_{timestamp}.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self._log(f"💾 Session data saved to: {session_file}")
        except Exception as e:
            self._log(f"❌ Error saving session data: {e}")
    
    def _generate_final_report(self):
        """Generate final session report."""
        self._log("")
        self._log("🏁 24-HOUR OPTIMIZED TRADING SESSION COMPLETE")
        self._log("=" * 60)
        
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        self._log(f"💰 Final Portfolio Value: ${final_value:,.2f}")
        self._log(f"📈 Total Return: {total_return:+.2f}%")
        self._log(f"📝 Total Trades: {len(self.trades)}")
        
        if self.trades:
            profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            win_rate = len(profitable_trades) / len([t for t in self.trades if 'pnl' in t]) * 100 if any('pnl' in t for t in self.trades) else 0
            self._log(f"📊 Win Rate: {win_rate:.1f}%")
        
        # Compare to baseline
        baseline_return = 2.82
        improvement = total_return - baseline_return
        self._log(f"🎯 vs Baseline (2.82%): {improvement:+.2f}% improvement")
        
        # Save final session data
        self._save_session_data()
        
        self._log("✅ Session completed successfully!")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self._log(f"🛑 Received signal {signum} - stopping gracefully...")
        self.running = False

def main():
    """Main entry point."""
    trader = OptimizedTradingDeployment()
    trader.run_24h_session()

if __name__ == "__main__":
    main()