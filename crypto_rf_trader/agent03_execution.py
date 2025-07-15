#!/usr/bin/env python3
"""
Agent03: Execution Engine Developer

Builds live/paper trading system using ccxt (or yfinance fallback),
enforces position sizing, handles live order flow.
"""

import time
import os
import json
import pickle
import signal
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Try to import ccxt for exchange integration
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

COMS_PATH = "crypto_rf_trader/coms.md"
MODEL_PATH = "crypto_rf_trader/models/enhanced_rf_models.pkl"
FEATURES_PATH = "crypto_rf_trader/models/feature_config.json"

class Agent03ExecutionEngine:
    """Live/paper trading execution engine."""
    
    def __init__(self):
        self.log("Agent03 execution engine starting")
        
        # Trading state
        self.initial_capital = 100000.0
        self.cash = 100000.0
        self.btc_position = 0.0
        self.trades = []
        self.portfolio_values = [100000.0]
        
        # Models and configuration
        self.models = {}
        self.scalers = {}
        self.feature_config = {}
        
        # Trading parameters (will be updated from parameter suggestions)
        self.trading_params = {
            'momentum_threshold': 1.780,
            'confidence_threshold': 0.65,
            'position_size_min': 0.464,
            'position_size_max': 0.800,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_duration = 24  # hours
        self.update_interval = 300  # 5 minutes
        self.running = True
        
        # Load models and setup
        self.load_models()
        self.setup_exchange()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def log(self, message):
        """Log message to communications file."""
        timestamp = datetime.utcnow().isoformat()
        with open(COMS_PATH, "a") as f:
            f.write(f"[agent03][{timestamp}] {message}\n")
        print(f"[agent03][{timestamp}] {message}")
    
    def load_models(self):
        """Load trained models from Agent02."""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    model_data = pickle.load(f)
                
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                
                self.log(f"Models loaded: {list(self.models.keys())}")
                
                # Load feature configuration
                if os.path.exists(FEATURES_PATH):
                    with open(FEATURES_PATH, "r") as f:
                        self.feature_config = json.load(f)
                    self.log("Feature configuration loaded")
                
                return True
            else:
                self.log("No models found, will use rule-based trading")
                return False
                
        except Exception as e:
            self.log(f"Error loading models: {str(e)}")
            return False
    
    def setup_exchange(self):
        """Setup exchange connection (paper trading)."""
        if CCXT_AVAILABLE:
            try:
                # Use Binance testnet for paper trading
                self.exchange = ccxt.binance({
                    'apiKey': 'test_api_key',
                    'secret': 'test_secret',
                    'sandbox': True,
                    'enableRateLimit': True,
                })
                self.log("CCXT exchange configured (testnet)")
                return True
            except Exception as e:
                self.log(f"CCXT setup error: {str(e)}")
        
        self.log("Using YFinance for data, internal position tracking")
        self.exchange = None
        return True
    
    def get_market_data(self):
        """Get current market data with indicators."""
        try:
            # Fetch BTC data
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period="7d", interval="1h")
            
            if data.empty:
                return None
            
            # Calculate indicators
            data = self._calculate_indicators(data)
            
            # Return latest data point
            return data.iloc[-1]
            
        except Exception as e:
            self.log(f"Market data error: {str(e)}")
            return None
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators."""
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Basic momentum
        data['momentum_1'] = data['close'].pct_change(1) * 100
        data['momentum_4'] = data['close'].pct_change(4) * 100
        data['momentum_strength'] = data['momentum_1'] / 4  # Hourly rate
        data['is_high_momentum'] = (data['momentum_strength'] > self.trading_params['momentum_threshold']).astype(int)
        
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
        
        # Bollinger Bands
        bb_period = 20
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        bb_std = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume indicators
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Market structure
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['higher_low'] = (data['low'] > data['low'].shift(1)).astype(int)
        data['market_structure'] = data['higher_high'] + data['higher_low'] - 1
        
        # Volatility
        data['volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_optimal_hour'] = (data['hour'] == 3).astype(int)
        
        # Success scoring
        data['success_score'] = (
            data['is_high_momentum'] * 3 +
            (data['market_structure'] >= 0).astype(int) * 2 +
            data['is_optimal_hour'] * 1 +
            (data['volume_ratio'] > 1.0).astype(int) * 1 +
            (data['rsi'] < 70).astype(int) * 1
        )
        
        data['success_probability'] = np.clip(data['success_score'] / 8.0 * 0.636, 0.0, 1.0)
        
        return data
    
    def generate_trading_signals(self, market_data):
        """Generate trading signals using loaded models."""
        signals = {
            'should_enter': False,
            'should_exit': False,
            'position_size': 0.5,
            'confidence': 0.5
        }
        
        try:
            # Use ML models if available
            if 'entry' in self.models and self.feature_config:
                signals.update(self._generate_ml_signals(market_data))
            else:
                signals.update(self._generate_rule_based_signals(market_data))
            
            # Ensure position size is within bounds
            signals['position_size'] = np.clip(
                signals['position_size'],
                self.trading_params['position_size_min'],
                self.trading_params['position_size_max']
            )
            
            return signals
            
        except Exception as e:
            self.log(f"Signal generation error: {str(e)}")
            return signals
    
    def _generate_ml_signals(self, market_data):
        """Generate signals using ML models."""
        signals = {}
        
        # Entry signal
        if 'entry' in self.models and 'entry_features' in self.feature_config:
            try:
                features = self.feature_config['entry_features']
                feature_vector = [market_data.get(f, 0) for f in features]
                
                X = self.scalers['entry'].transform([feature_vector])
                entry_prob = self.models['entry'].predict_proba(X)[0][1]
                
                signals['should_enter'] = entry_prob > self.trading_params['confidence_threshold']
                signals['confidence'] = entry_prob
                
            except Exception as e:
                self.log(f"ML entry signal error: {str(e)}")
                signals.update(self._generate_rule_based_signals(market_data))
        
        # Position sizing
        if 'position' in self.models and 'position_features' in self.feature_config:
            try:
                features = self.feature_config['position_features']
                feature_vector = [market_data.get(f, 0) for f in features]
                
                X = self.scalers['position'].transform([feature_vector])
                position_size = self.models['position'].predict(X)[0]
                
                signals['position_size'] = position_size
                
            except Exception as e:
                self.log(f"ML position sizing error: {str(e)}")
                signals['position_size'] = self._calculate_rule_based_position_size(market_data)
        
        # Exit signal
        if 'exit' in self.models and 'exit_features' in self.feature_config:
            try:
                features = self.feature_config['exit_features']
                feature_vector = [market_data.get(f, 0) for f in features]
                
                X = self.scalers['exit'].transform([feature_vector])
                exit_prob = self.models['exit'].predict_proba(X)[0][1]
                
                signals['should_exit'] = exit_prob > 0.5
                
            except Exception as e:
                self.log(f"ML exit signal error: {str(e)}")
                signals['should_exit'] = self._calculate_rule_based_exit(market_data)
        
        return signals
    
    def _generate_rule_based_signals(self, market_data):
        """Generate signals using rule-based logic."""
        # Entry signal
        momentum_signal = market_data.get('momentum_strength', 0) > self.trading_params['momentum_threshold']
        rsi_signal = 30 < market_data.get('rsi', 50) < 70
        volume_signal = market_data.get('volume_ratio', 1) > 1.0
        structure_signal = market_data.get('market_structure', 0) >= 0
        
        entry_signals = [momentum_signal, rsi_signal, volume_signal, structure_signal]
        signal_strength = sum(entry_signals) / len(entry_signals)
        
        should_enter = signal_strength >= 0.75  # Need 3/4 signals
        
        # Position sizing
        position_size = self._calculate_rule_based_position_size(market_data)
        
        # Exit signal
        should_exit = self._calculate_rule_based_exit(market_data)
        
        return {
            'should_enter': should_enter,
            'should_exit': should_exit,
            'position_size': position_size,
            'confidence': signal_strength
        }
    
    def _calculate_rule_based_position_size(self, market_data):
        """Calculate position size using rule-based logic."""
        momentum_strength = market_data.get('momentum_strength', 0)
        is_high_momentum = market_data.get('is_high_momentum', 0)
        success_probability = market_data.get('success_probability', 0.5)
        
        if is_high_momentum:
            base_size = self.trading_params['position_size_max']
        elif momentum_strength > 0.8:
            base_size = 0.620
        else:
            base_size = self.trading_params['position_size_min']
        
        # Adjust for success probability
        adjusted_size = base_size * (0.7 + success_probability * 0.6)
        
        return np.clip(adjusted_size, 0.3, 0.9)
    
    def _calculate_rule_based_exit(self, market_data):
        """Calculate exit signal using rule-based logic."""
        rsi_exit = market_data.get('rsi', 50) > 70
        momentum_exit = market_data.get('momentum_strength', 0) < -0.5
        bb_exit = market_data.get('bb_position', 0.5) > 0.8
        
        return rsi_exit or momentum_exit or bb_exit
    
    def execute_trade(self, action, market_data, quantity=None):
        """Execute a trade (paper trading)."""
        current_price = market_data.get('close', 0)
        
        if current_price <= 0:
            return None
        
        if action == 'BUY' and self.btc_position == 0 and self.cash > 1000:
            return self._execute_buy(current_price, quantity, market_data)
        
        elif action == 'SELL' and self.btc_position > 0.001:
            return self._execute_sell(current_price, market_data)
        
        return None
    
    def _execute_buy(self, price, quantity, market_data):
        """Execute buy order."""
        if quantity is None:
            signals = self.generate_trading_signals(market_data)
            position_fraction = signals['position_size']
            target_value = self.get_portfolio_value() * position_fraction
            quantity = min(target_value, self.cash * 0.99) / price
        
        if quantity > 0.001:
            cost = quantity * price
            self.cash -= cost
            self.btc_position += quantity
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'value': cost,
                'pnl': 0
            }
            
            self.trades.append(trade)
            self.log(f"🟢 BUY {quantity:.6f} BTC @ ${price:,.2f} (${cost:,.2f})")
            
            return trade
        
        return None
    
    def _execute_sell(self, price, market_data):
        """Execute sell order."""
        proceeds = self.btc_position * price
        
        # Calculate P&L
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        total_cost = sum(t['value'] for t in buy_trades)
        pnl = proceeds - total_cost
        
        self.cash += proceeds
        
        trade = {
            'timestamp': datetime.now(),
            'action': 'SELL',
            'quantity': self.btc_position,
            'price': price,
            'value': proceeds,
            'pnl': pnl
        }
        
        self.trades.append(trade)
        self.log(f"🔴 SELL {self.btc_position:.6f} BTC @ ${price:,.2f} (${proceeds:,.2f}) - P&L: ${pnl:+,.2f}")
        
        self.btc_position = 0
        return trade
    
    def get_portfolio_value(self):
        """Get current portfolio value."""
        try:
            if self.btc_position > 0:
                market_data = self.get_market_data()
                if market_data is not None:
                    current_price = market_data.get('close', 0)
                    btc_value = self.btc_position * current_price
                else:
                    btc_value = 0
            else:
                btc_value = 0
            
            return self.cash + btc_value
            
        except Exception as e:
            self.log(f"Portfolio value error: {str(e)}")
            return self.cash
    
    def check_risk_management(self, market_data):
        """Check and execute risk management rules."""
        if self.btc_position <= 0:
            return False
        
        current_price = market_data.get('close', 0)
        if current_price <= 0:
            return False
        
        # Get average buy price
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        if not buy_trades:
            return False
        
        total_cost = sum(t['value'] for t in buy_trades)
        total_quantity = sum(t['quantity'] for t in buy_trades)
        avg_buy_price = total_cost / total_quantity
        
        # Calculate P&L percentage
        pnl_pct = (current_price - avg_buy_price) / avg_buy_price
        
        # Stop loss
        if pnl_pct <= -self.trading_params['stop_loss_pct']:
            self.log(f"🛑 STOP LOSS triggered: {pnl_pct:.1%}")
            self.execute_trade('SELL', market_data)
            return True
        
        # Take profit
        if pnl_pct >= self.trading_params['take_profit_pct']:
            self.log(f"💰 TAKE PROFIT triggered: {pnl_pct:.1%}")
            self.execute_trade('SELL', market_data)
            return True
        
        return False
    
    def update_parameters(self):
        """Check for parameter updates from Agent01."""
        try:
            param_file = 'crypto_rf_trader/config/parameter_suggestions.json'
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    param_data = json.load(f)
                
                suggestions = param_data.get('suggestions', {})
                
                # Update parameters
                for key, value in suggestions.items():
                    if key in self.trading_params:
                        old_value = self.trading_params[key]
                        self.trading_params[key] = value
                        self.log(f"Parameter updated: {key} = {old_value} → {value}")
                
                return True
            
            return False
            
        except Exception as e:
            self.log(f"Parameter update error: {str(e)}")
            return False
    
    def save_session_data(self):
        """Save current session data."""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'session_start': self.session_start.isoformat(),
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'btc_position': self.btc_position,
            'portfolio_value': self.get_portfolio_value(),
            'total_trades': len(self.trades),
            'trading_parameters': self.trading_params,
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
        session_file = f"agent03_session_{timestamp}.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            self.log(f"Error saving session data: {str(e)}")
    
    def run_24h_trading_session(self):
        """Run 24-hour trading session."""
        session_end = self.session_start + timedelta(hours=self.session_duration)
        
        self.log(f"🚀 24-Hour Trading Session Started")
        self.log(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        self.log(f"🕐 Session End: {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"🤖 Models: {list(self.models.keys())}")
        
        last_update = datetime.now()
        hour_counter = 0
        
        try:
            while self.running and datetime.now() < session_end:
                current_time = datetime.now()
                
                # Update every interval
                if (current_time - last_update).total_seconds() >= self.update_interval:
                    
                    # Get market data
                    market_data = self.get_market_data()
                    
                    if market_data is not None:
                        
                        # Update parameters if needed
                        self.update_parameters()
                        
                        # Check risk management first
                        if not self.check_risk_management(market_data):
                            
                            # Generate and execute trading signals
                            signals = self.generate_trading_signals(market_data)
                            
                            if signals['should_enter'] and self.btc_position == 0:
                                self.execute_trade('BUY', market_data)
                            elif signals['should_exit'] and self.btc_position > 0:
                                self.execute_trade('SELL', market_data)
                        
                        # Update portfolio tracking
                        portfolio_value = self.get_portfolio_value()
                        self.portfolio_values.append(portfolio_value)
                    
                    last_update = current_time
                    
                    # Hourly reports
                    hours_elapsed = (current_time - self.session_start).total_seconds() / 3600
                    if hours_elapsed >= hour_counter + 1:
                        hour_counter = int(hours_elapsed)
                        self._generate_hourly_report(hour_counter)
                        self.save_session_data()
                    
                    # Log status
                    hours_remaining = (session_end - current_time).total_seconds() / 3600
                    self.log(f"⏳ Next update in {self.update_interval}s... ({hours_remaining:.1f} hours remaining)")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.log("🛑 Manual stop requested")
        
        except Exception as e:
            self.log(f"❌ Trading session error: {str(e)}")
        
        finally:
            self._generate_final_report()
    
    def _generate_hourly_report(self, hour):
        """Generate hourly status report."""
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        self.log("")
        self.log(f"📊 TRADING STATUS - Hour {hour:.1f}/24")
        self.log(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        self.log(f"📈 Total Return: {total_return:+.2f}%")
        self.log(f"💵 Cash: ${self.cash:,.2f}")
        
        if self.btc_position > 0:
            market_data = self.get_market_data()
            if market_data:
                current_price = market_data.get('close', 0)
                btc_value = self.btc_position * current_price
                self.log(f"₿ BTC Position: {self.btc_position:.6f} BTC (${btc_value:,.2f})")
                self.log(f"💲 BTC Price: ${current_price:,.2f}")
        else:
            self.log("₿ BTC Position: 0.000000 BTC ($0.00)")
        
        self.log(f"📝 Total Trades: {len(self.trades)}")
        self.log("-" * 60)
    
    def _generate_final_report(self):
        """Generate final session report."""
        self.log("")
        self.log("🏁 24-HOUR TRADING SESSION COMPLETE")
        self.log("=" * 60)
        
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        self.log(f"💰 Final Portfolio Value: ${final_value:,.2f}")
        self.log(f"📈 Total Return: {total_return:+.2f}%")
        self.log(f"📝 Total Trades: {len(self.trades)}")
        
        # Calculate win rate
        if self.trades:
            profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            total_trades_with_pnl = [t for t in self.trades if 'pnl' in t]
            
            if total_trades_with_pnl:
                win_rate = len(profitable_trades) / len(total_trades_with_pnl) * 100
                self.log(f"📊 Win Rate: {win_rate:.1f}%")
        
        # Performance vs targets
        baseline = 2.82
        target = 4.0
        
        if total_return >= target:
            self.log(f"🎯 TARGET ACHIEVED: {total_return:.2f}% ≥ {target}%")
        elif total_return >= baseline:
            self.log(f"📈 BASELINE EXCEEDED: {total_return:.2f}% > {baseline}%")
        else:
            self.log(f"📉 Below baseline: {total_return:.2f}% < {baseline}%")
        
        self.save_session_data()
        self.log("✅ Session completed")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log(f"Agent03 received signal {signum} - shutting down")
        self.running = False

def main():
    agent = Agent03ExecutionEngine()
    agent.run_24h_trading_session()

if __name__ == "__main__":
    main()