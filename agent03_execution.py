#!/usr/bin/env python3
"""
Agent03: Execution Engine Developer

Builds live/paper trading system using ccxt (or Alpaca/IBKR),
enforces position sizing, handles live order flow.
"""

import time
import os
import json
import signal
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import ccxt for exchange integration
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

COMS_PATH = "coms.md"

class Agent03ExecutionEngine:
    """Live/paper trading execution engine with position management."""
    
    def __init__(self):
        """Initialize Agent03 execution engine."""
        self.running = False
        
        # Trading parameters
        self.initial_capital = 100000.0
        self.cash = 100000.0
        self.btc_position = 0.0
        self.portfolio_values = [100000.0]
        self.trades = []
        
        # Risk management
        self.max_position_size = 0.95  # Max 95% of capital
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.05    # 5% take profit
        self.trailing_stop_pct = 0.015 # 1.5% trailing stop
        
        # Current trading parameters (loaded from Agent02)
        self.trading_params = {
            'momentum_threshold': 1.780,
            'confidence_threshold': 0.65,
            'exit_threshold': 0.55,
            'position_size_min': 0.464,
            'position_size_max': 0.800
        }
        
        # Models and scalers (loaded from Agent02)
        self.models = {}
        self.scalers = {}
        
        # Exchange setup (paper trading)
        self.exchange = None
        self.paper_trading = True
        
        # Session tracking
        self.session_start = None
        self.session_duration = 24  # hours
        self.update_interval = 300  # 5 minutes
        
        # Logging
        self.log_dir = "logs/agent03_trading"
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_log = os.path.join(self.log_dir, f'agent03_session_{timestamp}.log')
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.log("Agent03 execution engine initialized")
    
    def log(self, message: str):
        """Log message to communications file and session log."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[agent03][{timestamp}] {message}"
        
        # Log to communications file
        try:
            with open(COMS_PATH, "a") as f:
                f.write(f"{log_entry}\n")
        except Exception as e:
            print(f"[agent03] Coms logging error: {e}")
        
        # Log to session file
        try:
            with open(self.session_log, "a") as f:
                f.write(f"{log_entry}\n")
        except Exception as e:
            print(f"[agent03] Session logging error: {e}")
        
        print(log_entry)
    
    def setup_exchange(self) -> bool:
        """Setup exchange connection (paper trading mode)."""
        try:
            if CCXT_AVAILABLE:
                # Use Binance testnet for paper trading
                self.exchange = ccxt.binance({
                    'apiKey': 'test',
                    'secret': 'test',
                    'sandbox': True,  # Use testnet
                    'enableRateLimit': True,
                })
                self.log("CCXT exchange (Binance testnet) configured for paper trading")
            else:
                self.log("CCXT not available, using YFinance for data and internal position tracking")
            
            self.paper_trading = True
            return True
            
        except Exception as e:
            self.log(f"Exchange setup error: {e}")
            self.paper_trading = True
            return True  # Continue with paper trading
    
    def load_models(self) -> bool:
        """Load trained models from Agent02."""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                self.log("Models directory not found, will use simplified trading logic")
                return False
            
            # Find latest model files
            model_files = {}
            scaler_files = {}
            
            for file in os.listdir(models_dir):
                if file.endswith('.joblib'):
                    if '_model_' in file:
                        model_name = file.split('_model_')[0]
                        if model_name not in model_files or file > model_files[model_name]:
                            model_files[model_name] = file
                    elif '_scaler_' in file:
                        scaler_name = file.split('_scaler_')[0]
                        if scaler_name not in scaler_files or file > scaler_files[scaler_name]:
                            scaler_files[scaler_name] = file
            
            # Load models and scalers
            models_loaded = 0
            for model_name, model_file in model_files.items():
                try:
                    model_path = os.path.join(models_dir, model_file)
                    self.models[model_name] = joblib.load(model_path)
                    models_loaded += 1
                    
                    if model_name in scaler_files:
                        scaler_path = os.path.join(models_dir, scaler_files[model_name])
                        self.scalers[model_name] = joblib.load(scaler_path)
                    
                    self.log(f"Loaded {model_name} model: {model_file}")
                    
                except Exception as e:
                    self.log(f"Error loading {model_name} model: {e}")
            
            self.log(f"Models loaded: {models_loaded}/{len(model_files)}")
            return models_loaded > 0
            
        except Exception as e:
            self.log(f"Error loading models: {e}")
            return False
    
    def get_market_data(self) -> Optional[pd.Series]:
        """Get current market data with indicators."""
        try:
            # Fetch recent data
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period="7d", interval="1h")
            
            if data.empty:
                return None
            
            # Prepare features (simplified version)
            data = data.rename(columns={col.lower(): col.lower() for col in data.columns})
            data = self._calculate_indicators(data)
            
            # Return latest data point
            return data.iloc[-1]
            
        except Exception as e:
            self.log(f"Market data error: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading indicators."""
        
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
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
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
        
        # ATR
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        data['atr'] = true_range.rolling(14).mean()
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_optimal_hour'] = (data['hour'] == 3).astype(int)
        
        # Trend features
        sma_10 = data['close'].rolling(10).mean()
        sma_20 = data['close'].rolling(20).mean()
        data['trend_strength'] = np.abs(sma_10 - sma_20) / data['close']
        data['price_vs_sma10'] = (data['close'] - sma_10) / sma_10
        
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
    
    def generate_trading_signals(self, market_data: pd.Series) -> Dict[str, Any]:
        """Generate trading signals using loaded models or fallback logic."""
        
        signals = {
            'should_enter': False,
            'should_exit': False,
            'position_size': 0.5,
            'confidence': 0.5,
            'reasoning': []
        }
        
        try:
            # Entry signal
            if 'entry' in self.models and 'entry' in self.scalers:
                signals.update(self._generate_ml_entry_signal(market_data))
            else:
                signals.update(self._generate_rule_based_entry_signal(market_data))
            
            # Position sizing
            if 'position' in self.models and 'position' in self.scalers:
                signals['position_size'] = self._generate_ml_position_size(market_data)
            else:
                signals['position_size'] = self._generate_rule_based_position_size(market_data)
            
            # Exit signal
            if 'exit' in self.models and 'exit' in self.scalers:
                exit_signal = self._generate_ml_exit_signal(market_data)
                signals['should_exit'] = exit_signal['should_exit']
            else:
                signals['should_exit'] = self._generate_rule_based_exit_signal(market_data)
            
            # Ensure position size is within bounds
            signals['position_size'] = np.clip(
                signals['position_size'],
                self.trading_params['position_size_min'],
                self.trading_params['position_size_max']
            )
            
            return signals
            
        except Exception as e:
            self.log(f"Signal generation error: {e}")
            return signals
    
    def _generate_ml_entry_signal(self, market_data: pd.Series) -> Dict[str, Any]:
        """Generate entry signal using ML model."""
        try:
            entry_features = [
                'momentum_1', 'momentum_4', 'momentum_strength', 'rsi', 'macd',
                'volume_ratio', 'market_structure', 'bb_position', 'is_optimal_hour',
                'hour', 'day_of_week', 'volatility', 'trend_strength', 'success_probability'
            ]
            
            # Prepare feature vector
            feature_vector = []
            for feature in entry_features:
                value = market_data.get(feature, 0)
                if pd.isna(value):
                    value = 0
                feature_vector.append(value)
            
            # Scale and predict
            X = self.scalers['entry'].transform([feature_vector])
            probabilities = self.models['entry'].predict_proba(X)[0]
            
            entry_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            should_enter = entry_prob > self.trading_params['confidence_threshold']
            
            return {
                'should_enter': should_enter,
                'confidence': entry_prob,
                'reasoning': [f"ML entry probability: {entry_prob:.2f}"]
            }
            
        except Exception as e:
            self.log(f"ML entry signal error: {e}")
            return self._generate_rule_based_entry_signal(market_data)
    
    def _generate_rule_based_entry_signal(self, market_data: pd.Series) -> Dict[str, Any]:
        """Generate entry signal using rule-based logic."""
        
        # Momentum signal
        momentum_strength = market_data.get('momentum_strength', 0)
        momentum_signal = momentum_strength > self.trading_params['momentum_threshold']
        
        # RSI signal
        rsi = market_data.get('rsi', 50)
        rsi_signal = rsi < 70 and rsi > 30
        
        # Volume signal
        volume_ratio = market_data.get('volume_ratio', 1)
        volume_signal = volume_ratio > 1.0
        
        # Market structure signal
        market_structure = market_data.get('market_structure', 0)
        structure_signal = market_structure >= 0
        
        # Combine signals
        signal_count = sum([momentum_signal, rsi_signal, volume_signal, structure_signal])
        should_enter = signal_count >= 3  # Need at least 3/4 signals
        confidence = signal_count / 4.0
        
        reasoning = []
        if momentum_signal:
            reasoning.append(f"High momentum: {momentum_strength:.1f}%/h")
        if rsi_signal:
            reasoning.append(f"RSI favorable: {rsi:.0f}")
        if volume_signal:
            reasoning.append(f"Volume above average: {volume_ratio:.1f}x")
        if structure_signal:
            reasoning.append("Market structure bullish")
        
        return {
            'should_enter': should_enter,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _generate_ml_position_size(self, market_data: pd.Series) -> float:
        """Generate position size using ML model."""
        try:
            position_features = [
                'momentum_strength', 'is_high_momentum', 'volatility', 'atr',
                'rsi', 'market_structure', 'success_probability', 'volume_ratio'
            ]
            
            # Prepare feature vector
            feature_vector = []
            for feature in position_features:
                value = market_data.get(feature, 0)
                if feature == 'atr':
                    # Normalize ATR relative to price
                    value = value / market_data.get('close', 1) if market_data.get('close', 1) > 0 else 0
                if pd.isna(value):
                    value = 0
                feature_vector.append(value)
            
            # Scale and predict
            X = self.scalers['position'].transform([feature_vector])
            position_size = self.models['position'].predict(X)[0]
            
            return float(position_size)
            
        except Exception as e:
            self.log(f"ML position sizing error: {e}")
            return self._generate_rule_based_position_size(market_data)
    
    def _generate_rule_based_position_size(self, market_data: pd.Series) -> float:
        """Generate position size using rule-based logic."""
        
        momentum_strength = market_data.get('momentum_strength', 0)
        is_high_momentum = market_data.get('is_high_momentum', 0)
        volatility = market_data.get('volatility', 0.02)
        success_probability = market_data.get('success_probability', 0.5)
        
        # Base position size on momentum and success probability
        if is_high_momentum:
            base_size = self.trading_params['position_size_max']
        elif momentum_strength > 0.8:
            base_size = 0.620  # Medium position
        else:
            base_size = self.trading_params['position_size_min']
        
        # Adjust for volatility (reduce size if high volatility)
        volatility_adjustment = 1.0 - min(volatility * 10, 0.3)  # Max 30% reduction
        
        # Adjust for success probability
        success_adjustment = 0.7 + (success_probability * 0.6)  # 0.7 to 1.3 multiplier
        
        position_size = base_size * volatility_adjustment * success_adjustment
        
        return np.clip(position_size, 0.2, 0.95)
    
    def _generate_ml_exit_signal(self, market_data: pd.Series) -> Dict[str, Any]:
        """Generate exit signal using ML model."""
        try:
            exit_features = [
                'momentum_1', 'rsi', 'bb_position', 'market_structure',
                'volatility', 'volume_ratio', 'price_vs_sma10', 'macd'
            ]
            
            # Prepare feature vector
            feature_vector = []
            for feature in exit_features:
                value = market_data.get(feature, 0)
                if pd.isna(value):
                    value = 0
                feature_vector.append(value)
            
            # Scale and predict
            X = self.scalers['exit'].transform([feature_vector])
            probabilities = self.models['exit'].predict_proba(X)[0]
            
            exit_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            should_exit = exit_prob > self.trading_params['exit_threshold']
            
            return {
                'should_exit': should_exit,
                'exit_probability': exit_prob
            }
            
        except Exception as e:
            self.log(f"ML exit signal error: {e}")
            return {'should_exit': self._generate_rule_based_exit_signal(market_data)}
    
    def _generate_rule_based_exit_signal(self, market_data: pd.Series) -> bool:
        """Generate exit signal using rule-based logic."""
        
        # RSI overbought
        rsi = market_data.get('rsi', 50)
        rsi_exit = rsi > 70
        
        # Negative momentum
        momentum_strength = market_data.get('momentum_strength', 0)
        momentum_exit = momentum_strength < -0.5
        
        # Bollinger Band position
        bb_position = market_data.get('bb_position', 0.5)
        bb_exit = bb_position > 0.8
        
        # Market structure breakdown
        market_structure = market_data.get('market_structure', 0)
        structure_exit = market_structure < -1
        
        return rsi_exit or momentum_exit or bb_exit or structure_exit
    
    def execute_trade(self, action: str, market_data: pd.Series, quantity: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute a trade (paper trading)."""
        
        current_price = market_data.get('close', 0)
        if current_price <= 0:
            return None
        
        if action == 'BUY' and self.btc_position == 0 and self.cash > 1000:
            return self._execute_buy(current_price, quantity, market_data)
        
        elif action == 'SELL' and self.btc_position > 0.001:
            return self._execute_sell(current_price, market_data)
        
        return None
    
    def _execute_buy(self, price: float, quantity: Optional[float], market_data: pd.Series) -> Dict[str, Any]:
        """Execute buy order."""
        
        if quantity is None:
            # Calculate position size based on signals
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
                'pnl': 0,
                'portfolio_value_before': self.get_portfolio_value() + cost - (quantity * price)
            }
            
            self.trades.append(trade)
            self.log(f"🟢 BUY {quantity:.6f} BTC @ ${price:,.2f} (${cost:,.2f})")
            
            return trade
        
        return None
    
    def _execute_sell(self, price: float, market_data: pd.Series) -> Dict[str, Any]:
        """Execute sell order."""
        
        proceeds = self.btc_position * price
        
        # Calculate P&L
        buy_trades = [t for t in self.trades if t['action'] == 'BUY' and 'pnl' not in t or t['pnl'] == 0]
        total_cost = sum(t['value'] for t in buy_trades)
        pnl = proceeds - total_cost
        
        self.cash += proceeds
        
        trade = {
            'timestamp': datetime.now(),
            'action': 'SELL',
            'quantity': self.btc_position,
            'price': price,
            'value': proceeds,
            'pnl': pnl,
            'portfolio_value_before': self.get_portfolio_value() - proceeds + (self.btc_position * price)
        }
        
        self.trades.append(trade)
        self.log(f"🔴 SELL {self.btc_position:.6f} BTC @ ${price:,.2f} (${proceeds:,.2f}) - P&L: ${pnl:+,.2f}")
        
        self.btc_position = 0
        return trade
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            if self.btc_position > 0:
                # Get current BTC price
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
            self.log(f"Portfolio value calculation error: {e}")
            return self.cash
    
    def check_risk_management(self, market_data: pd.Series) -> bool:
        """Check risk management rules and execute if needed."""
        
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
        avg_buy_price = total_cost / total_quantity if total_quantity > 0 else 0
        
        if avg_buy_price <= 0:
            return False
        
        # Calculate current P&L percentage
        pnl_pct = (current_price - avg_buy_price) / avg_buy_price
        
        # Stop loss check
        if pnl_pct <= -self.stop_loss_pct:
            self.log(f"🛑 STOP LOSS triggered: {pnl_pct:.1%}")
            self.execute_trade('SELL', market_data)
            return True
        
        # Take profit check
        if pnl_pct >= self.take_profit_pct:
            self.log(f"💰 TAKE PROFIT triggered: {pnl_pct:.1%}")
            self.execute_trade('SELL', market_data)
            return True
        
        # Trailing stop (simplified)
        # TODO: Implement proper trailing stop logic
        
        return False
    
    def update_trading_parameters(self) -> bool:
        """Check for parameter updates from other agents."""
        try:
            # Check for parameter update files
            param_files = ['optimization_suggestions.json', 'parameter_update.json']
            
            for param_file in param_files:
                if os.path.exists(param_file):
                    with open(param_file, 'r') as f:
                        update_data = json.load(f)
                    
                    # Update parameters if they exist
                    if 'suggestions' in update_data:
                        suggestions = update_data['suggestions']
                        for key, value in suggestions.items():
                            if key in self.trading_params:
                                old_value = self.trading_params[key]
                                self.trading_params[key] = value
                                self.log(f"Parameter updated: {key} = {old_value} → {value}")
                    
                    return True
            
            return False
            
        except Exception as e:
            self.log(f"Parameter update error: {e}")
            return False
    
    def generate_hourly_report(self, hour: int):
        """Generate hourly trading report."""
        
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        self.log("")
        self.log(f"📊 AGENT03 TRADING STATUS - Hour {hour:.1f}/24")
        self.log(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        self.log(f"📈 Total Return: {total_return:+.2f}%")
        self.log(f"💵 Cash: ${self.cash:,.2f}")
        
        if self.btc_position > 0:
            market_data = self.get_market_data()
            if market_data is not None:
                current_price = market_data.get('close', 0)
                btc_value = self.btc_position * current_price
                
                # Position P&L
                buy_trades = [t for t in self.trades if t['action'] == 'BUY']
                if buy_trades:
                    total_cost = sum(t['value'] for t in buy_trades)
                    total_quantity = sum(t['quantity'] for t in buy_trades)
                    avg_buy_price = total_cost / total_quantity
                    position_pnl = (current_price - avg_buy_price) / avg_buy_price * 100
                else:
                    position_pnl = 0
                
                self.log(f"₿ BTC Position: {self.btc_position:.6f} BTC (${btc_value:,.2f})")
                self.log(f"💲 Current BTC Price: ${current_price:,.2f}")
                self.log(f"📊 Position P&L: {position_pnl:+.2f}%")
        else:
            self.log("₿ BTC Position: 0.000000 BTC ($0.00)")
        
        self.log(f"📝 Total Trades: {len(self.trades)}")
        self.log(f"📊 Models Loaded: {len(self.models)}")
        self.log("-" * 60)
        
        # Update portfolio values
        self.portfolio_values.append(portfolio_value)
    
    def save_session_data(self):
        """Save current session data."""
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'session_start': self.session_start.isoformat() if self.session_start else None,
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'btc_position': self.btc_position,
            'portfolio_value': self.get_portfolio_value(),
            'total_trades': len(self.trades),
            'trading_parameters': self.trading_params,
            'models_loaded': list(self.models.keys()),
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
            
            self.log(f"💾 Session data saved: {session_file}")
            
        except Exception as e:
            self.log(f"Error saving session data: {e}")
    
    def run_24h_trading_session(self):
        """Run 24-hour trading session."""
        
        self.log("Starting 24-hour trading session")
        
        # Setup
        self.setup_exchange()
        self.load_models()
        
        self.session_start = datetime.now()
        session_end = self.session_start + timedelta(hours=self.session_duration)
        
        self.log(f"🚀 24-Hour Trading Session Started")
        self.log(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        self.log(f"🕐 Session End: {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"⏱️ Update Interval: {self.update_interval} seconds")
        self.log(f"🤖 Models Loaded: {list(self.models.keys())}")
        self.log("Press Ctrl+C to stop early")
        
        self.running = True
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
                        
                        # Check for parameter updates
                        self.update_trading_parameters()
                        
                        # Check risk management first
                        if not self.check_risk_management(market_data):
                            
                            # Generate trading signals
                            signals = self.generate_trading_signals(market_data)
                            
                            # Execute trades based on signals
                            if signals['should_enter'] and self.btc_position == 0:
                                self.execute_trade('BUY', market_data)
                            elif signals['should_exit'] and self.btc_position > 0:
                                self.execute_trade('SELL', market_data)
                        
                        # Update portfolio value tracking
                        portfolio_value = self.get_portfolio_value()
                        self.portfolio_values.append(portfolio_value)
                    
                    last_update = current_time
                    
                    # Hourly reports
                    hours_elapsed = (current_time - self.session_start).total_seconds() / 3600
                    if hours_elapsed >= hour_counter + 1:
                        hour_counter = int(hours_elapsed)
                        self.generate_hourly_report(hour_counter)
                        self.save_session_data()
                    
                    # Log next update
                    hours_remaining = (session_end - current_time).total_seconds() / 3600
                    self.log(f"⏳ Next update in {self.update_interval}s... ({hours_remaining:.1f} hours remaining)")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.log("🛑 Manual stop requested")
        
        except Exception as e:
            self.log(f"❌ Trading session error: {e}")
        
        finally:
            self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate final session report."""
        
        self.log("")
        self.log("🏁 AGENT03 24-HOUR TRADING SESSION COMPLETE")
        self.log("=" * 60)
        
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        self.log(f"💰 Final Portfolio Value: ${final_value:,.2f}")
        self.log(f"📈 Total Return: {total_return:+.2f}%")
        self.log(f"📝 Total Trades: {len(self.trades)}")
        
        if self.trades:
            profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            total_trades_with_pnl = [t for t in self.trades if 'pnl' in t]
            
            if total_trades_with_pnl:
                win_rate = len(profitable_trades) / len(total_trades_with_pnl) * 100
                self.log(f"📊 Win Rate: {win_rate:.1f}%")
        
        # Target comparison
        baseline_return = 2.82
        target_return = 4.0
        
        if total_return >= target_return:
            self.log(f"🎯 TARGET ACHIEVED: {total_return:.2f}% ≥ {target_return}%")
        elif total_return >= baseline_return:
            self.log(f"📈 BASELINE EXCEEDED: {total_return:.2f}% > {baseline_return}%")
        else:
            self.log(f"📉 Below baseline: {total_return:.2f}% < {baseline_return}%")
        
        # Save final data
        self.save_session_data()
        
        self.log("✅ Agent03 session completed")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log(f"Agent03 received signal {signum} - shutting down")
        self.running = False

def main():
    """Entry point for Agent03."""
    agent = Agent03ExecutionEngine()
    agent.run_24h_trading_session()

if __name__ == "__main__":
    main()