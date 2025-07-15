#!/usr/bin/env python3
"""
Production Deployment Layer with CCXT/Alpaca Integration

Implements robust API handling, retry mechanisms, file locking, and
production-ready execution for the enhanced Random Forest trading system.

Features:
- Multi-exchange support via CCXT
- Comprehensive retry/failure handling
- File locking for concurrent access
- Rate limiting and API quota management
- Circuit breaker patterns
- Performance monitoring

Usage: from production_deployment_layer import ProductionTrader
"""

import ccxt
import time
import json
import fcntl
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class CircuitBreaker:
    """Circuit breaker for API failure handling."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if datetime.now().timestamp() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN - API temporarily unavailable")
            else:
                self.state = 'half_open'
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful API call."""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle API failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            print(f"🚨 Circuit breaker OPENED after {self.failure_count} failures")

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 60, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def acquire(self):
        """Acquire rate limit slot."""
        with self.lock:
            now = datetime.now().timestamp()
            
            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    print(f"⏱️ Rate limit reached, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    return self.acquire()
            
            self.calls.append(now)

class FileManager:
    """Thread-safe file operations with locking."""
    
    @staticmethod
    def safe_read(filepath: str, default_content: str = "{}") -> str:
        """Safely read file with locking."""
        try:
            with open(filepath, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                content = f.read()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
                return content
        except FileNotFoundError:
            return default_content
        except Exception as e:
            print(f"⚠️ File read error {filepath}: {e}")
            return default_content
    
    @staticmethod
    def safe_write(filepath: str, content: str, append: bool = False):
        """Safely write file with locking."""
        mode = 'a' if append else 'w'
        try:
            with open(filepath, mode) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                f.write(content)
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
        except Exception as e:
            print(f"❌ File write error {filepath}: {e}")
    
    @staticmethod
    def safe_json_read(filepath: str, default_data: Dict = None) -> Dict:
        """Safely read JSON with locking."""
        default_data = default_data or {}
        try:
            content = FileManager.safe_read(filepath, "{}")
            return json.loads(content) if content.strip() else default_data
        except json.JSONDecodeError:
            return default_data
    
    @staticmethod
    def safe_json_write(filepath: str, data: Dict):
        """Safely write JSON with locking."""
        content = json.dumps(data, indent=2, default=str)
        FileManager.safe_write(filepath, content)

class LogRotator:
    """Log rotation manager to prevent performance issues."""
    
    def __init__(self, max_size_mb: int = 50, max_files: int = 10):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
    
    def should_rotate(self, filepath: str) -> bool:
        """Check if log file should be rotated."""
        try:
            import os
            return os.path.getsize(filepath) > self.max_size_bytes
        except FileNotFoundError:
            return False
    
    def rotate_log(self, filepath: str):
        """Rotate log file."""
        import os
        import shutil
        
        if not os.path.exists(filepath):
            return
        
        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_file = f"{filepath}.{i}"
            new_file = f"{filepath}.{i + 1}"
            
            if os.path.exists(old_file):
                if i == self.max_files - 1:
                    os.remove(old_file)  # Remove oldest
                else:
                    shutil.move(old_file, new_file)
        
        # Move current file to .1
        shutil.move(filepath, f"{filepath}.1")
        print(f"📄 Log rotated: {filepath}")

class ProductionExchangeManager:
    """Production-ready exchange manager with CCXT."""
    
    def __init__(self, exchange_name: str = 'binance', api_credentials: Dict = None):
        self.exchange_name = exchange_name
        self.api_credentials = api_credentials or {}
        self.exchange = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.rate_limiter = RateLimiter(max_calls=50, time_window=60)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange with error handling."""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': self.api_credentials.get('api_key'),
                'secret': self.api_credentials.get('secret'),
                'password': self.api_credentials.get('passphrase'),  # For some exchanges
                'sandbox': self.api_credentials.get('sandbox', True),  # Use sandbox by default
                'enableRateLimit': True,
                'timeout': 30000,  # 30 second timeout
                'rateLimit': 1200  # 1.2 seconds between requests
            })
            
            print(f"✅ Initialized {self.exchange_name} exchange")
            
        except Exception as e:
            print(f"❌ Failed to initialize exchange: {e}")
            raise
    
    def _enforce_rate_limit(self):
        """Enforce minimum time between requests."""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_ticker(self, symbol: str, retries: int = 3) -> Optional[Dict]:
        """Fetch ticker with retry mechanism."""
        for attempt in range(retries):
            try:
                self.rate_limiter.acquire()
                self._enforce_rate_limit()
                
                ticker = self.circuit_breaker.call(self.exchange.fetch_ticker, symbol)
                return ticker
                
            except ccxt.NetworkError as e:
                print(f"🌐 Network error (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
                
            except ccxt.ExchangeError as e:
                print(f"🏪 Exchange error (attempt {attempt + 1}): {e}")
                if "rate limit" in str(e).lower():
                    time.sleep(5)
                    continue
                raise
                
            except Exception as e:
                print(f"❌ Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise
        
        return None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h', limit: int = 100, retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with retry mechanism."""
        for attempt in range(retries):
            try:
                self.rate_limiter.acquire()
                self._enforce_rate_limit()
                
                ohlcv = self.circuit_breaker.call(
                    self.exchange.fetch_ohlcv, 
                    symbol, 
                    timeframe, 
                    limit=limit
                )
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)
                    return df
                
            except ccxt.NetworkError as e:
                print(f"🌐 Network error fetching OHLCV (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
                
            except ccxt.ExchangeError as e:
                print(f"🏪 Exchange error fetching OHLCV (attempt {attempt + 1}): {e}")
                if "rate limit" in str(e).lower():
                    time.sleep(10)
                    continue
                raise
                
            except Exception as e:
                print(f"❌ Unexpected error fetching OHLCV (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise
        
        return None
    
    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None, retries: int = 3) -> Optional[Dict]:
        """Place order with retry mechanism."""
        for attempt in range(retries):
            try:
                self.rate_limiter.acquire()
                self._enforce_rate_limit()
                
                if order_type == 'market':
                    order = self.circuit_breaker.call(
                        self.exchange.create_market_order,
                        symbol, side, amount
                    )
                else:
                    order = self.circuit_breaker.call(
                        self.exchange.create_limit_order,
                        symbol, side, amount, price
                    )
                
                print(f"✅ Order placed: {side} {amount} {symbol}")
                return order
                
            except ccxt.InsufficientFunds as e:
                print(f"💸 Insufficient funds: {e}")
                raise  # Don't retry insufficient funds
                
            except ccxt.NetworkError as e:
                print(f"🌐 Network error placing order (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
                
            except ccxt.ExchangeError as e:
                print(f"🏪 Exchange error placing order (attempt {attempt + 1}): {e}")
                if "rate limit" in str(e).lower():
                    time.sleep(10)
                    continue
                raise
                
            except Exception as e:
                print(f"❌ Unexpected error placing order (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise
        
        return None

class ProductionTrader:
    """Production-ready trading system with enhanced reliability."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange_manager = ProductionExchangeManager(
            exchange_name=config.get('exchange', 'binance'),
            api_credentials=config.get('api_credentials', {})
        )
        self.log_rotator = LogRotator(max_size_mb=config.get('max_log_size_mb', 50))
        
        # File paths
        self.state_file = config.get('state_file', 'production_state.json')
        self.log_file = config.get('log_file', 'production_trading.log')
        self.coms_file = config.get('coms_file', 'coms.md')
        
        # Trading parameters
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.update_interval = config.get('update_interval', 300)
        self.position_size_range = config.get('position_size_range', (0.464, 0.800))
        
        # Load or initialize state
        self.state = self._load_state()
        
        print(f"🚀 Production trader initialized for {self.symbol}")
    
    def _load_state(self) -> Dict[str, Any]:
        """Load trading state from file."""
        default_state = {
            'session_start': datetime.now().isoformat(),
            'total_trades': 0,
            'current_position': 0.0,
            'cash_balance': 100000.0,
            'last_update': None,
            'errors': [],
            'performance_metrics': {}
        }
        
        return FileManager.safe_json_read(self.state_file, default_state)
    
    def _save_state(self):
        """Save trading state to file."""
        self.state['last_update'] = datetime.now().isoformat()
        FileManager.safe_json_write(self.state_file, self.state)
    
    def _log_message(self, message: str, level: str = "INFO"):
        """Log message with rotation support."""
        timestamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        log_line = f"{timestamp} [{level}] {message}\n"
        
        # Check if log rotation is needed
        if self.log_rotator.should_rotate(self.log_file):
            self.log_rotator.rotate_log(self.log_file)
        
        FileManager.safe_write(self.log_file, log_line, append=True)
        print(log_line.strip())
    
    def _update_coms(self, message: str):
        """Update communications log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        coms_entry = f"\n## {timestamp}\n{message}\n"
        
        FileManager.safe_write(self.coms_file, coms_entry, append=True)
    
    def fetch_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch current market data with error handling."""
        try:
            self._log_message(f"Fetching market data for {self.symbol}")
            
            data = self.exchange_manager.fetch_ohlcv(
                symbol=self.symbol,
                timeframe='4h',
                limit=50
            )
            
            if data is not None and len(data) > 0:
                self._log_message(f"✅ Fetched {len(data)} data points")
                return data
            else:
                self._log_message("⚠️ No market data received", "WARNING")
                return None
                
        except Exception as e:
            error_msg = f"❌ Market data fetch failed: {str(e)}"
            self._log_message(error_msg, "ERROR")
            self.state['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'function': 'fetch_market_data'
            })
            self._save_state()
            return None
    
    def execute_trading_logic(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute trading logic with comprehensive error handling."""
        try:
            # Simplified trading logic for demo
            # In production, this would use the enhanced Random Forest models
            
            latest_price = market_data['close'].iloc[-1]
            price_change = market_data['close'].pct_change().iloc[-1] * 100
            
            # Simple momentum-based logic
            signal = {
                'action': 'HOLD',
                'confidence': 0.0,
                'position_size': 0.0,
                'price': latest_price,
                'reasoning': 'No clear signal'
            }
            
            if price_change > 2.0:  # Simple buy signal
                signal.update({
                    'action': 'BUY',
                    'confidence': min(price_change / 5.0, 1.0),
                    'position_size': np.clip(price_change / 10.0, 0.1, 0.5),
                    'reasoning': f'Positive momentum: {price_change:.2f}%'
                })
            elif price_change < -2.0 and self.state['current_position'] > 0:
                signal.update({
                    'action': 'SELL',
                    'confidence': min(abs(price_change) / 5.0, 1.0),
                    'position_size': self.state['current_position'],
                    'reasoning': f'Negative momentum: {price_change:.2f}%'
                })
            
            self._log_message(f"Trading signal: {signal['action']} (conf: {signal['confidence']:.2f})")
            return signal
            
        except Exception as e:
            error_msg = f"❌ Trading logic failed: {str(e)}"
            self._log_message(error_msg, "ERROR")
            return {'action': 'HOLD', 'confidence': 0.0, 'error': str(e)}
    
    def execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Execute trade with comprehensive error handling."""
        if signal['action'] == 'HOLD' or signal['confidence'] < 0.6:
            return False
        
        try:
            # For demo purposes, simulate trade execution
            # In production, this would use exchange_manager.place_order()
            
            action = signal['action']
            size = signal['position_size']
            price = signal['price']
            
            if action == 'BUY':
                trade_value = size * price
                if trade_value <= self.state['cash_balance']:
                    self.state['current_position'] += size
                    self.state['cash_balance'] -= trade_value
                    self.state['total_trades'] += 1
                    
                    trade_msg = f"🟢 BUY {size:.6f} BTC @ ${price:,.2f} (${trade_value:,.2f})"
                    self._log_message(trade_msg)
                    self._update_coms(f"**TRADE EXECUTED**: {trade_msg}")
                    
                    return True
            
            elif action == 'SELL' and self.state['current_position'] >= size:
                trade_value = size * price
                self.state['current_position'] -= size
                self.state['cash_balance'] += trade_value
                self.state['total_trades'] += 1
                
                trade_msg = f"🔴 SELL {size:.6f} BTC @ ${price:,.2f} (${trade_value:,.2f})"
                self._log_message(trade_msg)
                self._update_coms(f"**TRADE EXECUTED**: {trade_msg}")
                
                return True
            
            self._save_state()
            return False
            
        except Exception as e:
            error_msg = f"❌ Trade execution failed: {str(e)}"
            self._log_message(error_msg, "ERROR")
            self.state['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'function': 'execute_trade'
            })
            self._save_state()
            return False
    
    def run_trading_cycle(self):
        """Run single trading cycle with full error handling."""
        try:
            cycle_start = datetime.now()
            self._log_message("🔄 Starting trading cycle")
            
            # Fetch market data
            market_data = self.fetch_market_data()
            if market_data is None:
                self._log_message("⚠️ Skipping cycle - no market data", "WARNING")
                return
            
            # Generate trading signal
            signal = self.execute_trading_logic(market_data)
            
            # Execute trade if signal is strong enough
            trade_executed = self.execute_trade(signal)
            
            # Update performance metrics
            current_value = self.state['cash_balance'] + (self.state['current_position'] * signal['price'])
            self.state['performance_metrics'] = {
                'current_value': current_value,
                'total_return_pct': ((current_value - 100000) / 100000) * 100,
                'total_trades': self.state['total_trades'],
                'cycle_duration': (datetime.now() - cycle_start).total_seconds()
            }
            
            # Save state
            self._save_state()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self._log_message(f"✅ Trading cycle complete ({cycle_duration:.1f}s)")
            
        except Exception as e:
            error_msg = f"❌ Trading cycle failed: {str(e)}"
            self._log_message(error_msg, "ERROR")
            self.state['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'function': 'run_trading_cycle'
            })
            self._save_state()

def main():
    """Demo of production deployment layer."""
    config = {
        'exchange': 'binance',
        'api_credentials': {
            'api_key': 'demo_key',
            'secret': 'demo_secret',
            'sandbox': True  # Use sandbox for testing
        },
        'symbol': 'BTC/USDT',
        'update_interval': 300,
        'max_log_size_mb': 10,
        'state_file': 'demo_production_state.json',
        'log_file': 'demo_production.log',
        'coms_file': 'demo_coms.md'
    }
    
    try:
        trader = ProductionTrader(config)
        
        print("🚀 Production deployment layer demo")
        print("Running 3 trading cycles...")
        
        for i in range(3):
            print(f"\n--- Cycle {i + 1} ---")
            trader.run_trading_cycle()
            time.sleep(2)  # Short delay for demo
        
        print(f"\n✅ Demo complete!")
        print(f"📊 Performance: {trader.state['performance_metrics']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()