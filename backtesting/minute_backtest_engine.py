"""Ultra-fast backtesting engine optimized for minute-level cryptocurrency trading."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import multiprocessing as mp

try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

try:
    import empyrical as emp
except ImportError:
    emp = None

from utils.config import BacktestConfig, Config
from models.minute_random_forest_model import MinuteRandomForestModel
from data.minute_data_manager import MinuteDataManager
from features.minute_feature_engineering import MinuteFeatureEngine


class MinuteBacktestEngine:
    """Ultra-fast backtesting engine optimized for minute-level data."""
    
    def __init__(self, config: Config = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization settings
        self.batch_size = 1440  # Process 1 day at a time (1440 minutes)
        self.parallel_processing = True
        self.use_numba_acceleration = NUMBA_AVAILABLE
        self.memory_efficient = True
        
        # Results storage
        self.results = {}
        self.portfolio_history = pd.DataFrame()
        self.trades_history = pd.DataFrame()
        self.performance_metrics = {}
        
        # Caching for performance
        self.prediction_cache = {}
        self.feature_cache = {}
        
        # Real-time tracking
        self.current_positions = {}
        self.current_portfolio_value = 0
        self.transaction_costs = 0
        
    def _get_default_config(self) -> Config:
        """Get default configuration optimized for minute backtesting."""
        from types import SimpleNamespace
        
        config = SimpleNamespace()
        config.backtest = SimpleNamespace()
        config.model = SimpleNamespace()
        config.strategy = SimpleNamespace()
        
        # Backtest settings
        config.backtest.initial_capital = 100000
        config.backtest.commission = 0.001  # 0.1% per trade
        config.backtest.slippage = 0.0005   # 0.05% slippage
        config.backtest.benchmark_symbol = 'BTC-USD'
        config.backtest.risk_free_rate = 0.02
        config.backtest.start_date = '2024-01-01'
        config.backtest.end_date = '2024-06-30'
        
        # Model settings
        config.model.target_horizon = 5  # 5 minutes
        config.model.retrain_frequency = 1440  # Retrain every 24 hours
        config.model.min_training_samples = 1000
        
        # Strategy settings
        config.strategy.max_positions = 5
        config.strategy.position_sizing = 'equal_weight'
        config.strategy.rebalance_frequency = 60  # Every hour
        config.strategy.stop_loss = 0.02  # 2%
        config.strategy.take_profit = 0.05  # 5%
        
        return config
    
    def run_ultra_fast_backtest(self, data_dict: Dict[str, pd.DataFrame], 
                               model: MinuteRandomForestModel, 
                               symbols: List[str]) -> Dict[str, Any]:
        """Run ultra-fast backtesting on minute-level data."""
        
        self.logger.info("Starting ultra-fast minute-level backtest")
        start_time = datetime.now()
        
        # Prepare data for fast processing
        aligned_data = self._prepare_aligned_data(data_dict, symbols)
        
        if aligned_data.empty:
            return {"error": "No aligned data available for backtesting"}
        
        # Initialize portfolio
        portfolio_tracker = FastPortfolioTracker(
            initial_capital=self.config.backtest.initial_capital,
            commission=self.config.backtest.commission,
            slippage=self.config.backtest.slippage
        )
        
        # Generate features in batches
        feature_engine = MinuteFeatureEngine()
        
        # Process data in batches for memory efficiency
        batch_results = []
        n_batches = len(aligned_data) // self.batch_size + 1
        
        self.logger.info(f"Processing {len(aligned_data)} minutes in {n_batches} batches")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(aligned_data))
            
            if start_idx >= len(aligned_data):
                break
            
            batch_data = aligned_data.iloc[start_idx:end_idx]
            
            try:
                batch_result = self._process_backtest_batch(
                    batch_data, model, symbols, feature_engine, 
                    portfolio_tracker, batch_idx
                )
                batch_results.append(batch_result)
                
                # Memory cleanup
                if batch_idx % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Compile final results
        final_results = self._compile_batch_results(batch_results, portfolio_tracker)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_ultra_fast_metrics(
            portfolio_tracker.get_portfolio_history(),
            aligned_data,
            symbols
        )
        
        final_results.update({
            'performance_metrics': self.performance_metrics,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'data_points_processed': len(aligned_data),
            'symbols_traded': symbols,
            'batches_processed': len(batch_results)
        })
        
        self.results = final_results
        
        self.logger.info(f"Ultra-fast backtest completed in {final_results['processing_time']:.2f} seconds")
        self.logger.info(f"Final portfolio value: ${final_results.get('final_value', 0):,.2f}")
        
        return final_results
    
    def _prepare_aligned_data(self, data_dict: Dict[str, pd.DataFrame], 
                            symbols: List[str]) -> pd.DataFrame:
        """Prepare aligned data for fast processing."""
        
        if not data_dict:
            return pd.DataFrame()
        
        # Find common time index
        common_index = None
        for symbol, df in data_dict.items():
            if symbol in symbols:
                if common_index is None:
                    common_index = df.index
                else:
                    common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()
        
        # Create aligned dataset
        aligned_data = pd.DataFrame(index=common_index)
        
        for symbol in symbols:
            if symbol in data_dict:
                df = data_dict[symbol]
                
                # Add price columns
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        aligned_data[f'{symbol}_{col.lower()}'] = df[col].reindex(common_index)
        
        # Forward fill missing values (limited to avoid lookahead bias)
        aligned_data = aligned_data.fillna(method='ffill', limit=3)
        aligned_data = aligned_data.dropna()
        
        self.logger.info(f"Aligned data shape: {aligned_data.shape}")
        return aligned_data
    
    def _process_backtest_batch(self, batch_data: pd.DataFrame, 
                              model: MinuteRandomForestModel,
                              symbols: List[str], 
                              feature_engine: MinuteFeatureEngine,
                              portfolio_tracker: 'FastPortfolioTracker',
                              batch_idx: int) -> Dict[str, Any]:
        """Process a single batch of backtest data."""
        
        # Generate features for this batch
        batch_features = self._generate_batch_features(batch_data, feature_engine, symbols)
        
        if batch_features.empty:
            return {'batch_idx': batch_idx, 'processed_minutes': 0, 'trades': 0}
        
        # Generate predictions
        try:
            predictions = model.predict_multi_horizon(batch_features, symbols)
        except Exception as e:
            self.logger.warning(f"Prediction failed for batch {batch_idx}: {e}")
            predictions = pd.DataFrame(index=batch_features.index)
        
        # Generate trading signals
        signals = self._generate_fast_signals(predictions, batch_data, symbols)
        
        # Execute trades and update portfolio
        trades_executed = 0
        
        for timestamp in batch_data.index:
            if timestamp in signals.index:
                current_prices = batch_data.loc[timestamp]
                signal_row = signals.loc[timestamp]
                
                # Execute trades
                trades_count = portfolio_tracker.update_portfolio(
                    signal_row, current_prices, timestamp
                )
                trades_executed += trades_count
        
        return {
            'batch_idx': batch_idx,
            'processed_minutes': len(batch_data),
            'trades': trades_executed,
            'final_value': portfolio_tracker.current_value
        }
    
    def _generate_batch_features(self, batch_data: pd.DataFrame, 
                               feature_engine: MinuteFeatureEngine,
                               symbols: List[str]) -> pd.DataFrame:
        """Generate features for a batch of data."""
        
        # Check cache first
        cache_key = f"features_{batch_data.index[0]}_{batch_data.index[-1]}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        all_features = pd.DataFrame(index=batch_data.index)
        
        for symbol in symbols:
            # Extract symbol data
            symbol_data = pd.DataFrame()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                col_name = f'{symbol}_{col}'
                if col_name in batch_data.columns:
                    symbol_data[col.title()] = batch_data[col_name]
            
            if symbol_data.empty:
                continue
            
            try:
                # Generate features for this symbol
                symbol_features = feature_engine.generate_minute_features(symbol_data, symbol)
                
                # Add to combined features with symbol prefix
                for col in symbol_features.columns:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        all_features[f'{symbol}_{col}'] = symbol_features[col]
                
            except Exception as e:
                self.logger.warning(f"Feature generation failed for {symbol}: {e}")
                continue
        
        # Cache features
        if len(self.feature_cache) < 100:  # Limit cache size
            self.feature_cache[cache_key] = all_features
        
        return all_features
    
    def _generate_fast_signals(self, predictions: pd.DataFrame, 
                             price_data: pd.DataFrame, 
                             symbols: List[str]) -> pd.DataFrame:
        """Generate trading signals using vectorized operations."""
        
        if predictions.empty:
            return pd.DataFrame(index=price_data.index)
        
        signals = pd.DataFrame(index=predictions.index)
        
        for symbol in symbols:
            # Use shortest horizon prediction for faster signals
            pred_col = f'{symbol}_1min_pred'
            
            if pred_col in predictions.columns:
                pred_values = predictions[pred_col]
                
                # Simple threshold-based signals (optimized for speed)
                buy_threshold = pred_values.quantile(0.7)   # Top 30% predictions
                sell_threshold = pred_values.quantile(0.3)  # Bottom 30% predictions
                
                signal = pd.Series(0, index=pred_values.index)
                signal[pred_values > buy_threshold] = 1    # Buy signal
                signal[pred_values < sell_threshold] = -1  # Sell signal
                
                signals[f'{symbol}_signal'] = signal
            else:
                signals[f'{symbol}_signal'] = 0
        
        return signals
    
    def _compile_batch_results(self, batch_results: List[Dict], 
                             portfolio_tracker: 'FastPortfolioTracker') -> Dict[str, Any]:
        """Compile results from all batches."""
        
        total_minutes = sum(batch['processed_minutes'] for batch in batch_results)
        total_trades = sum(batch['trades'] for batch in batch_results)
        
        # Get portfolio history
        portfolio_history = portfolio_tracker.get_portfolio_history()
        trades_history = portfolio_tracker.get_trades_history()
        
        return {
            'total_minutes_processed': total_minutes,
            'total_trades_executed': total_trades,
            'batches_processed': len(batch_results),
            'initial_value': self.config.backtest.initial_capital,
            'final_value': portfolio_tracker.current_value,
            'total_return': (portfolio_tracker.current_value / self.config.backtest.initial_capital) - 1,
            'portfolio_history': portfolio_history.to_dict('records') if not portfolio_history.empty else [],
            'trades_history': trades_history.to_dict('records') if not trades_history.empty else [],
            'transaction_costs': portfolio_tracker.total_costs
        }
    
    def _calculate_ultra_fast_metrics(self, portfolio_history: pd.DataFrame,
                                    price_data: pd.DataFrame, 
                                    symbols: List[str]) -> Dict[str, float]:
        """Calculate performance metrics optimized for speed."""
        
        if portfolio_history.empty:
            return {}
        
        # Portfolio returns
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics (vectorized for speed)
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Annualized metrics (assuming minute data)
        minutes_per_year = 365.25 * 24 * 60
        periods = len(returns)
        years = periods / minutes_per_year
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(minutes_per_year)
        
        # Risk metrics
        risk_free_rate = self.config.backtest.risk_free_rate
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Drawdown (vectorized)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Win rate approximation
        positive_returns = (returns > 0).sum()
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_periods': int(periods),
            'avg_return_per_minute': float(returns.mean()),
            'return_skewness': float(returns.skew()),
            'return_kurtosis': float(returns.kurtosis())
        }
    
    def run_parallel_backtest(self, data_dict: Dict[str, pd.DataFrame],
                            model: MinuteRandomForestModel,
                            symbols: List[str],
                            n_workers: int = None) -> Dict[str, Any]:
        """Run backtest using parallel processing."""
        
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers
        
        self.logger.info(f"Running parallel backtest with {n_workers} workers")
        
        # Prepare data
        aligned_data = self._prepare_aligned_data(data_dict, symbols)
        
        if aligned_data.empty:
            return {"error": "No aligned data available"}
        
        # Split data into chunks for parallel processing
        chunk_size = len(aligned_data) // n_workers
        data_chunks = []
        
        for i in range(n_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_workers - 1 else len(aligned_data)
            
            if start_idx < len(aligned_data):
                chunk = aligned_data.iloc[start_idx:end_idx]
                data_chunks.append((chunk, i))
        
        # Process chunks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_parallel_chunk, chunk, chunk_id, model, symbols): chunk_id
                for chunk, chunk_id in data_chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed chunk {chunk_id}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_id}: {e}")
        
        # Combine results
        combined_results = self._combine_parallel_results(results)
        
        return combined_results
    
    def _process_parallel_chunk(self, chunk_data: pd.DataFrame, chunk_id: int,
                              model: MinuteRandomForestModel, symbols: List[str]) -> Dict[str, Any]:
        """Process a chunk of data in parallel."""
        
        # Create a simplified portfolio tracker for this chunk
        chunk_tracker = FastPortfolioTracker(
            initial_capital=self.config.backtest.initial_capital / len(symbols),
            commission=self.config.backtest.commission,
            slippage=self.config.backtest.slippage
        )
        
        feature_engine = MinuteFeatureEngine()
        
        # Process this chunk
        chunk_features = self._generate_batch_features(chunk_data, feature_engine, symbols)
        
        try:
            predictions = model.predict_multi_horizon(chunk_features, symbols)
            signals = self._generate_fast_signals(predictions, chunk_data, symbols)
            
            trades_executed = 0
            for timestamp in chunk_data.index:
                if timestamp in signals.index:
                    current_prices = chunk_data.loc[timestamp]
                    signal_row = signals.loc[timestamp]
                    trades_count = chunk_tracker.update_portfolio(signal_row, current_prices, timestamp)
                    trades_executed += trades_count
            
            return {
                'chunk_id': chunk_id,
                'processed_minutes': len(chunk_data),
                'trades': trades_executed,
                'final_value': chunk_tracker.current_value,
                'portfolio_history': chunk_tracker.get_portfolio_history(),
                'success': True
            }
            
        except Exception as e:
            return {
                'chunk_id': chunk_id,
                'error': str(e),
                'success': False
            }
    
    def _combine_parallel_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from parallel processing."""
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful parallel chunks processed'}
        
        total_minutes = sum(r['processed_minutes'] for r in successful_results)
        total_trades = sum(r['trades'] for r in successful_results)
        
        # Combine portfolio histories
        all_histories = []
        for result in successful_results:
            if 'portfolio_history' in result and not result['portfolio_history'].empty:
                all_histories.append(result['portfolio_history'])
        
        combined_history = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
        
        return {
            'total_minutes_processed': total_minutes,
            'total_trades_executed': total_trades,
            'successful_chunks': len(successful_results),
            'combined_portfolio_history': combined_history.to_dict('records') if not combined_history.empty else [],
            'processing_method': 'parallel'
        }


class FastPortfolioTracker:
    """Ultra-fast portfolio tracker optimized for minute-level backtesting."""
    
    def __init__(self, initial_capital: float, commission: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.commission = commission
        self.slippage = slippage
        self.total_costs = 0
        
        # History tracking (optimized for speed)
        self.portfolio_snapshots = []
        self.trades_history = []
        
    def update_portfolio(self, signals: pd.Series, prices: pd.Series, timestamp: datetime) -> int:
        """Update portfolio based on signals. Returns number of trades executed."""
        
        trades_executed = 0
        
        for symbol in signals.index:
            if f'{symbol}_signal' in signals.index:
                signal = signals[f'{symbol}_signal']
                
                if signal == 0:  # No action
                    continue
                
                # Get current price
                price_col = f'{symbol}_close'
                if price_col not in prices.index:
                    continue
                
                current_price = prices[price_col]
                
                if pd.isna(current_price) or current_price <= 0:
                    continue
                
                # Execute trade
                if self._execute_trade(symbol, signal, current_price, timestamp):
                    trades_executed += 1
        
        # Update portfolio value
        self._update_portfolio_value(prices, timestamp)
        
        return trades_executed
    
    def _execute_trade(self, symbol: str, signal: float, price: float, timestamp: datetime) -> bool:
        """Execute a single trade."""
        
        current_position = self.positions.get(symbol, 0)
        
        # Calculate target position size (simplified)
        max_position_value = self.current_value * 0.2  # Max 20% per position
        target_quantity = 0
        
        if signal > 0:  # Buy signal
            target_quantity = max_position_value / price
        elif signal < 0 and current_position > 0:  # Sell signal
            target_quantity = 0  # Close position
        
        # Calculate trade quantity
        trade_quantity = target_quantity - current_position
        
        if abs(trade_quantity) < 0.001:  # Minimum trade size
            return False
        
        # Apply slippage
        trade_price = price * (1 + self.slippage * np.sign(trade_quantity))
        
        # Calculate costs
        trade_value = abs(trade_quantity * trade_price)
        cost = trade_value * self.commission
        
        # Check if we have enough cash for buying
        if trade_quantity > 0 and (trade_value + cost) > self.cash:
            # Adjust trade size to available cash
            available_for_trade = self.cash - cost
            if available_for_trade > 0:
                trade_quantity = available_for_trade / trade_price
            else:
                return False
        
        # Execute trade
        self.positions[symbol] = current_position + trade_quantity
        self.cash -= trade_quantity * trade_price + cost
        self.total_costs += cost
        
        # Record trade
        self.trades_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': trade_quantity,
            'price': trade_price,
            'value': trade_quantity * trade_price,
            'cost': cost,
            'signal': signal
        })
        
        return True
    
    def _update_portfolio_value(self, prices: pd.Series, timestamp: datetime):
        """Update current portfolio value."""
        
        # Calculate position values
        position_value = 0
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                price_col = f'{symbol}_close'
                if price_col in prices.index:
                    current_price = prices[price_col]
                    if not pd.isna(current_price) and current_price > 0:
                        position_value += quantity * current_price
        
        self.current_value = self.cash + position_value
        
        # Record portfolio snapshot
        self.portfolio_snapshots.append({
            'timestamp': timestamp,
            'total_value': self.current_value,
            'cash': self.cash,
            'position_value': position_value,
            'num_positions': len([p for p in self.positions.values() if p != 0])
        })
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        if not self.portfolio_snapshots:
            return pd.DataFrame()
        
        return pd.DataFrame(self.portfolio_snapshots).set_index('timestamp')
    
    def get_trades_history(self) -> pd.DataFrame:
        """Get trades history as DataFrame."""
        if not self.trades_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades_history).set_index('timestamp')


# Numba-accelerated functions for ultra-fast calculations

@jit(nopython=True)
def fast_returns_calculation(prices: np.ndarray) -> np.ndarray:
    """Calculate returns using Numba for speed."""
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
    return returns


@jit(nopython=True)
def fast_drawdown_calculation(cumulative_returns: np.ndarray) -> np.ndarray:
    """Calculate drawdown using Numba for speed."""
    drawdown = np.zeros(len(cumulative_returns))
    peak = cumulative_returns[0]
    
    for i in range(len(cumulative_returns)):
        if cumulative_returns[i] > peak:
            peak = cumulative_returns[i]
        drawdown[i] = (cumulative_returns[i] - peak) / peak
    
    return drawdown


@jit(nopython=True)
def fast_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate moving average using Numba for speed."""
    ma = np.zeros(len(prices))
    for i in range(window-1, len(prices)):
        ma[i] = np.mean(prices[i-window+1:i+1])
    return ma


# Utility functions

def create_minute_backtest_engine(symbols: List[str] = None, 
                                config: Config = None) -> MinuteBacktestEngine:
    """Create a configured minute backtest engine."""
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    return MinuteBacktestEngine(config)


def run_6_month_minute_backtest(symbols: List[str] = None,
                               model_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a complete 6-month minute-level backtest."""
    
    if symbols is None:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # Create components
    data_manager = MinuteDataManager()
    model = MinuteRandomForestModel()
    engine = MinuteBacktestEngine()
    
    # Fetch data
    print("Fetching 6-month minute data...")
    data_dict = data_manager.fetch_6_month_minute_data(symbols, '1m')
    
    if not data_dict:
        return {'error': 'No data fetched'}
    
    # Generate features and targets
    print("Generating features...")
    feature_engine = MinuteFeatureEngine()
    
    all_features = {}
    all_targets = {}
    
    for symbol, data in data_dict.items():
        features = feature_engine.generate_minute_features(data, symbol)
        targets = model.prepare_multi_horizon_targets(
            pd.DataFrame({f'{symbol}_close': data['Close']}), [symbol]
        )
        
        all_features[symbol] = features
        all_targets[symbol] = targets
    
    # Combine features
    combined_features = pd.concat(all_features.values(), axis=1, sort=True)
    combined_targets = pd.concat(all_targets.values(), axis=1, sort=True)
    
    # Train model
    print("Training models...")
    model.train_multi_horizon_models(combined_features, combined_targets, symbols)
    
    # Run backtest
    print("Running backtest...")
    results = engine.run_ultra_fast_backtest(data_dict, model, symbols)
    
    return results


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    symbols = ['BTC-USD', 'ETH-USD']
    
    # Create sample minute data
    dates = pd.date_range('2024-01-01', '2024-01-02', freq='1T')
    np.random.seed(42)
    
    sample_data = {}
    for symbol in symbols:
        price = 50000 + np.random.randn(len(dates)).cumsum() * 100
        df = pd.DataFrame({
            'Open': price + np.random.randn(len(dates)) * 10,
            'High': price + np.random.randn(len(dates)) * 10 + 20,
            'Low': price + np.random.randn(len(dates)) * 10 - 20,
            'Close': price,
            'Volume': np.random.exponential(1000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships
        df['High'] = np.maximum.reduce([df['Open'], df['Close'], df['High']])
        df['Low'] = np.minimum.reduce([df['Open'], df['Close'], df['Low']])
        
        sample_data[symbol] = df
    
    print(f"Sample data created for {len(symbols)} symbols, {len(dates)} minutes each")
    
    # Create and test engine
    engine = create_minute_backtest_engine(symbols)
    model = MinuteRandomForestModel()
    
    # Quick test (without full training)
    print("Running test backtest...")
    
    # Create dummy features
    n_features = 20
    dummy_features = pd.DataFrame(
        np.random.randn(len(dates), n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create simple dummy model for testing
    class DummyModel:
        def predict_multi_horizon(self, features, symbols):
            predictions = pd.DataFrame(index=features.index)
            for symbol in symbols:
                predictions[f'{symbol}_1min_pred'] = np.random.randn(len(features)) * 0.01
            return predictions
    
    dummy_model = DummyModel()
    
    # Test backtest
    results = engine.run_ultra_fast_backtest(sample_data, dummy_model, symbols)
    
    print("\nBacktest Results:")
    print(f"Processing time: {results.get('processing_time', 0):.2f} seconds")
    print(f"Total trades: {results.get('total_trades_executed', 0)}")
    print(f"Final value: ${results.get('final_value', 0):,.2f}")
    print(f"Total return: {results.get('total_return', 0):.2%}")
    
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Win rate: {metrics.get('win_rate', 0):.2%}")