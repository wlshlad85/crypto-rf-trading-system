"""Ultra-advanced target engineering for maximum prediction accuracy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import warnings
warnings.filterwarnings('ignore')

from utils.config import ModelConfig


class UltraTargetEngineer:
    """Ultra-advanced target engineering with risk-adjusted and regime-aware targets."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()
        self.quantile_transformer = QuantileTransformer(output_distribution='normal')
    
    def _get_price_columns(self, data: pd.DataFrame, symbol: str) -> Dict[str, str]:
        """Get the correct price column names for a symbol."""
        columns = {}
        
        # Try different naming conventions
        for price_type in ['close', 'high', 'low', 'open']:
            col = f"{symbol}_{price_type}"
            if col not in data.columns:
                col = f"{symbol}_primary_{price_type}"
            if col in data.columns:
                columns[price_type] = col
        
        return columns
        
    def create_ultra_targets(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Create ultra-advanced targets using sophisticated techniques."""
        self.logger.info("Creating ultra-advanced target variables")
        
        targets = pd.DataFrame(index=data.index)
        
        for symbol in symbols:
            price_cols = self._get_price_columns(data, symbol)
            if 'close' not in price_cols:
                self.logger.warning(f"No close price column found for {symbol}")
                continue
            
            self.logger.info(f"Creating ultra targets for {symbol}")
            
            # Basic return targets (only add this for now to test)
            targets = self._add_basic_targets(targets, data, symbol, price_cols)
            
            # For now, only add basic targets to test the system
            # Will add other targets later once basic targets work
        
        # Meta-targets combining multiple signals (commented out for now)
        # targets = self._add_meta_targets(targets, symbols)
        
        self.logger.info(f"Created {targets.shape[1]} ultra-advanced target variables")
        return targets
    
    def _add_basic_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str, price_cols: Dict[str, str]) -> pd.DataFrame:
        """Add basic return targets with multiple horizons."""
        close_prices = data[price_cols['close']]
        
        # Multiple prediction horizons
        horizons = [1, 3, 6, 12, 24, 48]  # Various time horizons
        
        for horizon in horizons:
            # Simple returns
            simple_returns = close_prices.pct_change(horizon).shift(-horizon)
            targets[f"{symbol}_return_{horizon}h"] = simple_returns
            
            # Log returns (more stable for large movements)
            log_returns = (np.log(close_prices) - np.log(close_prices.shift(horizon))).shift(-horizon)
            targets[f"{symbol}_log_return_{horizon}h"] = log_returns
            
            # Price direction (classification target)
            price_direction = np.sign(simple_returns)
            targets[f"{symbol}_direction_{horizon}h"] = price_direction
            
            # Magnitude of returns (regression on absolute returns)
            return_magnitude = abs(simple_returns)
            targets[f"{symbol}_magnitude_{horizon}h"] = return_magnitude
        
        return targets
    
    def _add_risk_adjusted_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add risk-adjusted target variables."""
        price_cols = self._get_price_columns(data, symbol)
        close_prices = data[price_cols['close']]
        returns = close_prices.pct_change()
        
        # Sharpe-ratio based targets
        for horizon in [6, 12, 24]:
            future_returns = close_prices.pct_change(horizon).shift(-horizon)
            
            # Historical volatility for scaling
            hist_vol = returns.rolling(100).std()
            
            # Sharpe-like target (return / volatility)
            sharpe_target = future_returns / (hist_vol * np.sqrt(horizon))
            targets[f"{symbol}_sharpe_target_{horizon}h"] = sharpe_target
            
            # Sortino-like target (return / downside volatility)
            downside_returns = returns.where(returns < 0, 0)
            downside_vol = downside_returns.rolling(100).std()
            sortino_target = future_returns / (downside_vol * np.sqrt(horizon))
            targets[f"{symbol}_sortino_target_{horizon}h"] = sortino_target
            
            # Information ratio target (excess return vs tracking error)
            # Using rolling mean as "benchmark"
            benchmark_return = returns.rolling(50).mean() * horizon
            excess_return = future_returns - benchmark_return.shift(-horizon)
            tracking_error = (returns - returns.rolling(50).mean()).rolling(100).std() * np.sqrt(horizon)
            info_ratio_target = excess_return / tracking_error
            targets[f"{symbol}_info_ratio_target_{horizon}h"] = info_ratio_target
        
        return targets
    
    def _add_regime_aware_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add regime-aware target variables."""
        price_cols = self._get_price_columns(data, symbol)
        close_prices = data[price_cols['close']]
        returns = close_prices.pct_change()
        
        # Volatility regime detection
        vol_short = returns.rolling(24).std()
        vol_long = returns.rolling(168).std()
        vol_regime = (vol_short > vol_long).astype(int)  # 1 = high vol, 0 = low vol
        
        # Trend regime detection
        sma_short = close_prices.rolling(24).mean()
        sma_long = close_prices.rolling(168).mean()
        trend_regime = (sma_short > sma_long).astype(int)  # 1 = uptrend, 0 = downtrend
        
        for horizon in [6, 12, 24]:
            future_returns = close_prices.pct_change(horizon).shift(-horizon)
            
            # Regime-conditional targets
            # High volatility regime
            high_vol_mask = vol_regime == 1
            targets[f"{symbol}_return_high_vol_{horizon}h"] = future_returns.where(high_vol_mask, np.nan)
            
            # Low volatility regime  
            low_vol_mask = vol_regime == 0
            targets[f"{symbol}_return_low_vol_{horizon}h"] = future_returns.where(low_vol_mask, np.nan)
            
            # Uptrend regime
            uptrend_mask = trend_regime == 1
            targets[f"{symbol}_return_uptrend_{horizon}h"] = future_returns.where(uptrend_mask, np.nan)
            
            # Downtrend regime
            downtrend_mask = trend_regime == 0
            targets[f"{symbol}_return_downtrend_{horizon}h"] = future_returns.where(downtrend_mask, np.nan)
            
            # Combined regime targets
            bull_market = (trend_regime == 1) & (vol_regime == 0)  # Uptrend + Low vol
            bear_market = (trend_regime == 0) & (vol_regime == 1)  # Downtrend + High vol
            
            targets[f"{symbol}_return_bull_{horizon}h"] = future_returns.where(bull_market, np.nan)
            targets[f"{symbol}_return_bear_{horizon}h"] = future_returns.where(bear_market, np.nan)
        
        return targets
    
    def _add_multi_horizon_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add multi-horizon aggregated targets."""
        close_col = f"{symbol}_close"
        close_prices = data[close_col]
        
        # Multi-horizon return aggregation
        horizons = [3, 6, 12, 24]
        
        # Compound returns over multiple horizons
        for end_horizon in [12, 24, 48]:
            compound_returns = []
            for h in range(1, end_horizon + 1):
                h_return = close_prices.pct_change(1).shift(-h)
                compound_returns.append(h_return)
            
            # Cumulative return
            if compound_returns:
                compound_return = pd.concat(compound_returns, axis=1).sum(axis=1)
                targets[f"{symbol}_compound_return_{end_horizon}h"] = compound_return
        
        # Maximum return over horizon
        for horizon in [12, 24, 48]:
            max_return = pd.Series(index=close_prices.index, dtype=float)
            for i in range(len(close_prices) - horizon):
                future_prices = close_prices.iloc[i+1:i+horizon+1]
                if len(future_prices) > 0:
                    max_ret = (future_prices.max() - close_prices.iloc[i]) / close_prices.iloc[i]
                    max_return.iloc[i] = max_ret
            targets[f"{symbol}_max_return_{horizon}h"] = max_return
        
        # Minimum return over horizon (for risk management)
        for horizon in [12, 24, 48]:
            min_return = pd.Series(index=close_prices.index, dtype=float)
            for i in range(len(close_prices) - horizon):
                future_prices = close_prices.iloc[i+1:i+horizon+1]
                if len(future_prices) > 0:
                    min_ret = (future_prices.min() - close_prices.iloc[i]) / close_prices.iloc[i]
                    min_return.iloc[i] = min_ret
            targets[f"{symbol}_min_return_{horizon}h"] = min_return
        
        return targets
    
    def _add_volatility_scaled_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add volatility-scaled target variables."""
        close_col = f"{symbol}_close"
        close_prices = data[close_col]
        returns = close_prices.pct_change()
        
        # Multiple volatility estimation methods
        vol_estimators = {
            'realized': returns.rolling(100).std(),
            'garman_klass': self._calculate_garman_klass_vol(data, symbol),
            'ewma': returns.ewm(span=50).std()
        }
        
        for vol_name, vol_est in vol_estimators.items():
            if vol_est is None:
                continue
                
            for horizon in [6, 12, 24]:
                future_returns = close_prices.pct_change(horizon).shift(-horizon)
                
                # Volatility-scaled returns
                vol_scaled_target = future_returns / (vol_est * np.sqrt(horizon))
                targets[f"{symbol}_vol_scaled_{vol_name}_{horizon}h"] = vol_scaled_target
                
                # Z-score like target
                expected_vol = vol_est * np.sqrt(horizon)
                z_score_target = (future_returns - 0) / expected_vol  # Assuming 0 expected return
                targets[f"{symbol}_zscore_{vol_name}_{horizon}h"] = z_score_target
        
        return targets
    
    def _add_momentum_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add momentum-based target variables."""
        close_col = f"{symbol}_close"
        close_prices = data[close_col]
        
        # Momentum persistence targets
        for lookback in [12, 24, 48]:
            historical_momentum = close_prices / close_prices.shift(lookback) - 1
            
            for horizon in [6, 12, 24]:
                future_returns = close_prices.pct_change(horizon).shift(-horizon)
                
                # Momentum continuation target
                momentum_sign = np.sign(historical_momentum)
                momentum_continuation = momentum_sign * future_returns
                targets[f"{symbol}_momentum_cont_{lookback}_{horizon}h"] = momentum_continuation
                
                # Momentum reversal target
                momentum_reversal = -momentum_sign * future_returns
                targets[f"{symbol}_momentum_rev_{lookback}_{horizon}h"] = momentum_reversal
        
        # Acceleration targets
        returns = close_prices.pct_change()
        for horizon in [6, 12, 24]:
            # Return acceleration (second derivative)
            return_accel = returns.diff(horizon).shift(-horizon)
            targets[f"{symbol}_return_accel_{horizon}h"] = return_accel
            
            # Momentum acceleration
            momentum = close_prices / close_prices.shift(12) - 1
            momentum_accel = momentum.diff(horizon).shift(-horizon)
            targets[f"{symbol}_momentum_accel_{horizon}h"] = momentum_accel
        
        return targets
    
    def _add_tail_risk_targets(self, targets: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add tail-risk aware target variables."""
        close_col = f"{symbol}_close"
        close_prices = data[close_col]
        returns = close_prices.pct_change()
        
        # Extreme movement prediction
        for horizon in [6, 12, 24]:
            future_returns = close_prices.pct_change(horizon).shift(-horizon)
            
            # Extreme positive movements (above 95th percentile)
            extreme_threshold = returns.rolling(500).quantile(0.95)
            extreme_positive = (future_returns > extreme_threshold).astype(int)
            targets[f"{symbol}_extreme_positive_{horizon}h"] = extreme_positive
            
            # Extreme negative movements (below 5th percentile)
            extreme_neg_threshold = returns.rolling(500).quantile(0.05)
            extreme_negative = (future_returns < extreme_neg_threshold).astype(int)
            targets[f"{symbol}_extreme_negative_{horizon}h"] = extreme_negative
            
            # Jump detection target
            vol_threshold = returns.rolling(100).std() * 3  # 3-sigma moves
            jump_target = (abs(future_returns) > vol_threshold).astype(int)
            targets[f"{symbol}_jump_{horizon}h"] = jump_target
            
            # Tail ratio target
            # Ratio of extreme positive to extreme negative movements
            pos_tail = future_returns.where(future_returns > extreme_threshold, 0)
            neg_tail = abs(future_returns.where(future_returns < extreme_neg_threshold, 0))
            tail_ratio = pos_tail / (neg_tail + 1e-8)  # Add small epsilon
            targets[f"{symbol}_tail_ratio_{horizon}h"] = tail_ratio
        
        return targets
    
    def _add_cross_sectional_targets(self, targets: pd.DataFrame, data: pd.DataFrame, 
                                   symbol: str, symbols: List[str]) -> pd.DataFrame:
        """Add cross-sectional (relative) target variables."""
        close_col = f"{symbol}_close"
        close_prices = data[close_col]
        
        # Calculate returns for all symbols
        all_returns = {}
        for sym in symbols:
            sym_close_col = f"{sym}_close"
            if sym_close_col in data.columns:
                all_returns[sym] = data[sym_close_col].pct_change()
        
        if len(all_returns) < 2:
            return targets
        
        for horizon in [6, 12, 24]:
            future_returns = close_prices.pct_change(horizon).shift(-horizon)
            
            # Relative performance vs other assets
            other_symbols = [s for s in symbols if s != symbol and f"{s}_close" in data.columns]
            
            if other_symbols:
                # Average return of other symbols
                other_returns = []
                for other_sym in other_symbols:
                    other_close = data[f"{other_sym}_close"]
                    other_ret = other_close.pct_change(horizon).shift(-horizon)
                    other_returns.append(other_ret)
                
                if other_returns:
                    avg_other_return = pd.concat(other_returns, axis=1).mean(axis=1)
                    
                    # Relative outperformance target
                    relative_performance = future_returns - avg_other_return
                    targets[f"{symbol}_relative_performance_{horizon}h"] = relative_performance
                    
                    # Ranking target (percentile rank among all assets)
                    all_future_returns = [future_returns] + other_returns
                    combined_returns = pd.concat(all_future_returns, axis=1)
                    ranking = combined_returns.rank(axis=1, pct=True).iloc[:, 0]  # Rank of current symbol
                    targets[f"{symbol}_ranking_{horizon}h"] = ranking
        
        return targets
    
    def _add_meta_targets(self, targets: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Add meta-targets combining multiple signals."""
        for symbol in symbols:
            # Combine risk-adjusted targets
            sharpe_cols = [col for col in targets.columns if f"{symbol}_sharpe_target" in col]
            if len(sharpe_cols) > 1:
                meta_sharpe = targets[sharpe_cols].mean(axis=1)
                targets[f"{symbol}_meta_sharpe"] = meta_sharpe
            
            # Combine regime targets
            regime_cols = [col for col in targets.columns if symbol in col and any(regime in col for regime in ['bull', 'bear', 'uptrend', 'downtrend'])]
            if len(regime_cols) > 1:
                # Weighted average of regime targets (handling NaN values)
                regime_targets = targets[regime_cols]
                meta_regime = regime_targets.mean(axis=1, skipna=True)
                targets[f"{symbol}_meta_regime"] = meta_regime
            
            # Combine directional targets
            direction_cols = [col for col in targets.columns if f"{symbol}_direction" in col]
            if len(direction_cols) > 1:
                # Majority vote for direction
                direction_vote = targets[direction_cols].mode(axis=1)
                if not direction_vote.empty:
                    targets[f"{symbol}_meta_direction"] = direction_vote[0] if len(direction_vote.columns) > 0 else np.nan
            
            # Confidence score (agreement between different target types)
            return_cols = [col for col in targets.columns if f"{symbol}_return" in col and "meta" not in col]
            if len(return_cols) > 3:
                # Calculate correlation between different return targets
                return_corr = targets[return_cols].corr().mean().mean()
                targets[f"{symbol}_target_confidence"] = return_corr
        
        return targets
    
    def _calculate_garman_klass_vol(self, data: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """Calculate Garman-Klass volatility estimator."""
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        open_col = f"{symbol}_open"
        close_col = f"{symbol}_close"
        
        if not all(col in data.columns for col in [high_col, low_col, open_col, close_col]):
            return None
        
        high = data[high_col]
        low = data[low_col]
        open_price = data[open_col]
        close = data[close_col]
        
        # Garman-Klass estimator
        gk_vol = np.sqrt(
            0.5 * (np.log(high / low)) ** 2 - 
            (2 * np.log(2) - 1) * (np.log(close / open_price)) ** 2
        )
        
        return gk_vol.rolling(30).mean()  # Smoothed version
    
    def get_primary_target(self, targets: pd.DataFrame, symbol: str, target_type: str = "meta_sharpe") -> str:
        """Get the primary target column for a symbol."""
        primary_target_col = f"{symbol}_{target_type}"
        
        if primary_target_col in targets.columns:
            return primary_target_col
        
        # Fallback to basic return target
        basic_target_col = f"{symbol}_return_{self.config.target_horizon}h"
        if basic_target_col in targets.columns:
            return basic_target_col
        
        # Final fallback
        return_cols = [col for col in targets.columns if f"{symbol}_return" in col]
        if return_cols:
            return return_cols[0]
        
        raise ValueError(f"No suitable target found for {symbol}")
    
    def prepare_targets_for_training(self, targets: pd.DataFrame, primary_target_col: str) -> Tuple[pd.Series, Dict]:
        """Prepare targets for training with preprocessing."""
        target_series = targets[primary_target_col].copy()
        
        # Remove infinite values
        target_series = target_series.replace([np.inf, -np.inf], np.nan)
        
        # Store statistics before preprocessing
        target_stats = {
            'mean': target_series.mean(),
            'std': target_series.std(),
            'skew': target_series.skew(),
            'kurt': target_series.kurtosis(),
            'quantiles': target_series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
        }
        
        # Winsorization (clip extreme values)
        lower_bound = target_series.quantile(0.01)
        upper_bound = target_series.quantile(0.99)
        target_series = target_series.clip(lower_bound, upper_bound)
        
        # Optional: Robust scaling or quantile transformation
        if self.config.target_type == "normalized":
            # Robust standardization
            target_series = (target_series - target_series.median()) / target_series.mad()
        elif self.config.target_type == "quantile":
            # Quantile transformation to normal distribution
            target_array = target_series.dropna().values.reshape(-1, 1)
            if len(target_array) > 100:  # Ensure sufficient data
                transformed = self.quantile_transformer.fit_transform(target_array)
                target_series.loc[target_series.dropna().index] = transformed.flatten()
        
        self.logger.info(f"Prepared target {primary_target_col}: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}")
        
        return target_series, target_stats