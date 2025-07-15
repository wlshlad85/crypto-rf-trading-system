"""Interactive visualization and reporting suite for minute-level cryptocurrency trading analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots will be disabled.")

from analytics.minute_performance_analytics import MinutePerformanceAnalytics


class MinuteVisualizationSuite:
    """Comprehensive visualization suite for minute-level trading analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Styling configuration
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Set default style (only if matplotlib is available)
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use('seaborn-v0_8')
                sns.set_palette(self.color_palette)
            except:
                pass  # Continue without styling
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default visualization configuration."""
        return {
            'figure_size': (15, 10),
            'dpi': 100,
            'save_format': 'png',
            'interactive_plots': PLOTLY_AVAILABLE,
            'color_scheme': 'professional',
            'font_size': 12,
            'line_width': 2,
            'alpha': 0.7
        }
    
    def create_comprehensive_dashboard(self, portfolio_history: pd.DataFrame,
                                     trades_history: pd.DataFrame = None,
                                     market_data: Dict[str, pd.DataFrame] = None,
                                     predictions: pd.DataFrame = None,
                                     analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a comprehensive dashboard with all key visualizations."""
        
        self.logger.info("Creating comprehensive minute-level trading dashboard")
        
        dashboard_components = {}
        
        # 1. Portfolio performance overview
        dashboard_components['portfolio_overview'] = self._create_portfolio_overview(
            portfolio_history, analysis_results
        )
        
        # 2. Risk analysis charts
        dashboard_components['risk_analysis'] = self._create_risk_analysis_charts(
            portfolio_history, analysis_results
        )
        
        # 3. Intraday patterns
        dashboard_components['intraday_patterns'] = self._create_intraday_analysis(
            portfolio_history, analysis_results
        )
        
        # 4. Trade analysis
        if trades_history is not None and not trades_history.empty:
            dashboard_components['trade_analysis'] = self._create_trade_analysis(trades_history)
        
        # 5. Market data analysis
        if market_data:
            dashboard_components['market_analysis'] = self._create_market_analysis(market_data)
        
        # 6. Model performance analysis
        if predictions is not None and not predictions.empty:
            dashboard_components['model_analysis'] = self._create_model_analysis(predictions)
        
        # 7. Feature importance visualization
        dashboard_components['feature_analysis'] = self._create_feature_analysis(analysis_results)
        
        # 8. Correlation analysis
        dashboard_components['correlation_analysis'] = self._create_correlation_analysis(
            portfolio_history, market_data, predictions
        )
        
        self.logger.info("Dashboard creation completed")
        return dashboard_components
    
    def _create_portfolio_overview(self, portfolio_history: pd.DataFrame,
                                 analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create portfolio performance overview charts."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle('Portfolio Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. Portfolio value over time
        ax1 = axes[0, 0]
        portfolio_values = portfolio_history['total_value']
        ax1.plot(portfolio_history.index, portfolio_values, linewidth=2, color=self.color_palette[0])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis for minute data
        if len(portfolio_history) > 1440:  # More than 1 day
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Returns distribution
        ax2 = axes[0, 1]
        returns = portfolio_values.pct_change().dropna()
        ax2.hist(returns, bins=50, alpha=0.7, color=self.color_palette[1], edgecolor='black')
        ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative returns
        ax3 = axes[1, 0]
        cumulative_returns = (1 + returns).cumprod() - 1
        ax3.plot(cumulative_returns.index, cumulative_returns * 100, 
                linewidth=2, color=self.color_palette[2])
        ax3.set_title('Cumulative Returns')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(cumulative_returns) > 1440:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        else:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Key metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if analysis_results and 'basic_metrics' in analysis_results:
            metrics = analysis_results['basic_metrics']
            metrics_text = f"""
Key Performance Metrics:

Total Return: {metrics.get('total_return', 0):.2%}
Annualized Return: {metrics.get('annualized_return', 0):.2%}
Volatility: {metrics.get('volatility', 0):.2%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Win Rate: {metrics.get('win_rate', 0):.2%}

Risk Metrics:
Max Drawdown: {analysis_results.get('risk_metrics', {}).get('max_drawdown', 0):.2%}
VaR (95%): {analysis_results.get('risk_metrics', {}).get('var_95', 0):.4f}
            """
        else:
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            volatility = returns.std() * np.sqrt(365 * 24 * 60)
            metrics_text = f"""
Basic Metrics:

Total Return: {total_return:.2%}
Volatility: {volatility:.2%}
Average Return: {returns.mean():.4f}
            """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        portfolio_overview_path = 'portfolio_overview.png'
        plt.savefig(portfolio_overview_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': portfolio_overview_path,
            'description': 'Portfolio performance overview with value, returns, and key metrics'
        }
    
    def _create_risk_analysis_charts(self, portfolio_history: pd.DataFrame,
                                   analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create risk analysis visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        # 1. Drawdown chart
        ax1 = axes[0, 0]
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax1.fill_between(drawdown.index, drawdown * 100, 0, 
                        alpha=0.7, color='red', label='Drawdown')
        ax1.set_title('Portfolio Drawdowns')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(drawdown) > 1440:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Rolling volatility
        ax2 = axes[0, 1]
        rolling_vol_1h = returns.rolling(60).std() * np.sqrt(365 * 24 * 60)
        rolling_vol_4h = returns.rolling(240).std() * np.sqrt(365 * 24 * 60)
        
        ax2.plot(rolling_vol_1h.index, rolling_vol_1h * 100, 
                label='1-Hour Rolling Vol', alpha=0.8, linewidth=1)
        ax2.plot(rolling_vol_4h.index, rolling_vol_4h * 100, 
                label='4-Hour Rolling Vol', linewidth=2)
        ax2.set_title('Rolling Volatility')
        ax2.set_ylabel('Annualized Volatility (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. VaR analysis
        ax3 = axes[1, 0]
        
        # Calculate rolling VaR
        rolling_var_95 = returns.rolling(240).quantile(0.05)  # 4-hour window
        rolling_var_99 = returns.rolling(240).quantile(0.01)
        
        ax3.plot(rolling_var_95.index, rolling_var_95 * 100, 
                label='VaR 95%', color='orange', linewidth=2)
        ax3.plot(rolling_var_99.index, rolling_var_99 * 100, 
                label='VaR 99%', color='red', linewidth=2)
        ax3.set_title('Rolling Value at Risk')
        ax3.set_ylabel('VaR (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Return vs Risk scatter
        ax4 = axes[1, 1]
        
        # Calculate rolling metrics for scatter plot
        window = 240  # 4-hour windows
        rolling_returns = returns.rolling(window).mean() * 365 * 24 * 60  # Annualized
        rolling_vols = returns.rolling(window).std() * np.sqrt(365 * 24 * 60)
        
        # Remove NaN values
        valid_mask = ~(rolling_returns.isna() | rolling_vols.isna())
        valid_returns = rolling_returns[valid_mask]
        valid_vols = rolling_vols[valid_mask]
        
        if len(valid_returns) > 0:
            ax4.scatter(valid_vols * 100, valid_returns * 100, 
                       alpha=0.6, c=range(len(valid_returns)), cmap='viridis')
            ax4.set_xlabel('Volatility (%)')
            ax4.set_ylabel('Return (%)')
            ax4.set_title('Risk-Return Profile (4h Windows)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        risk_analysis_path = 'risk_analysis.png'
        plt.savefig(risk_analysis_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': risk_analysis_path,
            'description': 'Risk analysis including drawdowns, volatility, and VaR'
        }
    
    def _create_intraday_analysis(self, portfolio_history: pd.DataFrame,
                                analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create intraday pattern analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle('Intraday Patterns Analysis', fontsize=16, fontweight='bold')
        
        portfolio_values = portfolio_history['total_value']
        returns = portfolio_values.pct_change().dropna()
        
        # Add time components
        df = pd.DataFrame({'returns': returns})
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 1. Hourly returns pattern
        ax1 = axes[0, 0]
        hourly_returns = df.groupby('hour')['returns'].mean()
        
        bars = ax1.bar(hourly_returns.index, hourly_returns * 100, 
                      alpha=0.7, color=self.color_palette[0])
        
        # Color bars by positive/negative
        for i, bar in enumerate(bars):
            if hourly_returns.iloc[i] < 0:
                bar.set_color('red')
        
        ax1.set_title('Average Returns by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # 2. Hourly volatility pattern
        ax2 = axes[0, 1]
        hourly_vol = df.groupby('hour')['returns'].std()
        
        ax2.plot(hourly_vol.index, hourly_vol * 100, 
                marker='o', linewidth=2, markersize=4, color=self.color_palette[1])
        ax2.set_title('Volatility by Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        
        # 3. Day of week patterns
        ax3 = axes[1, 0]
        daily_returns = df.groupby('day_of_week')['returns'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        bars = ax3.bar(range(7), daily_returns * 100, 
                      alpha=0.7, color=self.color_palette[2])
        
        # Color bars by positive/negative
        for i, bar in enumerate(bars):
            if daily_returns.iloc[i] < 0:
                bar.set_color('red')
        
        ax3.set_title('Average Returns by Day of Week')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Average Return (%)')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(day_names)
        ax3.grid(True, alpha=0.3)
        
        # 4. Intraday volatility heatmap
        ax4 = axes[1, 1]
        
        # Create volatility matrix (hour vs day of week)
        vol_matrix = df.groupby(['day_of_week', 'hour'])['returns'].std().unstack()
        
        if not vol_matrix.empty:
            im = ax4.imshow(vol_matrix.values * 100, cmap='YlOrRd', aspect='auto')
            
            ax4.set_title('Volatility Heatmap (Day vs Hour)')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Day of Week')
            ax4.set_yticks(range(7))
            ax4.set_yticklabels(day_names)
            ax4.set_xticks(range(0, 24, 4))
            ax4.set_xticklabels(range(0, 24, 4))
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Volatility (%)')
        
        plt.tight_layout()
        
        # Save plot
        intraday_path = 'intraday_analysis.png'
        plt.savefig(intraday_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': intraday_path,
            'description': 'Intraday patterns including hourly and daily seasonality'
        }
    
    def _create_trade_analysis(self, trades_history: pd.DataFrame) -> Dict[str, Any]:
        """Create trade analysis visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        # 1. Trade sizes distribution
        ax1 = axes[0, 0]
        if 'value' in trades_history.columns:
            trade_values = abs(trades_history['value'])
            ax1.hist(trade_values, bins=30, alpha=0.7, color=self.color_palette[0], edgecolor='black')
            ax1.set_title('Trade Size Distribution')
            ax1.set_xlabel('Trade Value ($)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # 2. Trading frequency over time
        ax2 = axes[0, 1]
        
        # Resample trades to hourly frequency
        hourly_trades = trades_history.resample('1H').size()
        ax2.plot(hourly_trades.index, hourly_trades.values, 
                linewidth=2, color=self.color_palette[1])
        ax2.set_title('Trading Frequency Over Time')
        ax2.set_ylabel('Trades per Hour')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(hourly_trades) > 48:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Symbol distribution
        ax3 = axes[1, 0]
        if 'symbol' in trades_history.columns:
            symbol_counts = trades_history['symbol'].value_counts()
            ax3.pie(symbol_counts.values, labels=symbol_counts.index, autopct='%1.1f%%',
                   colors=self.color_palette[:len(symbol_counts)])
            ax3.set_title('Trades by Symbol')
        
        # 4. Cumulative transaction costs
        ax4 = axes[1, 1]
        if 'cost' in trades_history.columns:
            cumulative_costs = trades_history['cost'].cumsum()
            ax4.plot(cumulative_costs.index, cumulative_costs.values, 
                    linewidth=2, color='red')
            ax4.set_title('Cumulative Transaction Costs')
            ax4.set_ylabel('Cumulative Costs ($)')
            ax4.grid(True, alpha=0.3)
            
            # Format x-axis
            if len(cumulative_costs) > 100:
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            else:
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        trade_analysis_path = 'trade_analysis.png'
        plt.savefig(trade_analysis_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': trade_analysis_path,
            'description': 'Trade analysis including sizes, frequency, and costs'
        }
    
    def _create_market_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create market data analysis visualizations."""
        
        n_symbols = len(market_data)
        fig, axes = plt.subplots(2, min(2, n_symbols), figsize=self.config['figure_size'])
        if n_symbols == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle('Market Data Analysis', fontsize=16, fontweight='bold')
        
        symbols = list(market_data.keys())[:4]  # Limit to 4 symbols
        
        for i, symbol in enumerate(symbols):
            if i >= 4:  # Maximum 4 subplots
                break
                
            data = market_data[symbol]
            
            row = i // 2
            col = i % 2
            
            if len(axes.shape) == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            # Price chart with volume
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                ax.plot(data.index, data['Close'], linewidth=2, label='Close Price')
                ax.set_title(f'{symbol} Price')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                if len(data) > 1440:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Remove empty subplots
        for i in range(len(symbols), 4):
            row = i // 2
            col = i % 2
            if len(axes.shape) == 1:
                if row < len(axes):
                    axes[row].remove()
            else:
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].remove()
        
        plt.tight_layout()
        
        # Save plot
        market_analysis_path = 'market_analysis.png'
        plt.savefig(market_analysis_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': market_analysis_path,
            'description': 'Market data analysis for key symbols'
        }
    
    def _create_model_analysis(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Create model performance analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Prediction distribution
        ax1 = axes[0, 0]
        pred_cols = [col for col in predictions.columns if '_pred' in col]
        
        if pred_cols:
            all_predictions = predictions[pred_cols].values.flatten()
            all_predictions = all_predictions[~np.isnan(all_predictions)]
            
            ax1.hist(all_predictions, bins=50, alpha=0.7, color=self.color_palette[0], edgecolor='black')
            ax1.axvline(np.mean(all_predictions), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_predictions):.4f}')
            ax1.set_title('Prediction Distribution')
            ax1.set_xlabel('Predicted Returns')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Prediction accuracy over time (simplified)
        ax2 = axes[0, 1]
        if pred_cols and len(pred_cols) > 0:
            # Use first prediction column as example
            pred_col = pred_cols[0]
            pred_values = predictions[pred_col].dropna()
            
            # Rolling statistics
            rolling_mean = pred_values.rolling(60).mean()
            rolling_std = pred_values.rolling(60).std()
            
            ax2.plot(rolling_mean.index, rolling_mean.values, 
                    linewidth=2, label='Rolling Mean', color=self.color_palette[1])
            ax2.fill_between(rolling_mean.index, 
                           (rolling_mean - rolling_std).values,
                           (rolling_mean + rolling_std).values,
                           alpha=0.3, label='±1 Std')
            ax2.set_title('Prediction Stability Over Time')
            ax2.set_ylabel('Prediction Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            if len(rolling_mean) > 1440:
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            else:
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Prediction correlation matrix
        ax3 = axes[1, 0]
        if len(pred_cols) > 1:
            pred_corr = predictions[pred_cols].corr()
            
            im = ax3.imshow(pred_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
            ax3.set_title('Prediction Correlations')
            ax3.set_xticks(range(len(pred_cols)))
            ax3.set_yticks(range(len(pred_cols)))
            ax3.set_xticklabels([col.replace('_pred', '') for col in pred_cols], rotation=45)
            ax3.set_yticklabels([col.replace('_pred', '') for col in pred_cols])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Correlation')
        
        # 4. Horizon comparison
        ax4 = axes[1, 1]
        
        # Group predictions by horizon
        horizon_stats = {}
        for col in pred_cols:
            if 'min_pred' in col:
                # Extract horizon from column name
                parts = col.split('_')
                horizon = next((p for p in parts if 'min' in p), 'unknown')
                
                pred_values = predictions[col].dropna()
                horizon_stats[horizon] = {
                    'mean': pred_values.mean(),
                    'std': pred_values.std(),
                    'count': len(pred_values)
                }
        
        if horizon_stats:
            horizons = list(horizon_stats.keys())
            means = [horizon_stats[h]['mean'] for h in horizons]
            stds = [horizon_stats[h]['std'] for h in horizons]
            
            x_pos = np.arange(len(horizons))
            ax4.bar(x_pos, means, yerr=stds, alpha=0.7, 
                   color=self.color_palette[:len(horizons)], capsize=5)
            ax4.set_title('Prediction Statistics by Horizon')
            ax4.set_xlabel('Prediction Horizon')
            ax4.set_ylabel('Mean Prediction')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(horizons)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        model_analysis_path = 'model_analysis.png'
        plt.savefig(model_analysis_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': model_analysis_path,
            'description': 'Model performance and prediction analysis'
        }
    
    def _create_feature_analysis(self, analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create feature importance analysis (placeholder)."""
        
        # This would typically show feature importance from model results
        # For now, create a placeholder chart
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Sample feature importance data (in real implementation, this would come from model)
        np.random.seed(42)
        feature_names = [f'Feature_{i}' for i in range(20)]
        importances = np.random.exponential(0.1, 20)
        importances = importances / importances.sum()
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:15]]
        top_importances = [importances[i] for i in sorted_idx[:15]]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_importances, alpha=0.7, color=self.color_palette[0])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Feature Importances')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{top_importances[i]:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        feature_analysis_path = 'feature_analysis.png'
        plt.savefig(feature_analysis_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': feature_analysis_path,
            'description': 'Feature importance analysis for model inputs'
        }
    
    def _create_correlation_analysis(self, portfolio_history: pd.DataFrame,
                                   market_data: Dict[str, pd.DataFrame] = None,
                                   predictions: pd.DataFrame = None) -> Dict[str, Any]:
        """Create correlation analysis between portfolio, market, and predictions."""
        
        fig, axes = plt.subplots(1, 2, figsize=self.config['figure_size'])
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio vs Market correlation
        ax1 = axes[0]
        
        if market_data:
            # Calculate portfolio returns
            portfolio_returns = portfolio_history['total_value'].pct_change().dropna()
            
            # Calculate market returns
            market_returns = {}
            for symbol, data in market_data.items():
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change().dropna()
                    # Align with portfolio returns
                    common_index = portfolio_returns.index.intersection(returns.index)
                    if len(common_index) > 0:
                        portfolio_aligned = portfolio_returns.loc[common_index]
                        market_aligned = returns.loc[common_index]
                        
                        if len(portfolio_aligned) > 10:
                            correlation = portfolio_aligned.corr(market_aligned)
                            market_returns[symbol] = correlation
            
            if market_returns:
                symbols = list(market_returns.keys())
                correlations = list(market_returns.values())
                
                bars = ax1.bar(range(len(symbols)), correlations, alpha=0.7, 
                              color=self.color_palette[:len(symbols)])
                
                ax1.set_title('Portfolio-Market Correlations')
                ax1.set_ylabel('Correlation')
                ax1.set_xticks(range(len(symbols)))
                ax1.set_xticklabels(symbols, rotation=45)
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Color bars by positive/negative correlation
                for i, bar in enumerate(bars):
                    if correlations[i] < 0:
                        bar.set_color('red')
        
        # 2. Prediction vs Actual correlation (simplified)
        ax2 = axes[1]
        
        if predictions is not None and not predictions.empty:
            # This is a simplified version - in practice would need actual returns
            pred_cols = [col for col in predictions.columns if '_pred' in col]
            
            if len(pred_cols) > 1:
                pred_corr = predictions[pred_cols].corr()
                
                # Show correlation matrix as heatmap
                im = ax2.imshow(pred_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
                ax2.set_title('Inter-Prediction Correlations')
                ax2.set_xticks(range(len(pred_cols)))
                ax2.set_yticks(range(len(pred_cols)))
                
                # Simplify labels
                labels = [col.replace('_pred', '').replace('_', ' ') for col in pred_cols]
                ax2.set_xticklabels(labels, rotation=45)
                ax2.set_yticklabels(labels)
                
                # Add text annotations
                for i in range(len(pred_cols)):
                    for j in range(len(pred_cols)):
                        text = ax2.text(j, i, f'{pred_corr.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Correlation')
        
        plt.tight_layout()
        
        # Save plot
        correlation_analysis_path = 'correlation_analysis.png'
        plt.savefig(correlation_analysis_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': correlation_analysis_path,
            'description': 'Correlation analysis between portfolio, market, and predictions'
        }
    
    def create_interactive_dashboard(self, portfolio_history: pd.DataFrame,
                                   trades_history: pd.DataFrame = None,
                                   market_data: Dict[str, pd.DataFrame] = None) -> str:
        """Create interactive Plotly dashboard."""
        
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Returns Distribution', 
                          'Drawdown Analysis', 'Trading Volume',
                          'Intraday Patterns', 'Risk Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        portfolio_values = portfolio_history['total_value']
        fig.add_trace(
            go.Scatter(x=portfolio_history.index, y=portfolio_values,
                      mode='lines', name='Portfolio Value', line=dict(width=2)),
            row=1, col=1
        )
        
        # Returns distribution
        returns = portfolio_values.pct_change().dropna()
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns Distribution',
                        opacity=0.7, marker_color=self.color_palette[1]),
            row=1, col=2
        )
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown * 100,
                      fill='tonexty', name='Drawdown (%)', 
                      line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
            row=2, col=1
        )
        
        # Trading volume (if available)
        if trades_history is not None and not trades_history.empty:
            daily_volume = trades_history.resample('1D')['value'].sum().abs()
            fig.add_trace(
                go.Bar(x=daily_volume.index, y=daily_volume.values,
                      name='Daily Trading Volume', marker_color=self.color_palette[2]),
                row=2, col=2
            )
        
        # Intraday patterns
        df = pd.DataFrame({'returns': returns})
        df['hour'] = df.index.hour
        hourly_returns = df.groupby('hour')['returns'].mean()
        
        fig.add_trace(
            go.Bar(x=hourly_returns.index, y=hourly_returns.values * 100,
                  name='Hourly Returns (%)', marker_color=self.color_palette[3]),
            row=3, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(60).std() * np.sqrt(365 * 24 * 60)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol * 100,
                      mode='lines', name='Rolling Volatility (%)',
                      line=dict(color=self.color_palette[4])),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Interactive Trading Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        # Save interactive plot
        interactive_path = 'interactive_dashboard.html'
        pyo.plot(fig, filename=interactive_path, auto_open=False)
        
        return interactive_path
    
    def generate_comprehensive_report(self, portfolio_history: pd.DataFrame,
                                    trades_history: pd.DataFrame = None,
                                    market_data: Dict[str, pd.DataFrame] = None,
                                    predictions: pd.DataFrame = None,
                                    analysis_results: Dict[str, Any] = None,
                                    output_path: str = 'trading_report.pdf') -> str:
        """Generate comprehensive PDF report with all visualizations."""
        
        self.logger.info("Generating comprehensive PDF report")
        
        # Create all visualizations
        dashboard_components = self.create_comprehensive_dashboard(
            portfolio_history, trades_history, market_data, predictions, analysis_results
        )
        
        # Create PDF report
        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Minute-Level Cryptocurrency Trading Analysis Report', 
                        fontsize=20, fontweight='bold', y=0.8)
            
            # Add summary information
            if analysis_results:
                period = analysis_results.get('analysis_period', {})
                basic_metrics = analysis_results.get('basic_metrics', {})
                
                summary_text = f"""
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Analysis Period:
  Start: {period.get('start', 'Unknown')}
  End: {period.get('end', 'Unknown')}
  Duration: {period.get('duration_days', 0):.1f} days

Key Performance Metrics:
  Total Return: {basic_metrics.get('total_return', 0):.2%}
  Annualized Return: {basic_metrics.get('annualized_return', 0):.2%}
  Sharpe Ratio: {basic_metrics.get('sharpe_ratio', 0):.2f}
  Maximum Drawdown: {analysis_results.get('risk_metrics', {}).get('max_drawdown', 0):.2%}
  Win Rate: {basic_metrics.get('win_rate', 0):.2%}
                """
                
                plt.text(0.1, 0.6, summary_text, transform=fig.transFigure, 
                        fontsize=14, verticalalignment='top', fontfamily='monospace')
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add each visualization to PDF
            for component_name, component_data in dashboard_components.items():
                if 'chart_path' in component_data:
                    # Load and add chart
                    img = plt.imread(component_data['chart_path'])
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(component_data.get('description', component_name), 
                               fontsize=14, fontweight='bold', pad=20)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        self.logger.info(f"Comprehensive report saved to {output_path}")
        return output_path


# Utility functions

def create_minute_visualization_suite(config: Dict[str, Any] = None) -> MinuteVisualizationSuite:
    """Create a configured visualization suite."""
    return MinuteVisualizationSuite(config)


def quick_portfolio_visualization(portfolio_history: pd.DataFrame,
                                analysis_results: Dict[str, Any] = None) -> Dict[str, str]:
    """Quick visualization of portfolio performance."""
    
    viz_suite = MinuteVisualizationSuite()
    
    results = {}
    
    # Portfolio overview
    overview = viz_suite._create_portfolio_overview(portfolio_history, analysis_results)
    results['portfolio_overview'] = overview['chart_path']
    
    # Risk analysis
    risk_analysis = viz_suite._create_risk_analysis_charts(portfolio_history, analysis_results)
    results['risk_analysis'] = risk_analysis['chart_path']
    
    return results


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-01-02', freq='1T')
    np.random.seed(42)
    
    # Sample portfolio data
    initial_value = 100000
    returns = np.random.normal(0.0001, 0.008, len(dates))
    
    portfolio_values = [initial_value]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    portfolio_history = pd.DataFrame({
        'total_value': portfolio_values[1:],
        'cash': portfolio_values[1:] * 0.2,
        'position_value': portfolio_values[1:] * 0.8,
        'num_positions': np.random.randint(1, 4, len(dates))
    }, index=dates)
    
    # Sample trades
    n_trades = 30
    trade_times = np.random.choice(dates, n_trades, replace=False)
    trades_history = pd.DataFrame({
        'symbol': np.random.choice(['BTC-USD', 'ETH-USD'], n_trades),
        'value': np.random.normal(0, 3000, n_trades),
        'cost': np.random.exponential(30, n_trades)
    }, index=trade_times)
    
    print(f"Sample data created: {len(portfolio_history)} minutes, {len(trades_history)} trades")
    
    # Create visualizations
    print("Creating visualization suite...")
    viz_suite = create_minute_visualization_suite()
    
    # Test portfolio overview
    overview = viz_suite._create_portfolio_overview(portfolio_history)
    print(f"Portfolio overview saved: {overview['chart_path']}")
    
    # Test risk analysis
    risk_analysis = viz_suite._create_risk_analysis_charts(portfolio_history)
    print(f"Risk analysis saved: {risk_analysis['chart_path']}")
    
    # Test trade analysis
    trade_analysis = viz_suite._create_trade_analysis(trades_history)
    print(f"Trade analysis saved: {trade_analysis['chart_path']}")
    
    # Test comprehensive dashboard
    print("Creating comprehensive dashboard...")
    dashboard = viz_suite.create_comprehensive_dashboard(
        portfolio_history, trades_history
    )
    
    print(f"Dashboard components created: {list(dashboard.keys())}")
    
    # Test interactive dashboard (if Plotly available)
    if PLOTLY_AVAILABLE:
        print("Creating interactive dashboard...")
        interactive_path = viz_suite.create_interactive_dashboard(portfolio_history, trades_history)
        if interactive_path:
            print(f"Interactive dashboard saved: {interactive_path}")
    
    print("Visualization suite testing completed!")