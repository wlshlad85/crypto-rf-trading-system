"""Visualization utilities for crypto RF trading system."""

import pandas as pd
import numpy as np
import importlib
plt = importlib.import_module('matplotlib.pyplot')
sns = importlib.import_module('seaborn')
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Some interactive visualizations will be unavailable.")


class CryptoTradingVisualizer:
    """Visualization class for crypto trading analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        self.logger = logging.getLogger(__name__)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Default figure settings
        self.figsize = (12, 8)
        self.dpi = 100
    
    def plot_portfolio_performance(self, portfolio_df: pd.DataFrame, 
                                 benchmark_data: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot portfolio performance over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df.index, portfolio_df['total_value'], 
                color=self.colors['primary'], linewidth=2, label='Portfolio Value')
        
        if benchmark_data is not None and 'benchmark_value' in benchmark_data.columns:
            ax1.plot(benchmark_data.index, benchmark_data['benchmark_value'],
                    color=self.colors['secondary'], linewidth=2, alpha=0.7, label='Benchmark')
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        ax2 = axes[0, 1]
        returns = portfolio_df['total_value'].pct_change().dropna()
        ax2.hist(returns, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax2.axvline(returns.mean(), color=self.colors['danger'], linestyle='--', 
                   label=f'Mean: {returns.mean():.4f}')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax3 = axes[1, 0]
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax3.fill_between(drawdown.index, drawdown * 100, 0, 
                        color=self.colors['danger'], alpha=0.3)
        ax3.plot(drawdown.index, drawdown * 100, color=self.colors['danger'], linewidth=1)
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Number of positions over time
        ax4 = axes[1, 1]
        ax4.plot(portfolio_df.index, portfolio_df['num_positions'], 
                color=self.colors['success'], linewidth=2)
        ax4.set_title('Number of Active Positions')
        ax4.set_ylabel('Number of Positions')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Portfolio performance chart saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=self.colors['primary'], alpha=0.7)
        
        # Customize
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Feature importance chart saved to {save_path}")
        
        return fig
    
    def plot_returns_analysis(self, portfolio_df: pd.DataFrame, 
                            benchmark_data: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot detailed returns analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Returns Analysis', fontsize=16, fontweight='bold')
        
        returns = portfolio_df['total_value'].pct_change().dropna()
        
        # Cumulative returns
        ax1 = axes[0, 0]
        cumulative_returns = (1 + returns).cumprod()
        ax1.plot(cumulative_returns.index, cumulative_returns, 
                color=self.colors['primary'], linewidth=2, label='Portfolio')
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['benchmark_value'].pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax1.plot(benchmark_cumulative.index, benchmark_cumulative,
                    color=self.colors['secondary'], linewidth=2, alpha=0.7, label='Benchmark')
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax2 = axes[0, 1]
        window = 720
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = np.where(rolling_std > 0, rolling_mean / rolling_std * np.sqrt(365*24), 0)
        rolling_sharpe = pd.Series(rolling_sharpe, index=returns.index)
        ax2.plot(rolling_sharpe.index, rolling_sharpe, color=self.colors['success'], linewidth=2)
        ax2.axhline(y=1.0, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        ax2.set_title('Rolling Sharpe Ratio (30-day)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        ax3 = axes[1, 0]
        monthly_returns = (1 + returns).resample('M').prod() - 1
        monthly_returns_pivot = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).mean().unstack()
        
        if len(monthly_returns_pivot) > 0:
            sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', 
                       cmap='RdYlGn', center=0, ax=ax3, cbar_kws={'label': 'Monthly Return'})
            ax3.set_title('Monthly Returns Heatmap')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Year')
        
        # Rolling volatility
        ax4 = axes[1, 1]
        rolling_vol = returns.rolling(window=720).std() * np.sqrt(365*24)
        ax4.plot(rolling_vol.index, rolling_vol, color=self.colors['warning'], linewidth=2)
        ax4.set_title('Rolling Volatility (30-day)')
        ax4.set_ylabel('Annualized Volatility')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Returns analysis chart saved to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, features_df: pd.DataFrame, symbols: List[str],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation matrix of cryptocurrency returns."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get return columns for each symbol
        return_cols = []
        for symbol in symbols:
            return_col = f"{symbol}_return_1h"
            if return_col in features_df.columns:
                return_cols.append(return_col)
        
        if not return_cols:
            self.logger.warning("No return columns found for correlation matrix")
            return fig
        
        # Calculate correlation matrix
        correlation_matrix = features_df[return_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Cryptocurrency Returns Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Clean up labels
        labels = [col.replace('_return_1h', '').upper() for col in return_cols]
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Correlation matrix saved to {save_path}")
        
        return fig
    
    def plot_trading_signals(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame,
                           symbol: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot trading signals for a specific symbol."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Trading Signals for {symbol.upper()}', fontsize=16, fontweight='bold')
        
        # Filter data by date range
        if start_date and end_date:
            mask = (prices_df.index >= start_date) & (prices_df.index <= end_date)
            prices_df = prices_df.loc[mask]
            signals_df = signals_df.loc[mask]
        
        price_col = f"{symbol}_close"
        signal_col = f"{symbol}_signal"
        
        if price_col not in prices_df.columns or signal_col not in signals_df.columns:
            self.logger.warning(f"Data not found for {symbol}")
            return fig
        
        prices = prices_df[price_col].dropna()
        signals = signals_df[signal_col].dropna()
        
        # Align data
        common_index = prices.index.intersection(signals.index)
        prices = prices.loc[common_index]
        signals = signals.loc[common_index]
        
        # Price chart
        ax1 = axes[0]
        ax1.plot(prices.index, prices, color=self.colors['dark'], linewidth=1, label='Price')
        
        # Add buy/sell signals
        buy_signals = signals[signals > 0]
        sell_signals = signals[signals < 0]
        
        if not buy_signals.empty:
            buy_prices = prices.loc[buy_signals.index]
            ax1.scatter(buy_signals.index, buy_prices, color=self.colors['success'], 
                       marker='^', s=100, label='Buy Signal', zorder=5)
        
        if not sell_signals.empty:
            sell_prices = prices.loc[sell_signals.index]
            ax1.scatter(sell_signals.index, sell_prices, color=self.colors['danger'], 
                       marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{symbol.upper()} Price and Trading Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Signals over time
        ax2 = axes[1]
        ax2.plot(signals.index, signals, color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
        ax2.axhline(y=0, color=self.colors['dark'], linestyle='-', alpha=0.5)
        ax2.axhline(y=1, color=self.colors['success'], linestyle='--', alpha=0.5, label='Buy')
        ax2.axhline(y=-1, color=self.colors['danger'], linestyle='--', alpha=0.5, label='Sell')
        ax2.set_title('Signal Strength Over Time')
        ax2.set_ylabel('Signal')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Trading signals chart saved to {save_path}")
        
        return fig
    
    def create_performance_dashboard(self, backtest_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Extract data
        performance_metrics = backtest_results.get('performance_metrics', {})
        
        # Key metrics display
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metrics_text = f"""
        Portfolio Performance Summary
        
        Total Return: {backtest_results.get('total_return', 0):.2%}
        Annualized Return: {performance_metrics.get('annualized_return', 0):.2%}
        Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
        Volatility: {performance_metrics.get('volatility', 0):.2%}
        Total Trades: {backtest_results.get('total_trades', 0)}
        """
        
        ax1.text(0.1, 0.5, metrics_text, fontsize=14, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light']))
        
        # Additional charts would be added here
        # This is a framework for a comprehensive dashboard
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    # Interactive plotly charts (if available)
    def create_interactive_portfolio_chart(self, portfolio_df: pd.DataFrame,
                                         save_path: Optional[str] = None):
        """Create interactive portfolio performance chart using Plotly."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for interactive charts")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value', 'Returns Distribution', 
                          'Drawdown', 'Number of Positions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=portfolio_df.index, y=portfolio_df['total_value'],
                      mode='lines', name='Portfolio Value',
                      line=dict(color=self.colors['primary'], width=2)),
            row=1, col=1
        )
        
        # Returns distribution
        returns = portfolio_df['total_value'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(x=returns, name='Returns Distribution',
                        marker_color=self.colors['primary'], opacity=0.7),
            row=1, col=2
        )
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown * 100,
                      mode='lines', name='Drawdown (%)',
                      fill='tonexty', fillcolor=f'rgba(214, 39, 40, 0.3)',
                      line=dict(color=self.colors['danger'], width=1)),
            row=2, col=1
        )
        
        # Number of positions
        fig.add_trace(
            go.Scatter(x=portfolio_df.index, y=portfolio_df['num_positions'],
                      mode='lines', name='Active Positions',
                      line=dict(color=self.colors['success'], width=2)),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Interactive Portfolio Performance Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig


def create_full_report(backtest_results: Dict[str, Any], portfolio_df: pd.DataFrame,
                      features_df: pd.DataFrame, symbols: List[str],
                      output_dir: str = "results/reports") -> Dict[str, str]:
    """Create a full visualization report."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = CryptoTradingVisualizer()
    
    saved_files = {}
    
    # Portfolio performance
    fig1 = visualizer.plot_portfolio_performance(
        portfolio_df, 
        save_path=f"{output_dir}/portfolio_performance.png"
    )
    saved_files['portfolio_performance'] = f"{output_dir}/portfolio_performance.png"
    plt.close(fig1)
    
    # Returns analysis
    fig2 = visualizer.plot_returns_analysis(
        portfolio_df,
        save_path=f"{output_dir}/returns_analysis.png"
    )
    saved_files['returns_analysis'] = f"{output_dir}/returns_analysis.png"
    plt.close(fig2)
    
    # Correlation matrix
    fig3 = visualizer.plot_correlation_matrix(
        features_df, symbols,
        save_path=f"{output_dir}/correlation_matrix.png"
    )
    saved_files['correlation_matrix'] = f"{output_dir}/correlation_matrix.png"
    plt.close(fig3)
    
    # Performance dashboard
    fig4 = visualizer.create_performance_dashboard(
        backtest_results,
        save_path=f"{output_dir}/performance_dashboard.png"
    )
    saved_files['performance_dashboard'] = f"{output_dir}/performance_dashboard.png"
    plt.close(fig4)
    
    # Interactive chart (if plotly available)
    if PLOTLY_AVAILABLE:
        interactive_fig = visualizer.create_interactive_portfolio_chart(
            portfolio_df,
            save_path=f"{output_dir}/interactive_dashboard.html"
        )
        if interactive_fig:
            saved_files['interactive_dashboard'] = f"{output_dir}/interactive_dashboard.html"
    
    return saved_files