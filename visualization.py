import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import json
import os

def plot_backtest_results(results_file: str, save_path: str = None):
    """
    Create interactive plots for backtest results
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert data to DataFrames
    trades_df = pd.DataFrame(results['trades'])
    equity_df = pd.DataFrame(results['equity_curve'])
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price and Trades', 'Equity Curve'),
        row_heights=[0.7, 0.3]
    )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['price'],
            name='Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add buy trades
    buy_trades = trades_df[trades_df['action'] == 'buy']
    fig.add_trace(
        go.Scatter(
            x=buy_trades['timestamp'],
            y=buy_trades['price'],
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=10)
        ),
        row=1, col=1
    )
    
    # Add sell trades
    sell_trades = trades_df[trades_df['action'] == 'sell']
    fig.add_trace(
        go.Scatter(
            x=sell_trades['timestamp'],
            y=sell_trades['price'],
            mode='markers',
            name='Sell',
            marker=dict(color='red', size=10)
        ),
        row=1, col=1
    )
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
            name='Equity',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Backtest Results',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='Equity',
        showlegend=True,
        height=800
    )
    
    # Save plot if path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")
    
    return fig

def plot_performance_metrics(results_files: List[str], save_path: str = None):
    """
    Create comparison plot of performance metrics across different scenarios
    """
    metrics_data = []
    
    for file in results_files:
        with open(file, 'r') as f:
            results = json.load(f)
            metrics = results['performance_metrics']
            
            # Extract scenario name from filename
            scenario_name = os.path.basename(file).replace('backtest_results_', '').replace('.json', '')
            
            metrics_data.append({
                'Scenario': scenario_name,
                'Total Return': metrics['total_return'],
                'Annual Return': metrics['annual_return'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate']
            })
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for each metric
    metrics = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=df['Scenario'],
                y=df[metric]
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Performance Metrics Comparison',
        xaxis_title='Scenario',
        yaxis_title='Value',
        barmode='group',
        height=600
    )
    
    # Save plot if path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")
    
    return fig

def main():
    # Example usage
    results_dir = 'backtest_results'
    results_files = [
        os.path.join(results_dir, f) for f in os.listdir(results_dir)
        if f.startswith('backtest_results_') and f.endswith('.json')
    ]
    
    # Create plots for each scenario
    for file in results_files:
        scenario_name = os.path.basename(file).replace('.json', '')
        plot_path = os.path.join(results_dir, f'{scenario_name}_plot.html')
        plot_backtest_results(file, plot_path)
    
    # Create comparison plot
    comparison_path = os.path.join(results_dir, 'performance_comparison.html')
    plot_performance_metrics(results_files, comparison_path)

if __name__ == "__main__":
    main() 