import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
from datetime import datetime
import numpy as np
from .code_guardian import create_guardian, CodeGuardian

def load_backtest_results(results_dir):
    """Load all backtest results from directory"""
    results = {}
    for file in os.listdir(results_dir):
        if file.startswith('backtest_results_') and file.endswith('.json'):
            with open(os.path.join(results_dir, file), 'r') as f:
                scenario_name = file.replace('backtest_results_', '').replace('.json', '')
                results[scenario_name] = json.load(f)
    return results

def create_dashboard(results_dir='backtest_results'):
    """Create and run the dashboard"""
    app = dash.Dash(__name__)
    
    # Load results
    results = load_backtest_results(results_dir)
    
    # Initialize Code Guardian
    guardian = create_guardian(os.path.dirname(os.path.dirname(__file__)))
    
    # Dashboard layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1('AI Trading Bot - Backtest Results & System Health Dashboard',
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px'}),
        
        # Tabs for different sections
        dcc.Tabs([
            # Backtest Results Tab
            dcc.Tab(label='Backtest Results', children=[
                # Scenario Selection
                html.Div([
                    html.Label('Select Scenario:'),
                    dcc.Dropdown(
                        id='scenario-dropdown',
                        options=[{'label': k, 'value': k} for k in results.keys()],
                        value=list(results.keys())[0] if results else None,
                        style={'width': '100%'}
                    ),
                ], style={'margin': '20px'}),
                
                # Main Charts
                html.Div([
                    # Trading Chart
                    html.Div([
                        dcc.Graph(id='trading-chart')
                    ], style={'width': '100%', 'marginBottom': '20px'}),
                    
                    # Performance Metrics Cards
                    html.Div([
                        html.Div([
                            html.Div(id='metrics-cards'),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
                    ], style={'marginBottom': '20px'}),
                    
                    # Trade Distribution Chart
                    html.Div([
                        dcc.Graph(id='trade-distribution')
                    ], style={'width': '100%', 'marginBottom': '20px'}),
                    
                    # Drawdown Chart
                    html.Div([
                        dcc.Graph(id='drawdown-chart')
                    ], style={'width': '100%'})
                ], style={'padding': '20px'})
            ]),
            
            # System Health Tab
            dcc.Tab(label='System Health', children=[
                html.Div([
                    # Code Health Overview
                    html.Div([
                        html.H2('Code Health Overview', style={'textAlign': 'center'}),
                        html.Div(id='code-health-cards', style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'justifyContent': 'space-around',
                            'margin': '20px'
                        })
                    ]),
                    
                    # Code Issues Table
                    html.Div([
                        html.H3('Active Code Issues'),
                        html.Div(id='code-issues-table')
                    ], style={'margin': '20px'}),
                    
                    # Performance Metrics
                    html.Div([
                        html.H3('Module Performance'),
                        dcc.Graph(id='performance-chart')
                    ], style={'margin': '20px'}),
                    
                    # Auto-Fix Controls
                    html.Div([
                        html.H3('Issue Resolution'),
                        html.Button('Auto-Fix Selected Issues', id='auto-fix-button'),
                        html.Div(id='auto-fix-status')
                    ], style={'margin': '20px'})
                ])
            ]),
            
            # System Logs Tab
            dcc.Tab(label='System Logs', children=[
                html.Div([
                    html.H2('System Logs'),
                    dcc.Interval(id='log-update', interval=5000),  # Update every 5 seconds
                    html.Pre(id='log-content', style={
                        'backgroundColor': '#2c3e50',
                        'color': '#ecf0f1',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'height': '500px',
                        'overflow': 'auto'
                    })
                ], style={'margin': '20px'})
            ])
        ])
    ])
    
    @app.callback(
        [Output('trading-chart', 'figure'),
         Output('metrics-cards', 'children'),
         Output('trade-distribution', 'figure'),
         Output('drawdown-chart', 'figure')],
        [Input('scenario-dropdown', 'value')]
    )
    def update_charts(selected_scenario):
        if not selected_scenario or selected_scenario not in results:
            return {}, [], {}, {}
        
        result = results[selected_scenario]
        trades_df = pd.DataFrame(result['trades'])
        equity_df = pd.DataFrame(result['equity_curve'])
        metrics = result['performance_metrics']
        
        # 1. Trading Chart
        trading_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price and Trades', 'Equity Curve'),
            row_heights=[0.7, 0.3]
        )
        
        # Add price line
        trading_fig.add_trace(
            go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['price'],
                name='Price',
                line=dict(color='#2980b9')
            ),
            row=1, col=1
        )
        
        # Add buy/sell markers
        for action, color in [('buy', '#27ae60'), ('sell', '#c0392b')]:
            mask = trades_df['action'] == action
            trading_fig.add_trace(
                go.Scatter(
                    x=trades_df[mask]['timestamp'],
                    y=trades_df[mask]['price'],
                    mode='markers',
                    name=action.capitalize(),
                    marker=dict(color=color, size=10)
                ),
                row=1, col=1
            )
        
        # Add equity curve
        trading_fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                name='Equity',
                line=dict(color='#8e44ad')
            ),
            row=2, col=1
        )
        
        trading_fig.update_layout(
            title='Trading Activity and Equity Curve',
            height=800,
            template='plotly_white'
        )
        
        # 2. Metrics Cards
        cards = []
        metrics_style = {
            'width': '200px',
            'margin': '10px',
            'padding': '15px',
            'borderRadius': '5px',
            'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
            'textAlign': 'center'
        }
        
        metrics_data = [
            ('Total Return', f"{metrics['total_return']:.2%}", '#27ae60'),
            ('Win Rate', f"{metrics['win_rate']:.2%}", '#2980b9'),
            ('Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}", '#8e44ad'),
            ('Max Drawdown', f"{metrics['max_drawdown']:.2%}", '#c0392b'),
            ('Total Trades', str(metrics['total_trades']), '#2c3e50')
        ]
        
        for label, value, color in metrics_data:
            cards.append(html.Div([
                html.H4(label, style={'color': '#7f8c8d'}),
                html.H3(value, style={'color': color})
            ], style=metrics_style))
        
        # 3. Trade Distribution
        returns = pd.Series([float(x) for x in trades_df['price'].pct_change().dropna()])
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=30,
            name='Returns Distribution',
            marker_color='#3498db'
        ))
        
        dist_fig.update_layout(
            title='Trade Returns Distribution',
            xaxis_title='Return',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        # 4. Drawdown Chart
        equity_series = pd.Series([float(x) for x in equity_df['equity']])
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=equity_df['timestamp'],
            y=drawdowns,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#e74c3c')
        ))
        
        dd_fig.update_layout(
            title='Drawdown Over Time',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            template='plotly_white'
        )
        
        return trading_fig, cards, dist_fig, dd_fig
    
    @app.callback(
        [Output('code-health-cards', 'children'),
         Output('code-issues-table', 'children'),
         Output('performance-chart', 'figure'),
         Output('auto-fix-status', 'children')],
        [Input('auto-fix-button', 'n_clicks')],
        [State('code-issues-table', 'selected_rows')]
    )
    def update_system_health(n_clicks, selected_rows):
        # Get guardian status
        status = guardian.get_status_report()
        
        # Create health cards
        health_cards = []
        health_metrics = [
            ('Total Issues', status['total_issues'], '#e74c3c'),
            ('Fixed Issues', status['fixed_issues'], '#27ae60'),
            ('Active Issues', status['active_issues'], '#f39c12'),
            ('Code Coverage', '85%', '#3498db')  # Example metric
        ]
        
        for label, value, color in health_metrics:
            health_cards.append(html.Div([
                html.H4(label),
                html.H3(str(value))
            ], style={
                'width': '200px',
                'margin': '10px',
                'padding': '15px',
                'borderRadius': '5px',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'textAlign': 'center',
                'backgroundColor': color,
                'color': 'white'
            }))
        
        # Create issues table
        issues_table = html.Table([
            html.Thead(html.Tr([
                html.Th('File'),
                html.Th('Line'),
                html.Th('Type'),
                html.Th('Severity'),
                html.Th('Description'),
                html.Th('Status')
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(issue.file_path),
                    html.Td(issue.line_number),
                    html.Td(issue.issue_type),
                    html.Td(issue.severity),
                    html.Td(issue.description),
                    html.Td('Fixed' if issue.fixed else 'Active')
                ]) for issue in guardian.issues
            ])
        ], style={'width': '100%'})
        
        # Create performance chart
        perf_data = status['performance_metrics']
        perf_fig = go.Figure()
        
        for module, metrics in perf_data.items():
            perf_fig.add_trace(go.Scatter(
                x=list(range(len(guardian.performance_metrics[module]))),
                y=guardian.performance_metrics[module],
                name=module,
                mode='lines+markers'
            ))
        
        perf_fig.update_layout(
            title='Module Performance Over Time',
            xaxis_title='Execution Count',
            yaxis_title='Execution Time (s)',
            template='plotly_white'
        )
        
        # Handle auto-fix status
        fix_status = None
        if n_clicks and selected_rows:
            fixed_count = sum(1 for i in selected_rows if guardian.fix_issue(guardian.issues[i]))
            fix_status = f"Fixed {fixed_count} out of {len(selected_rows)} selected issues"
        
        return health_cards, issues_table, perf_fig, fix_status
    
    @app.callback(
        Output('log-content', 'children'),
        [Input('log-update', 'n_intervals')]
    )
    def update_logs(_):
        try:
            with open('analysis.log', 'r') as f:
                logs = f.readlines()
                return ''.join(logs[-100:])  # Show last 100 lines
        except Exception as e:
            return f"Error reading logs: {str(e)}"
    
    return app

def run_dashboard(port=8050):
    """Run the dashboard"""
    app = create_dashboard()
    app.run_server(debug=True, port=port)

if __name__ == '__main__':
    run_dashboard() 