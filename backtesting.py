import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import ccxt
from concurrent.futures import ThreadPoolExecutor
import json
import os

from .trading_strategy import TradingStrategy
from .config import load_config

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config_path: str):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Initialize trading strategy
        self.strategy = TradingStrategy(self.config)
        
        # Initialize results storage
        self.results = {
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {}
        }
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize the cryptocurrency exchange"""
        try:
            exchange_class = getattr(ccxt, self.config['exchange']['name'])
            exchange = exchange_class({
                'apiKey': self.config['exchange']['api_key'],
                'secret': self.config['exchange']['api_secret'],
                'enableRateLimit': True
            })
            
            # Test connection
            exchange.load_markets()
            logger.info(f"Successfully connected to {self.config['exchange']['name']}")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            raise
    
    def fetch_historical_data(self, 
                            symbol: str, 
                            timeframe: str, 
                            start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
        """Fetch historical market data"""
        try:
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=start_timestamp,
                limit=1000  # Maximum limit per request
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def run_backtest(self, 
                    symbol: str,
                    timeframe: str,
                    start_date: datetime,
                    end_date: datetime,
                    initial_capital: float = 50.0) -> Dict:
        """Run backtest for a given period"""
        try:
            logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
            
            # Fetch historical data
            df = self.fetch_historical_data(symbol, timeframe, start_date, end_date)
            if df is None:
                raise ValueError("Failed to fetch historical data")
            
            # Initialize variables
            current_capital = initial_capital
            position = 0
            entry_price = 0
            trades = []
            equity_curve = []
            
            # Iterate through data
            for i in range(len(df)):
                current_data = df.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Generate trading signal
                signal = self.strategy.analyze_market(symbol, current_data)
                
                if signal:
                    # Execute trade if conditions are met
                    if signal.action == 'buy' and position <= 0:
                        position = 1
                        entry_price = current_price
                        trades.append({
                            'timestamp': current_data.index[-1],
                            'action': 'buy',
                            'price': current_price,
                            'position_size': signal.position_size
                        })
                    
                    elif signal.action == 'sell' and position >= 0:
                        position = -1
                        entry_price = current_price
                        trades.append({
                            'timestamp': current_data.index[-1],
                            'action': 'sell',
                            'price': current_price,
                            'position_size': signal.position_size
                        })
                
                # Update position PnL
                if position != 0:
                    pnl = position * (current_price - entry_price) * signal.position_size
                    current_capital += pnl
                
                # Record equity
                equity_curve.append({
                    'timestamp': current_data.index[-1],
                    'equity': current_capital
                })
            
            # Calculate performance metrics
            self.results['trades'] = trades
            self.results['equity_curve'] = equity_curve
            self.results['performance_metrics'] = self._calculate_performance_metrics(trades)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return None
    
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trades"""
        if not trades:
            return {}
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Calculate basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['action'] == 'buy'])
        losing_trades = len(df[df['action'] == 'sell'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        returns = df['price'].pct_change()
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def save_results(self, filepath: str):
        """Save backtest results to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, default=str)
            logger.info(f"Backtest results saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def load_results(self, filepath: str):
        """Load backtest results from file"""
        try:
            with open(filepath, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Backtest results loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return False

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    
    # Create backtester
    backtester = Backtester(config_path)
    
    # Define backtest parameters
    symbol = "BTC/USDT"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    initial_capital = 50.0
    
    # Run backtest
    results = backtester.run_backtest(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    if results:
        # Print performance metrics
        metrics = results['performance_metrics']
        print("\nBacktest Results:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        # Save results
        backtester.save_results('backtest_results.json')
    else:
        print("Backtest failed to complete")

if __name__ == "__main__":
    main() 