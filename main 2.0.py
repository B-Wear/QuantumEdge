import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import Dict, List
import pandas as pd
import ccxt
from concurrent.futures import ThreadPoolExecutor
import schedule
import threading
import signal
import queue

from .trading_strategy import TradingStrategy
from .config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config_path: str):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Initialize trading strategy
        self.strategy = TradingStrategy(self.config)
        
        # Initialize state
        self.is_running = False
        self.symbols = self.config['trading']['symbols']
        self.timeframes = self.config['trading']['timeframes']
        self.data_queue = queue.Queue()
        self.signal_queue = queue.Queue()
        
        # Load previous state if exists
        self._load_state()
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize the cryptocurrency exchange
        """
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
    
    def _load_state(self):
        """
        Load previous trading state
        """
        state_file = self.config['trading']['state_file']
        if os.path.exists(state_file):
            try:
                self.strategy.load_state(state_file)
                logger.info("Successfully loaded previous trading state")
            except Exception as e:
                logger.error(f"Error loading state: {str(e)}")
    
    def _save_state(self):
        """
        Save current trading state
        """
        try:
            self.strategy.save_state(self.config['trading']['state_file'])
            logger.info("Successfully saved trading state")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch market data from exchange
        """
        try:
            # Get OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=self.config['trading']['lookback_periods']
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
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None
    
    def process_market_data(self):
        """
        Process market data from queue
        """
        while self.is_running:
            try:
                # Get data from queue
                data = self.data_queue.get(timeout=1)
                symbol, timeframe, df = data
                
                # Generate trading signals
                signal = self.strategy.analyze_market(symbol, df)
                
                if signal:
                    # Add signal to queue
                    self.signal_queue.put(signal)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing market data: {str(e)}")
    
    def execute_signals(self):
        """
        Execute trading signals from queue
        """
        while self.is_running:
            try:
                # Get signal from queue
                signal = self.signal_queue.get(timeout=1)
                
                # Execute trade
                if self.strategy.execute_trade(signal):
                    logger.info(f"Executed {signal.action} trade for {signal.symbol}")
                    
                    # Update positions with current prices
                    current_prices = {
                        symbol: self.exchange.fetch_ticker(symbol)['last']
                        for symbol in self.symbols
                    }
                    self.strategy.update_positions(current_prices)
                    
                    # Save state after trade
                    self._save_state()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error executing signals: {str(e)}")
    
    def update_market_data(self):
        """
        Update market data for all symbols
        """
        with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
            futures = []
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    future = executor.submit(self.fetch_market_data, symbol, timeframe)
                    futures.append((symbol, timeframe, future))
            
            for symbol, timeframe, future in futures:
                try:
                    df = future.result()
                    if df is not None:
                        self.data_queue.put((symbol, timeframe, df))
                except Exception as e:
                    logger.error(f"Error updating market data for {symbol}: {str(e)}")
    
    def schedule_tasks(self):
        """
        Schedule periodic tasks
        """
        # Schedule market data updates
        for timeframe in self.timeframes:
            schedule.every().minute.at(":00").do(self.update_market_data)
        
        # Schedule state saving
        schedule.every().hour.at(":00").do(self._save_state)
        
        # Run the scheduler
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def start(self):
        """
        Start the trading bot
        """
        try:
            self.is_running = True
            
            # Start threads
            threads = [
                threading.Thread(target=self.process_market_data),
                threading.Thread(target=self.execute_signals),
                threading.Thread(target=self.schedule_tasks)
            ]
            
            for thread in threads:
                thread.daemon = True
                thread.start()
            
            # Initial market data update
            self.update_market_data()
            
            logger.info("Trading bot started successfully")
            
            # Wait for shutdown signal
            while self.is_running:
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {str(e)}")
            self.stop()
    
    def stop(self):
        """
        Stop the trading bot
        """
        try:
            self.is_running = False
            self._save_state()
            logger.info("Trading bot stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping trading bot: {str(e)}")

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    
    # Create trading bot
    bot = TradingBot(config_path)
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start trading bot
    bot.start()

if __name__ == "__main__":
    main() 