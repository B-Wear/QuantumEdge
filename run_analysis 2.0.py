import os
import sys
import logging
from datetime import datetime, timedelta
from src.backtesting import Backtester
from src.dashboard import run_dashboard
import threading
import webbrowser
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_backtests():
    """Run all backtest scenarios"""
    # Create results directory if it doesn't exist
    results_dir = 'backtest_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
    
    # Create backtester
    backtester = Backtester(config_path)
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'BTC/USDT 1h - Last 30 days',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'days': 30
        },
        {
            'name': 'ETH/USDT 4h - Last 60 days',
            'symbol': 'ETH/USDT',
            'timeframe': '4h',
            'days': 60
        },
        {
            'name': 'BTC/USDT 1d - Last 90 days',
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'days': 90
        },
        {
            'name': 'ETH/USDT 1d - Last 90 days',
            'symbol': 'ETH/USDT',
            'timeframe': '1d',
            'days': 90
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        logger.info(f"\nRunning scenario: {scenario['name']}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=scenario['days'])
        
        # Run backtest
        results = backtester.run_backtest(
            symbol=scenario['symbol'],
            timeframe=scenario['timeframe'],
            start_date=start_date,
            end_date=end_date,
            initial_capital=50.0
        )
        
        if results:
            # Save results
            filename = f"backtest_results_{scenario['symbol'].replace('/', '_')}_{scenario['timeframe']}.json"
            filepath = os.path.join(results_dir, filename)
            backtester.save_results(filepath)
            logger.info(f"Results saved to {filepath}")
        else:
            logger.error(f"Backtest failed for scenario: {scenario['name']}")

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)  # Wait for the server to start
    webbrowser.open('http://localhost:8050')

def main():
    """Run backtests and launch dashboard"""
    try:
        # Run backtests
        logger.info("Starting backtests...")
        run_backtests()
        logger.info("Backtests completed successfully")
        
        # Launch dashboard
        logger.info("Launching dashboard...")
        threading.Thread(target=open_browser).start()
        run_dashboard()
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 