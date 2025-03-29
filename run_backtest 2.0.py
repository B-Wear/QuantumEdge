import os
import sys
from datetime import datetime, timedelta
import logging
from src.backtesting import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_backtest_scenarios():
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
            # Print performance metrics
            metrics = results['performance_metrics']
            print(f"\nResults for {scenario['name']}:")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Annual Return: {metrics['annual_return']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            
            # Save results
            filename = f"backtest_results_{scenario['symbol'].replace('/', '_')}_{scenario['timeframe']}.json"
            backtester.save_results(filename)
            logger.info(f"Results saved to {filename}")
        else:
            logger.error(f"Backtest failed for scenario: {scenario['name']}")

if __name__ == "__main__":
    run_backtest_scenarios() 