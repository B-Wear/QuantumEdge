import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
from .system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    status: str = 'open'

class TradingSystem:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize system monitor
        self.monitor = SystemMonitor(os.path.dirname(os.path.dirname(__file__)))
        
        # Initialize trading parameters
        self.initial_capital = self.config['trading']['initial_capital']
        self.current_capital = self.initial_capital
        self.performance_history = []
        self.active_trades: List[Trade] = []
        
        # Initialize strategy parameters
        self.strategy_params = self.config['strategy']
        self.risk_params = self.config['risk_management']
        
        # Initialize performance metrics
        self.metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    @SystemMonitor.monitor_component("strategy_execution")
    def execute_strategy(self, market_data: pd.DataFrame) -> Optional[Dict]:
        """Execute trading strategy"""
        try:
            # Generate trading signals
            signal = self._generate_signals(market_data)
            
            if signal:
                # Validate trade
                if self._validate_trade(signal):
                    # Execute trade
                    trade = self._execute_trade(signal)
                    return {'status': 'success', 'trade': trade}
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    @SystemMonitor.monitor_component("risk_management")
    def _validate_trade(self, signal: Dict) -> bool:
        """Validate trade against risk parameters"""
        try:
            # Check if we have too many open trades
            if len(self.active_trades) >= self.risk_params['max_positions']:
                return False
            
            # Check if we have enough capital
            required_margin = self._calculate_margin(signal)
            if required_margin > self.current_capital * self.risk_params['max_position_size']:
                return False
            
            # Check if we're within daily loss limit
            if self._check_daily_loss_limit():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False
    
    @SystemMonitor.monitor_component("trade_execution")
    def _execute_trade(self, signal: Dict) -> Optional[Trade]:
        """Execute a trade"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            # Create trade object
            trade = Trade(
                symbol=signal['symbol'],
                direction=signal['direction'],
                entry_price=signal['price'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                position_size=position_size,
                entry_time=datetime.now(),
                exit_price=None,
                exit_time=None,
                pnl=None,
                status='open'
            )
            
            # Add to active trades
            self.active_trades.append(trade)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None
    
    @SystemMonitor.monitor_component("performance_monitoring")
    def update_performance(self):
        """Update performance metrics"""
        try:
            if not self.performance_history:
                return
            
            # Calculate basic metrics
            total_trades = len(self.performance_history)
            winning_trades = len([t for t in self.performance_history if t['pnl'] > 0])
            
            self.metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0
            })
            
            # Calculate advanced metrics
            self._calculate_advanced_metrics()
            
            # Check if we need to adjust strategy
            self._check_strategy_adjustment()
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    @SystemMonitor.monitor_component("strategy_adjustment")
    def _check_strategy_adjustment(self):
        """Check if strategy needs adjustment"""
        try:
            # Check win rate
            if self.metrics['win_rate'] < self.strategy_params['min_win_rate']:
                self._adjust_strategy('defensive')
            elif self.metrics['win_rate'] > self.strategy_params['target_win_rate']:
                self._adjust_strategy('aggressive')
            
            # Check drawdown
            if self.metrics['max_drawdown'] > self.risk_params['max_drawdown']:
                self._adjust_strategy('risk_reduction')
            
        except Exception as e:
            logger.error(f"Error checking strategy adjustment: {str(e)}")
    
    def _adjust_strategy(self, mode: str):
        """Adjust strategy parameters"""
        if mode == 'defensive':
            self.strategy_params['position_size'] *= 0.8
            self.strategy_params['stop_loss_multiplier'] *= 0.9
            
        elif mode == 'aggressive':
            self.strategy_params['position_size'] *= 1.2
            self.strategy_params['take_profit_multiplier'] *= 1.1
            
        elif mode == 'risk_reduction':
            self.strategy_params['position_size'] *= 0.7
            self.strategy_params['max_positions'] = max(1, self.strategy_params['max_positions'] - 1)
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            'capital': self.current_capital,
            'active_trades': len(self.active_trades),
            'metrics': self.metrics,
            'system_health': self.monitor.get_system_status()
        }
    
    def stop(self):
        """Stop the trading system"""
        self.monitor.stop()

def create_trading_system(config_path: str) -> TradingSystem:
    """Create and initialize a trading system"""
    return TradingSystem(config_path)

if __name__ == "__main__":
    # Example usage
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    trading_system = create_trading_system(config_path)
    
    try:
        # Simulate some trading
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        for i in range(len(market_data)):
            result = trading_system.execute_strategy(market_data.iloc[:i+1])
            if result and result['status'] == 'success':
                print(f"Executed trade: {result['trade']}")
        
        # Get final status
        status = trading_system.get_status()
        print("\nFinal Status:")
        print(json.dumps(status, indent=2, default=str))
        
    finally:
        trading_system.stop() 