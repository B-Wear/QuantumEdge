import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

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
    def __init__(self, initial_capital: float = 50.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.performance_history: List[Dict] = []
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.strategy_parameters = self._get_initial_parameters()
        self.risk_metrics = self._initialize_risk_metrics()
        self.last_recalibration = datetime.now()
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load configuration from config file"""
        try:
            with open('config/config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found. Using default parameters.")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            'risk_per_trade': 0.01,  # 1% risk per trade
            'max_positions': 2,
            'min_win_rate': 0.4,
            'recalibration_window': 20,
            'max_drawdown': 0.1,  # 10% maximum drawdown
            'leverage': 1,  # No leverage initially
            'position_sizing': {
                'method': 'fixed_fractional',
                'fraction': 0.01  # 1% of capital per trade
            }
        }
    
    def _get_initial_parameters(self) -> Dict:
        """Get initial strategy parameters"""
        return {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'atr_multiplier': 2
        }
    
    def _initialize_risk_metrics(self) -> Dict:
        """Initialize risk metrics tracking"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'avg_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    def monitor_performance(self, window_size: int = 20) -> bool:
        """
        Monitor recent performance and determine if recalibration is needed
        Returns True if recalibration is needed
        """
        if len(self.performance_history) < window_size:
            return False
            
        recent_performance = self.performance_history[-window_size:]
        win_rate = sum(1 for trade in recent_performance if trade['pnl'] > 0) / window_size
        
        # Check various performance metrics
        needs_recalibration = False
        
        # Win rate check
        if win_rate < self.config['min_win_rate']:
            logger.warning(f"Win rate {win_rate:.2%} below threshold {self.config['min_win_rate']:.2%}")
            needs_recalibration = True
        
        # Drawdown check
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > self.config['max_drawdown']:
            logger.warning(f"Current drawdown {current_drawdown:.2%} exceeds maximum {self.config['max_drawdown']:.2%}")
            needs_recalibration = True
        
        # Profit factor check
        profit_factor = self._calculate_profit_factor(recent_performance)
        if profit_factor < 1.0:
            logger.warning(f"Profit factor {profit_factor:.2f} below 1.0")
            needs_recalibration = True
        
        return needs_recalibration
    
    def recalibrate_strategy(self, market_data: pd.DataFrame):
        """
        Adjust strategy parameters based on recent market conditions
        """
        logger.info("Starting strategy recalibration")
        
        # Analyze market conditions
        volatility = self._calculate_volatility(market_data)
        trend_strength = self._calculate_trend_strength(market_data)
        
        # Adjust parameters based on market conditions
        new_parameters = self.strategy_parameters.copy()
        
        # Adjust RSI levels based on volatility
        if volatility > 0.02:  # High volatility
            new_parameters['rsi_overbought'] = 75
            new_parameters['rsi_oversold'] = 25
        else:  # Low volatility
            new_parameters['rsi_overbought'] = 70
            new_parameters['rsi_oversold'] = 30
        
        # Adjust ATR multiplier based on trend strength
        if trend_strength > 0.7:  # Strong trend
            new_parameters['atr_multiplier'] = 2.5
        else:  # Weak trend
            new_parameters['atr_multiplier'] = 2.0
        
        # Update parameters
        self.strategy_parameters = new_parameters
        self.last_recalibration = datetime.now()
        
        logger.info("Strategy recalibration completed")
        logger.info(f"New parameters: {new_parameters}")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules
        """
        risk_amount = self.current_capital * self.config['risk_per_trade']
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero. Cannot calculate position size.")
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        # Apply leverage if configured
        if self.config['leverage'] > 1:
            position_size *= self.config['leverage']
        
        # Ensure position size doesn't exceed maximum allowed
        max_position = self.current_capital * self.config['position_sizing']['fraction']
        position_size = min(position_size, max_position)
        
        return position_size
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        returns = data['close'].pct_change()
        return returns.std()
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        # Implementation would go here
        return 0.5  # Placeholder
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.performance_history:
            return 0.0
        
        peak = max(self.performance_history, key=lambda x: x['equity'])['equity']
        current = self.performance_history[-1]['equity']
        return (peak - current) / peak
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from recent trades"""
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf')
        
        return gross_profit / gross_loss
    
    def update_risk_metrics(self, trade: Trade):
        """Update risk metrics after a trade"""
        self.risk_metrics['total_trades'] += 1
        
        if trade.pnl and trade.pnl > 0:
            self.risk_metrics['winning_trades'] += 1
            self.risk_metrics['largest_win'] = max(
                self.risk_metrics['largest_win'],
                trade.pnl
            )
        elif trade.pnl and trade.pnl < 0:
            self.risk_metrics['losing_trades'] += 1
            self.risk_metrics['largest_loss'] = min(
                self.risk_metrics['largest_loss'],
                trade.pnl
            )
        
        # Update win rate
        if self.risk_metrics['total_trades'] > 0:
            self.risk_metrics['win_rate'] = (
                self.risk_metrics['winning_trades'] / 
                self.risk_metrics['total_trades']
            )
        
        # Update average trade
        if trade.pnl:
            self.risk_metrics['avg_trade'] = (
                (self.risk_metrics['avg_trade'] * (self.risk_metrics['total_trades'] - 1) + 
                 trade.pnl) / self.risk_metrics['total_trades']
            )
    
    def save_state(self):
        """Save current system state"""
        state = {
            'current_capital': self.current_capital,
            'strategy_parameters': self.strategy_parameters,
            'risk_metrics': self.risk_metrics,
            'last_recalibration': self.last_recalibration.isoformat(),
            'active_trades': {
                symbol: {
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'position_size': trade.position_size,
                    'entry_time': trade.entry_time.isoformat()
                }
                for symbol, trade in self.active_trades.items()
            }
        }
        
        try:
            with open('data/system_state.json', 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
    
    def load_state(self):
        """Load system state from file"""
        try:
            with open('data/system_state.json', 'r') as f:
                state = json.load(f)
                
            self.current_capital = state['current_capital']
            self.strategy_parameters = state['strategy_parameters']
            self.risk_metrics = state['risk_metrics']
            self.last_recalibration = datetime.fromisoformat(state['last_recalibration'])
            
            # Reconstruct active trades
            self.active_trades = {}
            for symbol, trade_data in state['active_trades'].items():
                self.active_trades[symbol] = Trade(
                    symbol=symbol,
                    direction=trade_data['direction'],
                    entry_price=trade_data['entry_price'],
                    stop_loss=trade_data['stop_loss'],
                    take_profit=trade_data['take_profit'],
                    position_size=trade_data['position_size'],
                    entry_time=datetime.fromisoformat(trade_data['entry_time'])
                )
        except FileNotFoundError:
            logger.info("No saved state found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'current_capital': self.current_capital,
            'total_trades': self.risk_metrics['total_trades'],
            'win_rate': self.risk_metrics['win_rate'],
            'profit_factor': self.risk_metrics['profit_factor'],
            'current_drawdown': self.risk_metrics['current_drawdown'],
            'active_trades': len(self.active_trades),
            'last_recalibration': self.last_recalibration.isoformat(),
            'strategy_parameters': self.strategy_parameters
        } 