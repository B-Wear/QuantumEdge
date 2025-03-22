import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Data class to store risk metrics"""
    daily_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    max_consecutive_wins: int

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config['initial_capital']
        self.current_capital = self.initial_capital
        self.risk_per_trade = config['risk_per_trade']
        self.max_daily_loss = config['max_daily_loss']
        self.max_drawdown = config['max_drawdown']
        self.max_positions = config['max_positions']
        self.leverage = config['leverage']
        self.stop_loss_pct = config['stop_loss_pct']
        self.take_profit_pct = config['take_profit_pct']
        
        # Performance tracking
        self.trades_history = []
        self.daily_pnl = []
        self.positions = {}
        self.risk_metrics = None
        
    def calculate_position_size(self, 
                              price: float,
                              stop_loss: float,
                              account_balance: float) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters
        """
        # Calculate risk amount in base currency
        risk_amount = account_balance * self.risk_per_trade
        
        # Calculate position size based on stop loss
        price_distance = abs(price - stop_loss)
        position_size = risk_amount / price_distance
        
        # Apply leverage
        position_size *= self.leverage
        
        # Calculate required margin
        required_margin = (position_size * price) / self.leverage
        
        # Ensure we don't exceed maximum position size
        max_position = account_balance * self.leverage / price
        position_size = min(position_size, max_position)
        
        return position_size, required_margin
    
    def validate_trade(self,
                      symbol: str,
                      position_size: float,
                      price: float,
                      stop_loss: float,
                      take_profit: float) -> bool:
        """
        Validate if a trade meets risk management criteria
        """
        # Check if we have too many open positions
        if len(self.positions) >= self.max_positions:
            logger.warning("Maximum number of positions reached")
            return False
        
        # Check if symbol is already in a position
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        # Calculate potential loss
        potential_loss = position_size * abs(price - stop_loss)
        
        # Check if potential loss exceeds daily limit
        if potential_loss > self.max_daily_loss:
            logger.warning("Trade exceeds maximum daily loss limit")
            return False
        
        # Check if stop loss is too far
        stop_loss_distance = abs(price - stop_loss) / price
        if stop_loss_distance > self.stop_loss_pct:
            logger.warning("Stop loss distance exceeds maximum allowed")
            return False
        
        # Check if take profit is reasonable
        take_profit_distance = abs(take_profit - price) / price
        if take_profit_distance > self.take_profit_pct:
            logger.warning("Take profit distance exceeds maximum allowed")
            return False
        
        return True
    
    def update_position(self,
                       symbol: str,
                       entry_price: float,
                       current_price: float,
                       position_size: float,
                       position_type: str) -> Optional[float]:
        """
        Update position and check for stop loss or take profit
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                'entry_price': entry_price,
                'current_price': current_price,
                'position_size': position_size,
                'position_type': position_type,
                'entry_time': datetime.now()
            }
            return None
        
        position = self.positions[symbol]
        pnl = 0
        
        # Calculate unrealized PnL
        if position_type == 'long':
            pnl = (current_price - entry_price) * position_size
        else:
            pnl = (entry_price - current_price) * position_size
        
        # Check stop loss
        if position_type == 'long':
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                pnl = -position_size * entry_price * self.stop_loss_pct
                del self.positions[symbol]
                return pnl
        else:
            if current_price >= entry_price * (1 + self.stop_loss_pct):
                pnl = -position_size * entry_price * self.stop_loss_pct
                del self.positions[symbol]
                return pnl
        
        # Check take profit
        if position_type == 'long':
            if current_price >= entry_price * (1 + self.take_profit_pct):
                pnl = position_size * entry_price * self.take_profit_pct
                del self.positions[symbol]
                return pnl
        else:
            if current_price <= entry_price * (1 - self.take_profit_pct):
                pnl = position_size * entry_price * self.take_profit_pct
                del self.positions[symbol]
                return pnl
        
        # Update current price
        position['current_price'] = current_price
        return None
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate various risk metrics
        """
        if not self.trades_history:
            return None
        
        # Convert trades history to DataFrame
        df = pd.DataFrame(self.trades_history)
        
        # Calculate daily returns
        daily_returns = df.groupby(df['exit_time'].dt.date)['pnl'].sum()
        
        # Calculate drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        # Calculate metrics
        daily_drawdown = drawdowns.min()
        max_drawdown = drawdowns.min()
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = daily_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # Calculate win rate and profit factor
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        win_rate = len(winning_trades) / len(df)
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
        
        # Calculate average win and loss
        avg_win = winning_trades['pnl'].mean()
        avg_loss = losing_trades['pnl'].mean()
        
        # Calculate consecutive wins and losses
        df['consecutive'] = (df['pnl'] > 0).astype(int)
        max_consecutive_wins = df['consecutive'].groupby((df['consecutive'] != df['consecutive'].shift()).cumsum()).sum().max()
        max_consecutive_losses = df['consecutive'].groupby((df['consecutive'] != df['consecutive'].shift()).cumsum()).sum().min()
        
        self.risk_metrics = RiskMetrics(
            daily_drawdown=daily_drawdown,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins
        )
        
        return self.risk_metrics
    
    def should_stop_trading(self) -> bool:
        """
        Check if trading should be stopped based on risk metrics
        """
        if not self.risk_metrics:
            return False
        
        # Stop if maximum drawdown is exceeded
        if self.risk_metrics.max_drawdown < -self.max_drawdown:
            logger.warning("Maximum drawdown exceeded. Stopping trading.")
            return True
        
        # Stop if win rate is too low
        if self.risk_metrics.win_rate < self.config['min_win_rate']:
            logger.warning("Win rate below minimum threshold. Stopping trading.")
            return True
        
        # Stop if profit factor is too low
        if self.risk_metrics.profit_factor < self.config['min_profit_factor']:
            logger.warning("Profit factor below minimum threshold. Stopping trading.")
            return True
        
        return False 