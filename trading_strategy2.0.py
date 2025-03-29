import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import json
import os

from .technical_analysis import TechnicalAnalyzer
from .machine_learning import MachineLearningModel
from .risk_management import RiskManager
from .sentiment_analysis import SentimentAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Data class to store trade signals"""
    symbol: str
    action: str  # 'buy', 'sell', or 'hold'
    confidence: float
    price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    technical_score: float
    ml_score: float
    sentiment_score: float
    risk_score: float

class TradingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.ml_model = MachineLearningModel(config)
        self.risk_manager = RiskManager(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        
        # Initialize state
        self.active_trades = {}
        self.trade_history = []
        self.performance_metrics = {}
        
    def analyze_market(self, symbol: str, data: pd.DataFrame) -> TradeSignal:
        """
        Analyze market conditions and generate trading signals
        """
        try:
            # Technical analysis
            technical_signals = self.technical_analyzer.generate_signals(data)
            technical_score = technical_signals['signal']
            
            # Machine learning prediction
            ml_predictions = self.ml_model.predict(data)
            ml_score = ml_predictions[-1] if ml_predictions is not None else 0
            
            # Sentiment analysis
            sentiment_score = self.sentiment_analyzer.get_combined_sentiment(symbol)
            if sentiment_score is None:
                sentiment_score = 0
            
            # Risk assessment
            risk_score = self.risk_manager.calculate_risk_metrics()
            
            # Combine signals
            combined_score = (
                technical_score * self.config['signal_weights']['technical'] +
                ml_score * self.config['signal_weights']['ml'] +
                sentiment_score * self.config['signal_weights']['sentiment'] +
                risk_score * self.config['signal_weights']['risk']
            )
            
            # Generate trading signal
            current_price = data['close'].iloc[-1]
            
            # Calculate stop loss and take profit levels
            stop_loss = self._calculate_stop_loss(current_price, combined_score)
            take_profit = self._calculate_take_profit(current_price, combined_score)
            
            # Calculate position size
            position_size, required_margin = self.risk_manager.calculate_position_size(
                current_price,
                stop_loss,
                self.risk_manager.current_capital
            )
            
            # Determine action
            if combined_score > self.config['signal_thresholds']['buy']:
                action = 'buy'
            elif combined_score < self.config['signal_thresholds']['sell']:
                action = 'sell'
            else:
                action = 'hold'
            
            # Create trade signal
            signal = TradeSignal(
                symbol=symbol,
                action=action,
                confidence=abs(combined_score),
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timestamp=datetime.now(),
                technical_score=technical_score,
                ml_score=ml_score,
                sentiment_score=sentiment_score,
                risk_score=risk_score
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return None
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """
        Execute a trade based on the signal
        """
        try:
            # Validate trade
            if not self.risk_manager.validate_trade(
                signal.symbol,
                signal.position_size,
                signal.price,
                signal.stop_loss,
                signal.take_profit
            ):
                return False
            
            # Execute trade
            if signal.action in ['buy', 'sell']:
                # Update position
                pnl = self.risk_manager.update_position(
                    signal.symbol,
                    signal.price,
                    signal.price,
                    signal.position_size,
                    signal.action
                )
                
                if pnl is not None:
                    # Record trade
                    self.trade_history.append({
                        'symbol': signal.symbol,
                        'action': signal.action,
                        'entry_price': signal.price,
                        'exit_price': signal.price,
                        'position_size': signal.position_size,
                        'pnl': pnl,
                        'entry_time': signal.timestamp,
                        'exit_time': datetime.now()
                    })
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update all open positions with current prices
        """
        try:
            for symbol, price in current_prices.items():
                if symbol in self.risk_manager.positions:
                    position = self.risk_manager.positions[symbol]
                    pnl = self.risk_manager.update_position(
                        symbol,
                        position['entry_price'],
                        price,
                        position['position_size'],
                        position['position_type']
                    )
                    
                    if pnl is not None:
                        # Record trade
                        self.trade_history.append({
                            'symbol': symbol,
                            'action': 'close',
                            'entry_price': position['entry_price'],
                            'exit_price': price,
                            'position_size': position['position_size'],
                            'pnl': pnl,
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.now()
                        })
                        
                        # Update performance metrics
                        self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def _calculate_stop_loss(self, price: float, signal: float) -> float:
        """
        Calculate stop loss level based on signal strength
        """
        # Adjust stop loss distance based on signal strength
        base_stop_loss = self.config['stop_loss_pct']
        signal_factor = abs(signal)
        stop_loss_distance = base_stop_loss * (1 + signal_factor)
        
        if signal > 0:  # Buy signal
            return price * (1 - stop_loss_distance)
        else:  # Sell signal
            return price * (1 + stop_loss_distance)
    
    def _calculate_take_profit(self, price: float, signal: float) -> float:
        """
        Calculate take profit level based on signal strength
        """
        # Adjust take profit distance based on signal strength
        base_take_profit = self.config['take_profit_pct']
        signal_factor = abs(signal)
        take_profit_distance = base_take_profit * (1 + signal_factor)
        
        if signal > 0:  # Buy signal
            return price * (1 + take_profit_distance)
        else:  # Sell signal
            return price * (1 - take_profit_distance)
    
    def _update_performance_metrics(self):
        """
        Update performance metrics based on trade history
        """
        if not self.trade_history:
            return
        
        # Convert trade history to DataFrame
        df = pd.DataFrame(self.trade_history)
        
        # Calculate metrics
        self.performance_metrics = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'win_rate': len(df[df['pnl'] > 0]) / len(df),
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'max_drawdown': self.risk_manager.risk_metrics.max_drawdown if self.risk_manager.risk_metrics else 0,
            'sharpe_ratio': self.risk_manager.risk_metrics.sharpe_ratio if self.risk_manager.risk_metrics else 0,
            'profit_factor': self.risk_manager.risk_metrics.profit_factor if self.risk_manager.risk_metrics else 0
        }
    
    def should_stop_trading(self) -> bool:
        """
        Check if trading should be stopped based on risk metrics
        """
        return self.risk_manager.should_stop_trading()
    
    def save_state(self, filepath: str):
        """
        Save trading strategy state
        """
        try:
            state = {
                'active_trades': self.active_trades,
                'trade_history': self.trade_history,
                'performance_metrics': self.performance_metrics,
                'risk_metrics': self.risk_manager.risk_metrics.__dict__ if self.risk_manager.risk_metrics else None
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, default=str)
            
            logger.info(f"Trading strategy state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return False
    
    def load_state(self, filepath: str):
        """
        Load trading strategy state
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.active_trades = state['active_trades']
            self.trade_history = state['trade_history']
            self.performance_metrics = state['performance_metrics']
            
            if state['risk_metrics']:
                self.risk_manager.risk_metrics = RiskMetrics(**state['risk_metrics'])
            
            logger.info(f"Trading strategy state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False 