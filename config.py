import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange Configuration
EXCHANGE_CONFIG = {
    'name': 'binance',
    'api_key': os.getenv('BINANCE_API_KEY', ''),
    'api_secret': os.getenv('BINANCE_API_SECRET', ''),
    'testnet': True  # Use testnet for development
}

# Trading Parameters
TRADING_CONFIG = {
    'symbols': ['BTC/USDT', 'ETH/USDT', 'EUR/USD'],
    'timeframes': ['1h', '4h', '1d'],
    'initial_capital': 10000,
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_positions': 3,
    'position_sizing': {
        'method': 'fixed_fractional',
        'fraction': 0.02  # 2% of capital per trade
    }
}

# Technical Analysis Parameters
TECHNICAL_CONFIG = {
    'indicators': {
        'bollinger_bands': {
            'period': 20,
            'std_dev': 2
        },
        'rsi': {
            'period': 14,
            'overbought': 70,
            'oversold': 30
        },
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'atr': {
            'period': 14
        }
    },
    'patterns': {
        'head_and_shoulders': True,
        'double_top_bottom': True,
        'triangles': True
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    'model_type': 'lstm',  # Options: 'lstm', 'rf', 'xgboost'
    'features': [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'rsi', 'macd', 'atr',
        'bb_upper', 'bb_lower', 'bb_width'
    ],
    'sequence_length': 10,
    'prediction_horizon': 1,
    'train_test_split': 0.8,
    'validation_split': 0.1
}

# Reinforcement Learning Configuration
RL_CONFIG = {
    'algorithm': 'PPO',
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5
}

# Risk Management Configuration
RISK_CONFIG = {
    'stop_loss': {
        'method': 'atr',
        'atr_multiplier': 2
    },
    'take_profit': {
        'method': 'risk_reward',
        'risk_reward_ratio': 2
    },
    'trailing_stop': {
        'enabled': True,
        'activation_percentage': 0.02,
        'trail_percentage': 0.01
    }
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'sources': [
        'reuters',
        'bloomberg',
        'forexfactory'
    ],
    'update_interval': 3600,  # 1 hour
    'weight': 0.2  # Weight in final decision
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 10000,
    'commission': 0.001,  # 0.1%
    'slippage': 0.0001    # 0.01%
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/trading_bot.log'
}

# Web Dashboard Configuration
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'update_interval': 5  # seconds
} 