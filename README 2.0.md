# Quantum Edge Trading System

A sophisticated, AI-powered trading system that combines machine learning, sentiment analysis, and advanced risk management to execute trades across multiple markets.

## Features

### 1. Multi-Bot Management
- Create and manage multiple trading bots
- Each bot can run different strategies
- Real-time monitoring and control
- Automatic performance optimization

### 2. Machine Learning Models
- LSTM for price prediction
- Reinforcement Learning (PPO) for strategy optimization
- Feature engineering and selection
- Automatic model retraining

### 3. Risk Management
- Dynamic position sizing
- Multi-level stop-loss system
- Portfolio correlation analysis
- Exposure management
- VaR calculations

### 4. Sentiment Analysis
- Real-time news analysis
- Social media sentiment tracking
- Market sentiment indicators
- Impact analysis

### 5. System Monitoring
- Performance metrics tracking
- Health monitoring
- Resource optimization
- Automatic issue detection and resolution

### 6. Code Guardian
- Continuous code quality monitoring
- Automatic issue detection
- Self-healing capabilities
- Performance optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum-edge-system.git
cd quantum-edge-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
- Copy `config/quantum_edge_config.json.example` to `config/quantum_edge_config.json`
- Update the configuration with your API keys and preferences

## Configuration

The system is configured through `quantum_edge_config.json`. Key sections include:

1. **System Settings**
   - Basic system configuration
   - Logging settings
   - Dashboard configuration

2. **Trading Settings**
   - Exchange configuration
   - Market selection
   - Order types
   - Position types

3. **ML Model Settings**
   - LSTM configuration
   - Reinforcement learning parameters
   - Feature selection
   - Training parameters

4. **Risk Management**
   - Position sizing rules
   - Stop-loss configuration
   - Risk limits
   - Exposure management

5. **Sentiment Analysis**
   - News API configuration
   - Social media settings
   - Analysis parameters
   - Update intervals

6. **Optimization Settings**
   - System optimization rules
   - Trading optimization parameters
   - ML model optimization
   - Performance thresholds

## Usage

1. Start the Quantum Edge System:
```python
from trading_bot.src.quantum_edge_system import create_quantum_edge_system

# Create system instance
system = create_quantum_edge_system(".", "config/quantum_edge_config.json")

# Create bot configurations
configs = [
    {
        "trading": {
            "initial_capital": 100000,
            "strategy": "trend_following",
            "markets": ["BTC/USD", "ETH/USD"]
        },
        "ml_model": {
            "type": "lstm",
            "features": ["price", "volume", "sentiment"]
        }
    },
    {
        "trading": {
            "initial_capital": 50000,
            "strategy": "mean_reversion",
            "markets": ["SOL/USD", "ADA/USD"]
        },
        "ml_model": {
            "type": "reinforcement",
            "features": ["technical_indicators", "order_flow"]
        }
    }
]

# Create and start bots
bot_ids = []
for config in configs:
    bot_id = system.create_quantum_bot(config)
    bot_ids.append(bot_id)
    system.start_bot(bot_id)

# Monitor system
while True:
    status = system.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    time.sleep(300)  # Update every 5 minutes
```

2. Access the dashboard:
   - Open your browser
   - Navigate to `http://localhost:8050`
   - Monitor system performance, bot status, and trading metrics

## Dashboard Features

1. **System Overview**
   - System health metrics
   - Resource utilization
   - Overall performance

2. **Bot Management**
   - Individual bot status
   - Performance metrics
   - Control panel

3. **Trading View**
   - Active positions
   - Order history
   - P&L tracking

4. **Risk Analytics**
   - Risk metrics
   - Exposure analysis
   - VaR calculations

5. **ML Model Insights**
   - Model performance
   - Feature importance
   - Training status

6. **Sentiment Analysis**
   - Market sentiment
   - News impact
   - Social media trends

## API Reference

### System Management
```python
create_quantum_bot(config: Dict) -> str
start_bot(bot_id: str)
stop_bot(bot_id: str)
get_system_status() -> Dict
get_bot_status(bot_id: str) -> Dict
```

### Performance Monitoring
```python
get_performance_metrics() -> Dict
get_risk_metrics() -> Dict
get_sentiment_metrics() -> Dict
get_ml_metrics() -> Dict
```

### System Control
```python
optimize_system()
optimize_ml_models()
optimize_risk_parameters()
handle_risk_breach()
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please:
1. Check the documentation
2. Open an issue
3. Contact the development team

## Disclaimer

This software is for educational purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred while using this system. 