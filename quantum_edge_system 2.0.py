import logging
import threading
import queue
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import traceback
import signal
import sys
from .trading_system import TradingSystem, create_trading_system
from .code_guardian import CodeGuardian, create_guardian
from .system_monitor import SystemMonitor
from .dashboard import create_dashboard
from .risk_management import RiskManager, RiskMetrics
from .sentiment_analysis import SentimentAnalyzer
from .machine_learning_model import MachineLearningModel
from .trading_environment import TradingEnvironment

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/quantum_edge_system.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            logger.error(f"All {max_retries} attempts failed. Last error: {str(last_exception)}")
            raise last_exception
        return wrapper
    return decorator

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    network_latency: float
    disk_usage: float
    error_rate: float
    last_check: datetime

@dataclass
class QuantumBotInstance:
    """Represents a Quantum Edge trading bot instance"""
    bot_id: str
    trading_system: TradingSystem
    ml_model: MachineLearningModel
    risk_manager: RiskManager
    sentiment_analyzer: SentimentAnalyzer
    trading_env: TradingEnvironment
    config: Dict
    status: str
    last_update: datetime
    performance_metrics: Dict
    message_queue: queue.Queue
    health_metrics: SystemHealth
    error_count: int
    recovery_attempts: int
    last_error: Optional[Exception] = None

class QuantumEdgeSystem:
    def __init__(self, project_root: str, config_path: str):
        """Initialize the Quantum Edge System"""
        self.project_root = project_root
        self.load_config(config_path)
        self.bots: Dict[str, QuantumBotInstance] = {}
        self.code_guardian = create_guardian(project_root)
        self.system_monitor = SystemMonitor(project_root)
        self.message_queue = queue.Queue()
        self.running = True
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.error_threshold = self.config['monitoring']['alert_thresholds']['error_rate']
        self.recovery_threshold = 3

        # Initialize system components
        self._init_system_components()
        self._start_management_threads()
        self._init_dashboard()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle system shutdown signals"""
        logger.info(f"Received shutdown signal {signum}")
        self.stop()

    @retry_on_failure(max_retries=3, delay=2.0)
    def load_config(self, config_path: str):
        """Load system configuration with retry"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.validate_config()
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def validate_config(self):
        """Validate system configuration with detailed checks"""
        required_sections = ['trading', 'ml_model', 'risk_management', 'sentiment_analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
            
            # Validate section-specific requirements
            if section == 'trading':
                self._validate_trading_config()
            elif section == 'ml_model':
                self._validate_ml_config()
            elif section == 'risk_management':
                self._validate_risk_config()
            elif section == 'sentiment_analysis':
                self._validate_sentiment_config()

    def _validate_trading_config(self):
        """Validate trading configuration"""
        trading_config = self.config['trading']
        required_fields = ['exchange', 'markets', 'timeframes']
        for field in required_fields:
            if field not in trading_config:
                raise ValueError(f"Missing required trading config field: {field}")
        
        if not trading_config['markets']:
            raise ValueError("No trading markets specified")
        
        if not trading_config['timeframes']:
            raise ValueError("No timeframes specified")

    def _validate_ml_config(self):
        """Validate ML configuration"""
        ml_config = self.config['ml_model']
        if 'default_type' not in ml_config:
            raise ValueError("ML model default type not specified")
        
        if ml_config['default_type'] not in ['lstm', 'reinforcement']:
            raise ValueError(f"Invalid ML model type: {ml_config['default_type']}")

    def _validate_risk_config(self):
        """Validate risk management configuration"""
        risk_config = self.config['risk_management']
        required_sections = ['position_sizing', 'risk_limits', 'stop_loss']
        for section in required_sections:
            if section not in risk_config:
                raise ValueError(f"Missing required risk config section: {section}")

    def _validate_sentiment_config(self):
        """Validate sentiment analysis configuration"""
        sentiment_config = self.config['sentiment_analysis']
        if 'providers' not in sentiment_config:
            raise ValueError("Sentiment analysis providers not specified")

    def _init_system_components(self):
        """Initialize system-wide components with error handling"""
        try:
            self.global_risk_manager = RiskManager(self.config['risk_management'])
            self.global_sentiment_analyzer = SentimentAnalyzer(self.config['sentiment_analysis'])
            self.performance_tracker = self._init_performance_tracker()
            self.health_metrics = SystemHealth(
                cpu_usage=0.0,
                memory_usage=0.0,
                network_latency=0.0,
                disk_usage=0.0,
                error_rate=0.0,
                last_check=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error initializing system components: {str(e)}")
            raise

    def _init_performance_tracker(self):
        """Initialize system-wide performance tracking with enhanced metrics"""
        return {
            'system_metrics': {},
            'bot_metrics': {},
            'risk_metrics': {},
            'sentiment_metrics': {},
            'ml_metrics': {},
            'error_metrics': {
                'total_errors': 0,
                'error_types': {},
                'recovery_success_rate': 1.0
            },
            'performance_metrics': {
                'average_latency': 0.0,
                'throughput': 0.0,
                'resource_utilization': {}
            }
        }

    def _start_management_threads(self):
        """Start system management threads with enhanced monitoring"""
        self.threads = {
            'monitor': threading.Thread(target=self._monitor_system),
            'message_handler': threading.Thread(target=self._handle_messages),
            'code_repair': threading.Thread(target=self._continuous_code_repair),
            'performance_tracker': threading.Thread(target=self._track_performance),
            'risk_monitor': threading.Thread(target=self._monitor_risk),
            'sentiment_monitor': threading.Thread(target=self._monitor_sentiment),
            'health_checker': threading.Thread(target=self._check_system_health)
        }
        
        for thread_name, thread in self.threads.items():
            thread.daemon = True
            thread.name = f"quantum_edge_{thread_name}"
            thread.start()
            logger.info(f"Started thread: {thread_name}")

    def _check_system_health(self):
        """Continuously check system health"""
        while self.running:
            try:
                health_metrics = {
                    'cpu_usage': self.system_monitor.get_cpu_usage(),
                    'memory_usage': self.system_monitor.get_memory_usage(),
                    'network_latency': self.system_monitor.get_network_latency(),
                    'disk_usage': self.system_monitor.get_disk_usage(),
                    'error_rate': self._calculate_error_rate()
                }
                
                self.health_metrics = SystemHealth(
                    **health_metrics,
                    last_check=datetime.now()
                )
                
                if self._is_system_unhealthy():
                    self._handle_system_health_issue()
                
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error checking system health: {str(e)}")
                time.sleep(300)

    def _calculate_error_rate(self) -> float:
        """Calculate system-wide error rate"""
        total_errors = sum(bot.error_count for bot in self.bots.values())
        total_operations = sum(bot.performance_metrics.get('total_operations', 0) 
                             for bot in self.bots.values())
        return total_errors / total_operations if total_operations > 0 else 0.0

    def _is_system_unhealthy(self) -> bool:
        """Determine if system is unhealthy"""
        thresholds = self.config['monitoring']['alert_thresholds']
        return (
            self.health_metrics.cpu_usage > thresholds['cpu_usage'] or
            self.health_metrics.memory_usage > thresholds['memory_usage'] or
            self.health_metrics.network_latency > thresholds['api_latency'] or
            self.health_metrics.error_rate > thresholds['error_rate']
        )

    def _handle_system_health_issue(self):
        """Handle system health issues"""
        logger.warning("System health issues detected")
        
        # Reduce system load
        self._optimize_system_performance()
        
        # Notify all bots
        self.broadcast_message({
            'type': 'system_alert',
            'action': 'reduce_load',
            'timestamp': datetime.now().isoformat()
        })
        
        # Log health metrics
        logger.warning(f"Current health metrics: {self.health_metrics}")

    @retry_on_failure(max_retries=3, delay=2.0)
    def create_quantum_bot(self, config: Dict) -> str:
        """Create a new Quantum Edge bot instance with enhanced error handling"""
        try:
            bot_id = f"quantum_bot_{len(self.bots) + 1}"
            
            # Create bot-specific config
            bot_config = self._create_bot_config(config)
            config_path = os.path.join(self.project_root, 'config', f'{bot_id}_config.json')
            
            # Save config with retry
            self._save_bot_config(config_path, bot_config)
            
            # Initialize bot components with parallel execution
            components = self._initialize_bot_components(bot_config)
            
            # Create bot instance with health monitoring
            bot = QuantumBotInstance(
                bot_id=bot_id,
                trading_system=components['trading_system'],
                ml_model=components['ml_model'],
                risk_manager=components['risk_manager'],
                sentiment_analyzer=components['sentiment_analyzer'],
                trading_env=components['trading_env'],
                config=bot_config,
                status='initialized',
                last_update=datetime.now(),
                performance_metrics={},
                message_queue=queue.Queue(),
                health_metrics=SystemHealth(
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    network_latency=0.0,
                    disk_usage=0.0,
                    error_rate=0.0,
                    last_check=datetime.now()
                ),
                error_count=0,
                recovery_attempts=0
            )
            
            self.bots[bot_id] = bot
            logger.info(f"Created new Quantum Edge bot instance: {bot_id}")
            
            return bot_id
            
        except Exception as e:
            logger.error(f"Error creating Quantum Edge bot: {str(e)}")
            raise

    def _save_bot_config(self, config_path: str, config: Dict):
        """Save bot configuration with retry"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving bot config: {str(e)}")
            raise

    def _initialize_bot_components(self, config: Dict) -> Dict:
        """Initialize bot components in parallel"""
        components = {}
        futures = {
            'trading_system': self.thread_pool.submit(
                create_trading_system, config['trading']
            ),
            'ml_model': self.thread_pool.submit(
                MachineLearningModel, config['ml_model']
            ),
            'risk_manager': self.thread_pool.submit(
                RiskManager, config['risk_management']
            ),
            'sentiment_analyzer': self.thread_pool.submit(
                SentimentAnalyzer, config['sentiment_analysis']
            ),
            'trading_env': self.thread_pool.submit(
                TradingEnvironment, config['trading']
            )
        }
        
        for name, future in futures.items():
            try:
                components[name] = future.result()
            except Exception as e:
                logger.error(f"Error initializing {name}: {str(e)}")
                raise
        
        return components

    def start_bot(self, bot_id: str):
        """Start a Quantum Edge bot with enhanced error handling"""
        if bot_id in self.bots:
            try:
                bot = self.bots[bot_id]
                
                # Initialize components with retry
                self._initialize_bot_components_with_retry(bot)
                
                # Start bot with health monitoring
                bot.status = 'running'
                bot.last_update = datetime.now()
                
                logger.info(f"Started Quantum Edge bot: {bot_id}")
                
                # Send start message to bot
                self._send_message_to_bot(bot_id, {
                    'type': 'command',
                    'action': 'start',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error starting Quantum Edge bot {bot_id}: {str(e)}")
                self._handle_bot_error(bot_id, e)

    def _initialize_bot_components_with_retry(self, bot: QuantumBotInstance):
        """Initialize bot components with retry mechanism"""
        try:
            bot.ml_model.initialize()
            bot.trading_env.reset()
            bot.trading_system.start()
        except Exception as e:
            logger.error(f"Error initializing bot components: {str(e)}")
            bot.error_count += 1
            bot.last_error = e
            raise

    def _handle_bot_error(self, bot_id: str, error: Exception):
        """Handle bot errors with recovery mechanism"""
        bot = self.bots[bot_id]
        bot.error_count += 1
        bot.last_error = error
        
        if bot.error_count >= self.recovery_threshold:
            logger.warning(f"Bot {bot_id} exceeded error threshold, attempting recovery")
            self._attempt_bot_recovery(bot_id)
        else:
            bot.status = 'error'
            logger.error(f"Bot {bot_id} encountered error: {str(error)}")

    def _attempt_bot_recovery(self, bot_id: str):
        """Attempt to recover a failed bot"""
        bot = self.bots[bot_id]
        try:
            # Reset bot state
            bot.error_count = 0
            bot.recovery_attempts += 1
            
            # Reinitialize components
            self._initialize_bot_components_with_retry(bot)
            
            # Restart bot
            bot.status = 'running'
            logger.info(f"Successfully recovered bot {bot_id}")
            
        except Exception as e:
            logger.error(f"Failed to recover bot {bot_id}: {str(e)}")
            bot.status = 'failed'
            self._handle_critical_bot_failure(bot_id)

    def _handle_critical_bot_failure(self, bot_id: str):
        """Handle critical bot failures"""
        logger.error(f"Critical failure for bot {bot_id}")
        
        # Notify system
        self.message_queue.put({
            'type': 'critical_error',
            'bot_id': bot_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update system metrics
        self.performance_tracker['error_metrics']['total_errors'] += 1
        error_type = type(self.bots[bot_id].last_error).__name__
        self.performance_tracker['error_metrics']['error_types'][error_type] = \
            self.performance_tracker['error_metrics']['error_types'].get(error_type, 0) + 1

    def stop(self):
        """Stop the Quantum Edge System with graceful shutdown"""
        logger.info("Initiating system shutdown")
        self.running = False
        
        # Stop all bots gracefully
        for bot_id in self.bots:
            self.stop_bot(bot_id)
        
        # Stop components
        self.code_guardian.stop()
        self.system_monitor.stop()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Wait for threads
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for thread {thread_name} to finish")
                thread.join(timeout=30)
                if thread.is_alive():
                    logger.warning(f"Thread {thread_name} did not terminate gracefully")
        
        logger.info("Quantum Edge System stopped")

def create_quantum_edge_system(project_root: str, config_path: str) -> QuantumEdgeSystem:
    """Create and initialize a Quantum Edge System"""
    return QuantumEdgeSystem(project_root, config_path)

if __name__ == "__main__":
    # Example usage with enhanced error handling
    config_path = "config/quantum_edge_config.json"
    system = None
    
    try:
        system = create_quantum_edge_system(".", config_path)
        
        # Create test bots with different strategies
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
            
    except KeyboardInterrupt:
        print("Shutting down Quantum Edge System...")
        if system:
            system.stop()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        if system:
            system.stop()
        raise 