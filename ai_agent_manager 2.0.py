import logging
import threading
import queue
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time
from .trading_system import TradingSystem, create_trading_system
from .code_guardian import CodeGuardian, create_guardian
from .system_monitor import SystemMonitor
from .dashboard import create_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_agent_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BotInstance:
    """Represents a trading bot instance"""
    bot_id: str
    trading_system: TradingSystem
    config: Dict
    status: str
    last_update: datetime
    performance_metrics: Dict
    message_queue: queue.Queue

class AIAgentManager:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.bots: Dict[str, BotInstance] = {}
        self.code_guardian = create_guardian(project_root)
        self.system_monitor = SystemMonitor(project_root)
        self.message_queue = queue.Queue()
        self.running = True

        # Start management threads
        self.threads = {
            'monitor': threading.Thread(target=self._monitor_bots),
            'message_handler': threading.Thread(target=self._handle_messages),
            'code_repair': threading.Thread(target=self._continuous_code_repair)
        }
        
        for thread in self.threads.values():
            thread.daemon = True
            thread.start()

        # Initialize dashboard
        self.dashboard = create_dashboard()
        self.dashboard_thread = threading.Thread(target=self._run_dashboard)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()

    def create_bot(self, config: Dict) -> str:
        """Create a new trading bot instance"""
        try:
            bot_id = f"bot_{len(self.bots) + 1}"
            
            # Save bot-specific config
            config_path = os.path.join(self.project_root, 'config', f'{bot_id}_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Create trading system
            trading_system = create_trading_system(config_path)
            
            # Create bot instance
            bot = BotInstance(
                bot_id=bot_id,
                trading_system=trading_system,
                config=config,
                status='initialized',
                last_update=datetime.now(),
                performance_metrics={},
                message_queue=queue.Queue()
            )
            
            self.bots[bot_id] = bot
            logger.info(f"Created new bot instance: {bot_id}")
            
            return bot_id
            
        except Exception as e:
            logger.error(f"Error creating bot: {str(e)}")
            raise

    def start_bot(self, bot_id: str):
        """Start a trading bot"""
        if bot_id in self.bots:
            try:
                bot = self.bots[bot_id]
                bot.status = 'running'
                bot.last_update = datetime.now()
                logger.info(f"Started bot: {bot_id}")
                
                # Send start message to bot
                self._send_message_to_bot(bot_id, {
                    'type': 'command',
                    'action': 'start',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error starting bot {bot_id}: {str(e)}")
                self.bots[bot_id].status = 'error'

    def stop_bot(self, bot_id: str):
        """Stop a trading bot"""
        if bot_id in self.bots:
            try:
                bot = self.bots[bot_id]
                bot.status = 'stopped'
                bot.last_update = datetime.now()
                bot.trading_system.stop()
                logger.info(f"Stopped bot: {bot_id}")
                
            except Exception as e:
                logger.error(f"Error stopping bot {bot_id}: {str(e)}")

    def send_message_to_bot(self, bot_id: str, message: Dict):
        """Send a message to a specific bot"""
        if bot_id in self.bots:
            self._send_message_to_bot(bot_id, message)
            logger.info(f"Sent message to bot {bot_id}: {message}")

    def broadcast_message(self, message: Dict):
        """Send a message to all bots"""
        for bot_id in self.bots:
            self._send_message_to_bot(bot_id, message)
        logger.info(f"Broadcast message to all bots: {message}")

    def get_bot_status(self, bot_id: str) -> Optional[Dict]:
        """Get status of a specific bot"""
        if bot_id in self.bots:
            bot = self.bots[bot_id]
            return {
                'bot_id': bot.bot_id,
                'status': bot.status,
                'last_update': bot.last_update.isoformat(),
                'performance_metrics': bot.performance_metrics,
                'trading_system_status': bot.trading_system.get_status()
            }
        return None

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'total_bots': len(self.bots),
            'active_bots': len([b for b in self.bots.values() if b.status == 'running']),
            'code_health': self.code_guardian.get_status_report(),
            'system_health': self.system_monitor.get_system_status(),
            'bots': {bot_id: self.get_bot_status(bot_id) for bot_id in self.bots}
        }

    def _send_message_to_bot(self, bot_id: str, message: Dict):
        """Internal method to send message to bot"""
        if bot_id in self.bots:
            self.bots[bot_id].message_queue.put(message)

    def _monitor_bots(self):
        """Continuously monitor bot performance and health"""
        while self.running:
            try:
                for bot_id, bot in self.bots.items():
                    if bot.status == 'running':
                        # Update bot metrics
                        status = bot.trading_system.get_status()
                        bot.performance_metrics = status['metrics']
                        bot.last_update = datetime.now()

                        # Check for issues
                        if status['system_health']['alerts']['unacknowledged'] > 0:
                            self._handle_bot_issue(bot_id, status)

                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in bot monitoring: {str(e)}")
                time.sleep(300)

    def _handle_messages(self):
        """Handle incoming messages from bots"""
        while self.running:
            try:
                for bot_id, bot in self.bots.items():
                    while not bot.message_queue.empty():
                        message = bot.message_queue.get_nowait()
                        self._process_bot_message(bot_id, message)
                        
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error handling messages: {str(e)}")
                time.sleep(60)

    def _continuous_code_repair(self):
        """Continuously monitor and repair code issues"""
        while self.running:
            try:
                # Get code issues
                status = self.code_guardian.get_status_report()
                
                # Attempt to fix issues
                for issue in self.code_guardian.issues:
                    if not issue.fixed and issue.suggested_fix:
                        if self.code_guardian.fix_issue(issue):
                            logger.info(f"Fixed code issue: {issue.description}")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in code repair: {str(e)}")
                time.sleep(600)

    def _handle_bot_issue(self, bot_id: str, status: Dict):
        """Handle issues with a specific bot"""
        try:
            # Create alert
            alert = {
                'type': 'alert',
                'bot_id': bot_id,
                'timestamp': datetime.now().isoformat(),
                'status': status
            }
            
            # Log alert
            logger.warning(f"Bot issue detected: {alert}")
            
            # Add to message queue
            self.message_queue.put(alert)
            
            # Take action based on issue severity
            if status['system_health']['alerts']['by_level'].get('error', 0) > 0:
                self.stop_bot(bot_id)
                logger.error(f"Stopped bot {bot_id} due to critical issues")
                
        except Exception as e:
            logger.error(f"Error handling bot issue: {str(e)}")

    def _process_bot_message(self, bot_id: str, message: Dict):
        """Process message from a bot"""
        try:
            if message['type'] == 'status_update':
                self.bots[bot_id].status = message['status']
                self.bots[bot_id].last_update = datetime.now()
                
            elif message['type'] == 'alert':
                self._handle_bot_issue(bot_id, message)
                
            elif message['type'] == 'performance_update':
                self.bots[bot_id].performance_metrics.update(message['metrics'])
                
        except Exception as e:
            logger.error(f"Error processing bot message: {str(e)}")

    def _run_dashboard(self):
        """Run the dashboard"""
        try:
            self.dashboard.run_server(debug=True, port=8050)
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")

    def stop(self):
        """Stop the AI Agent Manager"""
        self.running = False
        
        # Stop all bots
        for bot_id in self.bots:
            self.stop_bot(bot_id)
        
        # Stop components
        self.code_guardian.stop()
        self.system_monitor.stop()
        
        # Wait for threads
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join()

def create_agent_manager(project_root: str) -> AIAgentManager:
    """Create and initialize an AI Agent Manager"""
    return AIAgentManager(project_root)

if __name__ == "__main__":
    # Example usage
    manager = create_agent_manager(".")
    
    try:
        # Create some test bots
        config1 = {
            "trading": {"initial_capital": 100000},
            "strategy": {"name": "trend_following"},
            "risk_management": {"max_positions": 5}
        }
        
        config2 = {
            "trading": {"initial_capital": 50000},
            "strategy": {"name": "mean_reversion"},
            "risk_management": {"max_positions": 3}
        }
        
        # Create and start bots
        bot1_id = manager.create_bot(config1)
        bot2_id = manager.create_bot(config2)
        
        manager.start_bot(bot1_id)
        manager.start_bot(bot2_id)
        
        # Monitor for a while
        time.sleep(3600)  # Run for 1 hour
        
    finally:
        manager.stop() 