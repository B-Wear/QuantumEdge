import logging
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import functools
import traceback
from .code_guardian import CodeGuardian, create_guardian

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, project_root: str):
        self.guardian = create_guardian(project_root)
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.performance_data: Dict[str, Dict[str, float]] = {}
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def monitor_component(self, component_name: str):
        """Decorator to monitor component performance and errors"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Update performance metrics
                    self.guardian.monitor_performance(component_name, execution_time)
                    
                    # Update component status
                    self.component_status[component_name] = {
                        'status': 'healthy',
                        'last_execution': datetime.now().isoformat(),
                        'execution_time': execution_time,
                        'error': None
                    }
                    
                    return result
                    
                except Exception as e:
                    error_info = {
                        'type': type(e).__name__,
                        'message': str(e),
                        'traceback': traceback.format_exc()
                    }
                    
                    # Update component status
                    self.component_status[component_name] = {
                        'status': 'error',
                        'last_execution': datetime.now().isoformat(),
                        'execution_time': time.time() - start_time,
                        'error': error_info
                    }
                    
                    # Create alert
                    self._create_alert(
                        component_name,
                        f"Error in {component_name}: {str(e)}",
                        'error',
                        error_info
                    )
                    
                    logger.error(f"Error in {component_name}: {str(e)}")
                    raise
            
            return wrapper
        return decorator
    
    def _create_alert(self, 
                     component: str, 
                     message: str, 
                     level: str, 
                     details: Optional[Dict] = None):
        """Create a new alert"""
        alert_id = f"{component}_{datetime.now().timestamp()}"
        self.alerts[alert_id] = {
            'component': component,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'acknowledged': False
        }
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id]['acknowledged'] = True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'components': self.component_status,
            'alerts': {
                'total': len(self.alerts),
                'unacknowledged': len([a for a in self.alerts.values() if not a['acknowledged']]),
                'by_level': self._count_alerts_by_level()
            },
            'code_health': self.guardian.get_status_report(),
            'performance': self._get_performance_summary()
        }
    
    def _count_alerts_by_level(self) -> Dict[str, int]:
        """Count alerts by level"""
        counts = {'error': 0, 'warning': 0, 'info': 0}
        for alert in self.alerts.values():
            if alert['level'] in counts:
                counts[alert['level']] += 1
        return counts
    
    def _get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all components"""
        summary = {}
        for component, metrics in self.guardian.performance_metrics.items():
            if metrics:
                summary[component] = {
                    'avg_execution_time': sum(metrics) / len(metrics),
                    'max_execution_time': max(metrics),
                    'min_execution_time': min(metrics),
                    'total_executions': len(metrics)
                }
        return summary
    
    def _monitor_system(self):
        """Continuous system monitoring"""
        while self.running:
            try:
                # Check component health
                for component, status in self.component_status.items():
                    if status['status'] == 'error':
                        self._create_alert(
                            component,
                            f"Component {component} is in error state",
                            'error',
                            status['error']
                        )
                    
                    # Check for performance degradation
                    if component in self.guardian.performance_metrics:
                        metrics = self.guardian.performance_metrics[component]
                        if len(metrics) > 10:
                            recent_avg = sum(metrics[-10:]) / 10
                            overall_avg = sum(metrics) / len(metrics)
                            
                            if recent_avg > overall_avg * 1.5:  # 50% slower
                                self._create_alert(
                                    component,
                                    f"Performance degradation detected in {component}",
                                    'warning',
                                    {
                                        'recent_avg': recent_avg,
                                        'overall_avg': overall_avg
                                    }
                                )
                
                # Sleep for a while
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(300)  # Wait longer if there's an error
    
    def stop(self):
        """Stop the system monitor"""
        self.running = False
        self.guardian.stop()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

# Example usage
def example_usage():
    # Create system monitor
    monitor = SystemMonitor(".")
    
    # Example component with monitoring
    @monitor.monitor_component("example_component")
    def example_function(x: int) -> int:
        time.sleep(1)  # Simulate work
        return x * 2
    
    try:
        # Run component
        for i in range(5):
            result = example_function(i)
            print(f"Result: {result}")
        
        # Get system status
        status = monitor.get_system_status()
        print("\nSystem Status:")
        print(json.dumps(status, indent=2))
        
    finally:
        monitor.stop()

if __name__ == "__main__":
    example_usage() 