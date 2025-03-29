import inspect
import logging
import sys
import traceback
import ast
import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import importlib
import threading
import time
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class CodeIssue:
    """Represents a code issue found by the guardian"""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str
    suggested_fix: Optional[str] = None
    fixed: bool = False

class CodeGuardian:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.issues: List[CodeIssue] = []
        self.module_states: Dict[str, float] = {}  # Module path -> last modified timestamp
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_patterns: Dict[str, str] = self._load_error_patterns()
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._continuous_monitoring)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _load_error_patterns(self) -> Dict[str, str]:
        """Load known error patterns and their fixes"""
        return {
            "IndexError": "Check array bounds and data availability",
            "KeyError": "Verify dictionary keys exist before access",
            "AttributeError": "Ensure object attributes exist",
            "TypeError": "Check type compatibility",
            "ValueError": "Validate input values",
            "ZeroDivisionError": "Add zero-value checking",
            "ImportError": "Verify module installation and import path",
            "MemoryError": "Optimize memory usage or add cleanup",
            "RuntimeError": "Check execution flow and state",
        }
    
    def analyze_code(self, file_path: str) -> List[CodeIssue]:
        """Analyze code for potential issues"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            issues = []
            tree = ast.parse(code)
            
            # Check for common issues
            issues.extend(self._check_error_handling(tree, file_path))
            issues.extend(self._check_resource_management(tree, file_path))
            issues.extend(self._check_performance_patterns(tree, file_path))
            issues.extend(self._check_code_style(tree, file_path))
            
            return issues
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            return []
    
    def _check_error_handling(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Check for proper error handling"""
        issues = []
        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=handler.lineno,
                            issue_type="error_handling",
                            description="Bare except clause found",
                            severity="warning",
                            suggested_fix="Specify exception type(s) to catch"
                        ))
        return issues
    
    def _check_resource_management(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Check for proper resource management"""
        issues = []
        for node in ast.walk(tree):
            # Check for file operations without context manager
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ['open']:
                    if not isinstance(node.parent, ast.withitem):
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            issue_type="resource_management",
                            description="File opened without context manager",
                            severity="warning",
                            suggested_fix="Use 'with' statement for file operations"
                        ))
        return issues
    
    def _check_performance_patterns(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Check for performance-related issues"""
        issues = []
        for node in ast.walk(tree):
            # Check for inefficient list operations
            if isinstance(node, ast.For):
                if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Call):
                            if isinstance(node.iter.args[0].func, ast.Name) and node.iter.args[0].func.id == 'len':
                                issues.append(CodeIssue(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    issue_type="performance",
                                    description="Inefficient list iteration",
                                    severity="info",
                                    suggested_fix="Use 'for item in items' instead of range(len(items))"
                                ))
        return issues
    
    def _check_code_style(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Check for code style issues"""
        issues = []
        for node in ast.walk(tree):
            # Check for overly complex functions
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:  # Arbitrary threshold
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="code_style",
                        description="Function is too long",
                        severity="info",
                        suggested_fix="Consider breaking down into smaller functions"
                    ))
        return issues
    
    def monitor_performance(self, module_name: str, execution_time: float):
        """Monitor module performance"""
        if module_name not in self.performance_metrics:
            self.performance_metrics[module_name] = []
        
        self.performance_metrics[module_name].append(execution_time)
        
        # Check for performance degradation
        if len(self.performance_metrics[module_name]) > 10:
            recent_avg = np.mean(self.performance_metrics[module_name][-10:])
            overall_avg = np.mean(self.performance_metrics[module_name])
            
            if recent_avg > overall_avg * 1.5:  # 50% slower than average
                self.issues.append(CodeIssue(
                    file_path=module_name,
                    line_number=0,
                    issue_type="performance_degradation",
                    description=f"Performance degradation detected in {module_name}",
                    severity="warning"
                ))
    
    def fix_issue(self, issue: CodeIssue) -> bool:
        """Attempt to fix a code issue"""
        try:
            if not issue.suggested_fix:
                return False
            
            with open(issue.file_path, 'r') as f:
                lines = f.readlines()
            
            if issue.issue_type == "error_handling":
                # Fix bare except
                if "Bare except clause found" in issue.description:
                    lines[issue.line_number - 1] = lines[issue.line_number - 1].replace(
                        "except:", "except Exception:"
                    )
            
            elif issue.issue_type == "resource_management":
                # Fix file operations
                if "File opened without context manager" in issue.description:
                    indent = len(lines[issue.line_number - 1]) - len(lines[issue.line_number - 1].lstrip())
                    lines[issue.line_number - 1] = " " * indent + "with " + lines[issue.line_number - 1].lstrip()
            
            with open(issue.file_path, 'w') as f:
                f.writelines(lines)
            
            issue.fixed = True
            return True
            
        except Exception as e:
            logger.error(f"Error fixing issue: {str(e)}")
            return False
    
    def _continuous_monitoring(self):
        """Continuously monitor code for issues"""
        while self.running:
            try:
                # Check all Python files in project
                for root, _, files in os.walk(self.project_root):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            
                            # Check if file was modified
                            current_mtime = os.path.getmtime(file_path)
                            if file_path not in self.module_states or current_mtime > self.module_states[file_path]:
                                self.module_states[file_path] = current_mtime
                                
                                # Analyze code
                                new_issues = self.analyze_code(file_path)
                                self.issues.extend(new_issues)
                                
                                # Try to fix issues
                                for issue in new_issues:
                                    if issue.suggested_fix:
                                        self.fix_issue(issue)
                
                # Sleep for a while
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {str(e)}")
                time.sleep(300)  # Wait longer if there's an error
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate a status report"""
        return {
            'total_issues': len(self.issues),
            'fixed_issues': len([i for i in self.issues if i.fixed]),
            'active_issues': len([i for i in self.issues if not i.fixed]),
            'issues_by_type': self._count_issues_by_type(),
            'performance_metrics': self._summarize_performance_metrics()
        }
    
    def _count_issues_by_type(self) -> Dict[str, int]:
        """Count issues by type"""
        counts = {}
        for issue in self.issues:
            if issue.issue_type not in counts:
                counts[issue.issue_type] = 0
            counts[issue.issue_type] += 1
        return counts
    
    def _summarize_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Summarize performance metrics"""
        summary = {}
        for module, metrics in self.performance_metrics.items():
            if metrics:
                summary[module] = {
                    'avg': np.mean(metrics),
                    'min': np.min(metrics),
                    'max': np.max(metrics),
                    'std': np.std(metrics)
                }
        return summary
    
    def stop(self):
        """Stop the guardian"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

def create_guardian(project_root: str) -> CodeGuardian:
    """Create and start a code guardian instance"""
    return CodeGuardian(project_root)

if __name__ == "__main__":
    # Example usage
    guardian = create_guardian(".")
    try:
        while True:
            status = guardian.get_status_report()
            print(json.dumps(status, indent=2))
            time.sleep(300)  # Print status every 5 minutes
    except KeyboardInterrupt:
        guardian.stop() 