"""
Utility functions for AI Text Detection project.
Includes logging, Git automation, and project management utilities.
"""

import os
import logging
import json
import subprocess
import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys


class GitManager:
    """Git workflow automation manager."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
    
    def run_git_command(self, command: List[str]) -> Dict[str, Any]:
        """Run a git command and return the result."""
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return {
                'success': True,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'returncode': result.returncode
            }
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'stdout': e.stdout.strip() if e.stdout else '',
                'stderr': e.stderr.strip() if e.stderr else '',
                'returncode': e.returncode,
                'error': str(e)
            }
    
    def get_current_branch(self) -> str:
        """Get the current Git branch."""
        result = self.run_git_command(['branch', '--show-current'])
        return result['stdout'] if result['success'] else 'unknown'
    
    def create_feature_branch(self, feature_name: str) -> bool:
        """Create and switch to a new feature branch."""
        branch_name = f"feature/{feature_name}"
        
        # Check if branch already exists
        result = self.run_git_command(['branch', '--list', branch_name])
        if result['success'] and branch_name in result['stdout']:
            self.logger.info(f"Branch {branch_name} already exists, switching to it")
            switch_result = self.run_git_command(['checkout', branch_name])
            return switch_result['success']
        
        # Create and switch to new branch
        result = self.run_git_command(['checkout', '-b', branch_name])
        if result['success']:
            self.logger.info(f"Created and switched to branch: {branch_name}")
            return True
        else:
            self.logger.error(f"Failed to create branch {branch_name}: {result['stderr']}")
            return False
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Commit changes with a descriptive message."""
        # Add files
        if files:
            for file in files:
                add_result = self.run_git_command(['add', file])
                if not add_result['success']:
                    self.logger.error(f"Failed to add file {file}: {add_result['stderr']}")
                    return False
        else:
            # Add all changes
            add_result = self.run_git_command(['add', '.'])
            if not add_result['success']:
                self.logger.error(f"Failed to add changes: {add_result['stderr']}")
                return False
        
        # Commit changes
        commit_message = f"{message}\n\n🤖 Generated with Claude Code\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        result = self.run_git_command(['commit', '-m', commit_message])
        
        if result['success']:
            self.logger.info(f"Successfully committed changes: {message}")
            return True
        else:
            self.logger.error(f"Failed to commit changes: {result['stderr']}")
            return False
    
    def auto_commit_feature(self, feature_name: str, description: str, files: Optional[List[str]] = None) -> bool:
        """Automatically create feature branch and commit changes."""
        # Create feature branch
        if not self.create_feature_branch(feature_name):
            return False
        
        # Commit changes
        commit_msg = f"feat: {description}"
        return self.commit_changes(commit_msg, files)
    
    def merge_to_main(self, feature_branch: str, delete_branch: bool = True) -> bool:
        """Merge feature branch to main and optionally delete the feature branch."""
        # Switch to main
        checkout_result = self.run_git_command(['checkout', 'master'])
        if not checkout_result['success']:
            self.logger.error(f"Failed to checkout master: {checkout_result['stderr']}")
            return False
        
        # Merge feature branch
        merge_result = self.run_git_command(['merge', feature_branch])
        if not merge_result['success']:
            self.logger.error(f"Failed to merge {feature_branch}: {merge_result['stderr']}")
            return False
        
        # Delete feature branch if requested
        if delete_branch:
            delete_result = self.run_git_command(['branch', '-d', feature_branch])
            if delete_result['success']:
                self.logger.info(f"Deleted feature branch: {feature_branch}")
            else:
                self.logger.warning(f"Failed to delete feature branch {feature_branch}: {delete_result['stderr']}")
        
        self.logger.info(f"Successfully merged {feature_branch} to master")
        return True
    
    def get_git_status(self) -> Dict[str, Any]:
        """Get detailed Git status information."""
        status_result = self.run_git_command(['status', '--porcelain'])
        log_result = self.run_git_command(['log', '--oneline', '-5'])
        
        return {
            'current_branch': self.get_current_branch(),
            'status': status_result['stdout'] if status_result['success'] else '',
            'recent_commits': log_result['stdout'] if log_result['success'] else '',
            'has_changes': bool(status_result['stdout']) if status_result['success'] else False
        }


class ProjectLogger:
    """Enhanced logging system for the project."""
    
    def __init__(self, log_level: str = 'INFO', log_file: Optional[str] = None):
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            log_path = f"logs/{self.log_file}"
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Setup specific loggers
        self.setup_module_loggers()
    
    def setup_module_loggers(self):
        """Setup loggers for specific modules."""
        modules = ['src.data_processor', 'src.model_trainer', 'src.predictor', 'src.evaluator']
        
        for module in modules:
            logger = logging.getLogger(module)
            logger.setLevel(self.log_level)
    
    def log_system_info(self):
        """Log system and environment information."""
        import torch
        import platform
        
        logger = logging.getLogger(__name__)
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")


class ProjectManager:
    """Main project management class that coordinates Git and logging."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or {}
        self.git_manager = GitManager()
        self.logger_manager = ProjectLogger(
            log_level=self.config.get('log_level', 'INFO'),
            log_file=self.config.get('log_file', 'project.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Log system info on initialization
        self.logger_manager.log_system_info()
    
    def start_feature(self, feature_name: str, description: str) -> bool:
        """Start working on a new feature."""
        self.logger.info(f"Starting feature: {feature_name}")
        self.logger.info(f"Description: {description}")
        
        # Log current status
        git_status = self.git_manager.get_git_status()
        self.logger.info(f"Current branch: {git_status['current_branch']}")
        
        # Create feature branch
        success = self.git_manager.create_feature_branch(feature_name)
        
        if success:
            self.logger.info(f"Successfully started feature: {feature_name}")
            
            # Log feature start
            self.log_project_event({
                'event': 'feature_started',
                'feature_name': feature_name,
                'description': description,
                'timestamp': datetime.datetime.now().isoformat(),
                'branch': f"feature/{feature_name}"
            })
        
        return success
    
    def complete_feature(self, feature_name: str, files: Optional[List[str]] = None) -> bool:
        """Complete a feature and commit changes."""
        self.logger.info(f"Completing feature: {feature_name}")
        
        # Auto-commit changes
        success = self.git_manager.auto_commit_feature(
            feature_name, 
            f"Complete {feature_name} implementation", 
            files
        )
        
        if success:
            self.logger.info(f"Successfully completed feature: {feature_name}")
            
            # Log feature completion
            self.log_project_event({
                'event': 'feature_completed',
                'feature_name': feature_name,
                'timestamp': datetime.datetime.now().isoformat(),
                'commit_hash': self.get_latest_commit_hash()
            })
        
        return success
    
    def deploy_to_main(self, feature_name: str) -> bool:
        """Deploy completed feature to main branch."""
        feature_branch = f"feature/{feature_name}"
        
        self.logger.info(f"Deploying {feature_branch} to main")
        
        success = self.git_manager.merge_to_main(feature_branch, delete_branch=True)
        
        if success:
            self.logger.info(f"Successfully deployed {feature_name} to main")
            
            # Log deployment
            self.log_project_event({
                'event': 'feature_deployed',
                'feature_name': feature_name,
                'timestamp': datetime.datetime.now().isoformat(),
                'target_branch': 'master'
            })
        
        return success
    
    def get_latest_commit_hash(self) -> str:
        """Get the latest commit hash."""
        result = self.git_manager.run_git_command(['rev-parse', 'HEAD'])
        return result['stdout'][:8] if result['success'] else 'unknown'
    
    def log_project_event(self, event_data: Dict[str, Any]) -> None:
        """Log project events to a structured log file."""
        log_file = 'logs/project_events.jsonl'
        os.makedirs('logs', exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get comprehensive project status."""
        git_status = self.git_manager.get_git_status()
        
        status = {
            'timestamp': datetime.datetime.now().isoformat(),
            'git': git_status,
            'latest_commit': self.get_latest_commit_hash(),
        }
        
        # Add file system status
        src_files = list(Path('src').glob('*.py')) if Path('src').exists() else []
        status['src_files'] = [str(f) for f in src_files]
        status['src_file_count'] = len(src_files)
        
        return status
    
    def auto_workflow(self, action: str, description: str, files: Optional[List[str]] = None) -> bool:
        """Automated workflow for common actions."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_name = f"{action}_{timestamp}"
        
        # Start feature
        if not self.start_feature(feature_name, description):
            return False
        
        # Complete feature
        if not self.complete_feature(feature_name, files):
            return False
        
        # Deploy to main
        return self.deploy_to_main(feature_name)


def setup_project_management(config: Optional[Dict[str, Any]] = None) -> ProjectManager:
    """Setup project management with Git automation and logging."""
    return ProjectManager(config)


def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of a file for change detection."""
    import hashlib
    
    if not os.path.exists(filepath):
        return ""
    
    with open(filepath, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()


def monitor_file_changes(files: List[str]) -> Dict[str, str]:
    """Monitor file changes and return hash map."""
    return {file: get_file_hash(file) for file in files}


def detect_changes(old_hashes: Dict[str, str], new_hashes: Dict[str, str]) -> List[str]:
    """Detect which files have changed."""
    changed_files = []
    
    for file, new_hash in new_hashes.items():
        old_hash = old_hashes.get(file, "")
        if old_hash != new_hash:
            changed_files.append(file)
    
    return changed_files