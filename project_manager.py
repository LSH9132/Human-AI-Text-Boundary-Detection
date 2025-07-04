#!/usr/bin/env python3
"""
Project Manager Script - Automated Git workflow and logging
Usage: python project_manager.py [action] [description]
Actions: start, complete, deploy, status, auto
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import setup_project_management


def main():
    parser = argparse.ArgumentParser(description="AI Text Detection Project Manager")
    parser.add_argument("action", choices=["start", "complete", "deploy", "status", "auto"],
                       help="Action to perform")
    parser.add_argument("description", nargs="?", default="",
                       help="Description of the action")
    parser.add_argument("--feature", "-f", default="",
                       help="Feature name (for start/complete/deploy)")
    parser.add_argument("--files", nargs="*", default=None,
                       help="Specific files to commit")
    
    args = parser.parse_args()
    
    # Setup project manager
    pm = setup_project_management({
        'log_level': 'INFO',
        'log_file': 'project_manager.log'
    })
    
    if args.action == "start":
        if not args.feature or not args.description:
            print("Error: start action requires --feature and description")
            return 1
        
        success = pm.start_feature(args.feature, args.description)
        print(f"{'✅' if success else '❌'} Feature start: {args.feature}")
        return 0 if success else 1
    
    elif args.action == "complete":
        if not args.feature:
            print("Error: complete action requires --feature")
            return 1
        
        success = pm.complete_feature(args.feature, args.files)
        print(f"{'✅' if success else '❌'} Feature complete: {args.feature}")
        return 0 if success else 1
    
    elif args.action == "deploy":
        if not args.feature:
            print("Error: deploy action requires --feature")
            return 1
        
        success = pm.deploy_to_main(args.feature)
        print(f"{'✅' if success else '❌'} Feature deploy: {args.feature}")
        return 0 if success else 1
    
    elif args.action == "status":
        status = pm.get_project_status()
        print("📊 Project Status:")
        print(f"   Current branch: {status['git']['current_branch']}")
        print(f"   Latest commit: {status['latest_commit']}")
        print(f"   Source files: {status['src_file_count']}")
        print(f"   Has changes: {status['git']['has_changes']}")
        return 0
    
    elif args.action == "auto":
        if not args.description:
            print("Error: auto action requires description")
            return 1
        
        action_name = args.description.lower().replace(' ', '_')
        success = pm.auto_workflow(action_name, args.description, args.files)
        print(f"{'✅' if success else '❌'} Auto workflow: {args.description}")
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())