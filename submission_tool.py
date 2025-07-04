#!/usr/bin/env python3
"""
Submission Management Tool
CLI tool for managing and analyzing submission files.

Usage:
    python submission_tool.py list                    # List all submissions
    python submission_tool.py summary                 # Show summary
    python submission_tool.py compare file1 file2     # Compare two submissions
    python submission_tool.py cleanup --keep 5        # Keep only 5 latest
    python submission_tool.py report                  # Export detailed report
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.submission_manager import create_submission_manager


def main():
    parser = argparse.ArgumentParser(description="Submission Management Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all submissions')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show submission summary')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two submissions')
    compare_parser.add_argument('file1', help='First submission file')
    compare_parser.add_argument('file2', help='Second submission file')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old submissions')
    cleanup_parser.add_argument('--keep', type=int, default=10, help='Number of submissions to keep')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Export detailed report')
    report_parser.add_argument('--output', default='submission_report.json', help='Output file name')
    
    # Best command
    best_parser = subparsers.add_parser('best', help='Show best submission')
    best_parser.add_argument('--metric', default='mean_prediction', help='Metric to use for "best"')
    
    args = parser.parse_args()
    
    # Create submission manager
    sm = create_submission_manager()
    
    if args.command == 'list':
        submissions = sm.list_submissions()
        
        if not submissions:
            print("No submissions found.")
            return
        
        print(f"📋 Found {len(submissions)} submissions:\n")
        print(f"{'Filename':<40} {'Date':<20} {'Mean Pred':<10} {'Description'}")
        print("=" * 90)
        
        for sub in submissions:
            stats = sub['submission_stats']
            date = sub['datetime'][:19].replace('T', ' ')
            print(f"{sub['filename']:<40} {date:<20} {stats['mean_prediction']:<10.4f} {sub['description']}")
    
    elif args.command == 'summary':
        summary = sm.get_submission_summary()
        
        if summary['total_submissions'] == 0:
            print("No submissions found.")
            return
        
        print("📊 Submission Summary:")
        print(f"   Total submissions: {summary['total_submissions']}")
        print(f"   Latest: {summary['latest_submission']['filename']}")
        print(f"   Latest date: {summary['latest_submission']['datetime'][:19]}")
        print(f"   Latest mean prediction: {summary['latest_submission']['mean_prediction']:.4f}")
        print(f"   Overall mean prediction: {summary['mean_prediction_stats']['overall_mean']:.4f}")
        print(f"   Min/Max mean prediction: {summary['mean_prediction_stats']['min']:.4f} / {summary['mean_prediction_stats']['max']:.4f}")
    
    elif args.command == 'compare':
        try:
            comparison = sm.compare_submissions(args.file1, args.file2)
            
            print(f"🔍 Comparing submissions:")
            print(f"   File 1: {comparison['file1']}")
            print(f"   File 2: {comparison['file2']}")
            print(f"   Mean absolute difference: {comparison['mean_absolute_difference']:.4f}")
            print(f"   Max absolute difference: {comparison['max_absolute_difference']:.4f}")
            print(f"   Correlation: {comparison['correlation']:.4f}")
            print(f"   Samples with large diff (>0.1): {comparison['samples_with_large_diff']}")
            print(f"   Agreement rate (<0.05 diff): {comparison['agreement_rate']:.2%}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
    
    elif args.command == 'cleanup':
        print(f"🧹 Cleaning up submissions, keeping latest {args.keep}...")
        sm.cleanup_old_submissions(keep_latest=args.keep)
        print("✅ Cleanup completed.")
    
    elif args.command == 'report':
        print(f"📄 Exporting detailed report to {args.output}...")
        sm.export_submission_report(args.output)
        print("✅ Report exported.")
    
    elif args.command == 'best':
        best = sm.get_best_submission(args.metric)
        
        if not best:
            print("No submissions found.")
            return
        
        print(f"🏆 Best submission (by {args.metric}):")
        print(f"   Filename: {best['filename']}")
        print(f"   Date: {best['datetime'][:19]}")
        print(f"   Description: {best['description']}")
        print(f"   Mean prediction: {best['submission_stats']['mean_prediction']:.4f}")
        print(f"   Git commit: {best['git_info']['commit_hash']}")
        
        if 'metrics' in best and 'oof_auc' in best['metrics']:
            print(f"   OOF AUC: {best['metrics']['oof_auc']:.4f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()