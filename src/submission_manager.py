"""
Submission Management Module
Handles versioning, tracking, and analysis of submission files.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import logging
import subprocess


class SubmissionManager:
    """Manages submission files with versioning and metadata tracking."""
    
    def __init__(self, submission_dir: str = "submissions"):
        self.submission_dir = Path(submission_dir)
        self.submission_dir.mkdir(exist_ok=True)
        self.metadata_file = self.submission_dir / "submissions_metadata.json"
        self.logger = logging.getLogger(__name__)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load submission metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")
        
        return {"submissions": []}
    
    def _save_metadata(self) -> None:
        """Save submission metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git information."""
        git_info = {}
        
        try:
            # Get commit hash
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True)
            git_info['commit_hash'] = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Get branch name
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True)
            git_info['branch'] = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Get commit message
            result = subprocess.run(['git', 'log', '-1', '--pretty=%B'], 
                                  capture_output=True, text=True)
            git_info['commit_message'] = result.stdout.strip() if result.returncode == 0 else "unknown"
            
        except Exception as e:
            self.logger.warning(f"Failed to get git info: {e}")
            git_info = {'commit_hash': 'unknown', 'branch': 'unknown', 'commit_message': 'unknown'}
        
        return git_info
    
    def save_submission(self, submission_df: pd.DataFrame, 
                       config: Dict[str, Any] = None,
                       metrics: Dict[str, Any] = None,
                       description: str = "") -> str:
        """Save submission with full metadata tracking."""
        
        # Generate filename with timestamp and git hash
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        git_info = self._get_git_info()
        filename = f"submission_{timestamp}_{git_info['commit_hash']}.csv"
        
        # Save submission file
        submission_path = self.submission_dir / filename
        submission_df.to_csv(submission_path, index=False)
        
        # Calculate submission statistics
        stats = {
            'mean_prediction': float(submission_df['generated'].mean()),
            'std_prediction': float(submission_df['generated'].std()),
            'min_prediction': float(submission_df['generated'].min()),
            'max_prediction': float(submission_df['generated'].max()),
            'predictions_above_0.5': int((submission_df['generated'] > 0.5).sum()),
            'predictions_below_0.5': int((submission_df['generated'] <= 0.5).sum()),
            'total_samples': len(submission_df)
        }
        
        # Create metadata entry
        submission_metadata = {
            'filename': filename,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            'description': description,
            'git_info': git_info,
            'submission_stats': stats,
            'config': config or {},
            'metrics': metrics or {},
            'file_size_mb': os.path.getsize(submission_path) / 1024 / 1024
        }
        
        # Add to metadata
        self.metadata['submissions'].append(submission_metadata)
        self._save_metadata()
        
        self.logger.info(f"Submission saved: {submission_path}")
        self.logger.info(f"Mean prediction: {stats['mean_prediction']:.4f}")
        self.logger.info(f"Predictions > 0.5: {stats['predictions_above_0.5']}")
        
        return str(submission_path)
    
    def list_submissions(self) -> List[Dict[str, Any]]:
        """List all submissions with metadata."""
        return sorted(self.metadata['submissions'], key=lambda x: x['timestamp'], reverse=True)
    
    def get_submission_summary(self) -> Dict[str, Any]:
        """Get summary of all submissions."""
        submissions = self.list_submissions()
        
        if not submissions:
            return {"total_submissions": 0}
        
        # Calculate summary statistics
        total = len(submissions)
        latest = submissions[0]
        
        mean_preds = [s['submission_stats']['mean_prediction'] for s in submissions]
        
        summary = {
            'total_submissions': total,
            'latest_submission': {
                'filename': latest['filename'],
                'datetime': latest['datetime'],
                'description': latest['description'],
                'mean_prediction': latest['submission_stats']['mean_prediction']
            },
            'mean_prediction_stats': {
                'overall_mean': sum(mean_preds) / len(mean_preds),
                'min': min(mean_preds),
                'max': max(mean_preds),
                'std': pd.Series(mean_preds).std()
            }
        }
        
        return summary
    
    def compare_submissions(self, filename1: str, filename2: str) -> Dict[str, Any]:
        """Compare two submissions."""
        
        # Load submissions
        path1 = self.submission_dir / filename1
        path2 = self.submission_dir / filename2
        
        if not path1.exists() or not path2.exists():
            raise FileNotFoundError("One or both submission files not found")
        
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        
        # Calculate differences
        diff_df = df1.copy()
        diff_df['prediction_diff'] = df2['generated'] - df1['generated']
        diff_df['abs_diff'] = diff_df['prediction_diff'].abs()
        
        comparison = {
            'file1': filename1,
            'file2': filename2,
            'mean_absolute_difference': float(diff_df['abs_diff'].mean()),
            'max_absolute_difference': float(diff_df['abs_diff'].max()),
            'correlation': float(df1['generated'].corr(df2['generated'])),
            'samples_with_large_diff': int((diff_df['abs_diff'] > 0.1).sum()),
            'agreement_rate': float((diff_df['abs_diff'] < 0.05).mean())
        }
        
        return comparison
    
    def get_best_submission(self, metric: str = 'mean_prediction') -> Optional[Dict[str, Any]]:
        """Get the best submission based on a metric."""
        submissions = self.list_submissions()
        
        if not submissions:
            return None
        
        # Find submission with highest/lowest metric value
        if metric in ['mean_prediction']:
            # Higher is better for some metrics
            best = max(submissions, key=lambda x: x['submission_stats'].get(metric, 0))
        else:
            best = submissions[0]  # Default to latest
        
        return best
    
    def cleanup_old_submissions(self, keep_latest: int = 10) -> None:
        """Remove old submission files, keeping only the latest N."""
        submissions = self.list_submissions()
        
        if len(submissions) <= keep_latest:
            self.logger.info(f"Only {len(submissions)} submissions, no cleanup needed")
            return
        
        # Remove old files
        to_remove = submissions[keep_latest:]
        
        for submission in to_remove:
            file_path = self.submission_dir / submission['filename']
            if file_path.exists():
                os.remove(file_path)
                self.logger.info(f"Removed old submission: {submission['filename']}")
        
        # Update metadata
        self.metadata['submissions'] = submissions[:keep_latest]
        self._save_metadata()
        
        self.logger.info(f"Cleanup completed. Kept {keep_latest} latest submissions.")
    
    def export_submission_report(self, output_file: str = "submission_report.json") -> None:
        """Export comprehensive submission report."""
        
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'summary': self.get_submission_summary(),
            'all_submissions': self.list_submissions()
        }
        
        output_path = self.submission_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Submission report exported to: {output_path}")


def create_submission_manager(submission_dir: str = "submissions") -> SubmissionManager:
    """Factory function to create submission manager."""
    return SubmissionManager(submission_dir)