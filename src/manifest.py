"""
Reproducibility Manifest Generator
===================================
Auto-generate manifest.json for training runs with:
- run_id (timestamp + process ID)
- arguments, git commit, data hash
- CV scheme, validation window, seed
- start/end timestamps, exit code
===================================
"""
import json
import hashlib
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import argparse


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return 'unknown'


def get_data_hash(file_paths: List[str]) -> str:
    """
    Compute SHA256 hash of data files
    
    Args:
        file_paths: List of data file paths to hash
    
    Returns:
        SHA256 hash string
    """
    hasher = hashlib.sha256()
    
    for fpath in file_paths:
        if not Path(fpath).exists():
            continue
        
        try:
            with open(fpath, 'rb') as f:
                # Read in chunks for large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
        except:
            pass
    
    return hasher.hexdigest()[:16]  # Short hash


def generate_run_id(prefix: str = 'P') -> str:
    """
    Generate unique run ID
    
    Format: YYYY-MM-DDTHH-MMZ_P{pid}
    
    Args:
        prefix: Prefix character (P for Production, D for Dev, etc.)
    
    Returns:
        Run ID string
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H-%MZ')
    pid = os.getpid() % 10000  # Last 4 digits
    return f"{timestamp}_{prefix}{pid}"


class ManifestBuilder:
    """Build reproducibility manifest step-by-step"""
    
    def __init__(self, run_id: Optional[str] = None):
        self.manifest = {
            'run_id': run_id or generate_run_id(),
            'created_at': datetime.now().isoformat()
        }
        self.start_time = None
    
    def set_args(self, args: argparse.Namespace):
        """Record command-line arguments"""
        self.manifest['args'] = ' '.join(sys.argv[1:])
        self.manifest['args_dict'] = vars(args)
        return self
    
    def set_git_info(self):
        """Record git commit"""
        self.manifest['git_commit'] = get_git_commit()
        return self
    
    def set_data_hash(self, file_paths: List[str]):
        """Record data file hashes"""
        self.manifest['data_hash'] = f"sha256:{get_data_hash(file_paths)}"
        return self
    
    def set_cv_scheme(self, scheme: str, val_window: int = 12):
        """Record cross-validation scheme"""
        self.manifest['cv_scheme'] = scheme
        self.manifest['val_window'] = val_window
        return self
    
    def set_seed(self, seed: int):
        """Record random seed"""
        self.manifest['seed'] = seed
        return self
    
    def set_optuna_config(self, n_trials: int, timeout: Optional[int] = None):
        """Record Optuna configuration"""
        self.manifest['optuna'] = {
            'n_trials': n_trials,
            'timeout_min': timeout
        }
        return self
    
    def set_sparse_config(self, threshold: float, nonzero_min: float):
        """Record sparse series filtering config"""
        self.manifest['sparse_filter'] = {
            'avg_threshold': threshold,
            'nonzero_ratio_min': nonzero_min
        }
        return self
    
    def start(self):
        """Mark training start time"""
        self.start_time = datetime.now()
        self.manifest['start'] = self.start_time.isoformat()
        return self
    
    def finish(self, exit_code: int = 0):
        """Mark training end time and exit code"""
        end_time = datetime.now()
        self.manifest['end'] = end_time.isoformat()
        self.manifest['exit_code'] = exit_code
        
        if self.start_time:
            duration = (end_time - self.start_time).total_seconds()
            self.manifest['duration_sec'] = round(duration, 1)
            self.manifest['duration_human'] = format_duration(duration)
        
        return self
    
    def add_metadata(self, key: str, value):
        """Add custom metadata"""
        self.manifest[key] = value
        return self
    
    def save(self, output_path: str):
        """Save manifest to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        print(f"[SUCCESS] Manifest saved: {output_path}")
        return self
    
    def get_manifest(self) -> Dict:
        """Get manifest dictionary"""
        return self.manifest.copy()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}min"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{hours:.0f}h{mins:.0f}m"


def load_manifest(manifest_path: str) -> Dict:
    """Load existing manifest"""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def compare_manifests(manifest1_path: str, manifest2_path: str) -> Dict:
    """
    Compare two manifests for reproducibility check
    
    Returns:
        {
            'data_match': bool,
            'cv_match': bool,
            'seed_match': bool,
            'differences': dict
        }
    """
    m1 = load_manifest(manifest1_path)
    m2 = load_manifest(manifest2_path)
    
    data_match = m1.get('data_hash') == m2.get('data_hash')
    cv_match = (m1.get('cv_scheme') == m2.get('cv_scheme') and
                m1.get('val_window') == m2.get('val_window'))
    seed_match = m1.get('seed') == m2.get('seed')
    
    differences = {}
    for key in set(m1.keys()) | set(m2.keys()):
        if m1.get(key) != m2.get(key):
            differences[key] = {
                'manifest1': m1.get(key),
                'manifest2': m2.get(key)
            }
    
    return {
        'data_match': data_match,
        'cv_match': cv_match,
        'seed_match': seed_match,
        'differences': differences,
        'reproducible': data_match and cv_match and seed_match
    }


# Example usage template
def create_training_manifest_example():
    """Example of how to use ManifestBuilder in training script"""
    
    # At script start
    manifest = ManifestBuilder(run_id=generate_run_id('P'))
    
    # Parse arguments (example)
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-optimize', action='store_true')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    
    # Build manifest
    manifest.set_args(args) \
            .set_git_info() \
            .set_data_hash(['data/features/*.json']) \
            .set_cv_scheme('rolling_3fold', val_window=12) \
            .set_seed(42) \
            .set_optuna_config(n_trials=50, timeout=10) \
            .set_sparse_config(threshold=0.5, nonzero_min=0.3) \
            .start()
    
    # ... run training ...
    
    try:
        # Training code here
        exit_code = 0
    except Exception as e:
        print(f"[ERROR] {e}")
        exit_code = 1
    
    # Finish and save
    manifest.add_metadata('total_series', 337) \
            .add_metadata('successful_series', 220) \
            .finish(exit_code) \
            .save('artifacts/models/base_monthly/manifest.json')
    
    return manifest.get_manifest()


if __name__ == "__main__":
    # Test
    manifest = ManifestBuilder(run_id=generate_run_id('T'))
    manifest.set_git_info() \
            .set_cv_scheme('rolling_3fold', val_window=12) \
            .set_seed(42) \
            .set_optuna_config(n_trials=50) \
            .set_sparse_config(threshold=0.5, nonzero_min=0.3) \
            .start()
    
    # Simulate work
    import time
    time.sleep(2)
    
    manifest.add_metadata('test_run', True) \
            .finish(exit_code=0) \
            .save('test_manifest.json')
    
    print("\nManifest content:")
    print(json.dumps(manifest.get_manifest(), indent=2))
