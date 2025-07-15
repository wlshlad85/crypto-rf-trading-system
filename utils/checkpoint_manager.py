"""Automatic checkpoint and backup system for the crypto trading project."""

import os
import shutil
import json
import pickle
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import zipfile


class CheckpointManager:
    """Manages automatic checkpoints and backups for the trading system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.backup_dir = self.project_root / "backups"
        self.temp_dir = self.project_root / "temp"
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.checkpoint_interval = 300  # 5 minutes
        self.backup_interval = 1800     # 30 minutes
        self.max_checkpoints = 20
        self.max_backups = 10
        
        self.auto_checkpoint_thread = None
        self.auto_backup_thread = None
        self.running = False
        
        # Track file hashes to detect changes
        self.file_hashes = {}
        
    def start_auto_checkpoint(self):
        """Start automatic checkpoint system."""
        if self.running:
            return
            
        self.running = True
        self.auto_checkpoint_thread = threading.Thread(target=self._auto_checkpoint_loop)
        self.auto_backup_thread = threading.Thread(target=self._auto_backup_loop)
        
        self.auto_checkpoint_thread.daemon = True
        self.auto_backup_thread.daemon = True
        
        self.auto_checkpoint_thread.start()
        self.auto_backup_thread.start()
        
        self.logger.info("Automatic checkpoint system started")
    
    def stop_auto_checkpoint(self):
        """Stop automatic checkpoint system."""
        self.running = False
        if self.auto_checkpoint_thread:
            self.auto_checkpoint_thread.join(timeout=5)
        if self.auto_backup_thread:
            self.auto_backup_thread.join(timeout=5)
        self.logger.info("Automatic checkpoint system stopped")
    
    def _auto_checkpoint_loop(self):
        """Main loop for automatic checkpoints."""
        while self.running:
            try:
                if self._has_changes():
                    self.create_checkpoint()
                time.sleep(self.checkpoint_interval)
            except Exception as e:
                self.logger.error(f"Error in auto checkpoint: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _auto_backup_loop(self):
        """Main loop for automatic backups."""
        while self.running:
            try:
                self.create_full_backup()
                time.sleep(self.backup_interval)
            except Exception as e:
                self.logger.error(f"Error in auto backup: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _has_changes(self) -> bool:
        """Check if project files have changed since last checkpoint."""
        important_files = [
            "main.py", "ultra_main.py",
            "models/random_forest_model.py",
            "features/feature_engineering.py", "features/ultra_feature_engineering.py",
            "data/data_fetcher.py", "data/yfinance_fetcher.py",
            "strategies/long_short_strategy.py",
            "backtesting/backtest_engine.py",
            "utils/config.py", "utils/visualization.py",
            "configs/config.json", "configs/ultra_btc_config.json"
        ]
        
        changes_detected = False
        for file_path in important_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                current_hash = self._get_file_hash(full_path)
                if file_path not in self.file_hashes or self.file_hashes[file_path] != current_hash:
                    self.file_hashes[file_path] = current_hash
                    changes_detected = True
        
        return changes_detected
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def create_checkpoint(self, description: str = None) -> str:
        """Create a checkpoint of critical project data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        try:
            # Save critical files
            critical_files = {
                "main.py": "main.py",
                "ultra_main.py": "ultra_main.py",
                "models/": "models/",
                "features/": "features/",
                "data/": "data/",
                "strategies/": "strategies/",
                "backtesting/": "backtesting/",
                "utils/": "utils/",
                "configs/": "configs/"
            }
            
            for src, dest in critical_files.items():
                src_path = self.project_root / src
                dest_path = checkpoint_path / dest
                
                if src_path.exists():
                    if src_path.is_file():
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                    elif src_path.is_dir():
                        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            
            # Save checkpoint metadata
            metadata = {
                "timestamp": timestamp,
                "description": description or "Automatic checkpoint",
                "project_root": str(self.project_root),
                "files_backed_up": list(critical_files.keys()),
                "file_hashes": self.file_hashes.copy()
            }
            
            with open(checkpoint_path / "checkpoint_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self._cleanup_old_checkpoints()
            self.logger.info(f"Checkpoint created: {checkpoint_name}")
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            # Clean up partial checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            raise
    
    def create_full_backup(self) -> str:
        """Create a full compressed backup of the entire project."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"full_backup_{timestamp}.zip"
        backup_path = self.backup_dir / backup_name
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.project_root):
                    # Skip backup and temp directories
                    dirs[:] = [d for d in dirs if d not in ['backups', 'temp', '__pycache__', '.git']]
                    
                    for file in files:
                        if file.endswith(('.pyc', '.pyo', '.log')):
                            continue
                            
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.project_root)
                        zipf.write(file_path, arcname)
            
            # Save backup metadata
            metadata = {
                "timestamp": timestamp,
                "backup_size": backup_path.stat().st_size,
                "project_root": str(self.project_root),
                "backup_type": "full"
            }
            
            metadata_path = self.backup_dir / f"backup_metadata_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self._cleanup_old_backups()
            self.logger.info(f"Full backup created: {backup_name}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create full backup: {e}")
            if backup_path.exists():
                backup_path.unlink(missing_ok=True)
            raise
    
    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore from a specific checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_name}")
            return False
        
        try:
            # Create backup of current state before restore
            current_backup = self.create_checkpoint("Before restore")
            
            # Load checkpoint metadata
            metadata_path = checkpoint_path / "checkpoint_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.logger.info(f"Restoring checkpoint from {metadata['timestamp']}")
            
            # Restore files
            for item in checkpoint_path.iterdir():
                if item.name == "checkpoint_metadata.json":
                    continue
                    
                dest_path = self.project_root / item.name
                
                if item.is_file():
                    shutil.copy2(item, dest_path)
                elif item.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)
            
            self.logger.info(f"Checkpoint restored successfully: {checkpoint_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from a full backup."""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_name}")
            return False
        
        try:
            # Create emergency checkpoint before restore
            emergency_checkpoint = self.create_checkpoint("Emergency before restore")
            
            # Extract backup to temporary location
            temp_restore_dir = self.temp_dir / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_restore_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(temp_restore_dir)
            
            # Move restored files to project root
            for item in temp_restore_dir.iterdir():
                dest_path = self.project_root / item.name
                
                if dest_path.exists():
                    if dest_path.is_file():
                        dest_path.unlink()
                    else:
                        shutil.rmtree(dest_path)
                
                shutil.move(str(item), str(dest_path))
            
            # Cleanup temporary directory
            shutil.rmtree(temp_restore_dir, ignore_errors=True)
            
            self.logger.info(f"Backup restored successfully: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[self.max_checkpoints:]:
            shutil.rmtree(checkpoint, ignore_errors=True)
            self.logger.debug(f"Removed old checkpoint: {checkpoint.name}")
    
    def _cleanup_old_backups(self):
        """Remove old backups to save space."""
        backups = sorted(
            [f for f in self.backup_dir.iterdir() if f.suffix == '.zip'],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for backup in backups[self.max_backups:]:
            backup.unlink(missing_ok=True)
            # Also remove corresponding metadata
            metadata_file = backup.with_suffix('.json')
            metadata_file.unlink(missing_ok=True)
            self.logger.debug(f"Removed old backup: {backup.name}")
    
    def list_checkpoints(self) -> list:
        """List available checkpoints."""
        checkpoints = []
        for checkpoint_dir in sorted(self.checkpoint_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if checkpoint_dir.is_dir():
                metadata_path = checkpoint_dir / "checkpoint_metadata.json"
                metadata = {}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    except Exception:
                        pass
                
                checkpoints.append({
                    "name": checkpoint_dir.name,
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "description": metadata.get("description", "No description"),
                    "size": sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
                })
        
        return checkpoints
    
    def list_backups(self) -> list:
        """List available backups."""
        backups = []
        for backup_file in sorted(
            [f for f in self.backup_dir.iterdir() if f.suffix == '.zip'],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        ):
            metadata_file = backup_file.with_name(f"backup_metadata_{backup_file.stem.split('_', 2)[-1]}.json")
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                except Exception:
                    pass
            
            backups.append({
                "name": backup_file.name,
                "timestamp": metadata.get("timestamp", "unknown"),
                "size": backup_file.stat().st_size,
                "type": metadata.get("backup_type", "unknown")
            })
        
        return backups
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of checkpoint system."""
        return {
            "running": self.running,
            "checkpoint_interval": self.checkpoint_interval,
            "backup_interval": self.backup_interval,
            "project_root": str(self.project_root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "backup_dir": str(self.backup_dir),
            "num_checkpoints": len(self.list_checkpoints()),
            "num_backups": len(self.list_backups()),
            "last_checkpoint": max(
                [c["timestamp"] for c in self.list_checkpoints()],
                default="Never"
            ),
            "last_backup": max(
                [b["timestamp"] for b in self.list_backups()],
                default="Never"
            )
        }


# Global checkpoint manager instance
_checkpoint_manager = None

def get_checkpoint_manager(project_root: str = None) -> CheckpointManager:
    """Get or create global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(project_root)
    return _checkpoint_manager

def start_auto_checkpoints(project_root: str = None):
    """Start automatic checkpoints for the project."""
    manager = get_checkpoint_manager(project_root)
    manager.start_auto_checkpoint()
    return manager

def stop_auto_checkpoints():
    """Stop automatic checkpoints."""
    global _checkpoint_manager
    if _checkpoint_manager:
        _checkpoint_manager.stop_auto_checkpoint()

def emergency_checkpoint(description: str = "Emergency checkpoint"):
    """Create an emergency checkpoint immediately."""
    manager = get_checkpoint_manager()
    return manager.create_checkpoint(description)