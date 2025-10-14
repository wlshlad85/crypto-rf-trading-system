"""Optimized file I/O operations for large datasets."""

import pandas as pd
import numpy as np
import json
import pickle
import gzip
import bz2
import lz4.frame
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import gc

logger = logging.getLogger(__name__)


class OptimizedFileOperations:
    """Optimized file operations with compression and batch processing."""

    def __init__(self, temp_dir: str = None, compression: str = 'gzip'):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.compression = compression
        self.logger = logging.getLogger(__name__)

    def save_dataframe_compressed(self, df: pd.DataFrame, filepath: str,
                                 compression: str = None) -> str:
        """Save DataFrame with optimized compression."""
        compression = compression or self.compression

        try:
            if compression == 'gzip':
                df.to_pickle(filepath.replace('.pkl', '.pkl.gz'))
                return filepath.replace('.pkl', '.pkl.gz')
            elif compression == 'bz2':
                with bz2.BZ2File(filepath.replace('.pkl', '.pkl.bz2'), 'wb') as f:
                    pickle.dump(df, f)
                return filepath.replace('.pkl', '.pkl.bz2')
            elif compression == 'lz4':
                df.to_pickle(filepath.replace('.pkl', '.pkl.lz4'))
                return filepath.replace('.pkl', '.pkl.lz4')
            else:
                df.to_pickle(filepath)
                return filepath

        except Exception as e:
            self.logger.error(f"Error saving DataFrame: {e}")
            # Fallback to pickle
            df.to_pickle(filepath)
            return filepath

    def load_dataframe_compressed(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame with automatic compression detection."""
        try:
            if filepath.endswith('.pkl.gz'):
                return pd.read_pickle(filepath)
            elif filepath.endswith('.pkl.bz2'):
                with bz2.BZ2File(filepath, 'rb') as f:
                    return pickle.load(f)
            elif filepath.endswith('.pkl.lz4'):
                return pd.read_pickle(filepath)
            else:
                return pd.read_pickle(filepath)

        except Exception as e:
            self.logger.error(f"Error loading DataFrame: {e}")
            # Try alternative methods
            try:
                return pd.read_csv(filepath)
            except:
                raise FileNotFoundError(f"Could not load file: {filepath}")

    def batch_save_json(self, data_list: List[Dict], base_filepath: str,
                       batch_size: int = 100) -> List[str]:
        """Save large JSON data in batches."""
        saved_files = []

        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_num = i // batch_size

            filepath = f"{base_filepath}_batch_{batch_num}.json"

            try:
                with open(filepath, 'w') as f:
                    json.dump(batch, f, indent=2)
                saved_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error saving batch {batch_num}: {e}")

        return saved_files

    def batch_load_json(self, filepaths: List[str]) -> List[Dict]:
        """Load multiple JSON files efficiently."""
        all_data = []

        with ThreadPoolExecutor(max_workers=min(len(filepaths), 4)) as executor:
            future_to_file = {
                executor.submit(self._load_single_json, filepath): filepath
                for filepath in filepaths
            }

            for future in future_to_file:
                try:
                    data = future.result()
                    if data:
                        all_data.extend(data)
                except Exception as e:
                    self.logger.error(f"Error loading {future_to_file[future]}: {e}")

        return all_data

    def _load_single_json(self, filepath: str) -> List[Dict]:
        """Load a single JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON {filepath}: {e}")
            return []

    def save_model_artifacts(self, model_data: Dict[str, Any], filepath: str) -> str:
        """Save model artifacts efficiently."""
        # Create a compressed archive of model files
        import tarfile

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            with tarfile.open(tmp.name, 'w:gz') as tar:
                for name, data in model_data.items():
                    if isinstance(data, pd.DataFrame):
                        # Save DataFrame as pickle
                        temp_file = os.path.join(self.temp_dir, f"{name}.pkl")
                        data.to_pickle(temp_file)
                        tar.add(temp_file, arcname=f"{name}.pkl")
                        os.remove(temp_file)
                    elif isinstance(data, dict):
                        # Save dict as JSON
                        temp_file = os.path.join(self.temp_dir, f"{name}.json")
                        with open(temp_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        tar.add(temp_file, arcname=f"{name}.json")
                        os.remove(temp_file)

            # Move to final location
            os.rename(tmp.name, filepath)

        return filepath

    def load_model_artifacts(self, filepath: str) -> Dict[str, Any]:
        """Load model artifacts from compressed archive."""
        import tarfile

        artifacts = {}

        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        # Extract to temporary file
                        temp_file = tar.extractfile(member)
                        if temp_file:
                            if member.name.endswith('.pkl'):
                                # Load DataFrame
                                artifacts[member.name.replace('.pkl', '')] = pickle.load(temp_file)
                            elif member.name.endswith('.json'):
                                # Load JSON
                                artifacts[member.name.replace('.json', '')] = json.load(temp_file)

        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {e}")

        return artifacts

    def optimize_csv_save(self, df: pd.DataFrame, filepath: str,
                         chunk_size: int = 10000) -> str:
        """Save large DataFrame to CSV in chunks."""
        if len(df) <= chunk_size:
            df.to_csv(filepath, index=False)
            return filepath

        # Save in chunks
        first_chunk = True
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(filepath, mode=mode, header=header, index=False)
            first_chunk = False

        return filepath

    def memory_map_large_file(self, filepath: str) -> np.ndarray:
        """Memory map a large file for efficient reading."""
        try:
            return np.load(filepath, mmap_mode='r')
        except Exception as e:
            self.logger.error(f"Error memory mapping file: {e}")
            return None

    def create_memory_efficient_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a memory-efficient copy of DataFrame."""
        # Optimize dtypes
        df_optimized = df.copy()

        # Downcast numeric types
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            if df_optimized[col].dtype == np.float64:
                # Check if we can downcast
                if not df_optimized[col].isna().any():
                    col_min, col_max = df_optimized[col].min(), df_optimized[col].max()
                    if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)

            elif df_optimized[col].dtype == np.int64:
                if not df_optimized[col].isna().any():
                    col_min, col_max = df_optimized[col].min(), df_optimized[col].max()
                    if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)

        # Convert object columns to category if appropriate
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')

        return df_optimized


def save_results_efficiently(results: Dict[str, Any], filepath: str) -> str:
    """Save results dictionary efficiently."""
    with OptimizedFileOperations() as file_ops:
        return file_ops.save_model_artifacts(results, filepath)


def load_results_efficiently(filepath: str) -> Dict[str, Any]:
    """Load results dictionary efficiently."""
    with OptimizedFileOperations() as file_ops:
        return file_ops.load_model_artifacts(filepath)


def batch_process_files(input_dir: str, output_dir: str,
                       process_func: callable, file_pattern: str = "*.csv") -> List[str]:
    """Process multiple files in batch."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    processed_files = []

    # Get all input files
    input_files = list(input_path.glob(file_pattern))

    with ThreadPoolExecutor(max_workers=min(len(input_files), 4)) as executor:
        future_to_file = {
            executor.submit(process_single_file, file, output_path, process_func): file
            for file in input_files
        }

        for future in future_to_file:
            try:
                output_file = future.result()
                if output_file:
                    processed_files.append(output_file)
            except Exception as e:
                logger.error(f"Error processing {future_to_file[future]}: {e}")

    return processed_files


def process_single_file(input_file: Path, output_dir: Path, process_func: callable) -> Optional[str]:
    """Process a single file."""
    try:
        # Load file
        if input_file.suffix == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix == '.pkl':
            df = pd.read_pickle(input_file)
        else:
            return None

        # Process
        processed_df = process_func(df)

        # Save
        output_file = output_dir / f"processed_{input_file.name}"
        processed_df.to_csv(output_file, index=False)

        return str(output_file)

    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")
        return None