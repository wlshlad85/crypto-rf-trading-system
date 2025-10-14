"""Memory-efficient data processing utilities."""

import pandas as pd
import numpy as np
from typing import Iterator, Callable, Any, List, Dict
import logging
import gc
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class DataChunker:
    """Memory-efficient data processing with chunking."""

    def __init__(self, chunk_size: int = 10000, temp_dir: str = None):
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_files = []
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")

        self.temp_files.clear()

    def chunk_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split DataFrame into chunks."""
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size].copy()
            yield chunk

    def process_dataframe_in_chunks(
        self,
        df: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        combine_func: Callable[[List[pd.DataFrame]], pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Process DataFrame in chunks and combine results."""
        results = []

        for chunk in self.chunk_dataframe(df):
            try:
                processed_chunk = process_func(chunk)
                results.append(processed_chunk)

                # Memory management
                del chunk
                gc.collect()

            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}")
                continue

        if combine_func:
            return combine_func(results)
        else:
            # Default: concatenate along rows
            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def save_large_dataframe(self, df: pd.DataFrame, filepath: str, compression: str = 'gzip') -> str:
        """Save large DataFrame efficiently."""
        # Create temporary file for chunked saving
        temp_filepath = os.path.join(self.temp_dir, f"temp_{os.path.basename(filepath)}")

        if compression == 'gzip':
            df.to_csv(temp_filepath, compression='gzip', index=False)
        elif compression == 'bz2':
            df.to_csv(temp_filepath, compression='bz2', index=False)
        else:
            df.to_csv(temp_filepath, index=False)

        # Move to final location
        os.rename(temp_filepath, filepath)
        self.temp_files.append(filepath)

        return filepath

    def load_large_dataframe(self, filepath: str, chunksize: int = None) -> pd.DataFrame:
        """Load large DataFrame in chunks."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        chunksize = chunksize or self.chunk_size

        # Read in chunks and combine
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def memory_usage_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get memory usage summary for DataFrame."""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()

        return {
            'total_memory_mb': total_memory / (1024 * 1024),
            'memory_per_column_mb': (memory_usage / (1024 * 1024)).to_dict(),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict()
        }

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        df_optimized = df.copy()

        # Downcast numeric columns
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            # Try to downcast to smaller dtypes
            if df_optimized[col].dtype == np.float64:
                # Check if we can use float32
                if not df_optimized[col].isna().any():
                    col_min, col_max = df_optimized[col].min(), df_optimized[col].max()
                    if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)

            elif df_optimized[col].dtype == np.int64:
                # Check if we can use smaller int types
                if not df_optimized[col].isna().any():
                    col_min, col_max = df_optimized[col].min(), df_optimized[col].max()
                    if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)

        # Convert object columns to category where appropriate
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')

        return df_optimized


def process_large_dataset(
    input_path: str,
    output_path: str,
    process_func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10000,
    optimize_memory: bool = True
) -> str:
    """Process a large dataset file in chunks."""
    logger.info(f"Processing large dataset: {input_path}")

    with DataChunker(chunk_size=chunk_size) as chunker:
        # Process first chunk to get column structure
        first_chunk = pd.read_csv(input_path, nrows=1)
        processed_columns = process_func(first_chunk).columns

        # Initialize output file
        output_file = chunker.save_large_dataframe(
            pd.DataFrame(columns=processed_columns),
            output_path
        )

        # Process data in chunks
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            processed_chunk = process_func(chunk)

            # Optimize memory if requested
            if optimize_memory:
                processed_chunk = chunker.optimize_dataframe_memory(processed_chunk)

            # Append to output file
            processed_chunk.to_csv(
                output_file,
                mode='a',
                header=False,
                index=False
            )

            # Memory cleanup
            del chunk, processed_chunk
            gc.collect()

    logger.info(f"Large dataset processing completed: {output_path}")
    return output_path


# Utility functions for common operations
def calculate_rolling_features_chunked(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate rolling features efficiently in chunks."""
    result = df.copy()

    # Calculate rolling means for multiple windows efficiently
    for window in windows:
        for col in df.select_dtypes(include=[np.number]).columns:
            result[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()

    return result


def calculate_technical_indicators_chunked(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators in chunks."""
    result = df.copy()

    # Only calculate on numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col.endswith('_close'):  # Only for price columns
            prices = df[col]

            # RSI calculation
            delta = prices.diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            # Use exponential moving average for efficiency
            avg_gain = pd.Series(gain).ewm(alpha=1/14).mean()
            avg_loss = pd.Series(loss).ewm(alpha=1/14).mean()

            rs = avg_gain / avg_loss
            result[f"{col}_rsi"] = 100 - (100 / (1 + rs))

    return result