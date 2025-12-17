import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def save_embeddings_efficiently(df: pd.DataFrame, output_prefix: str) -> Tuple[str, str]:
    """
    Save embeddings and main data efficiently with separate storage
    
    Args:
        df: DataFrame containing embeddings and main data
        output_prefix: Prefix for output files
        
    Returns:
        Tuple of (embeddings_file_path, data_file_path)
    """
    
    # Extract embedding columns
    embedding_cols = [col for col in df.columns if col.endswith('_embedding')]
    
    if not embedding_cols:
        logger.warning("No embedding columns found")
        data_file = f'{output_prefix}_data.csv'
        df.to_csv(data_file, index=False)
        return None, data_file
    
    logger.info(f"Found {len(embedding_cols)} embedding columns: {embedding_cols}")
    
    # Prepare embeddings dictionary for numpy storage
    embeddings_data = {}
    storage_stats = {}
    
    for col in embedding_cols:
        embeddings_list = df[col].tolist()
        valid_embeddings = []
        valid_indices = []
        
        for i, emb in enumerate(embeddings_list):
            if emb is not None and len(emb) > 0:
                valid_embeddings.append(emb)
                valid_indices.append(i)
        
        if valid_embeddings:
            # Convert to numpy array with float32 for space efficiency
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            indices_array = np.array(valid_indices, dtype=np.int32)
            
            # Store both embeddings and their original indices
            embeddings_data[f'{col}_vectors'] = embeddings_array
            embeddings_data[f'{col}_indices'] = indices_array
            
            # Calculate storage stats
            dimensions = embeddings_array.shape[1]
            count = embeddings_array.shape[0]
            size_mb = embeddings_array.nbytes / (1024 * 1024)
            
            storage_stats[col] = {
                'count': count,
                'dimensions': dimensions,
                'size_mb': size_mb
            }
            
            logger.info(f"{col}: {count} embeddings, {dimensions}D, {size_mb:.1f} MB")
    
    # Save embeddings as compressed numpy archive
    embeddings_file = f'{output_prefix}_embeddings.npz'
    np.savez_compressed(embeddings_file, **embeddings_data)
    
    # Save main data without embeddings
    main_data = df.drop(columns=embedding_cols)
    data_file = f'{output_prefix}_data.csv'
    main_data.to_csv(data_file, index=False)
    
    # Log file sizes
    embeddings_size = os.path.getsize(embeddings_file) / (1024 * 1024)
    data_size = os.path.getsize(data_file) / (1024 * 1024)
    
    logger.info(f"\n=== STORAGE SUMMARY ===")
    logger.info(f"Embeddings file: {embeddings_file} ({embeddings_size:.1f} MB)")
    logger.info(f"Data file: {data_file} ({data_size:.1f} MB)")
    logger.info(f"Total: {embeddings_size + data_size:.1f} MB")
    
    # Save metadata about embeddings structure
    metadata = {
        'embedding_columns': embedding_cols,
        'storage_stats': storage_stats,
        'total_rows': len(df)
    }
    
    import json
    with open(f'{output_prefix}_embeddings_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return embeddings_file, data_file

def load_embeddings_efficiently(embeddings_file: str, data_file: str, 
                               metadata_file: str = None) -> pd.DataFrame:
    """
    Load embeddings and main data from efficient storage
    
    Args:
        embeddings_file: Path to .npz embeddings file
        data_file: Path to CSV data file
        metadata_file: Optional path to metadata JSON file
        
    Returns:
        DataFrame with reconstructed embeddings
    """
    
    logger.info("Loading main data...")
    df = pd.read_csv(data_file)
    
    if not os.path.exists(embeddings_file):
        logger.warning(f"Embeddings file not found: {embeddings_file}")
        return df
    
    logger.info("Loading embeddings...")
    with np.load(embeddings_file) as data:
        # Find all embedding column names
        vector_keys = [key for key in data.keys() if key.endswith('_vectors')]
        
        for vector_key in vector_keys:
            # Extract column name
            col_name = vector_key.replace('_vectors', '')
            indices_key = f'{col_name}_indices'
            
            if indices_key in data:
                embeddings_array = data[vector_key]
                indices_array = data[indices_key]
                
                # Reconstruct full embedding column with None values
                full_embeddings = [None] * len(df)
                for i, idx in enumerate(indices_array):
                    full_embeddings[idx] = embeddings_array[i].tolist()
                
                df[col_name] = full_embeddings
                logger.info(f"Restored {col_name}: {len(indices_array)} embeddings")
    
    return df

# Example usage functions
def convert_existing_csv_to_efficient_storage(csv_file: str, output_prefix: str):
    """Convert existing CSV with embeddings to efficient storage"""
    
    logger.info(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Convert string representations back to lists if needed
    embedding_cols = [col for col in df.columns if col.endswith('_embedding')]
    
    for col in embedding_cols:
        logger.info(f"Processing {col}...")
        embeddings = []
        for val in df[col]:
            if pd.isna(val) or val == 'None':
                embeddings.append(None)
            else:
                try:
                    # Handle string representation of lists
                    if isinstance(val, str):
                        # Remove brackets and split by comma
                        val = val.strip('[]').split(',')
                        val = [float(x.strip()) for x in val if x.strip()]
                    embeddings.append(val)
                except:
                    embeddings.append(None)
        
        df[col] = embeddings
    
    return save_embeddings_efficiently(df, output_prefix)

if __name__ == "__main__":
    # Example: Convert existing CSV to efficient storage
    # Paths relative to this script location
    script_dir = os.path.dirname(__file__)
    analysis_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(analysis_dir, 'data')

    input_csv = os.path.join(data_dir, "ideas_with_diversity_scores.csv")
    if os.path.exists(input_csv):
        print("Converting existing CSV to efficient storage...")
        output_prefix = os.path.join(data_dir, "ideas_efficient")
        emb_file, data_file = convert_existing_csv_to_efficient_storage(
            input_csv, output_prefix
        )
        
        # Test loading
        print("Testing loading...")
        df_restored = load_embeddings_efficiently(emb_file, data_file)
        print(f"Restored DataFrame: {len(df_restored)} rows")
    else:
        print(f"File {input_csv} not found - run the main script first")
