import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
import os
from typing import List, Optional, Tuple
import logging
from dotenv import load_dotenv
import random

random.seed(1234)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables only from analysis/env/.env
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
analysis_env_path = os.path.join(PROJECT_ANALYSIS_DIR, 'env', '.env')
if os.path.exists(analysis_env_path):
    load_dotenv(analysis_env_path)

# Initialize OpenAI client with API key from env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(text: str, model: str = "text-embedding-3-large") -> Optional[List[float]]:
    """
    Fetch the embedding of a text using the specified OpenAI model
    
    Args:
        text (str): Text to get the embedding for
        model (str): Name of the model to use
        
    Returns:
        List[float]: Embedding of the text, or None if failed
    """
    try:
        text = text.replace("\n", " ").strip()
        if not text:
            return None
        
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text: {e}")
        return None





def calculate_embeddings_for_column(df: pd.DataFrame, text_column: str) -> Tuple[List, List[int]]:
    """
    Calculate embeddings for a specific text column
    
    Args:
        df (pd.DataFrame): DataFrame containing the text column
        text_column (str): Name of the column to calculate embeddings for
        
    Returns:
        Tuple[List, List[int]]: List of embeddings and list of valid indices
    """
    logger.info(f"Calculating embeddings for {text_column}...")
    embeddings = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        if pd.isna(text) or str(text).strip() == '':
            embeddings.append(None)
            continue
            
        embedding = get_embedding(str(text))
        embeddings.append(embedding)
        
        if embedding is not None:
            valid_indices.append(idx)
            
        if len(valid_indices) % 10 == 0 and len(valid_indices) > 0:
            logger.info(f"Processed {len(valid_indices)} embeddings for {text_column}...")
    
    logger.info(f"Successfully generated {len(valid_indices)} embeddings for {text_column}")
    return embeddings, valid_indices

def calculate_diversity_scores(df: pd.DataFrame, embeddings: List, valid_indices: List[int], 
                              condition_column: str, prefix: str) -> pd.DataFrame:
    """
    Calculate diversity scores for given embeddings
    
    Args:
        df (pd.DataFrame): DataFrame to add diversity scores to
        embeddings (List): List of embeddings
        valid_indices (List[int]): List of valid indices with embeddings
        condition_column (str): Column name containing conditions for within-condition diversity
        prefix (str): Prefix for the diversity score column names
        
    Returns:
        pd.DataFrame: DataFrame with added diversity score columns
    """
    df_result = df.copy()
    
    # Initialize diversity columns
    all_div_col = f"{prefix}_div_allideas"
    condition_div_col = f"{prefix}_div_condition"
    df_result[all_div_col] = np.nan
    df_result[condition_div_col] = np.nan
    
    # Calculate diversity to all other ideas (leave-one-out)
    logger.info(f"Calculating diversity to all other ideas for {prefix}...")
    for idx in valid_indices:
        current_embedding = embeddings[idx]
        if current_embedding is None:
            continue
            
        # Get all other valid embeddings (leave-one-out)
        other_embeddings = [emb for i, emb in enumerate(embeddings) 
                          if emb is not None and i != idx]
        
        if len(other_embeddings) > 0:
            # Calculate average of all other embeddings
            average_others = np.mean(other_embeddings, axis=0)
            # Calculate cosine diversity (cosine distance)
            div_score = cosine(current_embedding, average_others)
            df_result.at[idx, all_div_col] = div_score
    
    # Calculate diversity to ideas within the same condition
    logger.info(f"Calculating diversity within same condition for {prefix}...")
    for condition in df_result[condition_column].unique():
        if pd.isna(condition):
            continue
            
        condition_indices = [idx for idx in valid_indices 
                           if df_result.at[idx, condition_column] == condition]
        
        for idx in condition_indices:
            current_embedding = embeddings[idx]
            if current_embedding is None:
                continue
                
            # Get embeddings from same condition (excluding current)
            same_condition_embeddings = [embeddings[i] for i in condition_indices 
                                       if i != idx and embeddings[i] is not None]
            
            if len(same_condition_embeddings) > 0:
                # Calculate average of same condition embeddings
                average_condition = np.mean(same_condition_embeddings, axis=0)
                # Calculate cosine diversity
                div_score = cosine(current_embedding, average_condition)
                df_result.at[idx, condition_div_col] = div_score
    
    return df_result

def calculate_cross_column_diversity(df: pd.DataFrame, source_embeddings: List, source_valid_indices: List[int],
                                     target_embeddings: List, target_valid_indices: List[int], 
                                     condition_column: str, output_column: str) -> pd.DataFrame:
    """
    Calculate diversity between source embeddings and centroid of target embeddings within same condition
    
    Args:
        df (pd.DataFrame): DataFrame to add diversity scores to
        source_embeddings (List): Source embeddings (e.g., refined idea embeddings)
        source_valid_indices (List[int]): Valid indices for source embeddings
        target_embeddings (List): Target embeddings (e.g., initial idea embeddings)
        target_valid_indices (List[int]): Valid indices for target embeddings
        condition_column (str): Column name containing conditions
        output_column (str): Name for the output diversity column
        
    Returns:
        pd.DataFrame: DataFrame with added cross-column diversity scores
    """
    df_result = df.copy()
    df_result[output_column] = np.nan
    
    logger.info(f"Calculating cross-column diversity for {output_column}...")
    
    # For each condition, calculate centroid of target embeddings
    for condition in df_result[condition_column].unique():
        if pd.isna(condition):
            continue
            
        # Find target embeddings for this condition
        target_condition_indices = [idx for idx in target_valid_indices 
                                  if df_result.at[idx, condition_column] == condition]
        
        target_condition_embeddings = [target_embeddings[i] for i in target_condition_indices 
                                     if target_embeddings[i] is not None]
        
        if len(target_condition_embeddings) == 0:
            logger.warning(f"No target embeddings found for condition {condition}")
            continue
        
        # Calculate centroid of target embeddings for this condition
        target_centroid = np.mean(target_condition_embeddings, axis=0)
        
        # Find source embeddings for this condition
        source_condition_indices = [idx for idx in source_valid_indices 
                                  if df_result.at[idx, condition_column] == condition]
        
        # Calculate similarity for each source embedding in this condition
        for idx in source_condition_indices:
            source_embedding = source_embeddings[idx]
            if source_embedding is None:
                continue
                
            # Calculate cosine diversity
            div_score = cosine(source_embedding, target_centroid)
            df_result.at[idx, output_column] = div_score
        
        logger.info(f"Processed condition {condition}: {len(target_condition_embeddings)} target embeddings, "
                   f"{len(source_condition_indices)} source embeddings")
    
    return df_result

def process_text_column_diversities(df: pd.DataFrame, text_column: str, condition_column: str, prefix: str) -> pd.DataFrame:
    """
    Process diversities for a single text column (combines embedding calculation and diversity scoring)
    
    Args:
        df (pd.DataFrame): DataFrame containing the text column
        text_column (str): Name of the text column to process
        condition_column (str): Column name containing conditions for within-condition diversity
        prefix (str): Prefix for the output column names
        
    Returns:
        pd.DataFrame: DataFrame with added diversity scores and embeddings
    """
    # Calculate embeddings
    embeddings, valid_indices = calculate_embeddings_for_column(df, text_column)
    
    # Add embeddings to dataframe
    embedding_col = f"{prefix}_embedding"
    df[embedding_col] = embeddings
    
    # Calculate diversity scores
    df_result = calculate_diversity_scores(df, embeddings, valid_indices, condition_column, prefix)
    
    return df_result

def main():
    """
    Main function to run the diversity analysis for original text columns
    """
    # Check if OpenAI API key is set
    if not client.api_key:
        logger.error("OPENAI_API_KEY not found.")
        logger.info("Create analysis/env/.env with OPENAI_API_KEY=... or export it in your shell.")
        return
    
    # Optional row limit for quick test runs
    try:
        limit_rows = int(os.getenv('LIMIT_ROWS', '0'))
    except ValueError:
        limit_rows = 0

    # Load the data (path relative to this script)
    data_path = os.path.join(PROJECT_ANALYSIS_DIR, 'data', 'data_clean.csv')
    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        if limit_rows and limit_rows > 0:
            df = df.head(limit_rows)
            logger.info(f"LIMIT_ROWS set → using first {len(df)} rows for test run")
        logger.info(f"Loaded {len(df)} rows from {data_path}")
    except FileNotFoundError:
        logger.error(f"{data_path} not found. Please make sure the file exists.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Show distribution by condition
    logger.info("Distribution by bot_type:")
    for condition, count in df['bot_type'].value_counts().items():
        logger.info(f"  {condition}: {count}")
    

    
    # Process diversities for original text columns only
    text_columns_to_process = [
        ('idea', 'idea'),
        ('refined_idea', 'refined')
    ]
    
    for text_column, prefix in text_columns_to_process:
        if text_column in df.columns:
            # Only process columns that have some non-null values
            non_null_count = df[text_column].notna().sum()
            if non_null_count > 0:
                logger.info(f"\n=== PROCESSING DIVERSITIES FOR {text_column.upper()} ===")
                logger.info(f"Found {non_null_count} non-null values in {text_column}")
                df = process_text_column_diversities(df, text_column, 'bot_type', prefix)
            else:
                logger.info(f"Skipping {text_column} - no non-null values found")
        else:
            logger.warning(f"Column {text_column} not found in DataFrame")
    
    # Calculate cross-column diversities: refined ideas vs initial idea centroids
    logger.info(f"\n=== CALCULATING CROSS-COLUMN DIVERSITIES ===")
    
    # Check if both idea and refined_idea embeddings exist
    if 'idea_embedding' in df.columns and 'refined_embedding' in df.columns:
        logger.info("Calculating refined idea diversity to initial idea centroids within condition...")
        
        # Get embeddings and valid indices for both columns
        idea_embeddings = df['idea_embedding'].tolist()
        refined_embeddings = df['refined_embedding'].tolist()
        
        idea_valid_indices = [i for i, emb in enumerate(idea_embeddings) if emb is not None]
        refined_valid_indices = [i for i, emb in enumerate(refined_embeddings) if emb is not None]
        
        # Calculate diversity between refined ideas and initial idea centroids
        df = calculate_cross_column_diversity(
            df, 
            source_embeddings=refined_embeddings,
            source_valid_indices=refined_valid_indices,
            target_embeddings=idea_embeddings,
            target_valid_indices=idea_valid_indices,
            condition_column='bot_type',
            output_column='refined_div_to_initial_centroid'
        )
    else:
        logger.warning("Could not calculate cross-column diversity - missing embeddings")
    

    
    # Save results with efficient embedding storage
    from improved_embedding_storage import save_embeddings_efficiently
    
    # Save with efficient storage (paths relative to this script)
    output_base = os.path.join(PROJECT_ANALYSIS_DIR, 'data', 'ideas_with_diversity_scores')
    embeddings_file, data_file = save_embeddings_efficiently(df, output_base)
    
    # Also save traditional CSV for compatibility (without embeddings)
    output_file = f"{output_base}.csv"
    embedding_cols = [col for col in df.columns if col.endswith('_embedding')]
    df_no_embeddings = df.drop(columns=embedding_cols)
    df_no_embeddings.to_csv(output_file, index=False)
    
    logger.info(f"Main data saved to {output_file}")
    if embeddings_file:
        logger.info(f"Embeddings saved separately to {embeddings_file}")
    logger.info(f"CSV file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Print embeddings summary for analysis
    logger.info("\n=== EMBEDDING SUMMARY ===")
    embedding_columns = [col for col in df.columns if col.endswith('_embedding')]
    for col in embedding_columns:
        valid_embeddings = df[col].dropna()
        if len(valid_embeddings) > 0:
            # Get first valid embedding to check dimensions
            first_valid = next(emb for emb in valid_embeddings if emb is not None)
            embedding_dim = len(first_valid) if first_valid is not None else 0
            
            logger.info(f"{col}: {len(valid_embeddings)}/{len(df)} valid ({len(valid_embeddings)/len(df)*100:.1f}%), {embedding_dim} dimensions")
    
    # Print summary statistics
    logger.info("\n=== SUMMARY STATISTICS ===")
    diversity_columns = [col for col in df.columns if '_div_' in col]
    
    for col in diversity_columns:
        valid_scores = df[col].dropna()
        if len(valid_scores) > 0:
            logger.info(f"{col}: Mean={valid_scores.mean():.3f}, Std={valid_scores.std():.3f}, N={len(valid_scores)}")
    
    # Print specific statistics for cross-column diversity
    logger.info("\n=== CROSS-COLUMN DIVERSITY STATISTICS ===")
    cross_column_cols = [col for col in diversity_columns if 'centroid' in col]
    
    if cross_column_cols:
        for col in cross_column_cols:
            valid_scores = df[col].dropna()
            if len(valid_scores) > 0:
                logger.info(f"{col}: Mean={valid_scores.mean():.3f}, Std={valid_scores.std():.3f}, "
                           f"Min={valid_scores.min():.3f}, Max={valid_scores.max():.3f}, N={len(valid_scores)}")
                
                # Show distribution by condition for cross-column diversities
                logger.info(f"  Distribution by bot_type:")
                for condition in df['bot_type'].unique():
                    if pd.notna(condition):
                        condition_scores = df[df['bot_type'] == condition][col].dropna()
                        if len(condition_scores) > 0:
                            logger.info(f"    {condition}: Mean={condition_scores.mean():.3f}, "
                                       f"Std={condition_scores.std():.3f}, N={len(condition_scores)}")
    else:
        logger.info("No cross-column diversity columns found.")

if __name__ == "__main__":
    main() 