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

# Load environment variables from backend/.env file
backend_env_path = os.path.join(os.path.dirname(__file__), '..', 'backend', '.env')
load_dotenv(backend_env_path)

# Initialize OpenAI client with API key from backend .env file
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

def generate_summary(text: str) -> Optional[str]:
    """
    Generate a one-sentence summary of the given idea using OpenAI's GPT model
    
    Args:
        text (str): The idea text to summarize
        
    Returns:
        str: One-sentence summary, or None if failed
    """
    try:
        text = text.strip()
        if not text:
            return None
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You create a one-sentence summary of the given idea. The summary should focus on the core semantic meaning of the idea. Do not include any other information. When the idea consists of several ideas, just summarize the first proposed idea. Start every sentence with 'The idea is...'"
                },
                {
                    "role": "user", 
                    "content": f"{text}"
                }
            ],
            max_tokens=70,
            temperature=0.0001
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary for text: {e}")
        return None

def generate_summaries_for_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Generate summaries for specified text columns
    
    Args:
        df (pd.DataFrame): DataFrame containing the text columns
        text_columns (List[str]): List of column names to generate summaries for
        
    Returns:
        pd.DataFrame: DataFrame with added summary columns
    """
    df_result = df.copy()
    
    for col in text_columns:
        if col not in df_result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
            
        summary_col = f"{col}_summary" 
        df_result[summary_col] = None
        
        logger.info(f"Generating summaries for {col}...")
        count = 0
        for idx, row in df_result.iterrows():
            text = row[col]
            
            if pd.notna(text) and str(text).strip() != '':
                summary = generate_summary(str(text))
                df_result.at[idx, summary_col] = summary
                count += 1
                
            if count % 10 == 0 and count > 0:
                logger.info(f"Processed {count} summaries for {col}...")
        
        logger.info(f"Generated {count} summaries for {col}")
    
    return df_result

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

def calculate_similarity_scores(df: pd.DataFrame, embeddings: List, valid_indices: List[int], 
                              condition_column: str, prefix: str) -> pd.DataFrame:
    """
    Calculate similarity scores for given embeddings
    
    Args:
        df (pd.DataFrame): DataFrame to add similarity scores to
        embeddings (List): List of embeddings
        valid_indices (List[int]): List of valid indices with embeddings
        condition_column (str): Column name containing conditions for within-condition similarity
        prefix (str): Prefix for the similarity score column names
        
    Returns:
        pd.DataFrame: DataFrame with added similarity score columns
    """
    df_result = df.copy()
    
    # Initialize similarity columns
    all_sim_col = f"{prefix}_sim_allideas"
    condition_sim_col = f"{prefix}_sim_condition"
    df_result[all_sim_col] = np.nan
    df_result[condition_sim_col] = np.nan
    
    # Calculate similarity to all other ideas (leave-one-out)
    logger.info(f"Calculating similarity to all other ideas for {prefix}...")
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
            # Calculate cosine similarity (1 - cosine distance)
            sim_score = 1 - cosine(current_embedding, average_others)
            df_result.at[idx, all_sim_col] = sim_score
    
    # Calculate similarity to ideas within the same condition
    logger.info(f"Calculating similarity within same condition for {prefix}...")
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
                # Calculate cosine similarity
                sim_score = 1 - cosine(current_embedding, average_condition)
                df_result.at[idx, condition_sim_col] = sim_score
    
    return df_result

def calculate_cross_column_similarity(df: pd.DataFrame, source_embeddings: List, source_valid_indices: List[int],
                                     target_embeddings: List, target_valid_indices: List[int], 
                                     condition_column: str, output_column: str) -> pd.DataFrame:
    """
    Calculate similarity between source embeddings and centroid of target embeddings within same condition
    
    Args:
        df (pd.DataFrame): DataFrame to add similarity scores to
        source_embeddings (List): Source embeddings (e.g., refined idea embeddings)
        source_valid_indices (List[int]): Valid indices for source embeddings
        target_embeddings (List): Target embeddings (e.g., initial idea embeddings)
        target_valid_indices (List[int]): Valid indices for target embeddings
        condition_column (str): Column name containing conditions
        output_column (str): Name for the output similarity column
        
    Returns:
        pd.DataFrame: DataFrame with added cross-column similarity scores
    """
    df_result = df.copy()
    df_result[output_column] = np.nan
    
    logger.info(f"Calculating cross-column similarity for {output_column}...")
    
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
                
            # Calculate cosine similarity
            sim_score = 1 - cosine(source_embedding, target_centroid)
            df_result.at[idx, output_column] = sim_score
        
        logger.info(f"Processed condition {condition}: {len(target_condition_embeddings)} target embeddings, "
                   f"{len(source_condition_indices)} source embeddings")
    
    return df_result

def process_text_column_similarities(df: pd.DataFrame, text_column: str, condition_column: str, prefix: str) -> pd.DataFrame:
    """
    Process similarities for a single text column (combines embedding calculation and similarity scoring)
    
    Args:
        df (pd.DataFrame): DataFrame containing the text column
        text_column (str): Name of the text column to process
        condition_column (str): Column name containing conditions for within-condition similarity
        prefix (str): Prefix for the output column names
        
    Returns:
        pd.DataFrame: DataFrame with added similarity scores and embeddings
    """
    # Calculate embeddings
    embeddings, valid_indices = calculate_embeddings_for_column(df, text_column)
    
    # Add embeddings to dataframe
    embedding_col = f"{prefix}_embedding"
    df[embedding_col] = embeddings
    
    # Calculate similarity scores
    df_result = calculate_similarity_scores(df, embeddings, valid_indices, condition_column, prefix)
    
    return df_result

def main():
    """
    Main function to run robustness checks with summary analysis
    """
    # Check if OpenAI API key is set
    if not client.api_key:
        logger.error("OpenAI API key not found in backend/.env file.")
        logger.info("Please make sure OPENAI_API_KEY is set in ../backend/.env")
        return
    
    # Load the data
    logger.info("Loading data from Data/data_clean.csv...")
    try:
        df = pd.read_csv('Data/data_clean.csv')
        logger.info(f"Loaded {len(df)} rows from Data/data_clean.csv")
    except FileNotFoundError:
        logger.error("Data/data_clean.csv not found. Please make sure the file exists.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Show distribution by condition
    logger.info("Distribution by bot_type:")
    for condition, count in df['bot_type'].value_counts().items():
        logger.info(f"  {condition}: {count}")
    
    # Generate summaries for both idea and refined_idea columns
    logger.info("\n=== ROBUSTNESS CHECK: GENERATING SUMMARIES ===")
    df = generate_summaries_for_columns(df, ['idea', 'refined_idea'])
    
    # Process similarities for summary columns only (as robustness check)
    text_columns_to_process = [
        ('idea_summary', 'idea_summary'),
        ('refined_idea_summary', 'refined_summary')
    ]
    
    for text_column, prefix in text_columns_to_process:
        if text_column in df.columns:
            # Only process columns that have some non-null values
            non_null_count = df[text_column].notna().sum()
            if non_null_count > 0:
                logger.info(f"\n=== PROCESSING SIMILARITIES FOR {text_column.upper()} (ROBUSTNESS CHECK) ===")
                logger.info(f"Found {non_null_count} non-null values in {text_column}")
                df = process_text_column_similarities(df, text_column, 'bot_type', prefix)
            else:
                logger.info(f"Skipping {text_column} - no non-null values found")
        else:
            logger.warning(f"Column {text_column} not found in DataFrame")
    
    # Calculate cross-column similarities for summaries: refined summaries vs initial summary centroids
    logger.info(f"\n=== CALCULATING CROSS-COLUMN SIMILARITIES FOR SUMMARIES (ROBUSTNESS CHECK) ===")
    
    # Check if both summary embeddings exist
    if 'idea_summary_embedding' in df.columns and 'refined_summary_embedding' in df.columns:
        logger.info("Calculating refined summary similarity to initial summary centroids within condition...")
        
        idea_summary_embeddings = df['idea_summary_embedding'].tolist()
        refined_summary_embeddings = df['refined_summary_embedding'].tolist()
        
        idea_summary_valid_indices = [i for i, emb in enumerate(idea_summary_embeddings) if emb is not None]
        refined_summary_valid_indices = [i for i, emb in enumerate(refined_summary_embeddings) if emb is not None]
        
        # Calculate similarity between refined summaries and initial summary centroids
        df = calculate_cross_column_similarity(
            df, 
            source_embeddings=refined_summary_embeddings,
            source_valid_indices=refined_summary_valid_indices,
            target_embeddings=idea_summary_embeddings,
            target_valid_indices=idea_summary_valid_indices,
            condition_column='bot_type',
            output_column='refined_summary_sim_to_initial_summary_centroid'
        )
    else:
        logger.warning("Could not calculate cross-column summary similarity - missing summary embeddings")
    
    # Save robustness check results
    output_file = 'Data/robustness_checks_with_summaries.csv'
    
    # Save robustness check data WITH summary embeddings
    df.to_csv(output_file, index=False)
    logger.info(f"Robustness check results saved to {output_file}")
    logger.info(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Print summary embeddings analysis
    logger.info("\n=== ROBUSTNESS CHECK: SUMMARY EMBEDDING ANALYSIS ===")
    summary_embedding_columns = [col for col in df.columns if col.endswith('_summary_embedding')]
    for col in summary_embedding_columns:
        valid_embeddings = df[col].dropna()
        if len(valid_embeddings) > 0:
            # Get first valid embedding to check dimensions
            first_valid = next(emb for emb in valid_embeddings if emb is not None)
            embedding_dim = len(first_valid) if first_valid is not None else 0
            
            logger.info(f"{col}: {len(valid_embeddings)}/{len(df)} valid ({len(valid_embeddings)/len(df)*100:.1f}%), {embedding_dim} dimensions")
    
    # Print summary similarity statistics
    logger.info("\n=== ROBUSTNESS CHECK: SUMMARY SIMILARITY STATISTICS ===")
    summary_similarity_columns = [col for col in df.columns if '_summary_sim_' in col or 'summary_sim_' in col]
    
    for col in summary_similarity_columns:
        valid_scores = df[col].dropna()
        if len(valid_scores) > 0:
            logger.info(f"{col}: Mean={valid_scores.mean():.3f}, Std={valid_scores.std():.3f}, N={len(valid_scores)}")
    
    # Compare summary vs original text similarities (robustness check)
    logger.info("\n=== ROBUSTNESS CHECK: COMPARING SUMMARY VS ORIGINAL TEXT SIMILARITIES ===")
    
    # Compare idea similarities
    if 'idea_sim_allideas' in df.columns and 'idea_summary_sim_allideas' in df.columns:
        original_scores = df['idea_sim_allideas'].dropna()
        summary_scores = df['idea_summary_sim_allideas'].dropna()
        
        if len(original_scores) > 0 and len(summary_scores) > 0:
            correlation = np.corrcoef(
                df.loc[df['idea_sim_allideas'].notna() & df['idea_summary_sim_allideas'].notna(), 'idea_sim_allideas'],
                df.loc[df['idea_sim_allideas'].notna() & df['idea_summary_sim_allideas'].notna(), 'idea_summary_sim_allideas']
            )[0, 1]
            
            logger.info(f"Idea similarities correlation (original vs summary): {correlation:.3f}")
            logger.info(f"Original idea similarities - Mean: {original_scores.mean():.3f}, Std: {original_scores.std():.3f}")
            logger.info(f"Summary idea similarities - Mean: {summary_scores.mean():.3f}, Std: {summary_scores.std():.3f}")
    
    # Compare refined idea similarities
    if 'refined_sim_allideas' in df.columns and 'refined_summary_sim_allideas' in df.columns:
        original_scores = df['refined_sim_allideas'].dropna()
        summary_scores = df['refined_summary_sim_allideas'].dropna()
        
        if len(original_scores) > 0 and len(summary_scores) > 0:
            correlation = np.corrcoef(
                df.loc[df['refined_sim_allideas'].notna() & df['refined_summary_sim_allideas'].notna(), 'refined_sim_allideas'],
                df.loc[df['refined_sim_allideas'].notna() & df['refined_summary_sim_allideas'].notna(), 'refined_summary_sim_allideas']
            )[0, 1]
            
            logger.info(f"Refined idea similarities correlation (original vs summary): {correlation:.3f}")
            logger.info(f"Original refined similarities - Mean: {original_scores.mean():.3f}, Std: {original_scores.std():.3f}")
            logger.info(f"Summary refined similarities - Mean: {summary_scores.mean():.3f}, Std: {summary_scores.std():.3f}")

if __name__ == "__main__":
    main()
