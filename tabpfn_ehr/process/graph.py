from typing import List
import networkx as nx
import numpy as np
import polars as pl
import Levenshtein

# Alternative approach using apply and storing as lists of integers
def convert_pathways_to_int_lists(encoded_process_df, pathway_column=None):
    """
    Convert pathway columns to lists of integers (more Polars-friendly).
    """
    df_with_lists = encoded_process_df.clone()
    df_with_lists = df_with_lists.with_columns(
        pl.col(pathway_column).map_elements(
            lambda x: [int(token) for token in x.replace(" [SEP] ", " 2 ").split() if token.strip()] if x else [],
            return_dtype=pl.List(pl.Int32)
        ).alias(f"{pathway_column}_list")
    )
    arrays = df_with_lists.select([f"{pathway_column}_list"]).to_numpy().tolist()
    return arrays

def calculate_ed_query_to_pool(query_pathway, pool_pathways, return_sorted=True):
    """
    Calculate edit distance from a query pathway to all pathways in the pool.
    
    Args:
        query_pathway: List of integers representing the query pathway
        pool_pathways: List of lists of integers representing the pathway pool
        return_sorted: Whether to return results sorted by edit distance score
        method: 'ged' for graph edit distance or 'levenshtein' for Levenshtein distance
        
    Returns:
        List of tuples (pool_index, distance_score) 
    """
    
    # Use Levenshtein distance - much faster
    
    results = []
    query_pathway = query_pathway[-1]  # Use the last element as the pathway
    if isinstance(query_pathway, np.ndarray):
        query_pathway = query_pathway.tolist()

    for pool_idx, pool_pathway in enumerate(pool_pathways):
        if len(pool_pathway):  # Only process non-empty pathways
            pool_pathway= pool_pathway[-1]  # Use the last element as the pathway
            if isinstance(pool_pathway, np.ndarray):
                pool_pathway = pool_pathway.tolist()

            distance = Levenshtein.distance(query_pathway, pool_pathway)
            results.append((pool_idx, distance))
        else:
            results.append((pool_idx, float('inf')))
    
    # Sort by distance if requested (ascending - lower is more similar)
    if return_sorted:
        results.sort(key=lambda x: x[1])
    
    return results
    

def calculate_ed_queries_to_pool(query_pathways, pool_pathways, return_sorted=True):
    """
    Calculate edit distance from multiple query pathways to all pathways in the pool.
    
    Args:
        query_pathways: List of lists of integers representing the query pathways
        pool_pathways: List of lists of integers representing the pathway pool
        return_sorted: Whether to return results sorted by edit distance score
        
    Returns:
        List of lists of tuples (pool_index, distance_score) for each query pathway
    """
    
    results = []
    for query_pathway in query_pathways:
        result = calculate_ed_query_to_pool(query_pathway, pool_pathways, return_sorted)
        results.append(result)
    
    return results


def main(encoded_process_df):
    """Example of how to use the edit distance functions."""

    process_arrays = convert_pathways_to_int_lists(
        encoded_process_df, 
        pathway_columns=[
            'pathway_encoded',
            'pathway_with_prev_1_encoded',
            'pathway_with_prev_2_encoded'
        ]
    )

    sample_pathway = process_arrays[0][-1]  # Example pathway as list of integers
    pool_pathways = [i[-1] for i in process_arrays[1:]]  # All other pathways
    
    lev_results = calculate_ed_query_to_pool(sample_pathway, pool_pathways)
    return lev_results
