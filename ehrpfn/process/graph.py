import numpy as np
import polars as pl
import Levenshtein
import multiprocessing as mp

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

def calculate_ed_query_to_pool(args):
    query_idx, query_pathway, pool_pathways, return_sorted, k = args
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
    
    return query_idx, results
    

def calculate_ed_queries_to_pool(query_pathways, pool_pathways, k=None, return_sorted=True):
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
    for query_id, query_pathway in enumerate(query_pathways):
        _, result = calculate_ed_query_to_pool((query_id, query_pathway, pool_pathways, return_sorted, k))
        results.append(result)
    
    return results


def _calculate_ed_worker(args):
    """
    Worker function for parallel edit distance calculation.
    """
    query_idx, query_pathway, pool_pathways, return_sorted, k = args
    
    query_pathway = query_pathway[-1]  # Use the last element as the pathway
    if isinstance(query_pathway, np.ndarray):
        query_pathway = query_pathway.tolist()
    
    results = []
    query_len = len(query_pathway)
    
    for pool_idx, pool_pathway in enumerate(pool_pathways):
        if len(pool_pathway):  # Only process non-empty pathways
            pool_pathway = pool_pathway[-1]  # Use the last element as the pathway
            if isinstance(pool_pathway, np.ndarray):
                pool_pathway = pool_pathway.tolist()
            
            pool_len = len(pool_pathway)
            
            # Early termination for top-k: if length difference > current k-th best, skip
            if k and len(results) >= k:
                current_kth_best = sorted(results, key=lambda x: x[1])[k-1][1]
                if abs(query_len - pool_len) > current_kth_best:
                    continue

            distance = Levenshtein.distance(query_pathway, pool_pathway)
            results.append((pool_idx, distance))
            
            # Keep only top k*2 to avoid frequent sorting
            if k and len(results) > k * 2:
                results = sorted(results, key=lambda x: x[1])[:k]
        else:
            results.append((pool_idx, float('inf')))
    
    # Final sort and trim to k if specified
    if return_sorted:
        results.sort(key=lambda x: x[1])
    if k:
        results = results[:k]
    
    return query_idx, results

def calculate_ed_queries_to_pool_parallel(query_pathways, pool_pathways, return_sorted=True, k=None, n_jobs=None):
    """
    Calculate edit distance from multiple query pathways to all pathways in the pool using parallel processing.
    
    Args:
        query_pathways: List of lists of integers representing the query pathways
        pool_pathways: List of lists of integers representing the pathway pool
        return_sorted: Whether to return results sorted by edit distance score
        k: If specified, only return top k results per query (faster)
        n_jobs: Number of parallel processes (default: CPU count)
        
    Returns:
        List of lists of tuples (pool_index, distance_score) for each query pathway
    """
   
    
    # Parallel processing for larger datasets
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), len(query_pathways))
    
    # Prepare arguments for each query
    worker_args = []
    for query_idx, query_pathway in enumerate(query_pathways):
        # worker_args.append((query_idx, query_pathway, pool_pathways, return_sorted, k))
        worker_args.append((query_idx, query_pathway, pool_pathways, return_sorted, k))
    
    # Process queries in parallel
    with mp.Pool(processes=n_jobs) as pool:
        worker_results = pool.map(calculate_ed_query_to_pool, worker_args)
    # Sort results by original query order
    worker_results.sort(key=lambda x: x[0])
    results = [result for _, result in worker_results]
    
    return results
