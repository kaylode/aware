import numpy as np
import torch
import pandas as pd
import polars as pl
import faiss
from ehrpfn.process.graph import convert_pathways_to_int_lists, calculate_ed_queries_to_pool

class FAISS:
    def __init__(self, X):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)

    def retrieve(self, queries, k, **kwargs):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)
        assert isinstance(k, int)

        # KNN retrieval
        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs



class ProcessRetriever:
    """
    Retrieval based on edit distance between patient activity pathways
    """
    def __init__(self, X, X_processes, pathway_column):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        self.X = X
        self.X_processes = X_processes
        self.n_instances, self.n_features = X.shape
        self.pathway_column = pathway_column

        self.pool_pathways = convert_pathways_to_int_lists(
            X_processes, 
            pathway_column=pathway_column
        )

    def retrieve(self, queries, queries_processes, k, **kwargs):

        query_pathways = convert_pathways_to_int_lists(
            queries_processes, 
            pathway_column=self.pathway_column
        )

        lev_results = calculate_ed_queries_to_pool(query_pathways, self.pool_pathways, return_sorted=True)

        results = []
        for query_results in lev_results:
            indices = []
            for pool_idx, distance in query_results:
                indices.append(pool_idx)
            indices = indices[:k]
            results.append(indices)
        results = np.array(results)
        return results


class TrajectoriesRetriever:
    def __init__(self, X, X_traj):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        self.X = X
        self.X_traj = X_traj
        self.n_instances, self.n_features = X.shape
        assert self.X_traj is not None, "X_traj must not be None"
        self.faiss = FAISS(self.X_traj)

    def retrieve(self, queries, queries_traj, k, **kwargs):
        return self.faiss.retrieve(queries_traj, k)
    
class ProcessTFIDFRetriever:
    def __init__(self, X, X_processes: pd.DataFrame):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        self.X = X
        self.X_processes = X_processes
        if isinstance(self.X_processes, pl.DataFrame):
            self.X_processes = self.X_processes.to_pandas()
        
        self.cluster_to_indices = {}
        unique_clusters = self.X_processes['cluster_id'].unique()
        for cluster_id in unique_clusters:
            indices = self.X_processes[self.X_processes['cluster_id'] == cluster_id].index.tolist()
            self.cluster_to_indices[cluster_id] = indices
        self.n_instances, self.n_features = X.shape

    def retrieve(self, queries, queries_processes, k, **kwargs):
        if kwargs.get('cluster_id') is None:
            results = []
            for cluster_id in queries_processes['cluster_id']:
                indices = self.cluster_to_indices.get(cluster_id, [])[:k]
                results.append(indices)
            results = np.array(results)
        else:
            results = []
            cluster_id = kwargs['cluster_id']
            indices = self.cluster_to_indices.get(cluster_id, [])[:k]
            results = np.array([indices] * len(queries))
            
        return results