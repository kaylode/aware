from typing import *
import json
import numpy as np
import torch
import faiss


class Retriever:
    k: int = 1024
    _is_fitted: bool = False
    def fit(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def retrieve(self, queries,  **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class FAISS(Retriever):
    def __init__(self, k:Union[float,int], distance:str='L2', max_context_size:int=None):
        self.k = k
        self.distance = distance
        self.max_context_size = max_context_size
        self._is_fitted = False

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        assert isinstance(X, np.ndarray), f"X must be a numpy array. Got {type(X)}"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)

        if self.distance == 'L2':
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif self.distance == 'cosine':
            self.index = faiss.IndexFlatIP(X.shape[1])
        else:
            raise ValueError(f'Unknown distance metric: {self.distance}')

        self.index.add(X)
        self._is_fitted = True

    def retrieve(self, queries, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("FAISS index is not fitted yet.")
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)

        # KNN retrieval
        if isinstance(self.k, float) and self.max_context_size is not None:
            k = min(int(self.k * self.index.ntotal), self.max_context_size)
        else:
            k = self.k
        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs

class WeightedRetriever(Retriever):
    """
    Retrieve similar instances based on a weighted distance metric.
    Weights are applied as feature scaling before indexing in FAISS.
    """
    def __init__(self, k:int, weights: np.ndarray, distance: str = 'L2'):
        self.k = k
        self.retriever = FAISS(k=k, distance=distance)
        self.weights = weights
        if isinstance(weights, list):
            self.weights = np.array(weights)

    def fit(self, X: np.ndarray, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        assert isinstance(X, np.ndarray), f"X must be a numpy array. Got {type(X)}"
        assert X.shape[1] == self.weights.shape[0], "weights must match feature dimension"

        # scale features by sqrt(weights) so that l2 distance corresponds to weighted l2
        self.X_scaled = X * np.sqrt(self.weights)
        self.retriever.fit(self.X_scaled)
        self._is_fitted = True

    def retrieve(self, queries: np.ndarray, **kwargs):
        """
        Retrieve k nearest neighbors for each query.
        queries: 2D numpy array (n_queries, n_features)
        Returns: distances, indices
        """
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        assert queries.shape[1] == self.weights.shape[0], "Query feature dimension must match weights"

        # scale queries the same way as database
        queries_scaled = queries * np.sqrt(self.weights)
        indices = self.retriever.retrieve(queries_scaled)

        return indices

class EmbeddingRetriever(Retriever):
    def __init__(self, k:int, distance:str='L2'):
        self.k = k
        self.distance = distance
        self.retriever = FAISS(k=k, distance=distance)
        self._is_fitted = False

    def fit(self, X, y=None, **kwargs):
        self.retriever.fit(kwargs.get('X_embs_train', X))  
        self._is_fitted = True

    def retrieve(self, queries, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("EmbeddingRetriever is not fitted yet.")
        return self.retriever.retrieve(kwargs.get('X_embs_test', queries))

class RandomRetriever(Retriever):
    def __init__(self, k:int):
        self.k = k

    def fit(self,X, y=None, **kwargs):
        self.context = X
        self._is_fitted = True

    def retrieve(self, queries, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("RandomRetriever is not fitted yet.")
        return np.random.choice(self.context.shape[0], size=self.k, replace=False)

def get_retriever(method:str, k:int, **kwargs):
    if method in ["faiss", 'knn']:
        retriever = FAISS(k=k, distance=kwargs.get('distance', 'L2'), max_context_size=kwargs.get('max_context_size', None))
    elif method in ["dbr", "embeddings"]:
        retriever = EmbeddingRetriever(k=k, distance=kwargs.get('distance', 'L2'))
    elif method in ['weighted']:
        assert 'weights' in kwargs, "weights must be provided for weighted retrieval"
        retriever = WeightedRetriever(
            k=k,
            weights=kwargs.get('weights'),
            distance=kwargs.get('distance', 'L2')
        )
    elif method == 'random':
        retriever = RandomRetriever(k=k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    
    return retriever




def get_retrieval_weight(weighted_type:str, task:str, feature_names:Optional[List[str]]):

    if weighted_type == 'feattype':
        feature_types = json.load(open("/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/feature_types2.json", 'r'))

        if task in ['Readmission', 'Length of Stay', 'Mortality']:
            mapping_weights = {
                "Diagnosis":0.15,
                "Procedure":0.15,
                "Medication":0.15,
                "Infusion":0.05,
                "Chart":0.40,
                "Demographics":0.10, 
                "Procedure ICD": 0.15
            }
        else:
            mapping_weights = {
                "Diagnosis":0.35,
                "Procedure":0.10,
                "Medication":0.20,
                "Infusion":0.05,
                "Chart":0.20,
                "Demographics":0.10,
                "Procedure ICD": 0.35
            }

        retrieval_weights = [mapping_weights[feature_types[name]] for name in feature_names]
    
    elif weighted_type == 'logistics':
        retrieval_weights = None
        retrieval_weights = json.load(open(
           f"/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/results/mimicpipe/{task}/logistics_mimicpipe_{task}_sel:shap_100/logistics_params.json"
        ))["coef"][0]
        retrieval_weights = np.absolute(retrieval_weights)
    else:
        raise ValueError(f"Unknown weighted type: {weighted_type}")
    
    return retrieval_weights