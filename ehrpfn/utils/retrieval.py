from typing import *
import json
import numpy as np
import torch
import faiss


class Retriever:
    k: int = 1024
    _is_fitted: bool = False
    return_distances: bool = False
    def fit(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def retrieve(self, queries, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class FAISS(Retriever):
    def __init__(self, k:Union[float,int], distance:str='L2', max_context_size:int=None, return_distances:bool=False):
        self.k = k
        self.distance = distance
        self.max_context_size = max_context_size
        self._is_fitted = False
        self.return_distances = return_distances

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

        if self.return_distances:
            distances_Xs, indices_Xs = knns
            return distances_Xs, indices_Xs
        else:
            indices_Xs = knns[1]
            return indices_Xs

class WeightedRetriever(Retriever):
    """
    Retrieve similar instances based on a weighted distance metric.
    Weights are applied as feature scaling before indexing in FAISS.
    """
    def __init__(self, k:int, weights: np.ndarray, distance: str = 'L2', return_distances: bool = False):
        self.k = k
        self.retriever = FAISS(k=k, distance=distance, return_distances=return_distances)
        self.weights = weights
        if isinstance(weights, list):
            self.weights = np.array(weights)

    def fit(self, X: np.ndarray, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        assert isinstance(X, np.ndarray), f"X must be a numpy array. Got {type(X)}"


        feature_order = kwargs.get('feature_order', None)
        if feature_order is not None:
            weights = self.weights[feature_order]
        else:
            weights = self.weights

        assert X.shape[1] == weights.shape[0], f"weights must match feature dimension. Got {weights.shape[0]} weights for {X.shape[1]} features"

        # scale features by sqrt(weights) so that l2 distance corresponds to weighted l2
        self.X_scaled = X * np.sqrt(weights)
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

        feature_order = kwargs.get('feature_order', None)
        if feature_order is not None:
            weights = self.weights[feature_order]
        else:
            weights = self.weights
        assert queries.shape[1] == weights.shape[0], "Query feature dimension must match weights"

        # scale queries the same way as database
        queries_scaled = queries * np.sqrt(weights)
        indices = self.retriever.retrieve(queries_scaled)

        return indices

class EmbeddingRetriever(Retriever):
    def __init__(self, k:int, distance:str='L2', return_distances:bool=False):
        self.k = k
        self.distance = distance
        self.retriever = FAISS(k=k, distance=distance, return_distances=return_distances)
        self._is_fitted = False

    def fit(self, X, y=None, **kwargs):
        self.retriever.fit(kwargs.get('X_embs_train', X))  
        self._is_fitted = True

    def retrieve(self, queries, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("EmbeddingRetriever is not fitted yet.")
        return self.retriever.retrieve(kwargs.get('X_embs_test', queries))

class RandomRetriever(Retriever):
    def __init__(self, k:int, **kwargs):
        self.k = k

    def fit(self,X, y=None, **kwargs):
        self.context = X
        self._is_fitted = True

    def retrieve(self, queries, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("RandomRetriever is not fitted yet.")
        return np.random.choice(self.context.shape[0], size=self.k, replace=False).reshape(1, -1).repeat(queries.shape[0], axis=0)

def get_retriever(method:str, k:int, **kwargs):
    if method in ["faiss", 'knn']:
        retriever = FAISS(
            k=k, distance=kwargs.get('distance', 'L2'), 
            max_context_size=kwargs.get('max_context_size', None),
            return_distances=kwargs.get('return_distances', False)
        )
    elif method in ["dbr", "embeddings"]:
        retriever = EmbeddingRetriever(
            k=k, distance=kwargs.get('distance', 'L2'),
            return_distances=kwargs.get('return_distances', False)
        )
    elif method in ['weighted']:
        assert 'weights' in kwargs, "weights must be provided for weighted retrieval"
        retriever = WeightedRetriever(
            k=k,
            weights=kwargs.get('weights'),
            distance=kwargs.get('distance', 'L2'),
            return_distances=kwargs.get('return_distances', False)
        )
    elif method == 'random':
        retriever = RandomRetriever(k=k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    
    return retriever




def get_retrieval_weight(weighted_type:str, data_name:str, task:str, feature_names:Optional[List[str]]):

    if weighted_type == 'feattype':
        if data_name == 'mimicpipe':
            feature_types = json.load(open(f"/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/feature_types_all.json", 'r'))
        else:
            feature_types = json.load(open(f"/home/mpham/workspace/source/ehrpfn/data/processed/eicu/feature_types.json", 'r'))

        if task not in ['Readmission', 'Length of Stay', 'Mortality', 'READMISSION', 'MORTALITY']:
            mapping_weights = {
                "Diagnosis":0.15,
                "Procedure":0.15,
                "Medication":0.15,
                "Infusion":0.05,
                "Chart":0.40,
                "Demographics":0.10, 
                "Procedure ICD": 0.15,
                "Vital": 0.40
            }
        else:
            mapping_weights = {
                "Diagnosis":0.35,
                "Procedure":0.10,
                "Medication":0.20,
                "Infusion":0.05,
                "Chart":0.20,
                "Demographics":0.10,
                "Procedure ICD": 0.35,
                "Vital": 0.2
            }

        retrieval_weights = [mapping_weights[feature_types[name]] for name in feature_names]
    
    elif weighted_type == 'logistics':
        retrieval_weights = None
        try:
            retrieval_dict = json.load(open(
            f"/home/mpham/workspace/source/ehrpfn/outputs/all/results/{data_name}/{task}/logistics_{data_name}_{task}_sel:shap_500/logistics_params.json"
            ))
            retrieval_weights = retrieval_dict["coef"][0]
            retrieval_weights = np.absolute(retrieval_weights)
            retrieval_names = retrieval_dict["feature_names"]
        except:
            raise FileNotFoundError("Logistics weights not found, consider running first with method=logistics")
        # retrieval_weights = [retrieval_weights[i] for i in range(len(feature_names))]
        retrieval_weights = [retrieval_weights[retrieval_names.index(name)] if name in retrieval_names else 0.0 for name in feature_names]

    else:
        raise ValueError(f"Unknown weighted type: {weighted_type}")
    
    return retrieval_weights