from typing import *
import json
import numpy as np
import torch
import faiss

import numpy as np
from .base import Retriever


class FAISS(Retriever):
    def __init__(self, 
                 k:Union[float,int], distance:str='L2', 
                 field_to_use:str='raw_feature',
                 return_distances:bool=False):
        self.k = k
        self.distance = distance
        self.field_to_use = field_to_use
        self._is_fitted = False
        self.return_distances = return_distances

    def fit(self, items:List[Dict], **kwargs):
        # Extract features from items based on field_to_use
        X = np.array([item[self.field_to_use] for item in items])

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        assert isinstance(X, np.ndarray), f"X must be a numpy array. Got {type(X)}"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)

        if self.distance.lower() == 'l2':
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif self.distance.lower() == 'cosine':
            self.index = faiss.IndexFlatIP(X.shape[1])
        elif self.distance.lower() == 'partitioned_l2':
            nlist = kwargs.get('nlist', 100)
            m = kwargs.get('m', 8)  # number of subquantizers
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(X.shape[1]), 
                X.shape[1], nlist, faiss.METRIC_L2
            )
            self.index.train(X)
        elif self.distance.lower() == 'partitioned_cosine':
            nlist = kwargs.get('nlist', 100)
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(X.shape[1]), 
                X.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(X)
            self.index.nprobe = kwargs.get('nprobe', 10)
        elif self.distance.lower() == 'partitioned_quantized_l2':
            nlist = kwargs.get('nlist', 100)
            m = kwargs.get('m', 8)  # number of subquantizers
            bits = kwargs.get('bits', 8)  # number of bits in each centroid
            quantizer = faiss.IndexFlatL2(X.shape[1])  # we keep the same L2 distance flat index
            self.index = faiss.IndexIVFPQ(quantizer, X.shape[1], nlist, m, bits)
            self.index.train(X)
            self.index.nprobe = kwargs.get('nprobe', 10)
        else:
            raise ValueError(f'Unknown distance metric: {self.distance}')

        self.index.add(X)
        self._is_fitted = True

    def retrieve(self, queries:List[Dict], **kwargs):
        
        queries = np.array([query[self.field_to_use] for query in queries])

        if not self._is_fitted:
            raise RuntimeError("FAISS index is not fitted yet.")
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)

        if kwargs.get('k', None):
            k = kwargs.get('k')
        else:
            k = self.k

        # KNN retrieval
        if isinstance(k, float):
            k = int(k * self.index.ntotal)
            
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
    def __init__(self, k:int, weights: np.ndarray, field_to_use: str = 'raw_feature', distance: str = 'L2', return_distances: bool = False):
        self.k = k
        self.field_to_use = field_to_use
        self.return_distances = return_distances
        self.retriever = FAISS(k=k, distance=distance, field_to_use=field_to_use, return_distances=return_distances)
        self.weights = weights
        if isinstance(weights, list):
            self.weights = np.array(weights)

    def fit(self, items:List[Dict], **kwargs):
        X = np.array([item[self.retriever.field_to_use] for item in items])
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
        self.retriever.fit([{self.retriever.field_to_use: x} for x in self.X_scaled])
        self._is_fitted = True

    def retrieve(self, queries: List[Dict], **kwargs):
        """
        Retrieve k nearest neighbors for each query.
        queries: 2D numpy array (n_queries, n_features)
        Returns: distances, indices
        """

        queries = np.array([query[self.retriever.field_to_use] for query in queries])
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
        indices = self.retriever.retrieve([{self.retriever.field_to_use: x} for x in queries_scaled])

        if self.return_distances:
            distances, indices = indices
            return distances, indices
        else:
            return indices


class RandomRetriever(Retriever):
    def __init__(self, k:int, field_to_use: str = 'raw_feature', **kwargs):
        self.k = k
        self.field_to_use = field_to_use

    def fit(self, items:List[Dict], **kwargs):
        self.context = np.array([item[self.field_to_use] for item in items])
        self._is_fitted = True

    def retrieve(self, queries:List[Dict], **kwargs):
        if not self._is_fitted:
            raise RuntimeError("RandomRetriever is not fitted yet.")

        retrieved = []
        for query in queries:
            indexes = np.random.choice(list(range(self.context.shape[0])), size=self.k, replace=False).reshape(1, -1)
            retrieved.append(indexes)
        return np.vstack(retrieved)

class BaselineRetriever(Retriever):
    """
    Docstring for BaselineRetriever

    A retriever that randomly selects shared k samples from the fitted context for all queries at once.
    """

    def __init__(self, k:int, field_to_use: str = 'raw_feature', **kwargs):
        self.k = k
        self.field_to_use = field_to_use

    def fit(self, items:List[Dict], **kwargs):
        self.context = np.array([item[self.field_to_use] for item in items])
        self._is_fitted = True

    def retrieve(self, queries:List[Dict], **kwargs):
        if not self._is_fitted:
            raise RuntimeError("RandomRetriever is not fitted yet.")

        indexes = np.random.choice(list(range(self.context.shape[0])), size=self.k, replace=False).reshape(1, -1)
        retrieved = np.repeat(indexes, repeats=len(queries), axis=0)
        return retrieved


def get_ehr_retrieval_weight(weighted_type:str, task:str, feature_names:Optional[List[str]], json_weight_path=None):

    if weighted_type == 'feattype':
        if task not in ['Readmission', 'Length of Stay', 'Mortality', 'READMISSION', 'MORTALITY']:
            mapping_weights = {
                "Diagnosis":0.15,
                "Procedure":0.15,
                "Medication":0.15,
                "Infusion":0.05,
                "Chart":0.40,
                "Demographics":0.10, 
                "Procedure ICD": 0.15,
                "Vital": 0.40,
                "Other": 0.1
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
                "Vital": 0.2,
                "Other": 0.1
            }

        feature_types = {}
        for name in feature_names:
            if name.startswith(('ICD//CM', "ICD10//CM")):
                feature_types[name] = 'Diagnosis'
            elif name.startswith(('ICD//PCS')):
                feature_types[name] = 'Procedure ICD'
            elif name.startswith(('PROCEDURES//', "TREATMENT//")):
                feature_types[name] = 'Procedure'
            elif name.startswith(('MED//', "MEDICATIONS//")):
                feature_types[name] = 'Medication'
            elif name.startswith('INFUSION//'):
                feature_types[name] = 'Infusion'
            elif name.startswith(('CHART//', 'LAB//')):
                feature_types[name] = 'Chart'
            elif name.startswith('VITAL//'):
                feature_types[name] = 'Vital'
            elif name.startswith('DEMOGRAPHICS//'):
                feature_types[name] = 'Demographics'
            elif name.startswith(("ICD//")):
                feature_types[name] = 'Diagnosis'
            else:
                feature_types[name] = 'Other'

        retrieval_weights = [mapping_weights[feature_types[name]] for name in feature_names]
    
    elif weighted_type == 'logistics':
        retrieval_weights = None
        try:
            retrieval_dict = json.load(open(json_weight_path, 'r'))
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