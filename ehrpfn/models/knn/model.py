import numpy as np
from typing import Optional, Any
from ehrpfn.utils.retrieval import Retriever
from sklearn.base import ClassifierMixin, RegressorMixin

class SimpleKNNClassifier(ClassifierMixin):
    def __init__(self, retriever: Retriever):
        super().__init__()
        self.retriever = retriever
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.retriever.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("KNNClassifier is not fitted yet.")
        indices = self.retriever.retrieve(X)
        # indices shape: (n_samples, k)
        if indices.ndim == 1:
            indices = indices[None, :]
        neighbor_labels = self.y[indices]
        # Majority vote
        preds = np.array([np.bincount(row.astype(int)).argmax() for row in neighbor_labels])
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("KNNClassifier is not fitted yet.")
        indices = self.retriever.retrieve(X)
        if indices.ndim == 1:
            indices = indices[None, :]
        neighbor_labels = self.y[indices]
        classes = np.unique(self.y)
        proba = np.zeros((X.shape[0], len(classes)))
        for i, row in enumerate(neighbor_labels):
            counts = np.bincount(row.astype(int), minlength=len(classes))
            proba[i] = counts / counts.sum()
        return proba

class SimpleKNNRegressor(RegressorMixin):
    def __init__(self, retriever: Retriever):
        super().__init__()
        self.retriever = retriever
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.retriever.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("KNNRegressor is not fitted yet.")
        indices = self.retriever.retrieve(X)
        if indices.ndim == 1:
            indices = indices[None, :]
        neighbor_targets = self.y[indices]
        preds = neighbor_targets.mean(axis=1)
        return preds