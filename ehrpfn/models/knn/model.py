import numpy as np
import torch
from typing import Optional, Any
from ehrpfn.retrieval import Retriever
from sklearn.base import ClassifierMixin, RegressorMixin

class SimpleKNNClassifier(ClassifierMixin):
	def __init__(self, retriever: Retriever):
		super().__init__()
		self.retriever = retriever
		self._is_fitted = False

	def fit(self, X: np.ndarray, y: np.ndarray, retriever_additional_fit: Optional[dict] = None, **kwargs):
		self.X = X
		self.y = y

		fit_items = []
		for i, (item, label) in enumerate(zip(X, y)):
			fit_dict = {"raw_feature": item, "label": label}
			if retriever_additional_fit is not None:
				for key, value in retriever_additional_fit.items():
					if self.retriever.field_to_use == key:
						assert len(value) == X.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in fit_dict, f"Key {key} already exists in fit_dict"
					try:
						fit_dict[key] = value[i]
					except Exception as e:
						continue
			fit_items.append(fit_dict)
		self.retriever.fit(fit_items)

		self._is_fitted = True
		return self

	def predict(self, X: np.ndarray, retriever_additional_query: Optional[dict] = None, **kwargs) -> np.ndarray:
		if not self._is_fitted:
			raise RuntimeError("KNNClassifier is not fitted yet.")
		# indices = self.retriever.retrieve(X, **kwargs)

		query_items = []
		for i in range(X.shape[0]):
			if isinstance(X, torch.Tensor):
				query_dict = {"raw_feature": X[i].cpu().numpy()}
			else:
				query_dict = {"raw_feature": X[i]}
			if retriever_additional_query is not None:
				for key, value in retriever_additional_query.items():
					if self.retriever.field_to_use == key:
						assert len(value) == X.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in query_dict, f"Key {key} already exists in query_dict"
					query_dict[key] = value[i]
			query_items.append(query_dict)

		indices = self.retriever.retrieve(query_items)
		

		# indices shape: (n_samples, k)
		if indices.ndim == 1:
			indices = indices[None, :]
		neighbor_labels = self.y[indices]
		# Majority vote
		preds = np.array([np.bincount(row.astype(int)).argmax() for row in neighbor_labels])
		return preds

	def predict_proba(self, X: np.ndarray, retriever_additional_query: Optional[dict] = None, **kwargs) -> np.ndarray:
		if not self._is_fitted:
			raise RuntimeError("KNNClassifier is not fitted yet.")
		
		query_items = []
		for i in range(X.shape[0]):
			if isinstance(X, torch.Tensor):
				query_dict = {"raw_feature": X[i].cpu().numpy()}
			else:
				query_dict = {"raw_feature": X[i]}
			if retriever_additional_query is not None:
				for key, value in retriever_additional_query.items():
					assert len(value) == X.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in query_dict, f"Key {key} already exists in query_dict"
					query_dict[key] = value[i]
			query_items.append(query_dict)

		indices = self.retriever.retrieve(query_items)


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

	def fit(self, X: np.ndarray, y: np.ndarray, retriever_additional_fit: Optional[dict] = None, **kwargs):
		self.X = X
		self.y = y

		fit_items = []
		for i, (item, label) in enumerate(zip(X, y)):
			fit_dict = {"raw_feature": item, "label": label}
			if retriever_additional_fit is not None:
				for key, value in retriever_additional_fit.items():
					if self.retriever.field_to_use == key:
						assert len(value) == X.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in fit_dict, f"Key {key} already exists in fit_dict"
					try:
						fit_dict[key] = value[i]
					except Exception as e:
						continue
			fit_items.append(fit_dict)
		self.retriever.fit(fit_items)

		self._is_fitted = True
		return self

	def predict_proba(self, X: np.ndarray, retriever_additional_query: Optional[dict] = None, **kwargs) -> np.ndarray:
		if not self._is_fitted:
			raise RuntimeError("KNNClassifier is not fitted yet.")
		
		query_items = []
		for i in range(X.shape[0]):
			if isinstance(X, torch.Tensor):
				query_dict = {"raw_feature": X[i].cpu().numpy()}
			else:
				query_dict = {"raw_feature": X[i]}
			if retriever_additional_query is not None:
				for key, value in retriever_additional_query.items():
					assert len(value) == X.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in query_dict, f"Key {key} already exists in query_dict"
					query_dict[key] = value[i]
			query_items.append(query_dict)

		indices = self.retriever.retrieve(query_items)

		if indices.ndim == 1:
			indices = indices[None, :]
			
		neighbor_targets = self.y[indices]
		preds = neighbor_targets.mean(axis=1)
		return preds

	def predict(self, X: np.ndarray, retriever_additional_query: Optional[dict] = None, **kwargs) -> np.ndarray:
		return self.predict_proba(X, retriever_additional_query, **kwargs)