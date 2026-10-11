from typing import Callable, Dict, List, Optional
import math
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from .utils import convert_to_torch_tensor, pad_x, generate_random_permutation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ehrpfn.retrieval import Retriever
from tqdm import tqdm
from scipy.special import softmax


class AWAREEstimator(BaseEstimator):
	"""Sklearn-compatible base estimator wrapping KNNPFNAWAREModel or TabDPTAWAREModel.

	Handles preprocessing (impute + standardize), retriever fitting, and batched
	inference with optional ensembling. Subclassed into AWAREClassifier (adds softmax
	and num_classes tracking) and AWARERegressor (raw scalar output).
	"""
	_DEFAULT_DEVICE = "cuda:0"

	def __init__(
		self,
		mode: str,
		infer_batch_size: int = 512,
		n_ensembles: int = 1,
		temperature: float = 0.8,
		device: str = _DEFAULT_DEVICE,
		use_flash: bool = True,
		compile: bool = True,
		retriever: Retriever = None,
		fit_retriever_before_fit: bool = True,
		base_model=None,
	):
		assert mode in ("cls", "reg"), "mode must be 'cls' or 'reg'"
		self.mode = mode
		self.infer_batch_size = infer_batch_size
		self.device = device
		self.use_flash = use_flash
		self.retriever = retriever
		self.fit_retriever_before_fit = fit_retriever_before_fit
		self.n_ensembles = n_ensembles
		self.temperature = temperature
		self.model = base_model
		self.model.eval()
		self.max_features = self.model.num_features
		self.max_num_classes = self.model.n_out
		self.compile = compile

	# ------------------------------------------------------------------
	# Helpers shared by fit() and predict_proba()
	# ------------------------------------------------------------------

	def _build_fit_items(
		self, X: np.ndarray, y: np.ndarray, extra: Optional[Dict] = None
	) -> List[dict]:
		"""Build item-dicts for retriever.fit()."""
		items = []
		for i, (feat, label) in enumerate(zip(X, y)):
			d: dict = {"raw_feature": feat, "label": label}
			if extra:
				for key, vals in extra.items():
					assert len(vals) == len(X), f"Length of '{key}' must match X"
					d[key] = vals[i]
			items.append(d)
		return items

	def _build_query_items(
		self, X_ori: np.ndarray, n: int, extra: Optional[Dict] = None
	) -> List[dict]:
		"""Build item-dicts for retriever.retrieve()."""
		items = []
		for i in range(n):
			d: dict = {"raw_feature": X_ori[i]}
			if extra:
				for key, vals in extra.items():
					d[key] = vals[i]
			items.append(d)
		return items

	# ------------------------------------------------------------------
	# fit — shared between classifier and regressor
	# ------------------------------------------------------------------

	def fit(self, X: np.ndarray, y: np.ndarray, retriever_additional_fit: Dict = None):
		"""Validate inputs, fit retriever, impute/scale, and store training data.

		AWAREClassifier overrides this to also compute self.num_classes.
		"""
		assert isinstance(X, np.ndarray), "X must be a numpy array"
		assert isinstance(y, np.ndarray), "y must be a numpy array"
		assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
		assert X.ndim == 2, "X must be 2-D"
		assert y.ndim == 1, "y must be 1-D"

		if self.fit_retriever_before_fit and self.retriever is not None:
			self.retriever.fit(self._build_fit_items(X, y, retriever_additional_fit))
			if getattr(self.retriever, "move_to_cpu", None) is not None:
				self.retriever.move_to_cpu()

		self.imputer = SimpleImputer(strategy="mean")
		X = self.imputer.fit_transform(X)
		self.scaler = StandardScaler()
		X = self.scaler.fit_transform(X)

		self.n_instances, self.n_features = X.shape
		self.X_train = X
		self.y_train = y
		self.is_fitted_ = True
		if self.compile:
			self.model = torch.compile(self.model)

	# ------------------------------------------------------------------
	# Shared pre-processing step for predict_proba
	# ------------------------------------------------------------------

	def _prepare_prediction(self, X: np.ndarray):
		"""Impute/scale test set and return GPU tensors.

		Returns:
			train_x  : (n_train, n_features) float32 on device
			train_y  : (n_train,) float32 on device
			test_x   : (n_test, n_features) float32 on device
			X_test_ori: original (unscaled) test array for retriever queries
		"""
		X_test_ori = X
		X_test = self.scaler.transform(self.imputer.transform(X))
		train_x = convert_to_torch_tensor(self.X_train).to(self.device).float()
		train_y = convert_to_torch_tensor(self.y_train).to(self.device).float()
		test_x  = convert_to_torch_tensor(X_test).to(self.device).float()
		return train_x, train_y, test_x, X_test_ori


class AWAREClassifier(AWAREEstimator, ClassifierMixin):
	"""Classifier wrapper — adds softmax post-processing and num_classes tracking."""

	def __init__(self, **kwargs):
		super().__init__(mode="cls", **kwargs)

	def fit(self, X: np.ndarray, y: np.ndarray, retriever_additional_fit: Dict = None):
		super().fit(X, y, retriever_additional_fit)
		self.num_classes = len(np.unique(self.y_train))

	def _predict_large_cls(self, X_train, X_test, y_train):
		"""Handle num_classes > max_num_classes via digit decomposition."""
		num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))
		digit_preds = []
		for i in range(num_digits):
			y_digit = (y_train // (self.max_num_classes ** i)) % self.max_num_classes
			eval_pos = X_train.shape[0]
			pred = self.model(
				x=torch.cat([X_train, X_test], dim=0),
				y=y_digit.unsqueeze(-1),
				eval_pos=eval_pos,
			)
			digit_preds.append(pred.float())

		full_pred = torch.zeros(
			(X_test.shape[0], X_test.shape[1], self.num_classes), device=X_train.device
		)
		for cls_idx in range(self.num_classes):
			cls_pred = torch.zeros_like(digit_preds[0][:, :, 0])
			for d_idx, d_pred in enumerate(digit_preds):
				digit_val = (cls_idx // (self.max_num_classes ** d_idx)) % self.max_num_classes
				cls_pred += d_pred[:, :, digit_val]
			full_pred[:, :, cls_idx] = cls_pred.transpose(0, 1)
		return full_pred

	@torch.no_grad()
	def predict_proba(
		self,
		X: np.ndarray,
		return_logits: bool = False,
		seed: int = None,
		retriever_additional_query: Dict = None,
	) -> np.ndarray:
		"""Predict class probabilities, averaging over n_ensembles.

		Temperature scaling and softmax are applied **once** after ensemble
		accumulation — never inside the loop — to avoid double-softmax.

		When seed is provided each ensemble uses seed+i so they differ in
		feature permutation order, providing genuine diversity.
		"""
		train_x, train_y, test_x, X_test_ori = self._prepare_prediction(X)
		pbar = tqdm(total=self.n_ensembles * len(X), desc="Predicting")
		context_size = self.retriever.k if self.retriever is not None else self.n_instances
		logits_cumsum = None

		for i in range(self.n_ensembles):
			# Each ensemble gets a different permutation when seed is set.
			seed_i = (seed + i) if seed is not None else None
			if seed_i is not None:
				perm = generate_random_permutation(train_x.shape[1], seed_i)
				X_train, X_test = train_x[:, perm], test_x[:, perm]
			else:
				X_train, X_test = train_x, test_x

			if not self.fit_retriever_before_fit and self.retriever is not None:
				self.retriever.fit([
					{self.retriever.field_to_use: row.cpu().numpy()} for row in X_train
				])

			if context_size >= self.n_instances:
				# Full-context path: all training rows in one context window.
				Xtr = pad_x(X_train[None], self.max_features).transpose(0, 1)  # (n,1,f)
				Xte = pad_x(X_test[None],  self.max_features).transpose(0, 1)  # (m,1,f)
				Ytr = train_y[None].float().transpose(0, 1)                     # (n,1)

				if seed_i is not None:
					idx = torch.from_numpy(np.random.permutation(Xtr.shape[0]))
					Xtr, Ytr = Xtr[idx], Ytr[idx]

				eval_pos = Xtr.shape[0]
				if self.num_classes <= self.max_num_classes:
					pred = self.model(
						x=torch.cat([Xtr, Xte], dim=0), y=Ytr, eval_pos=eval_pos
					)
				else:
					pred = self._predict_large_cls(Xtr, Xte, Ytr)

				# Accumulate raw logits — temperature + softmax applied once after loop.
				pred_val = pred[..., :self.num_classes].float().squeeze().detach().cpu().numpy()
				pbar.update(len(X))

			else:
				# Retrieval path: per-query context from k nearest neighbours.
				query_items = self._build_query_items(X_test_ori, len(X_test), retriever_additional_query)
				all_indices = torch.from_numpy(
					np.array(self.retriever.retrieve(query_items))
				).to(self.device)

				pred_list = []
				for b in range(math.ceil(len(X_test) / self.infer_batch_size)):
					start = b * self.infer_batch_size
					end   = min(len(X_test), start + self.infer_batch_size)

					idx   = all_indices[start:end]
					X_nni = train_x[idx]        # (batch, k, f)
					Y_nni = train_y[idx]         # (batch, k)
					X_ev  = test_x[start:end]    # (batch, f)

					if X_nni.shape[2] < self.max_features:
						X_nni = pad_x(X_nni, self.max_features)
						X_ev  = pad_x(X_ev.unsqueeze(1), self.max_features)
					else:
						X_ev = X_ev.unsqueeze(1)

					X_nni = X_nni.transpose(0, 1)   # (k, batch, f)
					X_ev  = X_ev.transpose(0, 1)    # (1, batch, f)
					Y_nni = Y_nni.transpose(0, 1)   # (k, batch)

					eval_pos = X_nni.shape[0]
					if self.num_classes <= self.max_num_classes:
						pred = self.model(
							x=torch.cat([X_nni, X_ev], dim=0), y=Y_nni, eval_pos=eval_pos
						)
					else:
						pred = self._predict_large_cls(X_nni, X_ev, Y_nni)

					pred_list.append(pred[..., :self.num_classes].float().squeeze(0))
					pbar.update(end - start)

				pred_val = (
					torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()
				)

			logits_cumsum = pred_val if logits_cumsum is None else logits_cumsum + pred_val

		pbar.close()

		# Temperature scaling + softmax exactly once after accumulating all ensembles.
		pred = logits_cumsum / self.n_ensembles
		if not return_logits:
			pred = pred / self.temperature
			pred = softmax(pred, axis=-1)
			pred /= pred.sum(axis=-1, keepdims=True)
		return pred

	def predict(self, X: np.ndarray, seed: int = None, retriever_additional_query: Dict = None):
		return self.predict_proba(X, seed=seed, retriever_additional_query=retriever_additional_query)


class AWARERegressor(AWAREEstimator, RegressorMixin):
	"""Regressor wrapper — returns raw scalar predictions from the last output logit."""

	def __init__(self, **kwargs):
		super().__init__(mode="reg", **kwargs)

	# fit() is fully inherited from AWAREEstimator — no override needed.

	@torch.no_grad()
	def predict_proba(
		self,
		X: np.ndarray,
		seed: int = None,
		retriever_additional_query: Dict = None,
	) -> np.ndarray:
		"""Return continuous predictions, averaged over n_ensembles.

		Uses the last output logit (index -1) following the same convention as
		compute_losses() during training.
		"""
		train_x, train_y, test_x, X_test_ori = self._prepare_prediction(X)
		pbar = tqdm(total=self.n_ensembles * len(X), desc="Predicting")
		context_size = self.retriever.k if self.retriever is not None else self.n_instances
		logits_cumsum = None

		for i in range(self.n_ensembles):
			seed_i = (seed + i) if seed is not None else None
			if seed_i is not None:
				perm = generate_random_permutation(train_x.shape[1], seed_i)
				X_train, X_test = train_x[:, perm], test_x[:, perm]
			else:
				X_train, X_test = train_x, test_x

			if not self.fit_retriever_before_fit and self.retriever is not None:
				self.retriever.fit([
					{self.retriever.field_to_use: row.cpu().numpy()} for row in X_train
				])

			if context_size >= self.n_instances:
				# Full-context path.
				Xtr = pad_x(X_train[None], self.max_features).transpose(0, 1)
				Xte = pad_x(X_test[None],  self.max_features).transpose(0, 1)
				Ytr = train_y[None].float().transpose(0, 1)

				eval_pos = Xtr.shape[0]
				pred = self.model(x=torch.cat([Xtr, Xte], dim=0), y=Ytr, eval_pos=eval_pos)
				# Regression uses the last output logit; squeeze batch dim.
				logits = pred[..., -1].float().squeeze().detach().cpu().numpy()
				pbar.update(len(X))

			else:
				# Retrieval path — batched like the classifier.
				query_items = self._build_query_items(X_test_ori, len(X_test), retriever_additional_query)
				indices_nni = self.retriever.retrieve(query_items)

				if seed_i is not None:
					col_perm = generate_random_permutation(np.array(indices_nni).shape[1], seed_i)
					indices_nni = torch.from_numpy(np.array(indices_nni))[:, col_perm]
				else:
					indices_nni = torch.from_numpy(np.array(indices_nni))

				pred_list = []
				for b in range(math.ceil(len(X_test) / self.infer_batch_size)):
					start = b * self.infer_batch_size
					end   = min(len(X_test), start + self.infer_batch_size)

					idx   = indices_nni[start:end].to(self.device)
					X_nni = train_x[idx]
					Y_nni = train_y[idx]
					X_ev  = test_x[start:end]

					if X_nni.shape[2] < self.max_features:
						X_nni = pad_x(X_nni, self.max_features)
						X_ev  = pad_x(X_ev.unsqueeze(1), self.max_features)
					else:
						X_ev = X_ev.unsqueeze(1)

					X_nni = X_nni.transpose(0, 1)
					X_ev  = X_ev.transpose(0, 1)
					Y_nni = Y_nni.transpose(0, 1)

					eval_pos = X_nni.shape[0]
					pred = self.model(
						x=torch.cat([X_nni, X_ev], dim=0), y=Y_nni, eval_pos=eval_pos
					)
					pred_list.append(pred[..., -1].float().squeeze(0))
					pbar.update(end - start)

				logits = (
					torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()
				)

			logits_cumsum = logits if logits_cumsum is None else logits_cumsum + logits

		pbar.close()
		return logits_cumsum / self.n_ensembles

	def predict(self, X: np.ndarray, seed: int = None, retriever_additional_query: Dict = None):
		return self.predict_proba(X, seed=seed, retriever_additional_query=retriever_additional_query)
