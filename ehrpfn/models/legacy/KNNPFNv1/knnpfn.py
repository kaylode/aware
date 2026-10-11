import numpy as np
from collections.abc import Sequence
from typing import Literal
import numpy as np
import torch
from tqdm import tqdm

from tabpfn.inference import (
	InferenceEngineOnDemand, _PerDeviceModelCache, 
	set_save_peak_memory, get_autocast_context,
)
from tabpfn.classifier import (
	_validate_eval_metric,
	infer_random_state,
	determine_precision,
)
from tabpfn.regressor import (
	REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
	FullSupportBarDistribution
)

from tabpfn.preprocessing import EnsembleConfig
from tabpfn.architectures.interface import Architecture

from ehrpfn.models.TabPFN import get_base_tabpfn

from ehrpfn.retrieval import Retriever


class InferenceWithRetrievalEngineOnDemand(InferenceEngineOnDemand):
	"""Inference engine that does not cache anything, computes everything as needed.

	This is one of the slowest ways to run inference, as computation that could be
	cached is recomputed on every call. However the memory demand is lowest and
	can be more trivially parallelized across GPUs with some work.
	"""

	X_train: np.ndarray
	y_train: np.ndarray
	cat_ix: list[int]
	static_seed: int
	n_preprocessing_jobs: int
	force_inference_dtype: torch.dtype | None
	ensemble_configs: list[EnsembleConfig]
	retriever: Retriever | None
	retriever_additional_fit: dict | None
	retriever_additional_query: dict | None
	batch_size: int = 1

	def __init__(self, batch_size, retriever: Retriever | None, retriever_additional_fit: dict | None, retriever_additional_query: dict | None, **kwargs):
		super().__init__(**kwargs)
		self.retriever = retriever
		self.retriever_additional_fit = retriever_additional_fit
		self.batch_size = batch_size
		self.retriever_additional_query = retriever_additional_query

	@classmethod
	def prepare(
		cls,
		X_train: np.ndarray,
		y_train: np.ndarray,
		*,
		batch_size: int = 1,
		retriever: Retriever,
		retriever_additional_fit: dict | None,
		retriever_additional_query: dict | None = None,
		cat_ix: list[int],
		models: list[Architecture],
		ensemble_configs: Sequence[EnsembleConfig],
		rng: np.random.Generator,
		n_preprocessing_jobs: int,
		dtype_byte_size: int,
		force_inference_dtype: torch.dtype | None,
		save_peak_mem: bool | Literal["auto"] | float | int,
	):
		"""Prepare the inference engine.

		Args:
			X_train: The training data.
			y_train: The training target.
			cat_ix: The categorical indices.
			models: The models to use.
			ensemble_configs: The ensemble configurations to use.
			rng: The random number generator.
			n_preprocessing_jobs: The number of workers to use.
			dtype_byte_size: The byte size of the dtype.
			force_inference_dtype: The dtype to force inference to.
			save_peak_mem: Whether to save peak memory usage.
		"""


		# We save it as a static seed to be reproducible across predicts
		static_seed = rng.integers(0, int(np.iinfo(np.int32).max))
		return cls(
			X_train=X_train,
			y_train=y_train,
			ensemble_configs=list(ensemble_configs),
			batch_size=batch_size,
			retriever=retriever,
			retriever_additional_fit=retriever_additional_fit,
			retriever_additional_query=retriever_additional_query,
			cat_ix=cat_ix,
			model_caches=[_PerDeviceModelCache(model) for model in models],
			static_seed=static_seed,
			n_preprocessing_jobs=n_preprocessing_jobs,
			dtype_byte_size=dtype_byte_size,
			force_inference_dtype=force_inference_dtype,
			save_peak_mem=save_peak_mem,
		)


	def _call_model(
		self,
		*,
		device: torch.device,
		is_parallel: bool,
		X_train: torch.Tensor | np.ndarray,
		X_test: torch.Tensor | np.ndarray,
		y_train: torch.Tensor | np.ndarray,
		cat_ix: list[int],
		autocast: bool,
		only_return_standard_out: bool,
		model_index: int,
		save_peak_mem: bool,
	) -> torch.Tensor | dict[str, torch.Tensor]:
		"""Execute a model forward pass on the provided device.

		Note that several instances of this function may be executed in parallel in
		different threads, one for each device in the system.
		"""
		# In parallel mode, the inference engine uses multiple devices. Otherwise, it
		# uses a single device.
		model = self.model_caches[model_index].get(device, multiple_devices=is_parallel)

		fit_items = []
		for i, (item, label) in enumerate(zip(X_train, y_train)):
			fit_dict = {"raw_feature": item, "label": label}
			if self.retriever_additional_fit is not None:
				for key, value in self.retriever_additional_fit.items():
					if self.retriever.field_to_use == key:
						assert len(value) == X_train.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in fit_dict, f"Key {key} already exists in fit_dict"
					try:
						fit_dict[key] = value[i]
					except Exception as e:
						continue
			fit_items.append(fit_dict)
		self.retriever.fit(fit_items)

		query_items = []
		for i in range(X_test.shape[0]):
			if isinstance(X_test, torch.Tensor):
				query_dict = {"raw_feature": X_test[i].cpu().numpy()}
			else:
				query_dict = {"raw_feature": X_test[i]}
			if self.retriever_additional_query is not None:
				for key, value in self.retriever_additional_query.items():
					if self.retriever.field_to_use == key:
						assert len(value) == X_test.shape[0], "Length of retriever_additional_fit values must match number of samples"
					assert key not in query_dict, f"Key {key} already exists in query_dict"
					query_dict[key] = value[i]
			query_items.append(query_dict)

		nn_indices = self.retriever.retrieve(query_items)
		dtype = self.force_inference_dtype if self.force_inference_dtype else torch.float32

		results = []
		# Split into batches
		test_batches = np.array_split(np.arange(X_test.shape[0]), max(1, X_test.shape[0] // self.batch_size))
		for batch_indices in tqdm(test_batches, desc="Predicting batches"):
			batch_query = X_test[batch_indices]
			batch_nn_indices = [nn_indices[i] for i in batch_indices]
			batch_context = []
			for nn_idx in batch_nn_indices:
				X_context = X_train[nn_idx, :]
				y_context = y_train[nn_idx]
				batch_context.append((X_context, y_context))

			input_tensor = []
			y_tensor = []
			for i in range(len(batch_context)):
				X_train_tmp = torch.as_tensor(batch_context[i][0], dtype=dtype, device=device)
				y_train_tmp = torch.as_tensor(batch_context[i][1], dtype=dtype, device=device)
				X_test_tmp = torch.as_tensor(batch_query[i][np.newaxis, :], dtype=dtype, device=device)
				X_full_tmp = torch.cat([X_train_tmp, X_test_tmp], dim=0).unsqueeze(1)
				input_tensor.append(X_full_tmp)
				y_tensor.append(y_train_tmp)
			input_tensor = torch.cat(input_tensor, dim=1)
			y_subset = torch.stack(y_tensor, dim=1)

			set_save_peak_memory(model, enabled=save_peak_mem)
			with get_autocast_context(device, enabled=autocast), torch.inference_mode():
				res = model(
					input_tensor,
					y_subset,
					only_return_standard_out=only_return_standard_out,
					categorical_inds=[cat_ix] * input_tensor.shape[1],
				)
				results.append(res.squeeze(0))
		results = torch.cat(results, dim=0)
		return results



class KNNPFN:
	"""
	Docstring for KNNPFN
	"""

	def __init__(
			self, 
			is_classification, 
			categorical_feature_indices, 
			retriever:Retriever,
			seed:int=42,
			device = 'cuda',
			batch_size:int=1,
		):
		self.is_classification = is_classification
		self.categorical_feature_indices = categorical_feature_indices
		self.retriever = retriever
		self.n_ensembles = 1
		self.seed = seed
		self.device = device
		self.batch_size = batch_size

	def fit(
			self,
			X_train, y_train, 
			retriever_additional_fit: dict | None = None,
			retriever_additional_query: dict | None = None,
			**kwargs
		):

		model = get_base_tabpfn(
			categorical_feature_indices=self.categorical_feature_indices,
			is_classification=self.is_classification,
			device=self.device,
			seed=self.seed
		)
		if self.is_classification:

			# Validate eval_metric here instead of in __init__ as per sklearn convention
			model.eval_metric_ = _validate_eval_metric(model.eval_metric)

			if model.fit_mode == "batched":
				model.fit_mode = "fit_preprocessors"

			if not hasattr(model, "models_") or not model.differentiable_input:
				byte_size, rng = model._initialize_model_variables()
				ensemble_configs, X_train, y_train = model._initialize_dataset_preprocessing(X_train, y_train, rng)
			else:  # already fitted and prompt_tuning mode: no cat. features
				_, rng = infer_random_state(model.random_state)
				_, _, byte_size = determine_precision(
					model.inference_precision, model.devices_
				)

			model._maybe_calibrate_temperature_and_tune_decision_thresholds(
				X=X_train,
				y=y_train,
			)

		else:
			
			if model.fit_mode == "batched":
				model.fit_mode = "fit_preprocessors"

			if not hasattr(model, "models_") or not model.differentiable_input:
				byte_size, rng = model._initialize_model_variables()
				ensemble_configs, X_train, y_train, model.znorm_space_bardist_ = (
					model._initialize_dataset_preprocessing(X_train, y_train, rng)
				)
			else:  # already fitted and prompt_tuning mode: no cat. features
				_, rng = infer_random_state(model.random_state)
				_, _, byte_size = determine_precision(
					model.inference_precision, model.devices_
				)

			assert len(ensemble_configs) == model.n_estimators

			model.is_constant_target_ = np.unique(y_train).size == 1
			model.constant_value_ = y_train[0] if model.is_constant_target_ else None

			if model.is_constant_target_:
				# Use relative epsilon, s.t. it works for small and large constant values
				border_adjustment = max(
					abs(model.constant_value_ * REGRESSION_CONSTANT_TARGET_BORDER_EPSILON),
					REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
				)

				model.znorm_space_bardist_ = FullSupportBarDistribution(
					borders=torch.tensor(
						[
							model.constant_value_ - border_adjustment,
							model.constant_value_ + border_adjustment,
						]
					)
				)
				# No need to create an inference engine for a constant prediction
				return model

			mean, std = np.mean(y_train), np.std(y_train)
			model.y_train_std_ = std.item() + 1e-20
			model.y_train_mean_ = mean.item()
			y_train = (y_train - model.y_train_mean_) / model.y_train_std_
			model.raw_space_bardist_ = FullSupportBarDistribution(
				model.znorm_space_bardist_.borders * model.y_train_std_ + model.y_train_mean_,
			).float()


		# Override inference engine
		model.executor_ = InferenceWithRetrievalEngineOnDemand.prepare(
			X_train=X_train,
			y_train=y_train,
			retriever=self.retriever,
			retriever_additional_fit=retriever_additional_fit,
			retriever_additional_query=retriever_additional_query,
			cat_ix=model.inferred_categorical_indices_,
			models=model.models_,
			ensemble_configs=ensemble_configs,
			rng=rng,
			batch_size=self.batch_size,
			n_preprocessing_jobs=model.n_preprocessing_jobs,
			dtype_byte_size=byte_size,
			force_inference_dtype=model.forced_inference_dtype_,
			save_peak_mem=model.memory_saving_mode,
		)

		self.model = model

	def predict(
			self,
			X_test,
			**kwargs,
		):
		if self.is_classification:
			y_pred = self.model.predict_proba(X_test)
		else:
			y_pred = self.model.predict(X_test)

		return y_pred
