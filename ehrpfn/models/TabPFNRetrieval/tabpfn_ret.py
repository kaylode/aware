from tqdm import tqdm
import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

import numpy as np
import torch

from tabpfn.architectures.base.memory import MemoryUsageEstimator
from tabpfn.preprocessing import fit_preprocessing
from tabpfn.utils import get_autocast_context, infer_random_state
from tabpfn.inference import InferenceEngine
from tabpfn.preprocessing import EnsembleConfig
from tabpfn.architectures.interface import Architecture
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution

from tabpfn.base import (
    determine_precision,
)

from ehrpfn.models.TabPFN.utils import (
    default_classifier_preprocessor_configs,
    default_regressor_preprocessor_configs,
)

from ehrpfn.utils.retrieval import Retriever


@dataclass
class InferenceWithRetrievalEngineOnDemand(InferenceEngine):
    """Inference engine that does not cache anything, computes everything as needed.

    This is one of the slowest ways to run inference, as computation that could be
    cached is recomputed on every call. However the memory demand is lowest and
    can be more trivially parallelized across GPUs with some work.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    ensemble_configs: Sequence[EnsembleConfig]
    cat_ix: list[int]
    static_seed: int
    n_workers: int
    model: Architecture
    force_inference_dtype: torch.dtype | None
    queries_embs: np.ndarray | None
    retriever: Retriever | None
    context_embs: np.ndarray | None
    batch_size: int

    @classmethod
    def prepare(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size:int,
        retriever: Retriever,
        context_embs: np.ndarray | None,
        queries_embs: np.ndarray | None,
        *,
        cat_ix: list[int],
        model: Architecture,
        ensemble_configs: Sequence[EnsembleConfig],
        rng: np.random.Generator,
        n_workers: int,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
    ):
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            rng: The random number generator.
            n_workers: The number of workers to use.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        # We save it as a static seed to be reproducible across predicts
        static_seed = rng.integers(0, int(np.iinfo(np.int32).max))
        return cls(
            X_train=X_train,
            y_train=y_train,
            batch_size=batch_size,
            retriever=retriever,
            context_embs=context_embs,
            queries_embs=queries_embs,
            ensemble_configs=ensemble_configs,
            cat_ix=cat_ix,
            model=model,
            static_seed=static_seed,
            n_workers=n_workers,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        devices: Sequence[torch.device],
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        rng = np.random.default_rng(self.static_seed)

        device = devices[0]

        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)

        itr = fit_preprocessing(
            configs=self.ensemble_configs,
            X_train=self.X_train,
            y_train=self.y_train,
            random_state=rng,
            cat_ix=self.cat_ix,
            n_workers=self.n_workers,
            parallel_mode="as-ready",
        )

        X_fulls = []
        y_contexts = []
        pbar = tqdm(
            total=len(self.ensemble_configs)*(X.shape[0]),
            desc="Predicting",
        )
        for config, preprocessor, X_train, y_train, cat_ix in itr:
            assert X_train.shape[0] == self.X_train.shape[0], "X_train shape mismatch"

            self.retriever.fit(X_train, y=y_train, X_embs_train=self.context_embs)#, feature_order=preprocessor.feature_order
            X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)  # noqa: PLW2901
            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            all_outputs = []

            for batch_id, query_item in enumerate(X_test):
                query_item = query_item.unsqueeze(0)
                nn_indices_ = self.retriever.retrieve(query_item, X_embs_test=self.queries_embs)#, feature_order=preprocessor.feature_order
                nn_indices = nn_indices_[0]

                X_context = X_train[nn_indices, :]
                y_context = y_train[nn_indices]

                X_full = torch.cat([X_context, query_item], dim=0).unsqueeze(1)
                y_context = torch.as_tensor(y_context, dtype=torch.float32, device=device)  # type: ignore  # noqa: PLW2901
                X_fulls.append(X_full)
                y_contexts.append(y_context)
                if len(X_fulls) % self.batch_size == 0 or (batch_id + 1) == X.shape[0]:
                    X_full = torch.cat(X_fulls, dim=1)
                    y_context = torch.stack(y_contexts, dim=1)
                    batched_cat_ix = [cat_ix] * X_full.shape[1]
                    X_fulls = []
                    y_contexts = []

                    MemoryUsageEstimator.reset_peak_memory_if_required(
                        save_peak_mem=self.save_peak_mem,
                        model=self.model,
                        X=X_full,
                        cache_kv=False,
                        dtype_byte_size=self.dtype_byte_size,
                        device=device,
                        safety_factor=1.2,  # TODO(Arjun): make customizable
                    )

                    if self.force_inference_dtype is not None:
                        X_full = X_full.type(self.force_inference_dtype)
                        y_context = y_context.type(self.force_inference_dtype)  # type: ignore  # noqa: PLW2901

                    with (
                        get_autocast_context(device, enabled=autocast),
                        torch.inference_mode(),
                    ):
                        output = self.model(
                            X_full,
                            y_context,
                            only_return_standard_out=only_return_standard_out,
                            categorical_inds=batched_cat_ix,
                        )

                    output = output if isinstance(output, dict) else output.squeeze(1)
                    all_outputs.append(output)
                    pbar.update(output.shape[1])
            output = torch.cat(all_outputs, dim=1).squeeze()
            yield output, config

        pbar.close()
        self.model = self.model.cpu()


def get_tabpfn_knn_results(
        X_train, y_train, X_test, 
        categorical_feature_indices, 
        is_classification, 
        retriever:Retriever,
        n_ensembles:int=8,
        seed:int=42,
        device = 'cuda',
        **kwargs
    ):


    # In case retrieval using extra embeddings
    context_embs = kwargs.get('X_embs_train', None)
    queries_embs = kwargs.get('X_embs_test', None)


    if is_classification:
        model = TabPFNClassifier(
            n_estimators=n_ensembles,
            ignore_pretraining_limits=True,
            categorical_features_indices=categorical_feature_indices,
            device=device,
            n_jobs=-1,
            random_state=seed,
            memory_saving_mode=True,
            fit_mode='low_memory',
            inference_config={
                "FEATURE_SHIFT_METHOD": "shuffle",
                'FINGERPRINT_FEATURE': False,
                "CLASS_SHIFT_METHOD": "shuffle",
                "PREPROCESS_TRANSFORMS": default_classifier_preprocessor_configs(),
            }
        )

        if not hasattr(model, "model_") or not model.differentiable_input:
            byte_size, rng = model._initialize_model_variables()
            ensemble_configs, X_train, y_train = model._initialize_dataset_preprocessing(X_train, y_train, rng)
        else:  # already fitted and prompt_tuning mode: no cat. features
            _, rng = infer_random_state(model.random_state)
            _, _, byte_size = determine_precision(
                model.inference_precision, model.device_
            )

    else:
        model = TabPFNRegressor(
            n_estimators=n_ensembles,
            ignore_pretraining_limits=True,
            categorical_features_indices=categorical_feature_indices,
            device=device,
            random_state=seed,
            memory_saving_mode=True,
            n_jobs=-1,
            fit_mode='low_memory',
            inference_config={
                "FEATURE_SHIFT_METHOD": "shuffle",
                'FINGERPRINT_FEATURE': False,
                "CLASS_SHIFT_METHOD": "shuffle",
                "PREPROCESS_TRANSFORMS": default_regressor_preprocessor_configs(),
            }
        )

        if not hasattr(model, "model_") or not model.differentiable_input:
            byte_size, rng = model._initialize_model_variables()
            ensemble_configs, X_train, y_train, model.znorm_space_bardist_ = (
                model._initialize_dataset_preprocessing(X_train, y_train, rng)
            )
        else:  # already fitted and prompt_tuning mode: no cat. features
            _, rng = infer_random_state(model.random_state)
            _, _, byte_size = determine_precision(
                model.inference_precision, model.device_
            )

        assert len(ensemble_configs) == model.n_estimators

        model.is_constant_target_ = np.unique(y_train).size == 1
        model.constant_value_ = y_train[0] if model.is_constant_target_ else None

        if model.is_constant_target_:
            model.znorm_space_bardist_ = FullSupportBarDistribution(
                borders=torch.tensor(
                    [model.constant_value_ - 1e-5, model.constant_value_ + 1e-5]
                )
            )
            # No need to create an inference engine for a constant prediction
        else:
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

        batch_size=128,
        retriever=retriever,
        context_embs=context_embs,
        queries_embs=queries_embs,

        cat_ix=model.inferred_categorical_indices_,
        ensemble_configs=ensemble_configs,
        rng=rng,
        model=model.model_,
        n_workers=model.n_jobs,
        dtype_byte_size=byte_size,
        force_inference_dtype=model.forced_inference_dtype_,
        save_peak_mem=model.memory_saving_mode,
    )

    if is_classification:
        y_pred = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test)

    return y_pred
