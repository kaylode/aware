from typing import *
import math
import torch
import numpy as np

from .utils import pad_x, generate_random_permutation
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .estimator import EHRPFNEstimator
from sklearn.base import RegressorMixin

from ehrret import Retriever


class EHRPFNRegressor(EHRPFNEstimator, RegressorMixin):
    def __init__(
            self, pretrained_path:str=None, inf_batch_size: int = 512,
            n_ensembles: int = 1, temperature: float = 0.8,
            device: str = EHRPFNEstimator._DEFAULT_DEVICE, 
            use_flash: bool = True, compile: bool = True, 
            categorical_feature_indices: list = None,
            retriever: Retriever = None, fit_retriever_before_fit: bool = False,
            base_model: Callable=None
        ):
        super().__init__(
            mode="reg", pretrained_path=pretrained_path, inf_batch_size=inf_batch_size, 
            n_ensembles=n_ensembles, temperature=temperature,
            device=device, use_flash=use_flash, compile=compile, categorical_feature_indices=categorical_feature_indices,
            fit_retriever_before_fit=fit_retriever_before_fit,
            base_model=base_model
        )
        self.retriever = retriever

    def fit(self, X, y, retriever_additional_fit:Dict=None, **kwargs):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        if self.fit_retriever_before_fit:
            fit_items = []
            for i, (item, label) in enumerate(zip(X, y)):
                fit_dict = {"raw_feature": item, "label": label}
                if retriever_additional_fit is not None:
                    for key, value in retriever_additional_fit.items():
                        assert len(value) == X.shape[0], "Length of retriever_additional_fit values must match number of samples"
                        assert key not in fit_dict, f"Key {key} already exists in fit_dict"
                        try:
                            fit_dict[key] = value[i]
                        except Exception as e:
                            continue
                fit_items.append(fit_dict)
            self.retriever.fit(fit_items)

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.num_classes = len(np.unique(self.y_train))
        self.is_fitted_ = True
        if self.compile:
            self.model = torch.compile(self.model)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, seed: int = None, retriever_additional_query=None, **kwargs):
        train_x, train_y, test_x = self._prepare_prediction(X)
        pbar = tqdm(total=self.n_ensembles*len(X), desc="Predicting")
        context_size = self.retriever.k if self.retriever is not None else self.n_instances

        logits_cumsum = None
        for _ in range(self.n_ensembles):
            # seed = int(np.random.SeedSequence().generate_state(1)[0])

            if seed is not None:
                feat_perm = generate_random_permutation(train_x.shape[1], seed)
                X_train = train_x[:, feat_perm]
                X_test = test_x[:, feat_perm]
            else:
                X_train = train_x
                X_test = test_x

            
            if not self.fit_retriever_before_fit:
                fit_items = []
                for item, label in zip(X_train, train_y):
                    fit_items.append({
                        self.retriever.field_to_use: item.cpu().numpy(),
                        "label": label.item()
                    })
                self.retriever.fit(fit_items)

            if context_size >= self.n_instances:
                X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
                X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
                y_train = train_y[None, :].float()

                X_train = X_train.transpose(0, 1)
                y_train = y_train.transpose(0, 1)
                X_test = X_test.transpose(0, 1)

                pred = self.model(
                    x=torch.cat([X_train, X_test], dim=0),
                    y=y_train,
                    task=self.mode,
                )

                logits = pred.float().squeeze().detach().cpu().float().numpy()
                pbar.update(X_test.shape[0])
            else:
                pred_list = []
                for b in range(math.ceil(len(X_test) / self.inf_batch_size)):
                    start = b * self.inf_batch_size
                    end = min(len(X_test), (b + 1) * self.inf_batch_size)

                    query_items = []
                    for i in range(start, end):
                        if isinstance(self.X_test_ori, torch.Tensor):
                            query_dict = {"raw_feature": self.X_test_ori[i].cpu().numpy()}
                        else:
                            query_dict = {"raw_feature": self.X_test_ori[i]}
                        if retriever_additional_query is not None:
                            for key, value in retriever_additional_query.items():
                                assert len(value) == self.X_test_ori.shape[0], "Length of retriever_additional_fit values must match number of samples"
                                assert key not in query_dict, f"Key {key} already exists in query_dict"
                                query_dict[key] = value[i]
                        query_items.append(query_dict)

                    indices_nni = self.retriever.retrieve(query_items)

                    if seed is not None:
                        shuffled_indices = generate_random_permutation(indices_nni.shape[1], seed)
                        indices_nni = torch.tensor(indices_nni)[:, shuffled_indices]
                    else:
                        indices_nni = torch.tensor(indices_nni)

                    X_nni = X_train[indices_nni]
                    y_nni = train_y[indices_nni]
                    X_eval = X_test[start:end]

                    if X_nni.shape[2] < self.max_features:
                        X_nni, y_nni = (
                            pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                            torch.Tensor(y_nni).to(self.device),
                        )
                        X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device)
                    else:
                        X_eval = X_eval.unsqueeze(1)
                    
                    X_nni = X_nni.transpose(0, 1)
                    X_eval = X_eval.transpose(0, 1)
                    y_nni = y_nni.transpose(0, 1)

                    pred = self.model(
                        x=torch.cat([X_nni, X_eval], dim=0),
                        y=y_nni,
                        task=self.mode,
                    )

                    pred_list.append(pred.squeeze())
                    pbar.update(end - start)

                logits = torch.cat(pred_list).detach().cpu().float().numpy()

            if logits_cumsum is None:
                logits_cumsum = logits
            else:
                logits_cumsum += logits

        pbar.close()
        return logits_cumsum / self.n_ensembles


    def predict(self, X, seed: int = None, retriever_additional_query=None, **kwargs):
        return self.predict_proba(X, seed=seed, retriever_additional_query=retriever_additional_query, **kwargs)