from typing import *
import numpy as np
import torch
import math
from .utils import pad_x, generate_random_permutation

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ehrret import Retriever
from tqdm import tqdm
from scipy.special import softmax

from sklearn.base import ClassifierMixin
from .estimator import EHRPFNEstimator



class EHRPFNClassifier(EHRPFNEstimator, ClassifierMixin):

    def __init__(
        self, pretrained_path:str=None, inf_batch_size: int = 512,
        n_ensembles: int = 1, temperature: float = 0.8,
        device: str = EHRPFNEstimator._DEFAULT_DEVICE, use_flash: bool = True, 
        compile: bool = True, categorical_feature_indices: list = None, 
        retriever:Retriever=None, fit_retriever_before_fit: bool = False,
        base_model: Callable=None
    ):
        super().__init__(
            mode="cls", 
            pretrained_path=pretrained_path, 
            inf_batch_size=inf_batch_size, 
            n_ensembles=n_ensembles, temperature=temperature,
            device=device, use_flash=use_flash, 
            compile=compile, 
            categorical_feature_indices=categorical_feature_indices,
            retriever=retriever,
            fit_retriever_before_fit=fit_retriever_before_fit,
            base_model=base_model
        )


    def fit(self, X, y, retriever_additional_fit:Dict=None, **kwargs):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        if self.fit_retriever_before_fit and self.retriever is not None:
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


    def _predict_large_cls(self, X_train, X_test, y_train):
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_train_digit = (y_train // (self.max_num_classes ** i)) % self.max_num_classes
            pred = self.model(
                x=torch.cat([X_train, X_test], dim=1),
                y=y_train_digit.unsqueeze(-1),
                task=self.mode,
            )
            digit_preds.append(pred.float())

        full_pred = torch.zeros((X_test.shape[0], X_test.shape[1], self.num_classes), device=X_train.device)
        for class_idx in range(self.num_classes):
            class_pred = torch.zeros_like(digit_preds[0][:, :, 0])
            for digit_idx, digit_pred in enumerate(digit_preds):
                digit_value = (class_idx // (self.max_num_classes ** digit_idx)) % self.max_num_classes
                class_pred += digit_pred[:, :, digit_value]
            full_pred[:, :, class_idx] = class_pred.transpose(0, 1)

        return full_pred
    

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, return_logits: bool = False, seed: int = None, retriever_additional_query=None, **kwargs):
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

            if not self.fit_retriever_before_fit and self.retriever is not None:
                fit_items = []
                for item, label in zip(X_train, train_y):
                    fit_items.append({
                        self.retriever.field_to_use: item.cpu().numpy(),
                        "label": label.item()
                    })
                self.retriever.fit(fit_items)

            if context_size >= self.n_instances:
                X_train = pad_x(X_train[None, :, :], self.max_features).to(self.device)
                X_test = pad_x(X_test[None, :, :], self.max_features).to(self.device)
                y_train = train_y[None, :].float()

                if seed is not None:
                    # shuffle X_train order
                    indices = np.arange(X_train.shape[1])
                    shuffled_indices = np.random.permutation(indices)
                    X_train = X_train[:, torch.from_numpy(shuffled_indices)]
                    y_train = y_train[:, torch.from_numpy(shuffled_indices)]

                X_train = X_train.transpose(0, 1)
                y_train = y_train.transpose(0, 1)
                X_test = X_test.transpose(0, 1)

                if self.num_classes <= self.max_num_classes:
                    pred = self.model(
                        x=torch.cat([X_train, X_test], dim=0),
                        y=y_train,
                        task=self.mode,
                    )
                else:
                    pred = self._predict_large_cls(X_train, X_test, y_train)

                pred = pred[..., :self.num_classes] / self.temperature
                if not return_logits:
                    pred = torch.nn.functional.softmax(pred.float(), dim=-1)
                pred_val = pred.squeeze().detach().cpu().float().numpy()
            else:
                pred_list = []
                for b in range(math.ceil(len(X_test) / self.inf_batch_size)):
                    start = b * self.inf_batch_size
                    end = min(len(X_test), (b + 1) * self.inf_batch_size)


                    query_items = []
                    for i in range(start, end):
                        if isinstance(self.X_test_ori, torch.Tensor):
                            if self.fit_retriever_before_fit:
                                query_dict = {"raw_feature": self.X_test_ori[i].cpu().numpy()}
                            else:
                                query_dict = {"raw_feature": X_test[i].cpu().numpy()}
                        else:
                            if self.fit_retriever_before_fit:
                                query_dict = {"raw_feature": self.X_test_ori[i]}
                            else:
                                query_dict = {"raw_feature": X_test[i]}
                        if retriever_additional_query is not None:
                            for key, value in retriever_additional_query.items():
                                if self.retriever.field_to_use == key:
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

                    if self.num_classes <= self.max_num_classes:
                        pred = self.model(
                            x=torch.cat([X_nni, X_eval], dim=0),
                            y=y_nni,
                            task=self.mode,
                        )
                    else:
                        pred = self._predict_large_cls(X_nni, X_eval, y_nni)

                    pred = pred.float()
                    if not return_logits:
                        pred = pred[..., :self.num_classes] / self.temperature
                        pred = torch.nn.functional.softmax(pred, dim=-1)
                        pred /= pred.sum(axis=-1, keepdims=True) + 1e-12 # numerical stability

                    pred_list.append(pred.squeeze())
                    pbar.update(end-start)
                pred_val = torch.cat(pred_list, dim=0).detach().cpu().float().numpy()
            
            if logits_cumsum is None:
                logits_cumsum = pred_val
            else:
                logits_cumsum += pred_val

        pbar.close()
        avg_pred = logits_cumsum / self.n_ensembles
        
        if return_logits:
            return avg_pred
        
        # If we already applied softmax inside the loop, avg_pred is already probabilities
        # Otherwise (unlikely with current flow), we would apply it here.
        # Ensure normalization for final probabilities
        avg_pred /= avg_pred.sum(axis=-1, keepdims=True) + 1e-12
        return avg_pred

    def predict(self, X, seed: int = None, retriever_additional_query=None, **kwargs):
        return self.predict_proba(X, seed=seed, retriever_additional_query=retriever_additional_query, **kwargs)