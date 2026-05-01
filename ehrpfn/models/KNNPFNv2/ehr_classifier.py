from typing import *
import numpy as np
import torch
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ehrret import Retriever
from tqdm import tqdm
from scipy.special import softmax
from sklearn.base import ClassifierMixin
from .estimator import KNNPFNv2Estimator

class KNNPFNv2Classifier(KNNPFNv2Estimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(mode="cls", **kwargs)

    def fit(self, X, y, retriever_additional_fit: Dict = None, **kwargs):
        assert X.ndim == 2, "X must be 2D"
        
        if self.fit_retriever_before_fit and self.retriever is not None:
            fit_items = []
            for i, (item, label) in enumerate(zip(X, y)):
                fit_dict = {"raw_feature": item, "label": label}
                if retriever_additional_fit is not None:
                    for key, value in retriever_additional_fit.items():
                        fit_dict[key] = value[i]
                fit_items.append(fit_dict)
            self.retriever.fit(fit_items)

        self.imputer = SimpleImputer(strategy="mean")
        X_imputed = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.X_train = X_scaled
        self.y_train = y
        self.num_classes = len(np.unique(y))
        
        # Initialize model if not already present
        if self.model is None:
            self.model = self._get_model(X_scaled.shape[1], self.num_classes)
        
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, seed: int = None, retriever_additional_query=None, **kwargs):
        train_x, train_y, test_x = self._prepare_prediction(X)
        context_size = self.k
        
        pbar = tqdm(total=self.n_ensembles * len(X), desc="KNNPFNv2 Predicting")
        probs_cumsum = None

        for ens_idx in range(self.n_ensembles):
            # Dynamic fit if needed
            if not self.fit_retriever_before_fit and self.retriever is not None:
                fit_items = []
                for i in range(len(train_x)):
                    fit_items.append({
                        self.retriever.field_to_use: train_x[i].cpu().numpy(),
                        "label": train_y[i].item()
                    })
                self.retriever.fit(fit_items)

            pred_list = []
            # Batch test samples
            for b in range(math.ceil(len(test_x) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(test_x), (b + 1) * self.inf_batch_size)
                
                # Retrieval
                query_items = []
                for i in range(start, end):
                    # Use raw if fitted on raw, otherwise scaled
                    feat = self.X_test_ori[i] if self.fit_retriever_before_fit else test_x[i].cpu().numpy()
                    query_dict = {"raw_feature": feat}
                    if retriever_additional_query is not None:
                        for k, v in retriever_additional_query.items():
                            query_dict[k] = v[i]
                    query_items.append(query_dict)
                
                indices = self.retriever.retrieve(query_items)
                indices = torch.LongTensor(indices).to(self.device) # (B, K)
                
                # Context selection
                # train_x: (N, D), indices: (B, K) -> (B, K, D)
                X_context = train_x[indices]
                y_context = train_y[indices]
                X_query = test_x[start:end].unsqueeze(1) # (B, 1, D)
                
                # Transpose for TabPFN: (K, B, D)
                X_context = X_context.transpose(0, 1)
                y_context = y_context.transpose(0, 1)
                X_query = X_query.transpose(0, 1)
                
                # Forward
                logits = self.model(
                    x_context=X_context,
                    y_context=y_context,
                    x_query=X_query,
                    task="cls"
                ) # (1, B, num_classes)
                
                logits = logits.squeeze(0) # (B, num_classes)
                probs = torch.softmax(logits / self.temperature, dim=-1)
                pred_list.append(probs.cpu())
                pbar.update(end - start)
            
            ens_probs = torch.cat(pred_list, dim=0).numpy()
            if probs_cumsum is None:
                probs_cumsum = ens_probs
            else:
                probs_cumsum += ens_probs
        
        pbar.close()
        final_probs = probs_cumsum / self.n_ensembles
        # Re-normalize for stability
        final_probs /= (final_probs.sum(axis=-1, keepdims=True) + 1e-12)
        return final_probs

    def predict(self, X, **kwargs):
        probs = self.predict_proba(X, **kwargs)
        return np.argmax(probs, axis=1)
