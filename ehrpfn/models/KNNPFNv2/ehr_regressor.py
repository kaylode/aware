from typing import *
import numpy as np
import torch
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ehrret import Retriever
from tqdm import tqdm
from sklearn.base import RegressorMixin
from .estimator import KNNPFNv2Estimator

class KNNPFNv2Regressor(KNNPFNv2Estimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(mode="reg", **kwargs)

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
        
        if self.model is None:
            # For regression, n_out might be different or we use a specific head
            self.model = self._get_model(X_scaled.shape[1], 1)
        
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray, seed: int = None, retriever_additional_query=None, **kwargs):
        train_x, train_y, test_x = self._prepare_prediction(X)
        context_size = self.k
        
        pbar = tqdm(total=self.n_ensembles * len(X), desc="KNNPFNv2 Regression")
        preds_cumsum = None

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
            for b in range(math.ceil(len(test_x) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(test_x), (b + 1) * self.inf_batch_size)
                
                # Retrieval
                query_items = []
                for i in range(start, end):
                    feat = self.X_test_ori[i] if self.fit_retriever_before_fit else test_x[i].cpu().numpy()
                    query_dict = {"raw_feature": feat}
                    if retriever_additional_query is not None:
                        for k, v in retriever_additional_query.items():
                            query_dict[k] = v[i]
                    query_items.append(query_dict)
                
                indices = self.retriever.retrieve(query_items)
                indices = torch.LongTensor(indices).to(self.device)
                
                # Context
                X_context = train_x[indices].transpose(0, 1)
                y_context = train_y[indices].transpose(0, 1)
                X_query = test_x[start:end].unsqueeze(1).transpose(0, 1)
                
                # Forward
                batch_preds = self.model(
                    x_context=X_context,
                    y_context=y_context,
                    x_query=X_query,
                    task="reg"
                ) # (1, B)
                
                pred_list.append(batch_preds.squeeze(0).cpu())
                pbar.update(end - start)
            
            ens_preds = torch.cat(pred_list, dim=0).numpy()
            if preds_cumsum is None:
                preds_cumsum = ens_preds
            else:
                preds_cumsum += ens_preds
        
        pbar.close()
        return preds_cumsum / self.n_ensembles
