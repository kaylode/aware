import torch
import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from .model import KNNPFNv2Model
from ehrret import Retriever

class KNNPFNv2Estimator(BaseEstimator):
    _DEFAULT_DEVICE = "cuda:0"

    def __init__(self, 
            mode: str, pretrained_path:str=None, inf_batch_size: int = 512, 
            n_ensembles: int = 1, temperature: float = 0.8,
            device: str = _DEFAULT_DEVICE,
            categorical_feature_indices: list = None,
            retriever: Retriever = None,
            fit_retriever_before_fit: bool = False,
            base_model: KNNPFNv2Model = None,
            k: int = 32
        ):
        self.mode = mode
        self.inf_batch_size = inf_batch_size
        self.device = device
        self.categorical_feature_indices = categorical_feature_indices
        self.retriever = retriever
        self.fit_retriever_before_fit = fit_retriever_before_fit
        self.n_ensembles = n_ensembles
        self.temperature = temperature
        self.k = k

        if base_model is not None:
            self.model = base_model
        else:
            # We will initialize the model in fit() if not provided,
            # or load it if pretrained_path is given
            self.model = None
            self.pretrained_path = pretrained_path

        self.is_fitted_ = False

    def _prepare_prediction(self, X: np.ndarray):
        check_is_fitted(self)
        self.X_test_ori = X
        X_test = self.imputer.transform(X)
        X_test = self.scaler.transform(X_test)
        
        train_x = torch.FloatTensor(self.X_train).to(self.device)
        train_y = torch.FloatTensor(self.y_train).to(self.device)
        test_x = torch.FloatTensor(X_test).to(self.device)

        return train_x, train_y, test_x

    def _get_model(self, n_features: int, num_classes: int):
        if self.model is not None:
            return self.model
        
        # Initialize default KNNPFNv2Model
        # ninp and other params usually come from config
        model = KNNPFNv2Model(
            n_out=num_classes,
            ninp=128, # Default TabPFN emsize
            num_features=100, # Default max features
            device=self.device
        )
        return model
