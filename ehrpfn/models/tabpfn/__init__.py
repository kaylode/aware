from .backbone_v1 import load_model_workflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..EHRPFN.ehr_classifier import EHRPFNClassifier
import numpy as np

_TABPFN_NUM_FEATURES = 100   # TabPFN v1 expects exactly 100 input features

class TabPFNModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_features = _TABPFN_NUM_FEATURES
        self.n_out = 10 # TabPFN v1 has 10 outputs

    def forward(self, x, y, task='cls'):
        # x: (T_train + T_test, B, F)
        # y: (T_train, B)
        # EHRPFNClassifier passes x as (T_train+T_test, B, F) and y as (T_train, B)
        T_train = y.shape[0]
        x_train = x[:T_train]
        x_test = x[T_train:]
        y_train = y
        
        # The internal Wrapper in backbone_v1 expects x_train, y_train, x_test
        # and it handles to(device) and float32 conversion.
        res = self.model(x_train, y_train, x_test)
        # res['logits'] is (T_test, B, 10)
        return res['logits']

class TabPFNv1(EHRPFNClassifier):
    def __init__(self, n_classes, context_size, device="cuda", retriever=None):
        base_model = load_model_workflow(device=device, only_inference=True)
        wrapped_model = TabPFNModelWrapper(base_model)
        
        # EHRPFNClassifier calls EHRPFNEstimator.__init__
        # EHRPFNEstimator.__init__ sets self.model = base_model if pretrained_path is None
        super().__init__(
            base_model=wrapped_model,
            device=device,
            retriever=retriever,
            fit_retriever_before_fit=True,
            n_ensembles=1,
        )
        self.n_classes = n_classes
        self.context_size = context_size
        if self.retriever is not None:
            self.retriever.k = context_size

    def fit(self, X, y, **kwargs):
        # Ensure X, y are numpy for EHRPFNClassifier.fit
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
            
        return super().fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):
        # Ensure X is numpy
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        output = super().predict_proba(X, **kwargs)
        return output

    def predict(self, X, **kwargs):
        # Ensure X is numpy
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        # EHRPFNClassifier.predict returns probabilities, but we want classes
        # to maintain backward compatibility with the old TabPFNv1.predict
        probs = self.predict_proba(X, **kwargs)
        return np.argmax(probs, axis=-1)
