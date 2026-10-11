import torch
import numpy as np
import os

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .model import EHRPFNModel
from .utils import convert_to_torch_tensor, download_model
from ehrpfn.retrieval import Retriever


class EHRPFNEstimator(BaseEstimator):
    _DEFAULT_DEVICE = "cuda:0"

    def __init__(self, 
            mode: str, pretrained_path:str=None, inf_batch_size: int = 512, 
            n_ensembles: int = 1, temperature: float = 0.8,
            device: str = _DEFAULT_DEVICE,
            use_flash: bool = True, compile: bool = True, categorical_feature_indices: list = None,
            retriever: Retriever = None,
            fit_retriever_before_fit: bool = True,
            base_model: EHRPFNModel = None, 
        ):
        self.mode = mode
        self.inf_batch_size = inf_batch_size
        self.device = device
        self.use_flash = use_flash
        path = pretrained_path
        self.categorical_feature_indices = categorical_feature_indices
        self.retriever = retriever
        self.fit_retriever_before_fit = fit_retriever_before_fit
        self.n_ensembles = n_ensembles
        self.temperature = temperature

        # automatically download model weight if path is not found
        if path is not None:
            if not os.path.exists(path):
                print(f"Path {path} does not exist in {os.getcwd()}.")
                path = download_model()
            self.path = path
            checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
            checkpoint["cfg"]["env"]["device"] = self.device
            self.model = EHRPFNModel.load(model_state=checkpoint["model"], config=checkpoint["cfg"], use_flash=use_flash)
        else:
            assert base_model is not None, "Either pretrained_path or base_model must be provided"
            self.model = base_model
        self.model.eval()

        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile
        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"

    def _prepare_prediction(self, X: np.ndarray):
        check_is_fitted(self)

        self.X_test_ori = X
        self.X_test = self.imputer.transform(X)
        self.X_test = self.scaler.transform(self.X_test)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA optionally to reduce the number of features
        if self.n_features > self.max_features:
            U, S, self.V = torch.pca_lowrank(train_x, q=self.max_features)
            train_x = train_x @ self.V
        else:
            self.V = None

        test_x = test_x @ self.V if self.V is not None else test_x
        return train_x, train_y, test_x
