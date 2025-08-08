import numpy as np
import torch
import math
from .utils import pad_x, generate_random_permutation

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ehrpfn.utils.retrieval import FAISS, ProcessRetriever, TrajectoriesRetriever
from tqdm import tqdm
from scipy.special import softmax

from sklearn.base import ClassifierMixin
from .estimator import TabDPTEstimator



class EHRPFNClassifier(TabDPTEstimator, ClassifierMixin):

    def __init__(
        self, pretrained_path:str=None, inf_batch_size: int = 512,
        device: str = TabDPTEstimator._DEFAULT_DEVICE, use_flash: bool = True, 
        compile: bool = True, categorical_feature_indices: list = None, 
        retrieval_method:str = "faiss"
    ):
        super().__init__(
            mode="cls", 
            pretrained_path=pretrained_path, 
            inf_batch_size=inf_batch_size, 
            device=device, use_flash=use_flash, 
            compile=compile, 
            categorical_feature_indices=categorical_feature_indices
        )

        self.retrieval_method = retrieval_method


        assert self.retrieval_method in ["faiss", "trajectories", "random", "activities", "knn"], \
            "retrieval_method must be one of ['faiss', 'trajectories', 'random', 'activities']"

    def fit(self, X, y, **kwargs):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

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


        if self.retrieval_method in ['knn',"faiss"]:
            self.retriever = FAISS(X)
        elif self.retrieval_method == "activities":
            self.retriever = ProcessRetriever(X, X_processes=kwargs.get('X_processes', None), pathway_column='pathway_with_prev_2_encoded')
        elif self.retrieval_method == "trajectories":
            self.retriever = TrajectoriesRetriever(X, X_traj=kwargs.get('X_traj', None))
        elif self.retrieval_method == "random":
            self.retriever = None
        else:
            raise ValueError(f"Invalid retrieval method: {self.retrieval_method}")

    def _predict_large_cls(self, X_train, X_test, y_train):
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_train_digit = (y_train // (self.max_num_classes ** i)) % self.max_num_classes
            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=1),
                y_src=y_train_digit.unsqueeze(-1),
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
    def ensemble_predict_proba(self, X: np.ndarray, n_ensembles:int=1, temperature: float = 0.8, context_size: int = 1024, return_logits: bool = False, seed: int = None, **kwargs):
        
        train_x, train_y, test_x = self._prepare_prediction(X)
        pbar = tqdm(total=n_ensembles*len(X), desc="Predicting")

        logits_cumsum = None
        for _ in range(n_ensembles):

            seed = int(np.random.SeedSequence().generate_state(1)[0])

            if seed is not None:
                feat_perm = generate_random_permutation(train_x.shape[1], seed)
                X_train = train_x[:, feat_perm]
                X_test = test_x[:, feat_perm]
            else:
                X_train = train_x
                X_test = test_x

            if context_size >= self.n_instances:
                X_train = pad_x(X_train[None, :, :], self.max_features).to(self.device)
                X_test = pad_x(X_test[None, :, :], self.max_features).to(self.device)
                y_train = train_y[None, :].float()

                if seed is not None:
                    # shuffle X_train order
                    indices = np.arange(X_train.shape[0])
                    shuffled_indices = np.random.permutation(indices)
                    X_train = X_train[torch.to_tensor(shuffled_indices)]
                    y_train = y_train[torch.to_tensor(shuffled_indices)]

                if self.num_classes <= self.max_num_classes:
                    pred = self.model(
                        x_src=torch.cat([X_train, X_test], dim=1),
                        y_src=y_train.unsqueeze(-1),
                        task=self.mode,
                    )
                else:
                    pred = self._predict_large_cls(X_train, X_test, y_train)

                pred = pred[..., :self.num_classes] / temperature
                pred = torch.nn.functional.softmax(pred.float(), dim=-1)
                pred_val = pred.squeeze().detach().cpu().numpy()
            else:
                pred_list = []
                for b in range(math.ceil(len(X_test) / self.inf_batch_size)):
                    start = b * self.inf_batch_size
                    end = min(len(X_test), (b + 1) * self.inf_batch_size)

                    if self.retrieval_method == "random":
                        indices_nni = np.random.choice(
                            self.n_instances, size=context_size, replace=False
                        )
                        indices_nni = indices_nni[np.newaxis, ...].repeat(end-start, axis=0)
                    else:
                        queries_processes = kwargs.get('X_processes', None)
                        if queries_processes is not None:
                            queries_processes = queries_processes[start:end]
                        queries_traj = kwargs.get('X_traj', None)
                        if queries_traj is not None:
                            queries_traj = queries_traj[start:end]

                        indices_nni = self.retriever.retrieve(
                            X_test[start:end], k=context_size,
                            queries_processes=queries_processes,
                            queries_traj=queries_traj
                        )

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

                    if self.num_classes <= self.max_num_classes:
                        pred = self.model(
                            x_src=torch.cat([X_nni, X_eval], dim=1),
                            y_src=y_nni.unsqueeze(-1),
                            task=self.mode,
                        )
                    else:
                        pred = self._predict_large_cls(X_nni, X_eval, y_nni)

                    pred = pred.float()
                    if not return_logits:
                        pred = pred[..., :self.num_classes] / temperature
                        pred = torch.nn.functional.softmax(pred, dim=-1)
                        pred /= pred.sum(axis=-1, keepdims=True)  # numerical stability

                    pred_list.append(pred.squeeze())
                    pbar.update(end-start)
                pred_val = torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()
            logits = pred_val
            if logits_cumsum is None:
                logits_cumsum = logits
            else:
                logits_cumsum += logits

        pbar.close()
        pred = (logits_cumsum / n_ensembles)[..., :self.num_classes] / temperature
        pred = softmax(pred, axis=-1)
        pred /= pred.sum(axis=-1, keepdims=True)  # numerical stability
        return pred

    def predict(self, X, n_ensembles: int = 1, temperature: float = 0.8, context_size: int = 128, seed: int = None, **kwargs):
        return self.ensemble_predict_proba(X, n_ensembles=n_ensembles, temperature=temperature, context_size=context_size, **kwargs)