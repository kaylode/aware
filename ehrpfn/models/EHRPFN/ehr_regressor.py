import math
import torch
import numpy as np

from .utils import pad_x, generate_random_permutation
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .estimator import TabDPTEstimator
from sklearn.base import RegressorMixin

from ehrpfn.utils.retrieval import FAISS, ProcessRetriever, TrajectoriesRetriever, ProcessTFIDFRetriever


class EHRPFNRegressor(TabDPTEstimator, RegressorMixin):
    def __init__(
            self, pretrained_path:str=None, inf_batch_size: int = 512,
            device: str = TabDPTEstimator._DEFAULT_DEVICE, 
            use_flash: bool = True, compile: bool = True, 
            categorical_feature_indices: list = None,
            retrieval_method:str = "faiss", 
        ):
        super().__init__(mode="reg", pretrained_path=pretrained_path, inf_batch_size=inf_batch_size, device=device, use_flash=use_flash, compile=compile, categorical_feature_indices=categorical_feature_indices)
        self.retrieval_method = retrieval_method

        assert self.retrieval_method in ["faiss", "trajectories", "random", "activities", "knn"], \
            "retrieval_method must be one of ['faiss', 'trajectories', 'random', 'activities', 'activities_tfidf_kmeans']"

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


        if self.retrieval_method in ['knn', "faiss"]:
            self.retriever = FAISS(X)
        elif self.retrieval_method == "activities":
            self.retriever = ProcessRetriever(X, X_processes=kwargs.get('X_processes', None), pathway_column='pathway_with_prev_2_encoded')
        elif self.retrieval_method == "random":
            self.retriever = None
        elif self.retrieval_method == "trajectories":
            self.retriever = TrajectoriesRetriever(X, X_traj=kwargs.get('X_traj', None))
        else:
            raise ValueError(f"Invalid retrieval method: {self.retrieval_method}")


    @torch.no_grad()
    def _ensemble_predict(self, X: np.ndarray, n_ensembles:int=1, context_size: int = 128, seed=None, **kwargs):
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
                X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
                X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
                y_train = train_y[None, :].float()
                pred = self.model(
                    x_src=torch.cat([X_train, X_test], dim=1),
                    y_src=y_train.unsqueeze(-1),
                    task=self.mode,
                )

                logits = pred.float().squeeze().detach().cpu().float().numpy()
                pbar.update(X_test.shape[0])
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

                    pred = self.model(
                        x_src=torch.cat([X_nni, X_eval], dim=1),
                        y_src=y_nni.unsqueeze(-1),
                        task=self.mode,
                    )

                    pred_list.append(pred.squeeze())
                    pbar.update(end - start)

                logits = torch.cat(pred_list).squeeze().detach().cpu().float().numpy()

            if logits_cumsum is None:
                logits_cumsum = logits
            else:
                logits_cumsum += logits

            logits_cumsum /= n_ensembles

        return logits_cumsum


    def predict(self, X: np.ndarray, n_ensembles: int = 1, context_size: int = 128, seed=None,  **kwargs):
        return self._ensemble_predict(X, n_ensembles=n_ensembles, context_size=context_size, **kwargs)