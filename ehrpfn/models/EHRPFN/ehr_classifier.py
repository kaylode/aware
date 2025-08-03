import numpy as np
import torch
import math
from .utils import pad_x, generate_random_permutation
from .classifier import TabDPTClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ehrpfn.utils.retrieval import FAISS, ProcessRetriever, TrajectoriesRetriever, ProcessTFIDFRetriever

class EHRPFNClassifier(TabDPTClassifier):

    def __init__(self, retrieval_method:str = "faiss", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_method = retrieval_method

        assert self.retrieval_method in ["faiss", "trajectories", "random", "activities", "activities_tfidf_kmeans"], \
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


        if self.retrieval_method == "faiss":
            self.retriever = FAISS(X)
        elif self.retrieval_method == "activities":
            self.retriever = ProcessRetriever(X, X_processes=kwargs.get('X_processes', None), pathway_column='pathway_with_prev_2_encoded')
        elif self.retrieval_method == "activities_tfidf_kmeans":
            self.retriever = ProcessTFIDFRetriever(X, X_processes=kwargs.get('X_processes', None))
            self.predict_proba = self.predict_proba_efficient
        elif self.retrieval_method == "trajectories":
            self.retriever = TrajectoriesRetriever(X, X_traj=kwargs.get('X_traj', None))
        elif self.retrieval_method == "random":
            self.retriever = None
        else:
            raise ValueError(f"Invalid retrieval method: {self.retrieval_method}")

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, temperature: float = 0.8, context_size: int = 1024, return_logits: bool = False, seed: int = None, **kwargs):
        train_x, train_y, test_x = self._prepare_prediction(X)

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
            y_train = train_y[None, :].float()

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
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

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
                        self.X_test[start:end], k=context_size,
                        queries_processes=queries_processes,
                        queries_traj=queries_traj
                    )

                X_nni = train_x[torch.tensor(indices_nni)]
                y_nni = train_y[torch.tensor(indices_nni)]

                X_nni, y_nni = (
                    pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                    torch.Tensor(y_nni).to(self.device),
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device)
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
            pred_val = torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()
        return pred_val

    @torch.no_grad()
    def predict_proba_efficient(self, X: np.ndarray, temperature: float = 0.8, context_size: int = 1024, return_logits: bool = False, seed: int = None, **kwargs):
        train_x, train_y, test_x = self._prepare_prediction(X)

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
            y_train = train_y[None, :].float()

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
            pred_list = {}

            # Efficient inference with batch processing
            # Batch test set based on cluster IDs
            if self.retrieval_method == "activities_tfidf_kmeans":
                queries_processes = kwargs.get('X_processes', None)
                
                if queries_processes is None:
                    raise ValueError("X_processes must be provided for activities_tfidf_kmeans retrieval method")
                cluster_ids = queries_processes['cluster_id'].to_numpy()
                unique_cluster_ids = np.unique(cluster_ids)
                cluster_to_indices = {cid: np.where(cluster_ids == cid)[0] for cid in unique_cluster_ids}
            else:
                raise ValueError(f"Invalid retrieval method for efficient inference: {self.retrieval_method}")
            
            for cluster_id, indices in cluster_to_indices.items():
                if len(indices) == 0: continue

                for b in range(math.ceil(len(indices) / self.inf_batch_size)):
                    start = b * self.inf_batch_size
                    end = min(len(indices), (b + 1) * self.inf_batch_size)
                    tmp_indices = indices[start:end]
                    queries_processes = kwargs.get('X_processes', None)
                    if queries_processes is not None:
                        queries_processes = queries_processes[tmp_indices]
                    queries_traj = kwargs.get('X_traj', None)
                    if queries_traj is not None:
                        queries_traj = queries_traj[tmp_indices]

                    indices_nni = self.retriever.retrieve(
                        self.X_test[tmp_indices], k=context_size,
                        queries_processes=queries_processes,
                        queries_traj=queries_traj,
                        cluster_id=cluster_id
                    )

                    # Get the indices for the current cluster
                    X_nni = train_x[torch.tensor(indices_nni)]
                    y_nni = train_y[torch.tensor(indices_nni)]

                    X_nni, y_nni = (
                        pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                        torch.Tensor(y_nni).to(self.device),
                    )
                    X_eval = test_x[tmp_indices]
                    X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device)
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

                    preds = pred.squeeze(1).detach().cpu().float()

                    for idx, p in zip(tmp_indices, preds):
                        if idx in pred_list:
                            raise ValueError(f"Duplicate index {idx} found in predictions.")
                        pred_list[idx] = p
            pred_list = {k:v for k,v in sorted(pred_list.items(), key=lambda item: item[0])}
            pred_list = list(pred_list.values())
            pred_list = [i.unsqueeze(0) if i.ndim == 1 else i for i in pred_list]
            pred_val = torch.cat(pred_list, dim=0).squeeze().numpy()
        return pred_val