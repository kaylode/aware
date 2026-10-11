import faiss
import numpy as np
import torch
from typing import *
from .aware import SNNLAttnRetriever

class ImbalanceAwareAWARE(SNNLAttnRetriever):
    def __init__(
        self,
        *args,
        candidate_multiplier: int = 5,
        class_weight_power: float = 1.0,   # 1.0 = inverse freq, 0.5 softer
        distance_weight_power: float = 1.0,
        deterministic: bool = False,
        use_gpu: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.candidate_multiplier = candidate_multiplier
        self.class_weight_power = class_weight_power
        self.distance_weight_power = distance_weight_power
        self.deterministic = deterministic
        self.use_gpu = use_gpu

    def fit(self, items, **kwargs):
        super().fit(items, **kwargs)
        self.labels = np.array([item["label"] for item in items])
        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_freq = dict(zip(unique, counts))

        # ── Pre-compute class weights for faster vectorized lookup ──
        max_label = int(self.labels.max())
        self.class_weights_arr = np.zeros(max_label + 1)
        for y, count in self.class_freq.items():
            self.class_weights_arr[int(y)] = 1.0 / (count ** self.class_weight_power)

        # ── Move FAISS index to GPU if requested ──
        if kwargs.get("use_gpu", self.use_gpu):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.use_gpu = True
                print("Moved FAISS index to GPU.")
            except Exception as e:
                print(f"Warning: Could not move FAISS index to GPU: {e}")

        return getattr(self, "_cv_metrics", None)

    def retrieve(self, items, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("Retriever is not fitted yet.")

        # 1. Get Embeddings (Vectorized)
        X_query = np.array([item[self.field_to_use] for item in items])
        device = next(self.model.parameters()).device
        X_tensor = torch.FloatTensor(X_query).to(device)
        
        # Get tensors directly from the base class
        query_embeddings = self._get_ensemble_embeddings(
            X_tensor, track_attention=False, return_tensor=True
        )

        if self.distance == "cosine":
            # Vectorized normalization on GPU
            norm = torch.norm(query_embeddings, p=2, dim=1, keepdim=True)
            query_embeddings = query_embeddings / (norm + 1e-12)

        k = kwargs.get("k", self.k)
        ntotal = self.index.ntotal
        
        # If we have fewer samples than requested k, settle for 32 (or ntotal if even less)
        if ntotal < k:
            k = min(ntotal, 32)
            
        k_candidate = min(k * self.candidate_multiplier, ntotal)

        # 2. Search all queries at once
        # Convert to numpy for search (most stable with FAISS GPU)
        query_embeddings_np = query_embeddings.cpu().numpy()
        scores, indices = self.index.search(query_embeddings_np, k_candidate)

        # 3. Vectorized Weighting & Sampling (using PyTorch for speed)
        scores_t = torch.from_numpy(scores).to(device)
        indices_t = torch.from_numpy(indices).to(device)
        labels_t = torch.from_numpy(self.labels).to(device)
        class_weights_t = torch.from_numpy(self.class_weights_arr).to(device)

        # Fast lookup of candidate labels and their class weights
        # We cast to .long() to ensure they can be used as indices
        cand_labels = labels_t[indices_t.long()].long()  # (batch, k_candidate)
        class_w = class_weights_t[cand_labels]           # (batch, k_candidate)

        if self.distance == "cosine":
            # For cosine: larger score is better. Scale by max for numerical stability.
            sim_w = torch.exp((scores_t - scores_t.max(dim=1, keepdim=True)[0]) * self.distance_weight_power)
        else:
            # For L2: smaller distance is better.
            sim_w = torch.exp(-(scores_t - scores_t.min(dim=1, keepdim=True)[0]) * self.distance_weight_power)

        weights = class_w * sim_w
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        k = min(k, weights.size(1))

        # 4. Selection
        # if self.deterministic:
        #     _, selected_idx = torch.topk(weights, k, dim=1)
        #     selected = torch.gather(indices_t, 1, selected_idx)
        # else:
        # torch.multinomial handles the entire batch at once on GPU
        selected_idx = torch.multinomial(weights, k, replacement=False)
        selected = torch.gather(indices_t, 1, selected_idx)

        return selected.cpu().numpy()
