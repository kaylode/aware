"""
ehrpfn.models.AWARE.cls_finetune — Retrieval-guided classification finetuning.

Classes
-------
* RetrievalContextSampler  — fixed retriever wrapper (fitted once on raw scaled features)
* AWARECLSModel            — abstract nn.Module: adapter (trainable) + frozen backbone + head
* TabPFNAWARECLSModel      — TabPFN v1 backbone variant
* TabDPTAWARECLSModel      — TabDPT backbone variant
* BaseClsFinetune          — training loop (CE loss, retrieval-guided context)
* TabPFNClsFinetune        — concrete subclass for TabPFN
* TabDPTClsFinetune        — concrete subclass for TabDPT
* KNNPFNClsFinetune        — alias of TabPFNClsFinetune (k-NN comes from the retriever)
* AWARECLSClassifier       — sklearn-compatible wrapper
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ehrret import Retriever

from .adapter import AWAREAdapter


# ---------------------------------------------------------------------------
# Constants & Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_TABPFN_NUM_FEATURES = 100   # TabPFN v1 expects exactly 100 input features


# ---------------------------------------------------------------------------
# RetrievalContextSampler
# ---------------------------------------------------------------------------

class RetrievalContextSampler:
    """Fixed retriever wrapper.

    Fitted **once** on raw scaled training features before finetuning starts.
    Never refitted — retrieval always operates in the original scaled space.
    """

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RetrievalContextSampler":
        items = [{"raw_feature": X[i], "label": float(y[i])} for i in range(len(X))]
        self.retriever.fit(items)
        if getattr(self.retriever, "move_to_cpu", None) is not None:
            self.retriever.move_to_cpu()
        return self

    def retrieve_indices(self, X_batch: np.ndarray) -> np.ndarray:
        """Return (B, k) integer array of training-set neighbor indices."""
        items = [{"raw_feature": X_batch[i]} for i in range(len(X_batch))]
        return np.array(self.retriever.retrieve(items))

    @property
    def k(self) -> int:
        return self.retriever.k


# ---------------------------------------------------------------------------
# AWARECLSModel — abstract base
# ---------------------------------------------------------------------------

class AWARECLSModel(nn.Module):
    """Backbone (frozen) + adapter + LayerNorm + classification head.

    Only ``adapter``, ``norm``, and ``cls_head`` carry gradients.
    Subclasses implement ``_get_query_embeddings`` for each backbone type.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: AWAREAdapter,
        n_classes: int,
        ninp: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        is_classification: bool = True,
    ) -> None:
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad = False
        self.n_classes = n_classes
        self.backbone = backbone
        self.adapter = adapter                  # trainable
        self.norm = nn.LayerNorm(ninp)          # trainable
        self.residual_head = nn.Sequential(          # trainable
            nn.Linear(ninp, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, n_classes),
        )
        self.is_classification = is_classification
        
        # Initialize the final layer to exactly zero. 
        # This guarantees that at initialization, the residual head outputs exactly 0.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        x_query: torch.Tensor,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            x_query: (B, n_feat) — raw scaled query features.
            x_ctx:   (B, K, n_feat) — unique context features per query.
            y_ctx:   (B, K)        — context labels as float.

        Returns:
            logits: (B, n_classes)
        """
        q_emb   = self.adapter(x_query)   # (B, emb_dim)
        ctx_emb = self.adapter(x_ctx)     # (B, K, emb_dim)
        # q_emb, ctx_emb = x_query, x_ctx
        
        q_logits, q_backbone_emb = self._get_query_embeddings(q_emb, ctx_emb, y_ctx)

        # Calculate parametric residual and add it to zero-shot logits
        residual = self.residual_head(self.norm(q_backbone_emb))
        if not self.is_classification:
            var_y = y_ctx.var(dim=1, keepdim=True, unbiased=False)
            std_y = torch.sqrt(var_y + 1e-12)
            residual = residual * std_y

        out = q_logits + residual
        return out

    def _get_query_embeddings(
        self,
        q_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        y_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-query backbone embeddings.

        Args:
            q_emb:   (B, emb_dim)
            ctx_emb: (K, emb_dim)
            y_ctx:   (K,) float

        Returns:
            (B, ninp)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TabPFN backbone subclass
# ---------------------------------------------------------------------------

class TabPFNAWARECLSModel(AWARECLSModel):
    """AWARE CLS model with frozen TabPFN v1 backbone.

    The adapter output is zero-padded to the backbone's expected 100-feature
    input. TabPFN's internal Linear(100, ninp) encoder then maps it to the
    transformer embedding space.
    """

    def _pad_to_tabpfn(self, x: torch.Tensor) -> torch.Tensor:
        """Pad/truncate last dim to _TABPFN_NUM_FEATURES."""
        n = x.shape[-1]
        if n < _TABPFN_NUM_FEATURES:
            pad = torch.zeros(
                *x.shape[:-1], _TABPFN_NUM_FEATURES - n,
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=-1)
        return x[..., :_TABPFN_NUM_FEATURES]

    def _get_query_embeddings(self, q_emb, ctx_emb, y_ctx):
        # Pad adapter outputs to what TabPFN's encoder expects
        # q_emb: (B, D), ctx_emb: (B, K, D), y_ctx: (B, K)
        ctx_p = self._pad_to_tabpfn(ctx_emb).transpose(0, 1)   # (K, B, 100)
        q_p   = self._pad_to_tabpfn(q_emb).unsqueeze(0)        # (1, B, 100)
        y_ctx_p = y_ctx.transpose(0, 1)                        # (K, B)

        if not self.is_classification:
            # 1. Standardize targets along sequence dimension (dim=0 of y_ctx_p)
            mean_y = y_ctx_p.mean(dim=0, keepdim=True)         # (1, B)
            std_y = y_ctx_p.std(dim=0, keepdim=True) + 1e-6    # (1, B)
            y_ctx_p_norm = (y_ctx_p - mean_y) / std_y          # (K, B)

            # 2. Discretize into 10 bins (max_num_classes of backbone is 10)
            borders = torch.linspace(-2.0, 2.0, 11, device=y_ctx.device)
            bin_centers = 0.5 * (borders[:-1] + borders[1:])   # (10,)
            
            y_ctx_p_norm_clamped = torch.clamp(y_ctx_p_norm, min=-2.0, max=2.0)
            y_ctx_digits = torch.bucketize(y_ctx_p_norm_clamped, borders)
            y_ctx_digits = torch.clamp(y_ctx_digits - 1, 0, 9).float()

            # 3. Backbone Inference
            output = self.backbone(ctx_p, y_ctx_digits, q_p)
            logits = output['logits']                          # (1, B, 10)
            emb = output["test_embeddings"].to(torch.float32)  # (1, B, ninp)

            # Squeeze sequence dimension
            if emb.ndim == 3:
                emb = emb.squeeze(0)                           # (B, ninp)
                logits = logits.squeeze(0)                     # (B, 10)

            # 4. Expectation & Denormalization
            probs = torch.softmax(logits, dim=-1)              # (B, 10)
            pred_norm = (probs * bin_centers.unsqueeze(0)).sum(dim=-1, keepdim=True) # (B, 1)

            mean_y_b = mean_y.transpose(0, 1)                  # (B, 1)
            std_y_b = std_y.transpose(0, 1)                    # (B, 1)
            pred = pred_norm * std_y_b + mean_y_b              # (B, 1)
            
            return pred, emb
        else:
            # Backbone Wrapper.forward(x_train, y_train, x_test)
            # Returns dict: test_embeddings shape (1, B, ninp)
            output = self.backbone(ctx_p, y_ctx_p, q_p)
            logits = output['logits'][..., :self.n_classes]
            emb = output["test_embeddings"].to(torch.float32)
            
            # Squeeze sequence dimension (TabPFN returns Seq, Batch, Dim)
            if emb.ndim == 3:
                emb = emb.squeeze(0)    # (B, ninp)
                logits = logits.squeeze(0)  # (B, n_classes)
            return logits, emb


# ---------------------------------------------------------------------------
# TabDPT backbone subclass
# ---------------------------------------------------------------------------

class TabDPTAWARECLSModel(AWARECLSModel):
    """AWARE CLS model with frozen TabDPT backbone.

    The adapter output is zero-padded to ``max_features`` and unsqueezed to
    (N, 1, max_features) — the format expected by TabDPT's sequence encoder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: AWAREAdapter,
        n_classes: int,
        ninp: int,
        max_features: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        is_classification: bool = True,
    ) -> None:
        super().__init__(backbone, adapter, n_classes, ninp, head_hidden_dim, dropout, is_classification)
        self.max_features = max_features

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Pad/truncate last dim to max_features."""
        n = x.shape[-1]
        if n < self.max_features:
            pad = torch.zeros(
                *x.shape[:-1], self.max_features - n,
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=-1)
        return x[..., : self.max_features]

    def _get_query_embeddings(self, q_emb, ctx_emb, y_ctx):
        # q_emb: (B, D), ctx_emb: (B, K, D), y_ctx: (B, K)
        K = ctx_emb.shape[1]
        ctx_seq = self._to_seq(ctx_emb).transpose(0, 1)     # (K, B, max_features)
        q_seq   = self._to_seq(q_emb).unsqueeze(0)          # (1, B, max_features)
        x_full  = torch.cat([ctx_seq, q_seq], dim=0)        # (K+1, B, max_features)
        y_ctx_p = y_ctx.transpose(0, 1)                     # (K, B)

        if not self.is_classification:
            # Standardize targets along sequence dimension (dim=0)
            mean_y = y_ctx_p.mean(dim=0, keepdim=True)
            std_y = y_ctx_p.std(dim=0, keepdim=True) + 1e-6
            y_ctx_p_norm = (y_ctx_p - mean_y) / std_y
        else:
            y_ctx_p_norm = y_ctx_p

        # TabDPTModel.forward(x, y, eval_pos, return_embeddings=True)
        # returns (pred[eval_pos:], src)
        logits, emb = self.backbone(
            x_full, y=y_ctx_p_norm.float(), eval_pos=K, return_embeddings=True
        )

        if not self.is_classification:
            # Slice the regression prediction channel (last index) and keep dimension for squeeze
            reg_logits = logits[..., -1:]  # (1, B, 1)
            # Denormalize
            reg_logits = reg_logits * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)
            logits = reg_logits
        else:
            logits = logits[..., :self.n_classes]

        return logits.squeeze(0), emb.squeeze(0).to(torch.float32)         # (B, ninp)


# ---------------------------------------------------------------------------
# ClassBalancedBatchSampler
# ---------------------------------------------------------------------------

class ClassBalancedBatchSampler:
    """Generates batch index arrays guaranteeing a minimum number of samples
    per batch for every class.
    """
    def __init__(
        self,
        y: np.ndarray,
        batch_size: int,
        n_classes: int,
        min_samples_per_class: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self._rng = np.random.default_rng(seed)
        self._n = len(self.y)

        self._class_indices = [np.where(self.y == c)[0] for c in range(n_classes)]
        
        for c, idxs in enumerate(self._class_indices):
            if len(idxs) == 0:
                raise ValueError(f"Training fold has no samples for class {c}.")

        default_min = max(1, batch_size // (n_classes * 4))
        raw = min_samples_per_class if min_samples_per_class is not None else default_min
        
        self.min_samples_per_class = int(min(raw, batch_size // n_classes))
        self._fill_size = batch_size - (self.min_samples_per_class * n_classes)

    def __iter__(self):
        class_perms = [self._rng.permutation(idxs) for idxs in self._class_indices]
        class_ptrs = [0] * self.n_classes
        all_perm = self._rng.permutation(self._n)
        f_ptr = 0

        n_batches = max(1, self._n // self.batch_size)
        
        for _ in range(n_batches):
            batch_chunks = []
            
            for c in range(self.n_classes):
                e_ptr = class_ptrs[c]
                e_perm = class_perms[c]
                e_end = e_ptr + self.min_samples_per_class
                
                if e_end <= len(e_perm):
                    batch_chunks.append(e_perm[e_ptr:e_end])
                    class_ptrs[c] = e_end
                else:
                    tail = e_perm[e_ptr:]
                    e_perm = self._rng.permutation(self._class_indices[c])
                    class_perms[c] = e_perm
                    need = self.min_samples_per_class - len(tail)
                    batch_chunks.append(tail)
                    if need > 0:
                        batch_chunks.append(e_perm[:need])
                    class_ptrs[c] = need

            if self._fill_size > 0:
                need = self._fill_size
                chunks = []
                while need > 0:
                    avail = all_perm[f_ptr:]
                    take = min(need, len(avail))
                    if take > 0:
                        chunks.append(avail[:take])
                    need -= take
                    f_ptr += take
                    if f_ptr >= len(all_perm):
                        all_perm = self._rng.permutation(self._n)
                        f_ptr = 0
                if chunks:
                    batch_chunks.append(np.concatenate(chunks))
            
            batch_np = np.concatenate(batch_chunks)
            self._rng.shuffle(batch_np)
            yield batch_np
            
    def __len__(self) -> int:
        return max(1, self._n // self.batch_size)


# ---------------------------------------------------------------------------
# RegressionBatchSampler
# ---------------------------------------------------------------------------

class RegressionBatchSampler:
    """Generates batch index arrays for regression tasks by shuffling the dataset using balance sampling."""
    def __init__(
        self,
        y: np.ndarray,
        batch_size: int,
        seed: Optional[int] = None,
        num_bins: int = 10,
    ):
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._n = len(self.y)
        self.num_bins = num_bins

        # Discretize targets into bins, compute bin frequencies, and assign inverse-frequency weights
        bins = np.quantile(self.y, np.linspace(0, 1, self.num_bins + 1))
        bin_ids = np.digitize(self.y, bins[1:-1])
        bin_counts = np.bincount(bin_ids)
        # Smooth bin counts to avoid extreme weights for rare target regions
        smoothing = max(1.0, len(self.y) / (self.num_bins * 10.0))
        weights = 1.0 / (bin_counts[bin_ids] + smoothing)
        
        # Normalize weights so they sum to 1.0
        self.probabilities = weights / np.sum(weights)

    def __iter__(self):
        n_batches = max(1, self._n // self.batch_size)
        indices = np.arange(self._n)
        
        for _ in range(n_batches):
            # Sample batch_size indices with replacement using computed probabilities
            batch_idx = self._rng.choice(
                indices,
                size=self.batch_size,
                replace=True,
                p=self.probabilities
            )
            yield batch_idx
            
    def __len__(self) -> int:
        return max(1, self._n // self.batch_size)



# ---------------------------------------------------------------------------
# BaseClsFinetune — training loop
# ---------------------------------------------------------------------------

class BaseClsFinetune:
    """Base class for retrieval-guided classification finetuning.

    Trains adapter + classification head with CE loss.
    The retriever is fitted once on raw scaled features and kept fixed for
    the entire training run — context selection always operates in the
    original scaled feature space.

    Subclasses implement ``_build_model`` to load the appropriate backbone.
    """

    def __init__(
        self,
        retriever: Retriever,
        learning_rate: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 64,
        context_size: int = 64,
        patience: Optional[int] = 10,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
        adapter_hidden_dim: int = 512,
        val_split: float = 0.1,
        device: str = "cuda:0",
        is_classification: bool = True,
        random_state: int = 42,
    ) -> None:
        self.retriever = retriever
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.context_size = context_size
        self.patience = patience
        self.dropout = dropout
        self.head_hidden_dim = head_hidden_dim
        self.adapter_hidden_dim = adapter_hidden_dim
        self.val_split = val_split
        self.device = device
        self.random_state = random_state
        self.is_classification = is_classification

        # Populated by fit()
        self.cls_model: Optional[AWARECLSModel] = None
        self.context_sampler: Optional[RetrievalContextSampler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.n_classes: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_model(self, n_features: int, n_classes: int) -> AWARECLSModel:
        """Instantiate and return the backbone-specific AWARECLSModel."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> "BaseClsFinetune":
        rng = np.random.default_rng(self.random_state)

        # ── 1. Preprocessing ──────────────────────────────────────────
        logger.info(f"Preprocessing {len(X)} samples...")
        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
 
        if self.is_classification:
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)
        else:
            self.classes_ = None
            self.n_classes = 1

        # Save full data before any splits so it can be used as context during inference
        X_full = X.copy()
        y_full = y.copy()

        # ── 2. Validation split ───────────────────────────────────────
        if val_data is None and self.val_split > 0:
            logger.info(f"Creating validation split (ratio={self.val_split})...")
            X, X_val, y, y_val = train_test_split(
                X, y,
                test_size=self.val_split,
                random_state=self.random_state,
                stratify=y if self.is_classification else None,
            )
            val_data = (X_val, y_val)

        N = len(X)

        # ── 3. Fit retriever (once, fixed) ────────────────────────────
        logger.info("Fitting retriever...")
        self.context_sampler = RetrievalContextSampler(self.retriever)
        self.context_sampler.fit(X, y)

        # ── 3.5 Pre-cache retrieval indices (eliminates retriever from hot path) ──
        # One retriever call per dataset instead of one per batch per epoch.
        logger.info("Pre-caching retrieval indices for training set...")
        train_ctx_cache = self.context_sampler.retrieve_indices(X)   # (N, k)

        val_ctx_cache: Optional[np.ndarray] = None
        X_val_s:       Optional[np.ndarray] = None
        X_val_t:       Optional[torch.Tensor] = None
        if val_data is not None:
            X_val_raw, _ = val_data
            X_val_s = self.scaler.transform(self.imputer.transform(X_val_raw))
            logger.info("Pre-caching retrieval indices for validation set...")
            val_ctx_cache = self.context_sampler.retrieve_indices(X_val_s)  # (N_val, k)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=self.device)

        # ── 4. Build model ────────────────────────────────────────────
        logger.info(f"Building AWARE model (n_features={X.shape[1]}, n_classes={self.n_classes})...")
        self.cls_model = self._build_model(X.shape[1], self.n_classes).to(self.device)

        # ── 5. Pinned training tensors ────────────────────────────────
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long if self.is_classification else torch.float32, device=self.device)

        # ── 6. Optimisation ───────────────────────────────────────────
        trainable_params = [p for p in self.cls_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.learning_rate * 0.01
        )

        if self.is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # ── 7. Training loop ──────────────────────────────────────────
        best_loss, best_state, patience_ctr = float("inf"), None, 0

        if self.is_classification:
            _train_sampler = ClassBalancedBatchSampler(
                y=y,
                batch_size=self.batch_size,
                n_classes=self.n_classes,
                seed=self.random_state,
            )
        else:
            _train_sampler = RegressionBatchSampler(
                y=y,
                batch_size=self.batch_size,
                seed=self.random_state,
            )

        pbar = tqdm(range(self.epochs), desc="Finetuning", disable=not verbose)
        for epoch in pbar:
            self.cls_model.train()
            epoch_loss, n_batches = 0.0, 0
            n_total_batches = len(_train_sampler)
            batch_pbar = tqdm(_train_sampler, desc=f"  Epoch {epoch}", total=n_total_batches, leave=False, disable=not verbose)
            for i, batch_idx in enumerate(batch_pbar):
                y_batch    = y_t[batch_idx]           # (B,) long

                cached_rows = train_ctx_cache[batch_idx]              # (B, k)
                # Get unique neighbors per query
                ctx_indices = cached_rows[:, :self.context_size]      # (B, K)
                x_ctx = X_t[ctx_indices]                              # (B, K, n_feat)
                y_ctx = y_t[ctx_indices].float()                      # (B, K)
                x_q   = X_t[batch_idx]                                # (B, n_feat)

                optimizer.zero_grad()
                logits = self.cls_model(x_q, x_ctx, y_ctx)

                if self.is_classification:
                    loss   = criterion(logits, y_batch)
                else:
                    loss   = criterion(logits.squeeze(-1), y_batch)

                if not torch.isfinite(loss):
                    breakpoint()
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cls_model.parameters(), max_norm=1.0)
                optimizer.step()

                if torch.isnan(trainable_params[-1]).any():
                    breakpoint()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches  += 1

                if verbose:
                    batch_pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
                    if i % max(1, n_total_batches // 5) == 0:
                        logger.debug(f"    Epoch {epoch} | Batch {i:3d}/{n_total_batches} | loss={batch_loss:.4f}")

            batch_pbar.close()
            scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)

            # ── Validation ────────────────────────────────────────────
            val_metrics = None
            if val_data is not None:
                _, y_val = val_data
                val_metrics = self._compute_val_metrics(
                    X_val_t, y_val, val_ctx_cache, X_t, y_t, criterion
                )

            # Early stopping is driven by val_loss when available, otherwise train_loss
            stop_loss = val_metrics["val_loss"] if val_metrics is not None else avg_loss

            if verbose:
                msg = f"epoch {epoch:3d}/{self.epochs}  train_loss={avg_loss:.4f}"
                postfix = {"train_loss": f"{avg_loss:.4f}"}
                if val_metrics is not None:
                    if self.is_classification:
                        msg += (
                            f"  val_loss={val_metrics['val_loss']:.4f}"
                            f"  auc={val_metrics['auc']:.4f}"
                            f"  auprc={val_metrics['auprc']:.4f}"
                            f"  f1={val_metrics['f1']:.4f}"
                        )
                        postfix.update({
                            "val_loss": f"{val_metrics['val_loss']:.4f}",
                            "auc":      f"{val_metrics['auc']:.4f}",
                            "auprc":    f"{val_metrics['auprc']:.4f}",
                            "f1":       f"{val_metrics['f1']:.4f}",
                        })
                    else:
                        msg += (
                            f"  val_loss={val_metrics['val_loss']:.4f}"
                            f"  mse={val_metrics['mse']:.4f}"
                            f"  r2={val_metrics['r2']:.4f}"
                        )
                        postfix.update({
                            "val_loss": f"{val_metrics['val_loss']:.4f}",
                            "mse":      f"{val_metrics['mse']:.4f}",
                            "r2":       f"{val_metrics['r2']:.4f}",
                        })
                pbar.set_postfix(postfix)
                logger.info(msg)

            if stop_loss < best_loss:
                best_loss    = stop_loss
                best_state   = copy.deepcopy(self.cls_model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if self.patience is not None and patience_ctr >= self.patience:
                    if verbose:
                        logger.info(
                            f"Early stopping at epoch {epoch}  "
                            f"(best val_loss={best_loss:.4f})"
                        )
                    break

        if best_state is not None:
            self.cls_model.load_state_dict(best_state)

        self.cls_model.eval()
        
        # Re-fit the retriever on the full data (train + val) for inference context
        if val_data is not None:
            logger.info("Re-fitting retriever on full training set (including validation split) for inference...")
            self.context_sampler.fit(X_full, y_full)
            
        self.X_train_scaled = X_full
        self.y_train = y_full
        return self

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_val_metrics(
        self,
        X_val_t: torch.Tensor,
        y_val: np.ndarray,
        val_ctx_cache: np.ndarray,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        criterion: nn.Module,
    ) -> dict:
        """Run one full validation pass and return loss + classification metrics.

        Accepts pre-scaled, pre-cached inputs so no retriever calls or
        re-scaling happen inside the loop.

        Args:
            X_val_t:       (N_val, n_feat) GPU tensor — pre-scaled validation features.
            y_val:         (N_val,) numpy array of true labels.
            val_ctx_cache: (N_val, k) pre-cached neighbor indices into X_t.
            X_t:           (N_train, n_feat) GPU tensor — training features.
            y_t:           (N_train,) GPU tensor — training labels (long).
            criterion:     loss function.

        Returns:
            dict with keys: val_loss, auc, auprc, f1
        """
        N_val = X_val_t.shape[0]
        self.cls_model.eval()
        total_loss, n_batches = 0.0, 0
        all_probs = []

        for start in range(0, N_val, self.batch_size):
            end  = min(N_val, start + self.batch_size)
            x_vb = X_val_t[start:end]                                    # already on device
            y_vb = torch.tensor(y_val[start:end], dtype=torch.long if self.is_classification else torch.float32, device=self.device)

            # Get unique neighbors per query
            ctx_indices = val_ctx_cache[start:end, :self.context_size]
            x_ctx = X_t[ctx_indices]
            y_ctx = y_t[ctx_indices].float()

            logits = self.cls_model(x_vb, x_ctx, y_ctx)                  # (B, n_classes)
            if self.is_classification:
                loss = criterion(logits, y_vb)
            else:
                loss = criterion(logits.squeeze(-1), y_vb)
            total_loss += loss.item()
            n_batches  += 1

            if self.is_classification:
                probs = torch.softmax(logits, dim=-1).cpu().float().numpy()
            else:
                probs = logits.cpu().float().numpy()
            all_probs.append(probs)

        val_loss = total_loss / max(1, n_batches)
        all_probs = np.concatenate(all_probs, axis=0)   # (N_val, n_classes)

        if self.is_classification:
            y_pred    = np.argmax(all_probs, axis=1)
            is_binary = self.n_classes == 2

            try:
                if is_binary:
                    auc   = roc_auc_score(y_val, all_probs[:, 1])
                    auprc = average_precision_score(y_val, all_probs[:, 1])
                else:
                    auc   = roc_auc_score(y_val, all_probs, multi_class="ovr", average="macro")
                    auprc = float(np.mean([
                        average_precision_score(
                            (y_val == c).astype(int), all_probs[:, c]
                        )
                        for c in range(self.n_classes)
                    ]))
            except ValueError:
                # Rare edge: a class missing from val fold
                auc = auprc = float("nan")

            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            return {"val_loss": val_loss, "auc": auc, "auprc": auprc, "f1": f1}
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            preds = all_probs.squeeze(-1)
            mse = mean_squared_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            return {"val_loss": val_loss, "mse": mse, "r2": r2}


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class TabPFNClsFinetune(BaseClsFinetune):
    """Classification finetuning with frozen TabPFN v1 backbone."""

    def _build_model(self, n_features: int, n_classes: int) -> TabPFNAWARECLSModel:
        from ehrpfn.models.AWARE.tabpfn.backbone_v1 import load_model_workflow
        # from ehrpfn.models.AWARE.tabpfn.backbone_v2 import load_model_workflow

        backbone = load_model_workflow(device=self.device, only_inference=True)

        # Probe actual transformer embedding dimension
        with torch.no_grad():
            d_x = torch.randn(1, 1, _TABPFN_NUM_FEATURES, device=self.device)
            d_y = torch.zeros(1, 1, device=self.device)
            d_out = backbone(d_x, d_y, d_x)
            ninp = d_out["test_embeddings"].shape[-1]

        adapter = AWAREAdapter(
            n_features=n_features,
            embedding_dim=_TABPFN_NUM_FEATURES,   # adapter → 100-dim → TabPFN encoder → ninp
            hidden_dim=self.adapter_hidden_dim,
            dropout=self.dropout,
        )

        return TabPFNAWARECLSModel(
            backbone=backbone,
            adapter=adapter,
            n_classes=n_classes,
            ninp=ninp,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
            is_classification=self.is_classification,
        )


class TabDPTClsFinetune(BaseClsFinetune):
    """Classification finetuning with frozen TabDPT backbone."""

    def _build_model(self, n_features: int, n_classes: int) -> TabDPTAWARECLSModel:
        from ehrpfn.models.AWARE.tabdpt.model import TabDPTModel, TABDPT_CHECKPOINT_PATH

        ckpt     = torch.load(TABDPT_CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        backbone = TabDPTModel.load(
            model_state=ckpt["model"],
            config=ckpt["cfg"],
        )
        backbone.to(self.device)

        ninp         = backbone.ninp
        max_features = backbone.num_features

        adapter = AWAREAdapter(
            n_features=n_features,
            embedding_dim=max_features,   # adapter → max_features-dim → TabDPT encoder → ninp
            hidden_dim=self.adapter_hidden_dim,
            dropout=self.dropout,
        )

        return TabDPTAWARECLSModel(
            backbone=backbone,
            adapter=adapter,
            n_classes=n_classes,
            ninp=ninp,
            max_features=max_features,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
            is_classification=self.is_classification,
        )


class KNNPFNClsFinetune(TabPFNClsFinetune):
    """TabPFN-based finetuning where context comes from a k-NN retriever.

    Functionally identical to ``TabPFNClsFinetune`` — the k-NN context
    selection is fully handled by the fixed ``RetrievalContextSampler``.
    """


# ---------------------------------------------------------------------------
# Sklearn wrapper — AWARECLSClassifier
# ---------------------------------------------------------------------------

class AWARECLSClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for the finetuned AWARE classification model.

    ``fit`` delegates to the wrapped ``BaseClsFinetune`` instance.
    ``predict_proba`` runs batched inference using the fixed retriever for
    context selection — identical mechanism to training, but without
    excluding batch members from the context pool.
    """

    def __init__(
        self,
        finetune: BaseClsFinetune,
        n_ensembles: int = 5,
        temperature: float = 0.8,
        infer_batch_size: int = 256,
    ) -> None:
        self.finetune = finetune
        self.n_ensembles = n_ensembles
        self.temperature = temperature
        self.infer_batch_size = infer_batch_size

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> "AWARECLSClassifier":
        self.finetune.fit(X, y, val_data=val_data, verbose=verbose)
        self.classes_ = self.finetune.classes_
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        ft = self.finetune
        assert ft.cls_model is not None, "Must call fit() before predict_proba()."
        ft.cls_model.eval()

        device     = ft.device
        X_scaled   = ft.scaler.transform(ft.imputer.transform(X))
        X_train_t  = torch.tensor(ft.X_train_scaled, dtype=torch.float32, device=device)
        y_train_t  = torch.tensor(ft.y_train,        dtype=torch.float32, device=device)

        # Use provided seed or fallback to a random one for ensembling variance
        base_rng = np.random.default_rng(seed)
        
        logits_cumsum = None
        
        for ens_idx in range(self.n_ensembles):
            # Create a per-ensemble RNG for reproducible but varied selection
            ens_seed = base_rng.integers(0, 2**32 - 1) if seed is not None else None
            ens_rng = np.random.default_rng(ens_seed)
            
            probs_ens = []
            for start in range(0, len(X), self.infer_batch_size):
                end     = min(len(X), start + self.infer_batch_size)
                X_batch = X_scaled[start:end]

                # Retrieve neighbors for each query
                per_q    = ft.context_sampler.retrieve_indices(X_batch)   # (B, k)
                
                # Randomized selection if n_ensembles > 1 to introduce variance
                ctx_indices = per_q[:, :ft.context_size].copy()
                if self.n_ensembles > 1:
                    # Pick a slightly different set or order of neighbors
                    # We can shuffle the neighbors for each query to add some noise
                    for b in range(len(ctx_indices)):
                        ens_rng.shuffle(ctx_indices[b])
                x_ctx = X_train_t[ctx_indices]                               # (B, K, n_feat)
                y_ctx = y_train_t[ctx_indices].float()                       # (B, K)
                x_q   = torch.tensor(X_batch, dtype=torch.float32, device=device)

                logits = ft.cls_model(x_q, x_ctx, y_ctx).cpu().float().numpy()
                if ft.is_classification:
                    probs  = softmax(logits / self.temperature, axis=-1)
                    probs /= probs.sum(axis=-1, keepdims=True) + 1e-12 # numerical stability
                else:
                    probs = logits
                probs_ens.append(probs)
            
            current_probs = np.concatenate(probs_ens, axis=0)
            
            if logits_cumsum is None:
                logits_cumsum = current_probs
            else:
                logits_cumsum += current_probs
                
        avg_pred = logits_cumsum / self.n_ensembles
        
        if ft.is_classification:
            # Ensure normalization for final probabilities
            avg_pred /= avg_pred.sum(axis=-1, keepdims=True) + 1e-12
        else:
            avg_pred = avg_pred.squeeze(-1)
        return avg_pred

    def predict(self, X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        probs = self.predict_proba(X, seed=seed)
        if self.finetune.is_classification:
            y_idx = np.argmax(probs, axis=1)
            return y_idx
        else:
            return probs
