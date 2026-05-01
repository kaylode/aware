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
    ) -> None:
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.adapter = adapter                  # trainable
        self.norm = nn.LayerNorm(ninp)          # trainable
        self.cls_head = nn.Sequential(          # trainable
            nn.Linear(ninp, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, n_classes),
        )

    def forward(
        self,
        x_query: torch.Tensor,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            x_query: (B, n_feat) — raw scaled query features.
            x_ctx:   (K, n_feat) — shared context features (raw scaled).
            y_ctx:   (K,)        — context labels as float.

        Returns:
            logits: (B, n_classes)
        """
        q_emb   = self.adapter(x_query)   # (B, emb_dim)
        ctx_emb = self.adapter(x_ctx)     # (K, emb_dim)
        q_backbone_emb = self._get_query_embeddings(q_emb, ctx_emb, y_ctx)  # (B, ninp)
        return self.cls_head(self.norm(q_backbone_emb))

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
        ctx_p = self._pad_to_tabpfn(ctx_emb)   # (K, 100)
        q_p   = self._pad_to_tabpfn(q_emb)     # (B, 100)

        # Backbone Wrapper.forward(x_train, y_train, x_test)
        # Returns dict: test_embeddings shape (B, 1, ninp) or (B, ninp)
        output = self.backbone(ctx_p, y_ctx, q_p)
        emb = output["test_embeddings"].to(torch.float32)
        if emb.ndim == 3:
            emb = emb.squeeze(1)    # (B, ninp)
        return emb


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
    ) -> None:
        super().__init__(backbone, adapter, n_classes, ninp, head_hidden_dim, dropout)
        self.max_features = max_features

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """(N, emb_dim) → (N, 1, max_features)."""
        x = x.unsqueeze(1)                          # (N, 1, emb_dim)
        n = x.shape[-1]
        if n < self.max_features:
            pad = torch.zeros(
                *x.shape[:-1], self.max_features - n,
                device=x.device, dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=-1)
        return x[..., : self.max_features]

    def _get_query_embeddings(self, q_emb, ctx_emb, y_ctx):
        K = ctx_emb.shape[0]
        ctx_seq = self._to_seq(ctx_emb)                     # (K, 1, max_features)
        q_seq   = self._to_seq(q_emb)                       # (B, 1, max_features)
        x_full  = torch.cat([ctx_seq, q_seq], dim=0)        # (K+B, 1, max_features)

        # TabDPTModel.forward(x, y, eval_pos, return_embeddings=True)
        # returns (pred[eval_pos:], src) where src is (K+B, 1, ninp)
        _, emb = self.backbone(
            x_full, y=y_ctx.float(), eval_pos=K, return_embeddings=True
        )
        return emb[K:].squeeze(1).to(torch.float32)         # (B, ninp)


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

        # Populated by fit()
        self.cls_model: Optional[AWARECLSModel] = None
        self.context_sampler: Optional[RetrievalContextSampler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
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
    # Helpers
    # ------------------------------------------------------------------

    def _sample_shared_context(
        self,
        X_batch: np.ndarray,
        batch_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve neighbors for the batch and return a shared context pool.

        Strategy: gather all retrieved neighbor indices across batch queries,
        deduplicate, exclude the current batch members, then sample up to
        ``context_size`` indices.

        Returns:
            ctx_sel: 1-D array of selected training indices
        """
        per_query_idx = self.context_sampler.retrieve_indices(X_batch)  # (B, k)
        pool = np.unique(per_query_idx.ravel())
        pool = pool[~np.isin(pool, batch_idx)]

        if len(pool) == 0:
            return np.empty(0, dtype=np.int64)

        k = min(self.context_size, len(pool))
        return rng.choice(pool, size=k, replace=False)

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
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        # ── 2. Validation split ───────────────────────────────────────
        if val_data is None and self.val_split > 0:
            logger.info(f"Creating validation split (ratio={self.val_split})...")
            X, X_val, y, y_val = train_test_split(
                X, y,
                test_size=self.val_split,
                random_state=self.random_state,
                stratify=y,
            )
            val_data = (X_val, y_val)

        N = len(X)

        # ── 3. Fit retriever (once, fixed) ────────────────────────────
        logger.info("Fitting retriever...")
        self.context_sampler = RetrievalContextSampler(self.retriever)
        self.context_sampler.fit(X, y)

        # ── 4. Build model ────────────────────────────────────────────
        logger.info(f"Building AWARE model (n_features={X.shape[1]}, n_classes={self.n_classes})...")
        self.cls_model = self._build_model(X.shape[1], self.n_classes).to(self.device)

        # ── 5. Pinned training tensors ────────────────────────────────
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,    device=self.device)

        # ── 6. Optimisation ───────────────────────────────────────────
        trainable_params = [p for p in self.cls_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.learning_rate * 0.01
        )
        criterion = nn.CrossEntropyLoss()

        # ── 7. Training loop ──────────────────────────────────────────
        best_loss, best_state, patience_ctr = float("inf"), None, 0

        pbar = tqdm(range(self.epochs), desc="Finetuning", disable=not verbose)
        for epoch in pbar:
            self.cls_model.train()
            epoch_loss, n_batches = 0.0, 0
            perm = rng.permutation(N)
            n_total_batches = (N + self.batch_size - 1) // self.batch_size
            batch_pbar = tqdm(range(0, N, self.batch_size), desc=f"  Epoch {epoch}", leave=False, disable=not verbose)
            for i, start in enumerate(batch_pbar):
                batch_idx = perm[start : start + self.batch_size]
                X_batch   = X[batch_idx]             # numpy, for retriever
                y_batch   = y_t[batch_idx]           # (B,) long

                ctx_sel = self._sample_shared_context(X_batch, batch_idx, rng)
                if len(ctx_sel) == 0:
                    continue

                x_ctx  = X_t[ctx_sel]                # (K, n_feat)
                y_ctx  = y_t[ctx_sel].float()         # (K,)
                x_q    = X_t[batch_idx]              # (B, n_feat)

                optimizer.zero_grad()
                logits = self.cls_model(x_q, x_ctx, y_ctx)
                loss   = criterion(logits, y_batch)

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cls_model.parameters(), max_norm=1.0)
                optimizer.step()

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
                val_metrics = self._compute_val_metrics(val_data, X_t, y_t, criterion)

            # Early stopping is driven by val_loss when available, otherwise train_loss
            stop_loss = val_metrics["val_loss"] if val_metrics is not None else avg_loss

            if verbose:
                msg = f"epoch {epoch:3d}/{self.epochs}  train_loss={avg_loss:.4f}"
                postfix = {"train_loss": f"{avg_loss:.4f}"}
                if val_metrics is not None:
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
        self.X_train_scaled = X
        self.y_train = y
        return self

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_val_metrics(
        self,
        val_data: Tuple[np.ndarray, np.ndarray],
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        criterion: nn.Module,
    ) -> dict:
        """Run one full validation pass and return loss + classification metrics.

        Uses the same retrieval-guided context selection as predict_proba so
        the reported numbers are representative of actual inference quality.

        Returns:
            dict with keys: val_loss, auc, auprc, f1
        """
        X_val, y_val = val_data
        X_val_s = self.scaler.transform(self.imputer.transform(X_val))

        self.cls_model.eval()
        total_loss, n_batches = 0.0, 0
        all_probs = []   # accumulate softmax probs across batches
        all_logits = []  # accumulate raw logits for CE loss

        for start in range(0, len(X_val), self.batch_size):
            end  = min(len(X_val), start + self.batch_size)
            X_vb = X_val_s[start:end]
            y_vb = torch.tensor(y_val[start:end], dtype=torch.long, device=self.device)

            per_query_idx = self.context_sampler.retrieve_indices(X_vb)  # (B, k)
            ctx_pool = np.unique(per_query_idx.ravel())
            k        = min(self.context_size, len(ctx_pool))
            ctx_sel  = ctx_pool[:k]

            x_ctx  = X_t[ctx_sel]
            y_ctx  = y_t[ctx_sel].float()
            x_vb_t = torch.tensor(X_vb, dtype=torch.float32, device=self.device)

            logits = self.cls_model(x_vb_t, x_ctx, y_ctx)   # (B, n_classes)
            total_loss += criterion(logits, y_vb).item()
            n_batches  += 1

            probs = torch.softmax(logits, dim=-1).cpu().float().numpy()
            all_probs.append(probs)

        val_loss = total_loss / max(1, n_batches)
        all_probs = np.concatenate(all_probs, axis=0)   # (N_val, n_classes)
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


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class TabPFNClsFinetune(BaseClsFinetune):
    """Classification finetuning with frozen TabPFN v1 backbone."""

    def _build_model(self, n_features: int, n_classes: int) -> TabPFNAWARECLSModel:
        from ehrpfn.models.AWARE.tabpfn.backbone_v1 import load_model_workflow

        backbone = load_model_workflow(device=self.device, only_inference=True)

        # Probe actual transformer embedding dimension
        with torch.no_grad():
            d_x = torch.randn(1, _TABPFN_NUM_FEATURES, device=self.device)
            d_y = torch.zeros(1, device=self.device)
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
        )


class TabDPTClsFinetune(BaseClsFinetune):
    """Classification finetuning with frozen TabDPT backbone."""

    def _build_model(self, n_features: int, n_classes: int) -> TabDPTAWARECLSModel:
        from ehrpfn.models.AWARE.tabdpt.tabdpt.model import TabDPTModel, TABDPT_CHECKPOINT_PATH

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
        temperature: float = 1.0,
        infer_batch_size: int = 256,
    ) -> None:
        self.finetune = finetune
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
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ft = self.finetune
        assert ft.cls_model is not None, "Must call fit() before predict_proba()."
        ft.cls_model.eval()

        device     = ft.device
        X_scaled   = ft.scaler.transform(ft.imputer.transform(X))
        X_train_t  = torch.tensor(ft.X_train_scaled, dtype=torch.float32, device=device)
        y_train_t  = torch.tensor(ft.y_train,        dtype=torch.float32, device=device)

        probs_list = []
        for start in range(0, len(X), self.infer_batch_size):
            end     = min(len(X), start + self.infer_batch_size)
            X_batch = X_scaled[start:end]

            # Retrieve neighbors for each query, use union as shared context
            per_q   = ft.context_sampler.retrieve_indices(X_batch)   # (B, k)
            ctx_pool = np.unique(per_q.ravel())
            k        = min(ft.context_size, len(ctx_pool))
            ctx_sel  = ctx_pool[:k]

            x_ctx = X_train_t[ctx_sel]                               # (K, n_feat)
            y_ctx = y_train_t[ctx_sel].float()                       # (K,)
            x_q   = torch.tensor(X_batch, dtype=torch.float32, device=device)

            logits = ft.cls_model(x_q, x_ctx, y_ctx).cpu().float().numpy()
            probs  = softmax(logits / max(self.temperature, 1e-8), axis=-1)
            probs_list.append(probs)

        result = np.concatenate(probs_list, axis=0)
        result /= result.sum(axis=-1, keepdims=True) + 1e-12
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
