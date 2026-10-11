"""
AWARE (Advanced Weighted Attention REtrieval) Retriever

AWARE leverages learned feature-level attention weights combined with metric learning 
(SNNL or SupCon) to build a robust retrieval space. It features cross-validation 
ensembles, focal loss for imbalanced clinical tasks, and support for tabular EHR 
augmentations like Mixup to outperform standard KNN-L2 baselines.
"""
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
import faiss
import os
import os.path as osp
import copy
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from ehrret.retrieval.base import Retriever

def ehr_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
	"""
	Tabular Mixup for EHR features.
	Returns mixed inputs, pairs of targets, and lambda.
	"""
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	index = torch.randperm(batch_size).to(x.device)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

class FocalLoss(nn.Module):
	def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, inputs, targets):
		ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
		pt = torch.exp(-ce_loss)
		focal_loss = (1 - pt) ** self.gamma * ce_loss

		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		else:
			return focal_loss


class BinaryFocalLoss(nn.Module):
	"""Focal Loss for binary classification with sigmoid outputs."""
	def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: (N,) or (N, 1) raw logits
			targets: (N,) binary labels (0 or 1)
		"""
		inputs = inputs.view(-1)
		targets = targets.view(-1).float()
		
		p = torch.sigmoid(inputs)
		ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
		p_t = p * targets + (1 - p) * (1 - targets)
		alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
		focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		else:
			return focal_loss
		
from .utils import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionSoftNearestNeighborLoss(nn.Module):
    """
    Regression version of Soft Nearest Neighbor Loss.

    Encourages embedding neighborhoods to reflect
    target-space neighborhoods.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        label_temperature: float = 1.0,
        use_cosine: bool = False,
    ):
        super().__init__()

        self.temperature = temperature
        self.label_temperature = label_temperature
        self.use_cosine = use_cosine

    def forward(self, z: torch.Tensor, y: torch.Tensor):

        device = z.device
        N = z.size(0)
        eps = 1e-12

        # ----------------------------------------
        # Embedding distances
        # ----------------------------------------

        if self.use_cosine:
            z_norm = F.normalize(z, p=2, dim=1)
            cos_sim = torch.matmul(z_norm, z_norm.T)
            pairwise_dist = 1 - cos_sim
        else:
            pairwise_dist = torch.cdist(z, z, p=2).pow(2)

        # ----------------------------------------
        # Embedding similarity
        # ----------------------------------------

        emb_sim = torch.exp(
            -pairwise_dist / self.temperature
        )

        # remove self
        dmask = torch.ones(N, N, device=device)
        dmask.fill_diagonal_(0)

        emb_sim = emb_sim * dmask

        # ----------------------------------------
        # Target similarity (continuous labels)
        # ----------------------------------------

        y = y.float().view(-1, 1)

        target_dist = torch.cdist(y, y, p=2).pow(2)

        target_sim = torch.exp(
            -target_dist / self.label_temperature
        )

        target_sim = target_sim * dmask

        # normalize target neighborhood
        target_sim = target_sim / (
            target_sim.sum(dim=1, keepdim=True) + eps
        )

        # normalize embedding neighborhood
        emb_prob = emb_sim / (
            emb_sim.sum(dim=1, keepdim=True) + eps
        )

        # ----------------------------------------
        # KL alignment
        # ----------------------------------------

        loss = (
            target_sim *
            (
                torch.log(target_sim + eps)
                - torch.log(emb_prob + eps)
            )
        ).sum(dim=1).mean()

        return loss

class SoftNearestNeighborLoss(nn.Module):
	"""
	Soft Nearest Neighbor Loss (Salakhutdinov & Hinton, 2007)
	Encourages embeddings of same-label samples to be close in metric space.
	
	Supports both L2 distance (original) and cosine distance (modern variant).
	"""
	def __init__(self, temperature: float = 0.1, use_cosine: bool = False):
		super().__init__()
		self.temperature = temperature
		self.use_cosine = use_cosine
	
	def forward(self, z: torch.Tensor, y: torch.Tensor):
		"""
		Args:
			z: (batch, embed_dim) - embeddings (normalized if use_cosine=True)
			y: (batch,) - integer labels
		Returns:
			loss: scalar SNNL loss
		"""
		device = z.device
		N = z.size(0)
		eps = 1e-12
		
		if self.use_cosine:
			# Cosine distance: 1 - cos_sim
			# Normalize embeddings
			z_norm = F.normalize(z, p=2, dim=1)
			# Cosine similarity matrix
			cos_sim = torch.matmul(z_norm, z_norm.T)
			# Convert to distance
			pairwise_dist = 1 - cos_sim
		else:
			# L2 distance (original SNNL)
			pairwise_dist = torch.cdist(z, z, p=2).pow(2)
		
		# Create diagonal mask (exclude self-similarity)
		dmask = torch.ones(N, N, device=device)
		dmask.fill_diagonal_(0)
		
		# Temperature-scaled similarities
		pairwise_dist = pairwise_dist / self.temperature
		negexpd = torch.exp(-pairwise_dist)
		
		# Same class mask
		mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
		
		# All class neighborhood (excluding self)
		alcn = (negexpd * dmask).sum(dim=1)
		
		# Same class neighborhood (excluding self)
		sacn = (negexpd * mask * dmask).sum(dim=1)
		
		# SNNL: -log( sum(exp(-d_same)) / sum(exp(-d_all)) )
		loss = -torch.log((sacn + eps) / (alcn + eps)).mean()
		
		return loss

class SupConLoss(nn.Module):
	"""
	Supervised Contrastive Loss (Khosla et al., 2020)
	"""
	def __init__(self, temperature=0.1):
		super().__init__()
		self.temperature = temperature

	def forward(self, embeddings, labels):
		"""
		embeddings: (B, D) normalized
		labels: (B,)
		"""
		device = embeddings.device
		B = embeddings.shape[0]
		embeddings = F.normalize(embeddings, dim=1)

		# Cosine similarity matrix
		sim_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)
		sim_matrix = sim_matrix / self.temperature

		# Mask out self-comparisons
		logits_mask = torch.ones_like(sim_matrix) - torch.eye(B, device=device)

		exp_sim = torch.exp(sim_matrix) * logits_mask

		# Label mask
		labels = labels.view(-1, 1)
		mask = torch.eq(labels, labels.T).float().to(device)
		mask = mask * logits_mask  # remove self

		# Compute log prob
		log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

		# Only positives
		mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

		loss = -mean_log_prob_pos.mean()
		return loss


class SNNLAttnEncoder(nn.Module):
	"""
	Lightweight encoder that learns embeddings optimized for retrieval via SNNL.
	Maps raw EHR features to a normalized embedding space.
	"""
	def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.1, num_classes: int = 2, use_attention: bool = True):
		super().__init__()
		self.n_features = n_features
		self.num_classes = num_classes
		self.use_attention = use_attention
		if self.use_attention:
			self.attention = nn.Sequential(
				nn.Linear(n_features, hidden_dim),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim, n_features),
				nn.Sigmoid()  # Outputs weights in [0, 1]
			)
		
		# Optional: task-specific head for supervised training
		if num_classes is None or num_classes <= 2: # regression
			output_dim = 1
		else:
			output_dim = num_classes

		self.task_head = nn.Sequential(
			nn.Linear(n_features, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, output_dim)
		)

		self.snnl_head = nn.Sequential(
			nn.Linear(n_features, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim)
		)
	
	def forward(self, x, return_embeddings_only=False):
		"""
		Args:
			x: (batch, n_features)
		Returns:
			weighted_x: (batch, n_features) - features scaled by learned importance
			weights: (batch, n_features) - attention weights for interpretability
			logits: (batch, 1) - task predictions
		"""
		if self.use_attention:
			weights = self.attention(x)  # (batch, n_features)
			weighted_x = x * weights      # Element-wise scaling
		else:
			weights = torch.ones_like(x)
			weighted_x = x
		logits = self.task_head(weighted_x)
		embeddings = self.snnl_head(weighted_x)  # (batch, hidden_dim)

		if return_embeddings_only:
			return embeddings
		
		return weighted_x, embeddings, logits
	


class SNNLAttnRetriever(Retriever):
	"""
	SNNL-based retriever with FAISS indexing.
	Trains embeddings using Soft Nearest Neighbor Loss for optimal retrieval quality.
	"""
	def __init__(
		self,
		model: SNNLAttnEncoder,
		field_to_use: str = 'raw_feature',
		distance: str = 'cosine',
		k: int = 32,
		temperature: float = 0.1,
		alpha: float = 0.2,
		feature_names:List[str] =	 None,  # Store feature names for visualization
		use_balance_sampling:bool=True,
		use_focal_loss: bool = False,
		use_snnl_loss: bool = True,
		focal_gamma: float = 2.0,
		use_ensemble:bool = False,
		sampling_method: str = None,
		pretrained_path: str = None,
		save_attn_path = None,
		use_cosine_loss: bool = False,
		return_distances: bool = False,
		use_mixup: bool = False,
		mixup_alpha: float = 0.2,
		use_gpu: bool = False,
		is_classification: bool = True
	):
		"""
		Args:
			model: SNNLEncoder instance
			field_to_use: Key to extract features from items dict
			distance: 'cosine' or 'L2' for FAISS indexing
			k: Number of neighbors to retrieve
			pretrained_path: Path to save/load model weights
			use_cosine_loss: If True, use cosine distance in SNNL loss; 
						   if False, use L2 distance (original SNNL)
		"""
		self.model = model
		self.distance = distance
		self.index = None
		self.sampling_method = sampling_method
		self._is_fitted = False
		self.k = k
		self.field_to_use = field_to_use
		self.pretrained_path = pretrained_path
		self.use_cosine_loss = use_cosine_loss
		self.return_distances = return_distances
		self.temperature = temperature
		self.alpha = alpha
		self.use_focal_loss = use_focal_loss
		self.focal_gamma = focal_gamma
		self.save_attn_path = save_attn_path
		self.use_snnl_loss = use_snnl_loss
		self.use_balance_sampling = use_balance_sampling
		self.use_mixup = use_mixup
		self.mixup_alpha = mixup_alpha
		self.use_gpu = use_gpu
		self.is_classification = is_classification
		
		# Ensemble: store best model state from each fold
		self.ensemble_model_states = []
		self.use_ensemble = use_ensemble  # Set to False to use single best model
		self.feature_names = feature_names  # Store feature names for visualization
		self.ensemble_attention_weights = None  # Store attention weights from ensemble
		
		if pretrained_path is not None and osp.isfile(pretrained_path):
			self.load_state_dict(pretrained_path)
	
	def save_state_dict(self, path: str):
		"""Save model weights to file (saves ensemble if available)"""
		if osp.dirname(path) != '':
			os.makedirs(osp.dirname(path), exist_ok=True)
		save_dict = {
			'model_state': self.model.state_dict(),
			'ensemble_model_states': self.ensemble_model_states,
			'use_ensemble': self.use_ensemble
		}
		torch.save(save_dict, path)
		print(f"Saved model weights to {path} (ensemble: {len(self.ensemble_model_states)} models)")
	
	def load_state_dict(self, path: str):
		"""Load pretrained model weights (loads ensemble if available)"""
		checkpoint = torch.load(path)
		
		# Handle both old format (just state_dict) and new format (dict with ensemble)
		if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
			self.model.load_state_dict(checkpoint['model_state'])
			self.ensemble_model_states = checkpoint.get('ensemble_model_states', [])
			self.use_ensemble = checkpoint.get('use_ensemble', True)
		else:
			# Old format: just the state dict
			self.model.load_state_dict(checkpoint)
			self.ensemble_model_states = []
			self.use_ensemble = False
		
		self.model.eval()
		self._is_fitted = True
		print(f"Loaded model weights from {path} (ensemble: {len(self.ensemble_model_states)} models)")
	
	def _evaluate_fold(
		self,
		model: nn.Module,
		X_val: torch.Tensor,
		y_val: torch.Tensor,
		retrieval_criterion: nn.Module,
		cls_criterion: nn.Module,
	) -> Dict:
		"""
		Evaluate model on validation set.
		
		Returns:
			Dict with loss and metrics (f1, precision, recall)
		"""
		model.eval()
		with torch.no_grad():
			_, embeddings, logits = model(X_val, return_embeddings_only=False)
			
			# Compute losses
			snnl_loss = retrieval_criterion(embeddings, y_val).item()
			
			if model.num_classes is not None:
				if model.num_classes == 2:
					cls_loss = cls_criterion(logits.view(-1), y_val.float().view(-1)).item()
					preds = (torch.sigmoid(logits.view(-1)) > 0.5).cpu().numpy().astype(int)
				else:
					cls_loss = cls_criterion(logits, y_val.long()).item()
					preds = torch.argmax(logits, dim=1).cpu().numpy()
			else:
				cls_loss = 0.0
				preds = None  # Dummy preds for regression
			total_loss = self.alpha * snnl_loss + (1 - self.alpha) * cls_loss
			
			# Compute metrics
			y_true = y_val.cpu().numpy()
			if model.num_classes is None:
				# Regression metrics can be added here (e.g., MSE, MAE)
				f1 = precision = recall = 0.0  # Placeholder for regression
			else:
				average = 'binary' if model.num_classes == 2 else 'macro'
				f1 = f1_score(y_true, preds, average=average, zero_division=0)
				precision = precision_score(y_true, preds, average=average, zero_division=0)
				recall = recall_score(y_true, preds, average=average, zero_division=0)
		
		return {
			'loss': total_loss,
			'snnl_loss': snnl_loss,
			'cls_loss': cls_loss,
			'f1': f1,
			'precision': precision,
			'recall': recall
		}
	
	def fit(
		self,
		items: List[Dict],
		epochs: int = 100,
		lr: float = 3e-4,
		batch_size: int = 4096,
		n_folds: int = 5,
		early_stopping_patience: int = 15,
		early_stopping_metric: str = 'f1',
		verbose: bool = True,
		**kwargs
	):
		"""
		Train encoder with SNNL + optional classification loss using stratified k-fold cross-validation.
		
		Args:
			items: List of dicts with 'raw_feature' and 'label' keys
			epochs: Maximum number of training epochs
			lr: Learning rate
			batch_size: Batch size for training
			temperature: SNNL temperature parameter
						- For L2 distance: typically 0.1 - 1.0
						- For cosine distance: typically 0.05 - 0.5
			alpha: Weight for SNNL loss (0-1), (1-alpha) for classification
			n_folds: Number of folds for cross-validation
			early_stopping_patience: Number of epochs without improvement before stopping
			early_stopping_metric: Metric to monitor ('f1', 'precision', 'recall', 'loss')
			use_focal_loss: If True, use FocalLoss instead of BCE/CrossEntropy
			focal_gamma: Gamma parameter for focal loss (default: 2.0)
			verbose: Whether to print detailed training info
			
		Returns:
			Dict with average metrics across all folds
		"""
		X_all = np.array([item[self.field_to_use] for item in items])
		y_all = np.array([item['label'] for item in items])

		if self.model.num_classes is None or self.alpha >= 0.99:
			early_stopping_metric = 'loss'  # For regression or metric-only training
		
		# Store feature names for visualization

		if self.sampling_method is not None:
			from ehrmeds import synthetic_sampling
			sampling_type, sampling_method = self.sampling_method.split('_')
			X_all, y_all = synthetic_sampling(
				X_all, y_all, 
				oversampling=sampling_method if sampling_type == "oversampling" else None,
				undersampling=sampling_method if sampling_type == "undersampling" else None,
				seed=2000,
				negative_ratio=5.0
			)

		X_tensor = torch.FloatTensor(X_all)
		y_tensor = torch.LongTensor(y_all)
		
		if not self._is_fitted:
			import torch.optim as optim
			from torch.utils.data import DataLoader, TensorDataset
			from ehrmeds import BalanceSampler, BalanceRegressionSampler
			

			if self.use_snnl_loss:
				if self.is_classification:
					retrieval_criterion = SoftNearestNeighborLoss(
						temperature=self.temperature,
						use_cosine=self.use_cosine_loss
					)
				else:
					retrieval_criterion = RegressionSoftNearestNeighborLoss(
						temperature=self.temperature,
						use_cosine=self.use_cosine_loss
					)
			else:
				retrieval_criterion = SupConLoss(temperature=self.temperature)

			if self.use_focal_loss:
				if self.model.num_classes == 2:
					cls_criterion = BinaryFocalLoss(gamma=self.focal_gamma)
				else:
					cls_criterion = FocalLoss(gamma=self.focal_gamma)
			elif self.model.num_classes == 2:
				cls_criterion = nn.BCEWithLogitsLoss() 
			else:
				cls_criterion = nn.CrossEntropyLoss()

			# Stratified K-Fold Cross Validation (preserves class distribution)
			fold_metrics = []
			best_fold_score = -float('inf') if early_stopping_metric != 'loss' else float('inf')
			best_model_state = None
			
			# Ensemble: store best model from each fold
			self.ensemble_model_states = []
			
			loss_type = "Cosine" if self.use_cosine_loss else "L2"

			if self.model.num_classes is None:
				kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
				splits = kfold.split(X_all)  # For regression, just use regular KFold without stratification
			else:
				kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
				splits = kfold.split(X_all, y_all)
			
			for fold, (train_idx, val_idx) in enumerate(splits):
				if verbose:
					print(f"\n{'='*50}")
					print(f"Fold {fold + 1}/{n_folds}")
					print(f"{'='*50}")
				
				# Split data
				X_train, X_val = X_all[train_idx], X_all[val_idx]
				y_train, y_val = y_all[train_idx], y_all[val_idx]
				
				X_train_t = torch.FloatTensor(X_train)
				y_train_t = torch.LongTensor(y_train)
				X_val_t = torch.FloatTensor(X_val)
				y_val_t = torch.LongTensor(y_val)
				
				# Reset model weights for each fold
				def reset_weights(m):
					if hasattr(m, 'reset_parameters'):
						m.reset_parameters()
				self.model.apply(reset_weights)
				
				# Initialize optimizer
				optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
				
				# Initialize early stopping
				early_stopping_mode = 'max' if early_stopping_metric != 'loss' else 'min'
				early_stopping = EarlyStopping(
					patience=early_stopping_patience,
					mode=early_stopping_mode
				)
				
				# Create DataLoader for training
				dataset = TensorDataset(X_train_t, y_train_t)
				if self.model.num_classes is not None and self.is_classification:
					if self.use_balance_sampling:
						sampler = BalanceSampler(labels=y_train)
					else:
						sampler = None
					loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
				else:
					if self.use_balance_sampling:
						sampler = BalanceRegressionSampler(labels=y_train)
					else:
						sampler = None
					loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
				
				bar = tqdm(range(epochs), desc=f"Fold {fold+1} Training ({loss_type})", disable=not verbose)
				best_epoch_metrics = None
				
				for epoch in bar:
					self.model.train()
					total_snnl_loss = 0.0
					total_cls_loss = 0.0
					num_batches = 0
					
					for xb, yb in loader:
						optimizer.zero_grad()
						
						if self.use_mixup:
							xb, yb_a, yb_b, lam = ehr_mixup(xb, yb, alpha=self.mixup_alpha)
							_, embeddings, logits = self.model(xb, return_embeddings_only=False)
							
							# Retrieval Loss
							snnl_loss = lam * retrieval_criterion(embeddings, yb_a) + (1 - lam) * retrieval_criterion(embeddings, yb_b)
							# Task Loss
							if self.model.num_classes is not None:
								if self.model.num_classes == 2:
									cls_loss = lam * cls_criterion(logits.view(-1), yb_a.float().view(-1)) + (1 - lam) * cls_criterion(logits.view(-1), yb_b.float().view(-1))
								else:
									cls_loss = lam * cls_criterion(logits, yb_a.long()) + (1 - lam) * cls_criterion(logits, yb_b.long())
							else:
								cls_loss = 0.0
						else:
							# Forward pass
							_, embeddings, logits = self.model(xb, return_embeddings_only=False)
							# SNNL loss (main retrieval objective)
							snnl_loss = retrieval_criterion(embeddings, yb)
							
							# Classification loss (auxiliary task)
							if self.model.num_classes is not None:
								if self.model.num_classes == 2:
									cls_loss = cls_criterion(logits.view(-1), yb.float().view(-1))
								else:
									cls_loss = cls_criterion(logits, yb.long())
							else:
								cls_loss = 0.0

						# Combined loss
						total_loss = self.alpha * snnl_loss + (1 - self.alpha) * cls_loss
						
						total_loss.backward()
						optimizer.step()
						
						total_snnl_loss += snnl_loss.item()
						total_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
						num_batches += 1
					
					avg_snnl = total_snnl_loss / num_batches
					avg_cls = total_cls_loss / num_batches
					
					# Evaluate on validation set
					val_metrics = self._evaluate_fold(
						self.model, X_val_t, y_val_t,
						retrieval_criterion, cls_criterion
					)
					
					bar.set_postfix({
						'snnl': f'{avg_snnl:.4f}',
						'cls': f'{avg_cls:.4f}',
						'val_f1': f"{val_metrics['f1']:.4f}",
						'val_prec': f"{val_metrics['precision']:.4f}",
						'val_rec': f"{val_metrics['recall']:.4f}"
					})
					
					# Early stopping check
					current_metric = val_metrics[early_stopping_metric]
					if early_stopping(current_metric, self.model):
						if verbose:
							print(f"\nEarly stopping triggered at epoch {epoch + 1}")
						break
					
					# Track best metrics for this fold
					if best_epoch_metrics is None:
						best_epoch_metrics = val_metrics.copy()
					else:
						if early_stopping_mode == 'max':
							if val_metrics[early_stopping_metric] > best_epoch_metrics[early_stopping_metric]:
								best_epoch_metrics = val_metrics.copy()
						else:
							if val_metrics[early_stopping_metric] < best_epoch_metrics[early_stopping_metric]:
								best_epoch_metrics = val_metrics.copy()
				
				# Restore best model for this fold
				early_stopping.restore_best_model(self.model)
				
				# Store fold metrics
				fold_metrics.append(best_epoch_metrics)
				
				# Save best model state from this fold for ensemble
				self.ensemble_model_states.append(copy.deepcopy(self.model.state_dict()))
				
				if verbose:
					print(f"Fold {fold + 1} Best - F1: {best_epoch_metrics['f1']:.4f}, "
						  f"Precision: {best_epoch_metrics['precision']:.4f}, "
						  f"Recall: {best_epoch_metrics['recall']:.4f}")
				
				# Track best model across all folds (for fallback single-model mode)
				fold_score = best_epoch_metrics[early_stopping_metric]
				if early_stopping_mode == 'max':
					is_best = fold_score > best_fold_score
				else:
					is_best = fold_score < best_fold_score
				
				if is_best:
					best_fold_score = fold_score
					best_model_state = copy.deepcopy(self.model.state_dict())
			
			# Restore best model across all folds
			if best_model_state is not None:
				self.model.load_state_dict(best_model_state)
			
			# Compute average metrics across folds
			avg_metrics = {
				'f1': np.mean([m['f1'] for m in fold_metrics]),
				'precision': np.mean([m['precision'] for m in fold_metrics]),
				'recall': np.mean([m['recall'] for m in fold_metrics]),
				'loss': np.mean([m['loss'] for m in fold_metrics]),
				'std_f1': np.std([m['f1'] for m in fold_metrics]),
				'std_precision': np.std([m['precision'] for m in fold_metrics]),
				'std_recall': np.std([m['recall'] for m in fold_metrics]),
			}
			
			if verbose:
				print(f"\n{'='*50}")
				print(f"Cross-Validation Results ({n_folds} folds)")
				print(f"{'='*50}")
				print(f"F1: {avg_metrics['f1']:.4f} (+/- {avg_metrics['std_f1']:.4f})")
				print(f"Precision: {avg_metrics['precision']:.4f} (+/- {avg_metrics['std_precision']:.4f})")
				print(f"Recall: {avg_metrics['recall']:.4f} (+/- {avg_metrics['std_recall']:.4f})")
				print(f"Best {early_stopping_metric}: {best_fold_score:.4f}")
				print(f"Ensemble models saved: {len(self.ensemble_model_states)}")
			
			self._is_fitted = True
			self._cv_metrics = avg_metrics
			
			if self.pretrained_path is not None:
				self.save_state_dict(self.pretrained_path)
		
		# Build FAISS index with learned embeddings (using ensemble if available)
		embeddings = self._get_ensemble_embeddings(X_tensor)

		if self.save_attn_path:
			self.visualize_attention_weights(
				feature_names = self.feature_names,
				top_k = 10,
				save_path = self.save_attn_path,
				figsize = (12, 4),
				title = None
			)
		
		if self.distance == 'cosine':
			# Normalize for cosine similarity
			faiss.normalize_L2(embeddings)
			self.index = faiss.IndexFlatIP(embeddings.shape[1])
		else:
			self.index = faiss.IndexFlatL2(embeddings.shape[1])
		
		self.index.add(embeddings)
		
		# Move to GPU if requested
		if self.use_gpu:
			try:
				res = faiss.StandardGpuResources()
				self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
				print("Moved FAISS index to GPU.")
			except Exception as e:
				print(f"Warning: Could not move FAISS index to GPU: {e}")

		self.train_data = embeddings
		mode = "ensemble" if self.use_ensemble and len(self.ensemble_model_states) > 1 else "single"
		print(f"Built FAISS Index ({self.distance}, {mode}) with {self.index.ntotal} vectors")
	
	def _get_ensemble_embeddings(self, X_tensor: torch.Tensor, track_attention: bool = True, return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
		"""
		Get embeddings by averaging across all ensemble models.
		If ensemble is not available or disabled, uses single model.
		Also tracks and stores attention weights.
		
		Args:
			X_tensor: Input features tensor
			track_attention: Whether to compute and store attention weights
			
		Returns:
			Averaged embeddings as contiguous numpy array
		"""

		track_attention = track_attention and getattr(self.model, 'attention', None) is not None

		if self.use_ensemble and len(self.ensemble_model_states) > 1:
			# Ensemble mode: average embeddings from all fold models
			all_embeddings = []
			all_attention_weights = [] if track_attention else None
			
			for state_dict in self.ensemble_model_states:
				self.model.load_state_dict(state_dict)
				self.model.eval()
				
				with torch.no_grad():
					emb = self.model(X_tensor, return_embeddings_only=True)
					all_embeddings.append(emb)
					
					if track_attention:
						# Get attention weights for this model
						weights = self.model.attention(X_tensor)  # (batch, n_features)
						# Average across samples
						weights_mean = weights.mean(dim=0)  # (n_features,)
						all_attention_weights.append(weights_mean)
			
			# Average embeddings across all models
			stacked = torch.stack(all_embeddings, dim=0)  # (n_models, batch, embed_dim)
			averaged = stacked.mean(dim=0)  # (batch, embed_dim)
			
			# Average attention weights across all models
			if track_attention:
				stacked_weights = torch.stack(all_attention_weights, dim=0)  # (n_models, n_features)
				self.ensemble_attention_weights = stacked_weights.mean(dim=0).cpu().numpy()  # (n_features,)
			
			if return_tensor:
				return averaged
			return np.ascontiguousarray(averaged.cpu().numpy())
		else:
			# Single model mode
			self.model.eval()
			with torch.no_grad():
				embeddings = self.model(X_tensor, return_embeddings_only=True)
				
				if track_attention and self.model.use_attention:
					weights = self.model.attention(X_tensor)
					self.ensemble_attention_weights = weights.mean(dim=0).cpu().numpy()
				else:
					self.ensemble_attention_weights = np.ones(X_tensor.shape[1])
				
				if return_tensor:
					return embeddings
				return np.ascontiguousarray(embeddings.cpu().numpy())
	
	def retrieve(self, items: List[Dict], **kwargs) -> Union[np.ndarray, tuple]:
		"""Retrieve k nearest neighbors using learned embeddings (ensemble averaged)"""
		X_query = np.array([item[self.field_to_use] for item in items])
		
		if not self._is_fitted:
			raise RuntimeError("Retriever is not fitted yet.")
		
		X_tensor = torch.FloatTensor(X_query).to(next(self.model.parameters()).device)
		query_embeddings = self._get_ensemble_embeddings(X_tensor, return_tensor=self.use_gpu)
		
		if self.distance == 'cosine':
			if self.use_gpu:
				# FAISS normalize_L2 doesn't work on tensors directly without contrib
				norm = torch.norm(query_embeddings, p=2, dim=1, keepdim=True)
				query_embeddings = query_embeddings / (norm + 1e-12)
			else:
				faiss.normalize_L2(query_embeddings)

		if kwargs.get('k', None) is not None:
			k = kwargs['k']
		else:
			k = self.k

		if self.use_gpu and torch.is_tensor(query_embeddings):
			# Convert to numpy for standard search if not using search_torch
			# or use a helper if available. FAISS GPU search with tensors is faster.
			query_embeddings_np = query_embeddings.cpu().numpy()
			distances, indices = self.index.search(query_embeddings_np, k)
		else:
			distances, indices = self.index.search(query_embeddings, k)

		if self.return_distances:
			return distances, indices
		else:
			return indices
	
	@staticmethod
	def _standardize_feature_name(raw_name: str) -> tuple:
		"""
		Standardize feature names from format like 'LAB//tidal_volume_(set)_max'
		to human-readable format like 'Tidal Volume (Set) Max'.
		
		Args:
			raw_name: Raw feature name (e.g., 'LAB//tidal_volume_(set)_max')
			
		Returns:
			Tuple of (feature_type, standardized_name)
		"""
		# Extract feature type and name
		if '//' in raw_name:
			parts = raw_name.split('//', 1)
			feature_type = parts[0].upper()
			feature_name = parts[1] if len(parts) > 1 else raw_name
		else:
			feature_type = 'OTHER'
			feature_name = raw_name
		
		# Standardize the feature name
		# Replace underscores with spaces
		standardized = feature_name.replace('_', ' ')
		# Handle parentheses - ensure spaces around them
		standardized = standardized.replace('(', ' (').replace(')', ') ')
		# Remove multiple spaces
		standardized = ' '.join(standardized.split())
		# Title case each word
		standardized = standardized.title()
		feature_type += '//'

		
		return feature_type, standardized

	def visualize_attention_weights(
		self,
		feature_names: List[str] = None,
		top_k: int = None,
		save_path: str = 'attention_weights.pdf',
		figsize: tuple = (4, 12),
		title: str = 'Attention Weight Distribution',
		cmap: str = 'Greys'
	):
		"""
		Visualize attention weights as a histogram-style distribution plot with gradient colors based on weight values.
		Uses the stored ensemble_attention_weights computed during _get_ensemble_embeddings.
		
		Args:
			feature_names: List of feature names (uses stored names if None)
			top_k: Not used (kept for API compatibility)
			save_path: Path to save the PDF file
			figsize: Figure size tuple (width, height)
			title: Plot title
			cmap: Colormap name for gradient coloring (default: 'Blues')
			
		Returns:
			Dict with feature names and their attention weights (sorted by name)
		"""
		if self.ensemble_attention_weights is None:
			raise RuntimeError(
				"Attention weights not available. Call fit() first to compute embeddings and track attention weights."
			)
		
		# Use stored feature names if not provided
		if feature_names is None:
			feature_names = self.feature_names
		
		if feature_names is None:
			# Generate default feature names
			feature_names = [f'Feature_{i}' for i in range(len(self.ensemble_attention_weights))]
		
		attention_weights = self.ensemble_attention_weights
		
		# Create DataFrame-like structure with standardized names
		feature_data = []
		for name, weight in zip(feature_names, attention_weights):
			feat_type, std_name = self._standardize_feature_name(name)
			feature_data.append({
				'raw_name': name,
				'std_name': std_name,
				'type': feat_type,
				'weight': weight
			})
		
		# Sort by feature name (alphabetically)
		sorted_data = sorted(feature_data, key=lambda x: x['raw_name'])
		
		# Extract data for plotting
		weights = np.array([d['weight'] for d in sorted_data])
		
		# Normalize weights to [0, 1] for colormap
		weights_min, weights_max = weights.min(), weights.max()
		if weights_max > weights_min:
			weights_normalized = (weights - weights_min) / (weights_max - weights_min)
		else:
			weights_normalized = np.ones_like(weights) * 0.5
		
		# Get colormap and generate gradient colors based on weight values
		colormap = plt.cm.get_cmap(cmap)
		# Use normalized weights to get colors (higher weight = darker color)
		colors = [colormap(0.3 + 0.7 * w) for w in weights_normalized]  # Scale to 0.3-1.0 range for better visibility
		
		# Apply paper style
		plt.rcParams['font.family'] = 'serif'
		plt.rcParams['font.size'] = 12
		plt.style.use('seaborn-v0_8-paper')
		
		# Create histogram-style distribution plot
		fig, ax = plt.subplots(figsize=figsize)
		
		# Create bars with gradient colors based on weight values (no gaps for histogram look)
		x_pos = np.arange(len(weights))
		bars = ax.bar(x_pos, weights, color=colors, edgecolor='none', alpha=0.85, width=1.0)
		
		# Remove x-axis and y-axis tickers
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		
		# Remove spines for cleaner look
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		
		# Adjust layout
		plt.tight_layout()
		
		# Save as PDF
		if osp.dirname(save_path) != '':
			os.makedirs(osp.dirname(save_path), exist_ok=True)
		fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
		print(f"Saved attention weights plot to {save_path}")
		
		# Save plot data as pickle for later use in combined figures
		import pickle
		plot_data = {
			'weights': weights.tolist(),
			'weights_normalized': weights_normalized.tolist(),
			'colors': colors,
			'cmap': cmap,
			'sorted_data': sorted_data,
		}
		pickle.dump(plot_data, open(f'{save_path.replace(".pdf", "")}.pickle', 'wb'))
		print(f"Saved attention weights data to {save_path.replace('.pdf', '')}.pickle")

		plt.close(fig)
		
		# Return results
		return {
			'feature_names': [d['raw_name'] for d in sorted_data],
			'standardized_names': [d['std_name'] for d in sorted_data],
			'feature_types': [d['type'] for d in sorted_data],
			'attention_weights': [d['weight'] for d in sorted_data],
		}

