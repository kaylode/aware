import torch
import torch.nn as nn
import copy

class EarlyStopping:
	"""
	Early stopping to halt training when validation metric stops improving.
	"""
	def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
		"""
		Args:
			patience: Number of epochs to wait for improvement before stopping
			min_delta: Minimum change to qualify as an improvement
			mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_model_state = None
	
	def __call__(self, score: float, model: nn.Module) -> bool:
		"""
		Check if training should stop.
		"""
		if self.best_score is None:
			self.best_score = score
			self.best_model_state = copy.deepcopy(model.state_dict())
			return False
		
		if self.mode == 'min':
			improved = score < self.best_score - self.min_delta
		else:  # mode == 'max'
			improved = score > self.best_score + self.min_delta
		
		if improved:
			self.best_score = score
			self.best_model_state = copy.deepcopy(model.state_dict())
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
				return True
		
		return False
	
	def restore_best_model(self, model: nn.Module):
		"""Restore the model to its best state."""
		if self.best_model_state is not None:
			model.load_state_dict(self.best_model_state)
