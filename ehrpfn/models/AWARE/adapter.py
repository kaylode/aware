import torch
import torch.nn as nn

class AWAREAdapter(nn.Module):
	"""Lightweight MLP adapter mapping raw EHR features to a compact embedding.

	Sits between the raw feature vector and the frozen PFN backbone. Trained
	jointly with the SNNL retriever so that the embedding space is simultaneously
	good for retrieval (context selection) and for in-context prediction.

	Architecture: Linear → ReLU → Dropout → Linear
	Output z is passed directly to the backbone encoder (unnormalized).
	"""
	def __init__(
		self,
		n_features: int,
		embedding_dim: int = 256,
		hidden_dim: int = 512,
		dropout: float = 0.2,
	):
		"""
		Args:
			n_features: number of raw input features (dataset-specific).
			embedding_dim: output embedding dimension (= backbone ninp).
			hidden_dim: hidden layer width.
			dropout: dropout rate applied after ReLU.
		"""
		super().__init__()

		self.attention = nn.Sequential(
			nn.Linear(n_features, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, embedding_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Map raw features to an unnormalized embedding vector.

		Args:
			x: input tensor of shape (..., n_features).

		Returns:
			z: unnormalized embedding of shape (..., embedding_dim).
			   Used directly as the token sequence for the PFN backbone.
		"""
		return self.attention(x)