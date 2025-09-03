import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache


class BaseModel(nn.Module):
    def __init__(
        self, **kwargs
    ):

        super().__init__()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @lru_cache
    def num_parameters(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, *args, **kwargs):
        return self.forward_batch(*args, **kwargs)
    
    def forward_batch(self, batch):
        """
        Forward pass for the model. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @torch.no_grad()
    def get_next_token(self, x: torch.Tensor, return_probs: bool = False, top_k: int | None = None):
        logits = self(x).logits
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if return_probs:
            return next_token, probs
        return next_token