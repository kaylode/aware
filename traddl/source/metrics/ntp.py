from typing import *
from collections import defaultdict
from collections.abc import Callable
import numpy as np
import torch as th
from theseus.base.metrics.metric_template import Metric



class NextTokenPredictionMetric(Metric):
    def __init__(self, tokens_of_interest: dict, **kwargs):
        super().__init__()
        self.tokens_of_interest = tokens_of_interest
        self.block_size = 2048
        self.block_thresh = self.block_size // 2
        self.reset()

    def update_model(self, model):
        self.model = model

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        
        # Get the predictions and targets from the outputs
        logits = outputs["predictions"].detach().cpu()
        targets = batch["targets"].cpu()

        # Append the predictions and targets to the lists
        self.predictions.append(logits)
        self.targets.append(targets)    

    def reset(self):
        self.targets = []
        self.predictions = []
        self.model = None    

    def reset(self):
        self.targets = []
        self.predictions = []
        self.model = None    
        

    def value(self):
        predictions = th.cat(self.predictions, dim=0)
        targets = th.cat(self.targets, dim=0)
        results1 = defaultdict(list)
        results2 = defaultdict(list)
        for k in [1, 3, 5]:
            results1[f"acc_top/all/{self.block_thresh}/k={k}"].append(
                top_k_accuracy(predictions, targets, k=k, threshold=self.block_thresh)
            )
        for stoken, token in self.tokens_of_interest.items():
            for k in [1, 3, 10]:
                results2[f"acc_top/{stoken}/{self.block_thresh}/k={k}"].append(
                    top_k_acc_special(predictions, targets, token, k=k, threshold=self.block_thresh)
                )
        
        results = {}
        results.update({test_name: np.mean(v) for test_name, v in results1.items()})
        results.update({test_name: compute_weighted_mean(v) for test_name, v in results2.items()})
        return results


def compute_weighted_mean(v):
    weights = sum(x[1] for x in v)
    if weights == 0:
        return th.nan
    return sum(x[0] * x[1] for x in v) / weights


@th.jit.script
def top_k_accuracy(logits: th.Tensor, y: th.Tensor, k: int, threshold: int) -> float:
    # logits: (batch_size, self.block_size, vocab_size)
    logits = logits[:, threshold - 1 :, :]
    y_true = y[:, threshold - 1 :, None]
    k = min(k, logits.size(-1))
    _, indices = th.topk(logits, k)
    correct = (indices == y_true).any(dim=-1)
    if correct.numel() == 0:
        return 0.0
    return float(correct.sum() / correct.numel())


@th.jit.script
def top_k_acc_special(
    logits: th.Tensor, y: th.Tensor, token_of_interest: int, k: int, threshold: int
) -> tuple[float, int]:
    """
    Calculate top-k accuracy for a specific token of interest in the logits.
    """
    logits = logits[:, threshold - 1 :, :]
    y_true = y[:, threshold - 1 :, None]
    interested = y_true == token_of_interest
    if interested.sum() == 0:
        return 0.0, 0
    _, indices = th.topk(logits, min(k, logits.size(-1)))
    correct = (indices == y_true).any(dim=-1, keepdim=True)
    weight = interested.sum()
    if weight == 0:
        return 0.0, 0
    score = (correct & interested).sum() / weight
    return score.item(), weight.item()

