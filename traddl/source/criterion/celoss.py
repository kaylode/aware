from typing import *

import torch
from torch import nn

from theseus.base.utilities.cuda import move_to


class MLMCELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, weight: List = None, **kwargs):
        super(MLMCELoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=kwargs.get("ignore_index", -100),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        device: torch.device = None,
    ):
        pred = outputs["outputs"]
        if device is not None:
            target = move_to(batch["targets"], device)
        else:
            target = batch["targets"]

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target.view(-1).contiguous())

        loss_dict = {"CE": loss.item()}
        return loss, loss_dict
