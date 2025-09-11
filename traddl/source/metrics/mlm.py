from typing import *
import numpy as np
import torch as th

from theseus.base.metrics.metric_template import Metric



class MLMAccuracyMetric(Metric):
    def __init__(self, masking_target_idx:int=-100, **kwargs):
        super().__init__()
        self.masking_target_idx = masking_target_idx
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        
        # Get the predictions and targets from the outputs
        logits = outputs["outputs"].detach().cpu().numpy()
        targets = batch["targets"].view(-1).cpu().numpy()

        ind = np.where(targets!=-self.masking_target_idx)[0]
        truepred = logits[ind]
        truelabel = targets[ind]

        truepred = th.softmax(th.tensor(truepred), dim=1)
        outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
        corrects = np.sum(np.array(outs) == np.array(truelabel))
        self.no_corrects += corrects
        self.no_samples += len(outs)

    def reset(self):
        self.no_corrects = 0
        self.no_samples = 0

    def value(self):
        if self.no_samples == 0:
            return 0.0
        acc = self.no_corrects / self.no_samples
        return {
            "mlm_acc": acc
        }
    

class AccuracyAtK(Metric):
    def __init__(self, topks=[1,5,10], masking_target_idx:int=-100, **kwargs):
        """Computes the acc@k for the specified value of k"""
        self.masking_target_idx = masking_target_idx
        self.topk = topks
        super().__init__()
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        logits = outputs["outputs"].detach()
        targets = batch["targets"].view(-1)

        ind = th.where((targets != self.masking_target_idx) & (targets != 0))

        logits = logits[ind]
        targets = targets[ind]

        for k in self.topk:
            _, pred = logits.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets)
            self.no_corrects[k] += correct.sum().item()

        self.no_samples += targets.size(0)

    def reset(self):
        self.no_corrects = {
            k: 0 for k in self.topk
        }
        self.no_samples = 0

    def value(self):
        if self.no_samples == 0:
            return {
                f"acc@{f}": 0.0 for f in self.topk
            }
        acc = [self.no_corrects[k] / self.no_samples for k in self.topk]
        return {
            f"acc@{k}": acc for k, acc in zip(self.topk, acc)
        }