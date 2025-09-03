
from typing import *
from copy import deepcopy
from loguru import logger
from ehrpfn.utils.constants import Task, SpecialToken as ST, Reason
from nosos.datasets.cohort.cohort import translate_icdcm_tokens
from transformers import PreTrainedModel
import torch as th
from collections import defaultdict
import numpy as np
from collections import abc

class RetrievalMetric:
    """
    Retrieval metric base class.
    """
    name:str = None

    def __init__(self, topk:int=1):
        self.topk = topk

    def __call__(self, targets:List[List[Dict[str, Any]]], preds: List[List[Dict[str, Any]]]):
        y_set_true = deepcopy(targets)
        y_set_pred = deepcopy(preds)
        for i in range(len(targets)):
            y_set_true[i] = set(targets[i])
            for j in range(len(preds[i])):
                y_set_pred[i][j] = set(preds[i][j])

        k_preds = []
        for i in range(len(y_set_pred)):
            k_preds.append(y_set_pred[i][:self.topk])

        score = self.compute(y_set_true, k_preds)
        if isinstance(score, dict):
            return score
        elif isinstance(score, (float, int)):
            return {
                f"{self.name}@{self.topk}": score
            }
        else:
            raise ValueError(f"Invalid score type: {type(score)}")
    
class PrecisionAtK(RetrievalMetric):
    """
    Precision at k metric.
    """
    name: str = "all_codes_precision"
    def compute(self, targets:List[str], preds:List[List[str]]):
        precision = 0
        if len(targets) == 0: return 1
        for i in range(len(targets)):
            max_precision = 0
            for j in range(len(preds[i])):
                if len(preds[i][j]) == 0:
                    continue
                max_precision = max(
                    max_precision, 
                    len(targets[i].intersection(preds[i][j])) / len(preds[i][j])
                )
            precision += max_precision
        return precision / len(targets) if len(targets) > 0 else 1
        

class RecallAtK(RetrievalMetric):
    """
    Recall at k metric.
    """
    name: str = "all_codes_recall"
    def compute(self, targets, preds):
        recall = 0
        for i in range(len(targets)):
            max_recall = 0
            if len(targets[i]) == 0:
                continue
            for j in range(len(preds[i])):
                if len(preds[i][j]) == 0:
                    continue
                max_recall = max(
                    max_recall, 
                    len(targets[i].intersection(preds[i][j])) / len(targets[i])
                )
            recall += max_recall
        return recall / len(targets) if len(targets) > 0 else 1
        

class F1ScoreAtK(RetrievalMetric):
    """
    F1 score metric.
    """
    name: str = "all_codes_f1"
    def compute(self, targets, preds):
        f1 = 0
        for i in range(len(targets)):
            max_f1 = 0
            if len(targets[i]) == 0:
                continue
            for j in range(len(preds[i])):
                if len(preds[i][j]) == 0:
                    continue
                precision = len(targets[i].intersection(preds[i][j])) / len(preds[i][j])
                recall = len(targets[i].intersection(preds[i][j])) / len(targets[i])
                if precision + recall == 0:
                    continue
                f1_score = 2 * (precision * recall) / (precision + recall)
                max_f1 = max(max_f1, f1_score)
            f1 += max_f1
        return f1 / len(targets) if len(targets) > 0 else 1
        

class NTPCodeOfInterestAccuracy():
    """
    NTP loss metric.
    """
    name: str = "ntp_loss"

    def __init__(self, tokens_of_interest: dict, k:int, block_thresh:int):
        super().__init__()
        self.tokens_of_interest = tokens_of_interest
        self.k = k
        self.block_thresh = block_thresh

    def __call__(self, targets: th.Tensor, preds: th.Tensor):
        """
        Top k accuracy metric.
        Args:
            targets: ground truth labels (batch_size, block_size, vocab_size)
            preds: predicted logits (batch_size, block_size, vocab_size)
        """
        general_acc = self.top_k_accuracy(preds, targets, k=self.k, threshold=self.block_thresh)

        toi_res = defaultdict(list)
        for stoken, token in self.tokens_of_interest.items():
            toi_res[f"{stoken}@{self.k}"].append(
                self.top_k_acc_special(preds, targets, token, k=self.k, threshold=self.block_thresh)
            )
        
        toi_res = {test_name: self.compute_weighted_mean(v) for test_name, v in toi_res.items()}
        return {
           f"all_tokens_acc@{self.k}": general_acc,
            **toi_res
        }

    def compute_weighted_mean(self, v:List):
        weights = sum(x[1] for x in v)
        if weights == 0:
            return th.nan
        return sum(x[0] * x[1] for x in v) / weights
    
    def top_k_accuracy(self, logits: th.Tensor, y: th.Tensor, k: int, threshold: int) -> float:
        # logits: (batch_size, block_size, vocab_size)
        logits = logits[:, threshold - 1 :, :]
        y_true = y[:, threshold - 1 :, None]
        k = min(k, logits.size(-1))
        _, indices = th.topk(logits, k)
        correct = (indices == y_true).any(dim=-1)
        return float(correct.sum() / correct.numel())


    def top_k_acc_special(
            self,
        logits: th.Tensor, y: th.Tensor, token_of_interest: int, k: int, threshold: int
    ) -> tuple[float, int]:
        logits = logits[:, threshold - 1 :, :]
        y_true = y[:, threshold - 1 :, None]
        interested = y_true == token_of_interest
        if interested.sum() == 0:
            return 0.0, 0
        _, indices = th.topk(logits, min(k, logits.size(-1)))
        correct = (indices == y_true).any(dim=-1, keepdim=True)
        weight = interested.sum()
        score = (correct & interested).sum() / weight
        return score.item(), weight.item()





class Evaluator:
    def __init__(self, get_batch: Callable, eval_iters: int, tokens_of_interest: dict):
        self.get_batch = get_batch
        self.eval_iters = eval_iters
        self.tokens_of_interest = tokens_of_interest
        self.block_size = 2048
        self.block_thresh = self.block_size // 2
        self.metrics = [
            NTPCodeOfInterestAccuracy(self.tokens_of_interest, k=1, block_thresh=self.block_thresh),
            NTPCodeOfInterestAccuracy(self.tokens_of_interest, k=5, block_thresh=self.block_thresh),
            NTPCodeOfInterestAccuracy(self.tokens_of_interest, k=10, block_thresh=self.block_thresh),
        ]

    @th.no_grad()
    def __call__(self, model: PreTrainedModel, ctx) -> dict:
        
        out = {}
        model.eval()
        for split in ["train", "val"]:
            prefix = "Validation" if split == "val" else "Training"
            losses = th.empty(self.eval_iters)
            predictions = []
            targets = []

            for i in range(self.eval_iters):
                X, attention_mask, Y = self.get_batch(split)
                with ctx:
                    output = model(input_ids=X, labels=Y, attention_mask=attention_mask)
                    loss, logits = output.loss, output.logits
                losses[i] = loss.item()
                out[f"{prefix}/CELoss"] = losses.mean().item()

                if split == "val":
                    predictions.append(logits.detach().cpu())
                    targets.append(Y.cpu())

            if split == "val":
                metric_scores = self.run(targets, predictions)
                for metric_name, score in metric_scores.items():
                    out[f"{prefix}/{metric_name}"] = score

        model.train()
        return out

    def run(self, targets:List, preds:List):
        """
        Evaluate the predictions and targets.
        """
        targets = th.cat(targets, axis=0)
        preds = th.cat(preds, axis=0)

        results = {}
        for metric in self.metrics:
            results.update(metric(targets, preds))
        return results


import numpy as np
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, 
    confusion_matrix, matthews_corrcoef,
    precision_recall_curve, auc
)
        


def run_binary_classification_metrics(y_pred, y):
    """
    Run binary classification metrics.
    """
    if isinstance(y_pred[0], abc.Iterable):
        y_pred = np.array(y_pred)
        try:
            if y_pred.shape[-1] == 2:
                auc_score = roc_auc_score(y, y_pred[:,1])
                _precision, _recall, thresholds = precision_recall_curve(y, y_pred[:,1])
                auc_precision_recall = auc(_recall, _precision)
            else:
                auc_score = roc_auc_score(y, y_pred)
                _precision, _recall, thresholds = precision_recall_curve(y, y_pred[:,1])
                auc_precision_recall = auc(_recall, _precision)
        except ValueError as e:
            # If the prediction is not valid for AUC, set it to None
            logger.warning(f"Error calculating AUC: {e}")
            auc_score = None
            auc_precision_recall = None
        y_pred = np.argmax(y_pred, axis=1).tolist()
    else:
        auc_score = None

    def binary_sensitivity_specificity_precision_f1(cm):
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        def sensitivity():
            return TP / (TP + FN) if (TP + FN) > 0 else 0
        def specificity():
            return TN / (TN + FP) if (TN + FP) > 0 else 0
        def precision():
            return TP / (TP + FP) if (TP + FP) > 0 else 0
        def f1():
            return 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
        return sensitivity(), specificity(), precision(), f1()
    

    blacc_score = balanced_accuracy_score(y, y_pred)
    cfm = confusion_matrix(y, y_pred, labels=[0,1])

    sen, spec, precision, f1 = binary_sensitivity_specificity_precision_f1(cfm)
    mcc = matthews_corrcoef(y, y_pred)

    # class_distribution = np.bincount(np.array(y))
    # class_distribution = class_distribution / np.sum(class_distribution)
    # class_distribution = class_distribution.tolist()
    # class_distribution = {str(i): float(class_distribution[i]) for i in range(len(class_distribution))}
    # class_distribution = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: item[0])}
    
    score = {
        # 'class_distribution': class_distribution,
        'bl_acc': float(blacc_score),
        'cfm': cfm.tolist(),
        'sensitivity': float(sen),
        'specificity': float(spec),
        'mcc': float(mcc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(sen),
        'auc': float(auc_score) if auc_score is not None else None,
        'auprc': float(auc_precision_recall) if auc_precision_recall is not None else None,
    }

    return score


from nosos.datasets.cohort.cohort import DISEASE_GROUP_TO_DETAILED_ICDS

class HCAIMetric(RetrievalMetric):
    """
    Overall metric for all tasks.
    """
    name: str = "hcai_metric"
    def __init__(self, topk:int=1, aggregate_mode:str='max'):
        super().__init__(topk=topk)
        self.group_to_icd_mapping = {
            k:v for k,v in DISEASE_GROUP_TO_DETAILED_ICDS.items()
            if k in ['HCAI', 'SSI', 'SEPSIS', 'UTI', 'BSI', 'VAP']
        }
        self.aggregate_mode= aggregate_mode
        assert self.aggregate_mode in ['max', 'most', 'prob'], f"Invalid aggregate mode: {self.aggregate_mode}"


    def compute(self, targets:List[str], preds:List[List[str]]):
        
        score_dict = defaultdict(dict)
        for group_name, icds in self.group_to_icd_mapping.items():
            icds = set(icds)
            binary_predictions = []
            binary_targets = []
            for i in range(len(targets)):
                if len(targets[i]) == 0:
                    continue

                is_hcai_in_target = len(icds.intersection(targets[i])) > 0
                binary_targets.append(int(is_hcai_in_target))

                is_hcai_in_preds = []
                for j in range(len(preds[i])): # loop through top k predictions
                    if len(preds[i][j]) == 0:
                        continue
                    is_hcai_in_preds.append(int(len(icds.intersection(preds[i][j])) > 0))
                    
                if len(is_hcai_in_preds) > 0:
                    if self.aggregate_mode == 'max':
                        is_hcai_in_pred = max(is_hcai_in_preds) # prediction is 1 if any of the top k predictions are in the group
                    elif self.aggregate_mode == 'most':
                        is_hcai_in_pred = 1 if sum(is_hcai_in_preds) >= (len(is_hcai_in_preds) / 2) else 0
                    elif self.aggregate_mode == 'prob':
                        pos_prob = sum(is_hcai_in_preds) / len(is_hcai_in_preds)
                        neg_prob = 1 - pos_prob
                        is_hcai_in_pred = (neg_prob, pos_prob)
                    else:
                        raise ValueError(f"Invalid aggregate mode: {self.aggregate_mode}")
                else:
                    if self.aggregate_mode == 'prob':
                        is_hcai_in_pred = (1.0, 0)
                    else:
                        is_hcai_in_pred = 0
                    
                binary_predictions.append(is_hcai_in_pred)

            hcai_score = run_binary_classification_metrics(
                binary_predictions, 
                binary_targets
            )
            if self.topk > 1:
                hcai_score = {
                    f"{k}@{self.topk}": v for k, v in hcai_score.items()
                }

            score_dict[group_name] = hcai_score
                
        return score_dict
        



def run_evaluate(results, task:str, aggregate_mode:str='max'):
    """
    Evaluate the results of the inference.
    """
    ## Gather prediction and ground truth data
    all_predictions = []
    all_ground_truth = []
    for k_result in results:
        predictions = k_result['predictions']
        ground_truth = k_result['ground_truth']['expected']
        if task == Task.ICD_PREDICTION or task == Task.ICD_PREDICTION_NEXT_TW:
            ground_truth = [token for token in ground_truth if token not in [ST.START_ICD, ST.END_ICD]]
            ground_truth = translate_icdcm_tokens(ground_truth)

        k_preds = []
        for result in predictions:
            pred = result['decoded_generated_tokens']
            if task == Task.ICD_PREDICTION or task == Task.ICD_PREDICTION_NEXT_TW:
                pred = [token for token in pred if token not in [ST.START_ICD, ST.END_ICD]]
                pred = translate_icdcm_tokens(pred)
            k_preds.append(pred)

        all_predictions.append(k_preds)
        all_ground_truth.append(ground_truth)

    results = {
        'Overall': {}
    }
    for k in [1,3,5,10]:
        recall_score = RecallAtK(topk=k)(all_ground_truth, all_predictions)
        precision_score = PrecisionAtK(topk=k)(all_ground_truth, all_predictions)
        f1_score = F1ScoreAtK(topk=k)(all_ground_truth, all_predictions)
        hcai_score = HCAIMetric(topk=k, aggregate_mode=aggregate_mode)(all_ground_truth, all_predictions)

        for key in hcai_score.keys():
            if key in results.keys():
                results[key].update(hcai_score[key])
            else:
                results[key] = hcai_score[key]

        overall_result = {}
        overall_result.update(recall_score)
        overall_result.update(precision_score)
        overall_result.update(f1_score)
        results['Overall'].update(overall_result)

    return results