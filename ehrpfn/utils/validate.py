import sys
from typing import *
import json
import os.path as osp
import numpy as np
from .visualize import make_cm_fig
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, 
    confusion_matrix, matthews_corrcoef,
)
from scikitplot.metrics import plot_precision_recall, plot_roc
import matplotlib.pyplot as plt

def binary_sensitivity_specificity_precision_f1(cm):
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    def sensitivity():
        return TP / (TP + FN)
    def specificity():
        return TN / (TN + FP)
    def precision():
        return TP / (TP + FP)
    def f1():
        return 2*TP / (2*TP + FP + FN)
    return sensitivity(), specificity(), precision(), f1()

def validate(proba_function, X, y, output_folder:str=None):
    y_prob = proba_function(X)
    y_pred = np.argmax(y_prob, axis=1)
    if y_prob.shape[-1] == 2:
        auc_score = roc_auc_score(y, y_prob[:,1], average="macro", multi_class='ovr')
    else:
        auc_score = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
    blacc_score = balanced_accuracy_score(y, y_pred)
    cfm = confusion_matrix(y, y_pred)

    sen, spec, precision, f1 = binary_sensitivity_specificity_precision_f1(cfm)
    mcc = matthews_corrcoef(y, y_pred)

    class_distribution = np.bincount(np.array(y))
    class_distribution = class_distribution / np.sum(class_distribution)
    class_distribution = class_distribution.tolist()
    class_distribution = {str(i): float(class_distribution[i]) for i in range(len(class_distribution))}
    class_distribution = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: item[0])}
    
    score = {
        'class_distribution': class_distribution,
        'auc': float(auc_score),
        'bl_acc': float(blacc_score),
        'cfm': cfm.tolist(),
        'sensitivity': float(sen),
        'specificity': float(spec),
        'mcc': float(mcc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(sen),
    }

    if output_folder is not None:
        json.dump(score, open(osp.join(output_folder, "score.json"), "w"), indent=4)
    
    # score['cfm_fig'] = cfm_fig
    # score['y_prob'] = y_prob
    return score


def cross_validate(proba_functions, Xs, ys, output_folder:str=None):

    def get_mean_std(scores):
        return (
            np.mean(scores), \
            np.std(scores)
        )

    scores = []
    for proba_function, X, y in zip(proba_functions, Xs, ys):
        scores.append(validate(proba_function, X, y, output_folder=None))

    avg_score = {
        'auc': get_mean_std([score['auc'] for score in scores]),
        'bl_acc': get_mean_std([score['bl_acc'] for score in scores]),
        'sensitivity': get_mean_std([score['sensitivity'] for score in scores]),
        'specificity': get_mean_std([score['specificity'] for score in scores]),
        'mcc': get_mean_std([score['mcc'] for score in scores]),
        'f1': get_mean_std([score['f1'] for score in scores]),
        'precision': get_mean_std([score['precision'] for score in scores]),
        'recall': get_mean_std([score['recall'] for score in scores])
    }

    if output_folder is not None:
        json.dump(avg_score, open(osp.join(output_folder, "cross_validation_score.json"), "w"))
    
    return avg_score