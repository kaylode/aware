
from __future__ import annotations
from collections import abc
import numpy as np
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, 
    confusion_matrix, matthews_corrcoef,
    precision_recall_curve, auc,
    average_precision_score
)
from sklearn.metrics import mean_squared_log_error, median_absolute_error, explained_variance_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
        
from ehrpfn.utils import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

def run_binary_classification_metrics(y_pred, y, average='weighted', plot_save_path=None):
    """
    Run binary classification metrics.
    """
    if isinstance(y_pred[0], abc.Iterable):
        y_pred = np.array(y_pred)
        try:
            if y_pred.shape[-1] == 2 or len(np.unique(y)) == 2:
                auc_score = roc_auc_score(y, y_pred[:,1], average=average, multi_class='ovr')
        
            else:
                auc_score = roc_auc_score(y, y_pred, average=average, multi_class='ovo')
        except ValueError as e:
            # If the prediction is not valid for AUC, set it to None
            LOGGER.text(f"Error calculating AUROC: {e}", level=LoggerObserver.WARN)
            auc_score = None

        try:
            if y_pred.shape[-1] == 2 or len(np.unique(y)) == 2:
                _precision, _recall, thresholds = precision_recall_curve(y, y_pred[:,1])
                auc_precision_recall = auc(_recall, _precision)
            else:
                y_bin = label_binarize(y, classes=np.unique(y))
                auc_precision_recall = average_precision_score(y_bin, y_pred, average=average)
        except ValueError as e:
            LOGGER.text(f"Error calculating AUPRC: {e}", level=LoggerObserver.WARN)
            auc_precision_recall = None

        if plot_save_path:
            plot_prediction_distribution(y_pred, y, save_path=plot_save_path)

        y_pred = np.argmax(y_pred, axis=1).tolist()
    else:
        auc_score = None
        auc_precision_recall = None

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
    cfm = confusion_matrix(y, y_pred)

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


def run_regression_metrics(y_pred, y):
    """
    Run regression metrics.
    """
    if isinstance(y_pred[0], abc.Iterable):
        y_pred = np.array(y_pred)
        y_pred = y_pred[:,0] if y_pred.shape[-1] > 1 else y_pred[:,0]
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(mse)
    # mape = np.mean(np.abs((y - y_pred) / y)) * 100 if np.any(y) else np.nan
    evs = explained_variance_score(y, y_pred)
    medae = median_absolute_error(y, y_pred)
    msle = mean_squared_log_error(y, y_pred) if np.all(y >= 0) and np.all(y_pred >= 0) else np.nan
    
    score = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'rmse': float(rmse),
        # 'mape': float(mape),
        'evs': float(evs),
        'medae': float(medae),
        'msle': float(msle),
    }
    return score

def plot_prediction_distribution(y_pred, y, save_path=None):
    """
    Plot the smooth histogram (KDE + histogram) of predicted probabilities 
    for the positive class, stratified by the true label.
    This helps visualize the model's confidence and separability.
    """
    if isinstance(y_pred[0], abc.Iterable):
        y_pred = np.array(y_pred)
        if y_pred.shape[-1] >= 2:
            probs = y_pred[:, 1]
        else:
            probs = y_pred
    else:
        probs = np.array(y_pred)
        
    y = np.array(y)
    plt.figure(figsize=(8, 6))
    
    # Plot for negative class
    sns.histplot(probs[y == 0], color='blue', label='Class 0 (Negative)', kde=True, 
                 stat='density', alpha=0.4, bins=30)
    
    # Plot for positive class
    sns.histplot(probs[y == 1], color='red', label='Class 1 (Positive)', kde=True, 
                 stat='density', alpha=0.4, bins=30)
                 
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability of Positive Class")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        # Ensure .pdf extension if specified or just use what's provided
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        LOGGER.text(f"Saved prediction distribution plot to {save_path}", level=LoggerObserver.INFO)
    
    plt.close()