import numpy as np
from datasets.buildin import TAX_ORDER_REVERSE_DICT
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_MSE(gt, pred):
    return np.mean((gt - pred) ** 2)

def compute_RMSE(gt, pred):
    return compute_MSE(gt, pred) ** 0.5

def compute_BCE(gt, pred):
    eps = 1e-15  # Small epsilon for numerical stability
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    
def compute_weighted_BCE(gt, pred):
    """
    Compute weighted Binary Cross-Entropy (BCE) loss.

    Args:
        gt (np.ndarray): Ground truth labels (binary, 0 or 1), shape (N,).
        pred (np.ndarray): Predicted probabilities, shape (N,).

    Returns:
        float: Weighted BCE loss.
    """
    eps = 1e-15  # Small epsilon for numerical stability
    pred = np.clip(pred, eps, 1 - eps)

    # Compute class frequencies
    n_0 = np.sum(gt == 0.)  # Count of class 0
    n_1 = np.sum(gt == 1.)  # Count of class 1

    # Compute inverse class weights (to give more weight to the minority class)
    w_0 = 1.0 if n_0 == 0 else (n_0 + n_1) / (2.0 * n_0)
    w_1 = 1.0 if n_1 == 0 else (n_0 + n_1) / (2.0 * n_1)
    # Compute weighted BCE loss
    loss = -np.mean(w_1 * gt * np.log(pred) + w_0 * (1 - gt) * np.log(1 - pred))
    return loss


def compute_binary_ACC(gt, pred):
    threshold = 0.5
    binary_pred = (pred >= threshold).astype(int)
    binary_gt = gt.astype(int)
    accuracy = np.sum(binary_pred == binary_gt) * 1. / binary_gt.shape[0]
    
    TP = np.sum((binary_pred == 0) & (binary_gt == 0))  # True Positives
    FP = np.sum((binary_pred == 0) & (binary_gt == 1))  # False Positives
    FN = np.sum((binary_pred == 1) & (binary_gt == 0))  # False Negatives
    p0 = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r0 = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f0 = 2 * (p0 * r0) / (p0 + r0 + 1e-12)
    
    TP = np.sum((binary_pred == 1) & (binary_gt == 1))  # True Positives
    FP = np.sum((binary_pred == 1) & (binary_gt == 0))  # False Positives
    FN = np.sum((binary_pred == 0) & (binary_gt == 1))  # False Negatives
    p1 = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r1 = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (p1 * r1) / (p1 + r1)

    auc_roc = roc_auc_score(gt, pred)
    
    return {'neg_accuracy': -accuracy,
            'accuracy': accuracy,
            'precision0': p0, 
            'recall0': r0, 
            'fscore0': f0,
            'precision1': p1, 
            'recall1': r1,
            'fscore1': f1, 
            'auc_roc': auc_roc}
    
def compute_multi_ACC(gt, pred, num_classes):
    pred = pred.astype(int)
    gt = gt.astype(int)
    accuracy = np.sum(pred == gt) * 1. / gt.shape[0]
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gt, pred):
        cm[t, p] += 1

    output = {}
    precisions, recalls, f1s = [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_i = 2 * (p_i * r_i) / (p_i + r_i + 1e-12)
        precisions.append(p_i)
        recalls.append(r_i)
        f1s.append(f_i)
        output[TAX_ORDER_REVERSE_DICT[i]] = {'precision': p_i, 'recall': r_i, 'f1': f_i}
    
    p = np.mean(precisions)
    r = np.mean(recalls)
    f1 = np.mean(f1s)
    output['neg_accuracy'] = -accuracy
    output['accuracy'] = accuracy
    output['precision'] = p
    output['recall'] = r
    output['fscore'] = f1

    precision_weighted = precision_score(gt, pred, average='weighted')
    recall_weighted = recall_score(gt, pred, average='weighted')
    f1_weighted = f1_score(gt, pred, average='weighted')
    output['precision_weighted'] = precision_weighted
    output['recall_weighted'] = recall_weighted
    output['f1_weighted'] = f1_weighted

    return output
    