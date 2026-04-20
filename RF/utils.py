from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score, classification_report
)

import os
import json
import pickle

#-------------- DEFINING DATA CONSTANTS --------------
TAX_ORDER_DICT = {
    'Andisols': 0,     
    'Entisols': 1, 
    'Gelisols': 2,
    'Histosols': 3, 
    'Inceptisols': 4,
    'Mollisols': 5,    
    'Spodosols': 6,
}

TAX_ORDER_REVERSE_DICT = {
    0: 'Andisols',
    1: 'Entisols',
    2: 'Gelisols',
    3: 'Histosols',
    4: 'Inceptisols',
    5: 'Mollisols',
    6: 'Spodosols',
}

PEAT_LEVEL_DICT = {
    'no': 0,
    'shallow': 1,
    'deep': 2,
}

PEAT_LEVEL_REVERSE_DICT = {
    0: 'no',
    1: 'shallow',
    2: 'deep',
}

EPS = 1e-15

#-------------- DATA CLEANING FUNCS --------------
def _collapse_value(v):
    """
    Returns:
      - scalar (str/int/float/...) if v is scalar
      - v[0] if v is a list and all elements are equal
      - None otherwise (meaning: ambiguous / invalid)
    """
    if isinstance(v, list):
        if not v:
            return None
        first = v[0]
        # all-same check without building a set
        if all(x == first for x in v):
            return first
        return None
    return v


def _get_tax_order_category(value):
    v = _collapse_value(value)
    if isinstance(v, str):
        return TAX_ORDER_DICT.get(v, None)
    else:
        return None
    
def _get_peat_level(value):
    v = _collapse_value(value)
    if isinstance(v, str):
        return PEAT_LEVEL_DICT.get(v, None)
    else:
        return None
    
def _get_peat_level_binary(value):
    binary_value = []
    if isinstance(value, str):
        value = [value]
    for v in value:
        if v == 'no':
            binary_value.append(0)
        if v in ("deep", "shallow", "deep/shallow"):
            binary_value.append(1)

    v = _collapse_value(binary_value)
    if v is not None:
        return v
    else:
        return None
    
def _get_aksdb_pf1m_bin(value):
    v = _collapse_value(value)
    if pd.isna(v):
        return None

    if isinstance(v, (int, float)):
        return int(v)

    return None
    
#-------------- EVALUATION METRIC FUNCS --------------
def compute_binary_ACC(gt, pred):
    threshold = 0.5
    gt = np.asarray(gt).astype(int).ravel()
    
    pred = np.asarray(pred).ravel()

    y_pred = (pred >= threshold).astype(int)
    y_true = gt

    # Support (counts) for each class
    n = y_true.size
    n0 = int(np.sum(y_true == 0))
    n1 = int(np.sum(y_true == 1))

    # Confusion matrix components (standard: positive is class 1)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Accuracy
    accuracy = (TP + TN) / max(n, 1)

    # Per-class precision/recall/F1
    # Class 1 metrics (positive class = 1)
    precision1 = TP / (TP + FP + EPS)
    recall1 = TP / (TP + FN + EPS)
    fscore1 = 2 * precision1 * recall1 / (precision1 + recall1 + EPS)

    # Class 0 metrics (treat class 0 as the "positive" class)
    # Equivalent to swapping roles: TP0=TN, FP0=FN, FN0=FP
    TP0 = TN
    FP0 = FN
    FN0 = FP
    precision0 = TP0 / (TP0 + FP0 + EPS)
    recall0 = TP0 / (TP0 + FN0 + EPS)
    fscore0 = 2 * precision0 * recall0 / (precision0 + recall0 + EPS)

    # Balanced accuracy = mean of recalls (a.k.a. (TPR + TNR)/2)
    balanced_accuracy = 0.5 * (recall0 + recall1)

    # Weighted F1 (support-weighted average of per-class F1)
    # sklearn's average="weighted" behavior for binary
    weighted_f1 = (n0 * fscore0 + n1 * fscore1) / max(n0 + n1, 1)

    # AUC metrics (use raw scores)
    # Handle degenerate case where only one class exists in y_true
    if len(np.unique(y_true)) < 2:
        auc_roc = float("nan")
        pr_auc = float("nan")
    else:
        auc_roc = roc_auc_score(y_true, pred)
        pr_auc = average_precision_score(y_true, pred)

    return {'neg_accuracy': -accuracy,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision0': precision0, 
            'recall0': recall0, 
            'fscore0': fscore0,
            'precision1': precision1, 
            'recall1': recall1,
            'fscore1': fscore1, 
            'weighted_f1': weighted_f1,
            'auc_roc': auc_roc,
            'pr_auc': pr_auc}

        
def compute_multi_ACC(gt, pred, num_classes, task_name):
    y_true = np.asarray(gt).astype(int).ravel()
    y_pred = np.asarray(pred).astype(int).ravel()
    labels = np.arange(num_classes)
    n = y_true.size
    
    accuracy = float(np.mean(y_true == y_pred))
    # Confusion matrix: rows=true, cols=pred
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for i in labels:
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        p_i = tp / (tp + fp + EPS)
        r_i = tp / (tp + fn + EPS)
        f_i = 2 * p_i * r_i / (p_i + r_i + EPS)

        precisions.append(p_i)
        recalls.append(r_i)
        f1s.append(f_i)
        supports.append(support)
        
        if task_name == 'tax_order':
            per_class[TAX_ORDER_REVERSE_DICT[i]] = {'precision': p_i, 'recall': r_i, 'f1': f_i, 'support': int(support)}
        elif task_name == 'peat_level':
            per_class[PEAT_LEVEL_REVERSE_DICT[i]] = {'precision': p_i, 'recall': r_i, 'f1': f_i, 'support': int(support)}

    # Macro averages (unweighted mean over classes)
    precision_macro = float(np.mean(precisions))
    recall_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))

    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Weighted averages (support-weighted)
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    # Micro averages (global TP/FP/FN over all classes)
    # For single-label multiclass: micro-F1 == accuracy, but still nice to report explicitly.
    precision_micro = float(precision_score(y_true, y_pred, average="micro", labels=labels, zero_division=0))
    recall_micro = float(recall_score(y_true, y_pred, average="micro", labels=labels, zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0))

    print(f"{task_name} Yijun metrics")
    print(
        f"{precision_macro:.5f},"
        f"{recall_macro:.5f},"
        f"{f1_macro:.5f},"
        f"{precision_micro:.5f},"
        f"{recall_micro:.5f},"
        f"{f1_micro:.5f},"
        f"{precision_weighted:.5f},"
        f"{recall_weighted:.5f},"
        f"{f1_weighted:.5f},"
        f"{accuracy:.5f},"
        f"{balanced_acc:.5f},"
        f"{per_class['Andisols']['precision']:.5f},"
        f"{per_class['Andisols']['recall']:.5f},"
        f"{per_class['Entisols']['precision']:.5f},"
        f"{per_class['Entisols']['recall']:.5f},"
        f"{per_class['Gelisols']['precision']:.5f},"
        f"{per_class['Gelisols']['recall']:.5f},"
        f"{per_class['Histosols']['precision']:.5f},"
        f"{per_class['Histosols']['recall']:.5f},"
        f"{per_class['Inceptisols']['precision']:.5f},"
        f"{per_class['Inceptisols']['recall']:.5f},"
        f"{per_class['Mollisols']['precision']:.5f},"
        f"{per_class['Mollisols']['recall']:.5f},"
        f"{per_class['Spodosols']['precision']:.5f},"
        f"{per_class['Spodosols']['recall']:.5f}"
    )
    
    return {
        "neg_accuracy": -accuracy,   # need "lower is better" optimization
        "accuracy": accuracy, 
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,  # should be the same as accuracy
        "per_class": per_class,
    }

def run_binary_metric(all_ids, kfold_indices, y, pred_folder):
    indices = np.array(all_ids)

    for fold, train_test_dict in kfold_indices.items():
        fold_save_path = pred_folder
        pred_df = pd.read_csv(os.path.join(pred_folder, f"test_preds_{fold}.csv"), index_col=0)
        pred = pred_df["pred"].to_numpy()

        test_indices = train_test_dict['test']

        test_rows = np.where(np.isin(indices, test_indices))[0]
        y_test = y[test_rows] 
        
        yijun_metric_dict = compute_binary_ACC(y_test, pred)
        
        print(
            f"{yijun_metric_dict['precision0']:.5f},"
            f"{yijun_metric_dict['recall0']:.5f},"
            f"{yijun_metric_dict['fscore0']:.5f},"
            f"{yijun_metric_dict['precision1']:.5f},"
            f"{yijun_metric_dict['recall1']:.5f},"
            f"{yijun_metric_dict['fscore1']:.5f},"
            f"{yijun_metric_dict['weighted_f1']:.5f},"
            f"{yijun_metric_dict['accuracy']:.5f},"
            f"{yijun_metric_dict['balanced_accuracy']:.5f},"
            f"{yijun_metric_dict['auc_roc']:.5f},"
            f"{yijun_metric_dict['pr_auc']:.5f}"
        )
        yijun_metric_df = pd.DataFrame([yijun_metric_dict])
        yijun_metric_df.to_csv(os.path.join(fold_save_path, fold + '_yijun_metric.csv'))
    
def run_multi_metric(all_ids, kfold_indices, y, pred_folder):
    indices = np.array(all_ids)
    num_classes = 7
    task_name = 'tax_order'

    for fold, train_test_dict in kfold_indices.items():
        fold_save_path = pred_folder
        pred_df = pd.read_csv(os.path.join(pred_folder, f"test_preds_{fold}.csv"), index_col=0)
        pred = pred_df["pred"].to_numpy()

        test_indices = train_test_dict['test']
        test_rows = np.where(np.isin(indices, test_indices))[0]
        y_test = y[test_rows] 
        
        yijun_metric_dict = compute_multi_ACC(y_test, pred, num_classes, task_name)
        yijun_metric_df = pd.DataFrame([yijun_metric_dict])
        yijun_metric_df.to_csv(os.path.join(fold_save_path, fold + '_yijun_metric.csv'))
    
#-------------- SKLEARN RUN FUNCS --------------
def run_rf_binary(all_ids, kfold_indices, X, y, hyperparameters, outroot=".", save_rf=None):
    """
    Binary Classification for x_folds
    """
    indices = np.array(all_ids)
    
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_precision_0 = []
    fold_precision_1 = []
    fold_recall_0 = []
    fold_recall_1 = []
    
    for fold, train_test_dict in kfold_indices.items():
        print('Training Fold: ', fold)

        train_indices = train_test_dict['train']
        test_indices = train_test_dict['test']

        train_rows = np.where(np.isin(indices, train_indices))[0]
        test_rows = np.where(np.isin(indices, test_indices))[0]

        X_train = X[train_rows]
        X_test = X[test_rows]
        y_train = y[train_rows]
        y_test = y[test_rows] 

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        rf = RandomForestClassifier(
            n_estimators=hyperparameters['est'],
            max_depth=hyperparameters['max_depth'],
            min_samples_split=hyperparameters['min_samples_split'],
            n_jobs=hyperparameters['n_jobs'],
            verbose=hyperparameters['verbose']
        )
        rf.fit(X_train, y_train.ravel())
        y_pred = rf.predict(X_test)

        ################ PREDICTION TYPE SWAP ################
#         preds = rf.predict_proba(X_test)
#         y_proba = preds[:, 1]
        ################ PREDICTION TYPE SWAP ################

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        class_precisions = precision_score(y_test, y_pred, average=None)
        class_recalls = recall_score(y_test, y_pred, average=None)
        
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_precision_0.append(class_precisions[0])
        fold_precision_1.append(class_precisions[1])
        fold_recall_0.append(class_recalls[0])
        fold_recall_1.append(class_recalls[1])
        
        fold_num = int(fold.split('_')[-1])
        print(f"Fold {fold_num} Accuracy: {accuracy:.4f}")
        print(f"Fold {fold_num} Precision: {precision:.4f}")
        print(f"Fold {fold_num} Recall: {recall:.4f}")
        
        #Saving preds
        test_ids = indices[np.isin(indices, test_indices)]
        preds = {
            "id": test_ids,  
            "gt": y_test.flatten(), 
            "pred": y_pred  
        }
        df_predictions = pd.DataFrame(preds)
        preds_outpath = os.path.join(outroot, f"test_preds_{fold}.csv")
        df_predictions.to_csv(preds_outpath, index=False)
        print(f"Test predictions saved to {preds_outpath}")
        
        if save_rf:
            model_file = os.path.join(save_rf, f"rf_weights_{fold}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(rf, f)
            print(f"Model saved to {model_file}")
            
    #Save results csv
    results = {
            "Fold": list(range(0, len(fold_accuracies))),
            "Accuracy": fold_accuracies,
            "Precision": fold_precisions,
            "Recall": fold_recalls,
            "Precision_0": fold_precision_0,
            "Recall_0": fold_recall_0,
            "Precision_1": fold_precision_1,
            "Recall_1": fold_recall_1,

        }
    results_df = pd.DataFrame(results)
    result_outpath = os.path.join(outroot, "sklearn_metric_summary.csv")

def run_rf_multiclass(all_ids, kfold_indices, X, y, hyperparameters, outroot=".", save_rf=None):
    """
    Multiclass Classification SKLEARN Run Functions
    """
    indices = np.array(all_ids)
    
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_weighted_f1s = []
    class_precisions = {}
    class_recalls = {}
    
    unique_classes = np.unique(y)
    
    for class_label in unique_classes:
        class_precisions[class_label] = []
        class_recalls[class_label] = []
    
    for fold, train_test_dict in kfold_indices.items():
        print('Training Fold: ', fold)

        train_indices = train_test_dict['train']
        test_indices = train_test_dict['test']

        train_rows = np.where(np.isin(indices, train_indices))[0]
        test_rows = np.where(np.isin(indices, test_indices))[0]

        X_train = X[train_rows]
        X_test = X[test_rows]
        y_train = y[train_rows]
        y_test = y[test_rows] 

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        est = hyperparameters['est']
        verbose = hyperparameters['verbose']
        n_jobs = hyperparameters['n_jobs']
        rf = RandomForestClassifier(n_estimators=est, verbose=verbose, n_jobs=n_jobs,)
        rf.fit(X_train, y_train.ravel())
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        precisions_per_class = precision_score(y_test, y_pred, average=None)
        recalls_per_class = recall_score(y_test, y_pred, average=None)
        
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_weighted_f1s.append(weighted_f1)
        
        unique_y_test = np.unique(y_test)
        for i, class_label in enumerate(np.unique(y_test)):
            class_precision = precisions_per_class[i]
            class_recall = recalls_per_class[i]
            class_precisions[class_label].append(class_precision)
            class_recalls[class_label].append(class_recall)
        
        fold_num = int(fold.split('_')[-1])
        print(f"Fold {fold_num} Accuracy: {accuracy:.4f}")
        print(f"Fold {fold_num} Precision: {precision:.4f}")
        print(f"Fold {fold_num} Recall: {recall:.4f}")
        print(f"Fold {fold_num} F1: {f1:.4f}")
        
        #Save predictions
        test_ids = indices[np.isin(indices, test_indices)]
        preds = {
            "id": test_ids,  
            "gt": y_test.flatten(), 
            "pred": y_pred  
        }
        df_predictions = pd.DataFrame(preds) 
        preds_outpath = os.path.join(outroot, f"test_preds_{fold}.csv")
        df_predictions.to_csv(preds_outpath, index=False)
        print(f"Test predictions saved to {preds_outpath}")
            
        if save_rf:
            model_file = os.path.join(save_rf, f"rf_weights_{fold}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(rf, f)
            print(f"Model saved to {model_file}")
            
    # Save general results file
    results = {
        "Fold": list(range(0, len(fold_accuracies))),
        "Accuracy": fold_accuracies,
        "Precision": fold_precisions,
        "Recall": fold_recalls,
        "F1": fold_f1s,
        "Weighted_F1": fold_weighted_f1s
    }
    for class_idx, precision_val_list in class_precisions.items():
        recall_val_list = class_recalls[class_idx]
        p_key = "Precision_" + TAX_ORDER_REVERSE_DICT[class_idx]
        r_key = "Recall_" + TAX_ORDER_REVERSE_DICT[class_idx]
        results[p_key] = precision_val_list
        results[r_key] = recall_val_list

        results_df = pd.DataFrame(results)
        result_outpath = os.path.join(outroot, "sklearn_metric_summary.csv")
        results_df.to_csv(result_outpath, index = False)
    print(f"Metrics saved to {result_outpath}")