from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

def calculate_all_metrics(y_true, y_pred, y_probs=None):
    """Computes a dictionary of standard classification metrics."""
    metrics = {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_probs is not None:
        metrics['AUC_ROC'] = roc_auc_score(y_true, y_probs)
        
    # False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return metrics

def get_metrics_summary_table(model_results_dict):
    """
    Takes a dictionary mapping model names to (y_true, y_pred, y_probs) 
    and returns a formatted DataFrame for comparison.
    """
    rows = []
    for name, data in model_results_dict.items():
        y_t, y_p, y_pb = data
        m = calculate_all_metrics(y_t, y_p, y_pb)
        m['Model'] = name
        rows.append(m)
        
    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ['Model', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC', 'FPR']
    return df[[c for c in cols if c in df.columns]]
