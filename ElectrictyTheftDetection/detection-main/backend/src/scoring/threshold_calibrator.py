from sklearn.metrics import precision_recall_curve
import numpy as np

def find_optimal_threshold(y_true, y_probs, target_precision=0.85):
    """Finds the threshold that achieves a specific target precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find first index where precision meets target
    idx = np.where(precisions >= target_precision)[0]
    
    if len(idx) > 0:
        best_idx = idx[0]
        # thresholds is 1 element shorter than precisions/recalls
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return best_threshold, precisions[best_idx], recalls[best_idx]
    
    return 0.5, precisions.max(), 0.0
