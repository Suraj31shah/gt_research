import numpy as np
from sklearn.metrics import roc_curve

def calibrate_thresholds(y_cal, probs_cal, fpr_targets=(0.05, 0.01, 0.001)):
    """
    Calibrates thresholds to achieve target False Positive Rates.
    y_cal: true labels (0=ham, 1=spam)
    probs_cal: predicted probabilities for spam class
    fpr_targets: list of desired maximum FPRs (e.g. permissive, balanced, strict)
    
    Returns: dict mapping target FPR to threshold
    """
    fpr, tpr, thresholds = roc_curve(y_cal, probs_cal)
    
    results = {}
    for target in fpr_targets:
        # Find the threshold that gives an FPR closest to, but not exceeding, the target
        # roc_curve returns FPR in increasing order
        valid_indices = np.where(fpr <= target)[0]
        if len(valid_indices) == 0:
            best_idx = 0
        else:
            best_idx = valid_indices[-1]
            
        results[target] = thresholds[best_idx]
        
    return results
