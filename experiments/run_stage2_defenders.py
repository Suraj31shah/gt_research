import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.defenders.naive_bayes import NaiveBayesDefender
from src.defenders.logistic import LogisticDefender
from src.defenders.svm import SVMDefender
from src.defenders.threshold import calibrate_thresholds

def evaluate_defender(defender, X, y, split_name):
    preds = defender.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    dr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return dr, fpr

def main():
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = 'enron'
    print(f"=== Stage 2: Training Defenders on {dataset.capitalize()} Dataset ===")
    
    # Load processed text and labels
    data_path = config['datasets'][dataset]['processed_path']
    if not os.path.exists(data_path) and os.path.exists(data_path.replace('.parquet', '.csv')):
        data_path = data_path.replace('.parquet', '.csv')
        
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        
    X_full = df['body'].fillna('').values
    y_full = df['label'].values
    
    # Load indices
    splits_dir = config['splits']['split_indices_dir']
    idx_train = np.load(os.path.join(splits_dir, f'{dataset}_train.npy'))
    idx_cal = np.load(os.path.join(splits_dir, f'{dataset}_cal.npy'))
    idx_test = np.load(os.path.join(splits_dir, f'{dataset}_test.npy'))
    
    X_train, y_train = X_full[idx_train], y_full[idx_train]
    X_cal, y_cal = X_full[idx_cal], y_full[idx_cal]
    X_test, y_test = X_full[idx_test], y_full[idx_test]
    
    # Define models
    models = [
        NaiveBayesDefender(),
        LogisticDefender(),
        SVMDefender()
    ]
    
    # Desired FPR operating points: permissive (5%), balanced (1%), strict (0.1%)
    fpr_targets = [0.05, 0.01, 0.001]
    
    results = []
    models_dir = 'results/models'
    os.makedirs(models_dir, exist_ok=True)
    
    for model in models:
        print(f"\nTraining {model.name}...")
        model.fit(X_train.tolist(), y_train)
        
        print(f"Calibrating thresholds on Calibration Set...")
        probs_cal = model.predict_proba(X_cal.tolist())[:, 1]
        thresholds = calibrate_thresholds(y_cal, probs_cal, fpr_targets)
        
        for target_fpr, tau in thresholds.items():
            model.set_threshold(tau)
            
            # Evaluate on Test Set
            dr_test, fpr_test = evaluate_defender(model, X_test.tolist(), y_test, 'Test')
            
            results.append({
                'Filter': model.name,
                'Target_FPR': target_fpr,
                'Actual_Threshold': tau,
                'Test_DR': dr_test,
                'Test_FPR': fpr_test
            })
            
            print(f"  [{target_fpr:.3f} Target] Threshold={tau:.4f} -> Test DR: {dr_test:.4f}, Test FPR: {fpr_test:.4f}")
            
        # Save model for later usage
        joblib.dump(model, os.path.join(models_dir, f'{model.name}_{dataset}.pkl'))
        
    print("\n=== Summary of Defender Strategies (Pure Strategies for Game) ===")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    os.makedirs('results/logs', exist_ok=True)
    df_results.to_csv(f'results/logs/defender_performance_{dataset}.csv', index=False)
    print("\nStage 2 completed successfully!")

if __name__ == "__main__":
    main()
