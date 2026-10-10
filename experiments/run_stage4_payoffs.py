import os
import yaml
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.payoffs.engine import estimate_payoffs
from src.attacks.base import NoAttack
from src.attacks.obfuscation import ObfuscationAttack
from src.attacks.synonym_swap import SynonymSwapAttack
from src.attacks.ham_injection import HamInjectionAttack

def load_all_defenders(dataset):
    models = []
    base_models = [
        joblib.load(f'results/models/NaiveBayes_{dataset}.pkl'),
        joblib.load(f'results/models/LogisticRegression_{dataset}.pkl'),
        joblib.load(f'results/models/SVM_{dataset}.pkl')
    ]
    df_perf = pd.read_csv(f'results/logs/defender_performance_{dataset}.csv')
    
    for base in base_models:
        rows = df_perf[df_perf['Filter'] == base.name]
        for _, row in rows.iterrows():
            import copy
            model = copy.deepcopy(base)
            model.set_threshold(row['Actual_Threshold'])
            models.append(model)
            
    return models

def main():
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/costs.yaml', 'r') as f:
        costs_cfg = yaml.safe_load(f)
    with open('configs/types.yaml', 'r') as f:
        types_cfg = yaml.safe_load(f)
        
    dataset = 'enron'
    print(f"=== Stage 4: Estimating Payoff Tensors on {dataset.capitalize()} ===")
    
    data_path = config['datasets'][dataset]['processed_path'].replace('.parquet', '.csv')
    df = pd.read_csv(data_path)
    X_full = df['body'].fillna('').values
    y_full = df['label'].values
    
    splits_dir = config['splits']['split_indices_dir']
    idx_test = np.load(os.path.join(splits_dir, f'{dataset}_test.npy'))
    
    np.random.seed(42)
    # Use subset for faster estimation in development
    sample_idx = np.random.choice(idx_test, size=min(400, len(idx_test)), replace=False)
    
    X_test = X_full[sample_idx]
    y_test = y_full[sample_idx]
    
    ham_emails = X_test[y_test == 0].tolist()
    threat_emails = X_test[y_test == 1].tolist()
    
    email_splits = {
        'ham': ham_emails,
        'threat': threat_emails
    }
    
    defenders = load_all_defenders(dataset)
    
    attacks = [
        NoAttack(),                         # 0
        ObfuscationAttack(strength=0.15),   # 1
        SynonymSwapAttack(strength=0.5),    # 2
        SynonymSwapAttack(strength=1.0),    # 3
        HamInjectionAttack(strength=1),     # 4
        HamInjectionAttack(strength=3),     # 5
    ]
    
    U_D, U_A = estimate_payoffs(defenders, attacks, types_cfg['types'], costs_cfg, email_splits)
    
    os.makedirs('results/tensors', exist_ok=True)
    np.save(f'results/tensors/U_D_{dataset}.npy', U_D)
    np.save(f'results/tensors/U_A_{dataset}.npy', U_A)
    
    print(f"Saved payoff tensors of shape {U_D.shape} (Types x Defenders x Attacks)")
    print("\nStage 4 completed successfully!")

if __name__ == "__main__":
    main()
