import os
import yaml
import numpy as np
import pandas as pd
import joblib
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.base import NoAttack
from src.attacks.obfuscation import ObfuscationAttack
from src.attacks.synonym_swap import SynonymSwapAttack
from src.attacks.ham_injection import HamInjectionAttack
from src.attacks.similarity import get_similarity_score, similarity_check
from src.data.loaders import get_active_dataset, load_processed_data

def main():
    random.seed(42)
    np.random.seed(42)
    
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = get_active_dataset(config)
    print(f"=== Stage 3: Evaluating Attacks on {dataset.capitalize()} ===")
    
    df = load_processed_data(config, dataset)
    X_full = df['body'].fillna('').values
    y_full = df['label'].values
    
    splits_dir = config['splits']['split_indices_dir']
    idx_cal = np.load(os.path.join(splits_dir, f'{dataset}_cal.npy'))
    
    spam_indices = [i for i in idx_cal if y_full[i] == 1]
    sample_indices = random.sample(spam_indices, min(100, len(spam_indices)))
    X_spam = X_full[sample_indices]
    
    attacks = [
        NoAttack(),
        ObfuscationAttack(strength=0.05),
        ObfuscationAttack(strength=0.15),
        SynonymSwapAttack(strength=0.5),
        SynonymSwapAttack(strength=1.0),
        HamInjectionAttack(strength=1),
        HamInjectionAttack(strength=3),
    ]
    
    defender_model_path = f'results/models/LogisticRegression_{dataset}.pkl'
    if not os.path.exists(defender_model_path):
        print(f"Error: Model file {defender_model_path} not found. Please run Stage 2 first.")
        return
        
    defender = joblib.load(defender_model_path)
    
    # Dynamically find the calibrated threshold for target FPR = 0.010
    tau = 0.5
    perf_path = f'results/logs/defender_performance_{dataset}.csv'
    if os.path.exists(perf_path):
        df_perf = pd.read_csv(perf_path)
        lr_rows = df_perf[(df_perf['Filter'] == 'LogisticRegression') & np.isclose(df_perf['Target_FPR'], 0.010)]
        if not lr_rows.empty:
            tau = float(lr_rows.iloc[0]['Actual_Threshold'])
    defender.set_threshold(tau)
    print(f"Calibrated threshold for Logistic Regression (Target FPR=0.010): {tau:.4f}")
    
    print(f"\nEvaluating against Logistic Regression (Target FPR=0.010)")
    print(f"{'Attack':<25} | {'Valid %':<10} | {'Evasion Success':<15}")
    print("-" * 55)
    
    for attack in attacks:
        valid_count = 0
        evasions = 0
        total = len(X_spam)
        
        for email in X_spam:
            attacked = attack.apply(email)
            
            if similarity_check(email, attacked, min_sim=0.75):
                valid_count += 1
                
                pred = defender.predict([attacked])[0]
                if pred == 0:  
                    evasions += 1
                    
        valid_rate = valid_count / total * 100
        evasion_rate = (evasions / valid_count * 100) if valid_count > 0 else 0
        
        print(f"{attack.name + '_' + str(attack.strength):<25} | {valid_rate:>6.1f}%    | {evasion_rate:>13.1f}%")
        
    print("\nStage 3 completed successfully!")

if __name__ == "__main__":
    main()
