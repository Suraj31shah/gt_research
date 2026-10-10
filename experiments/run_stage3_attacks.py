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

def main():
    random.seed(42)
    np.random.seed(42)
    
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = 'enron'
    print(f"=== Stage 3: Evaluating Attacks on {dataset.capitalize()} ===")
    
    data_path = config['datasets'][dataset]['processed_path'].replace('.parquet', '.csv')
    df = pd.read_csv(data_path)
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
    
    defender = joblib.load(f'results/models/LogisticRegression_{dataset}.pkl')
    # Use the 0.010 target FPR threshold
    defender.set_threshold(0.6691) 
    
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
