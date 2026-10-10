import os
import yaml
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.bayesian_stackelberg import solve as solve_dobss
from src.payoffs.engine import estimate_payoffs
from src.attacks.base import NoAttack
from src.attacks.obfuscation import ObfuscationAttack
from src.attacks.synonym_swap import SynonymSwapAttack
from src.attacks.ham_injection import HamInjectionAttack
from experiments.run_stage4_payoffs import load_all_defenders
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def sweep_priors(U_D, U_A):
    print("\n--- Sweeping Prior over Threat Types ---")
    print(f"{'Spammer %':<10} | {'Phisher %':<10} | {'Value_D':<10}")
    print("-" * 35)
    for spam_prior in [0.1, 0.3, 0.5, 0.7, 0.9]:
        phish_prior = 1.0 - spam_prior
        prior = [spam_prior, phish_prior]
        res = solve_dobss(U_D, U_A, prior)
        print(f"{spam_prior*100:>8.0f}% | {phish_prior*100:>8.0f}% | {res.model_value_D:>10.4f}")

def sweep_costs(dataset):
    print("\n--- Sweeping Cost Ratios (c_miss : c_FP) ---")
    print("Re-evaluating utility tensors across different ratios... (this takes ~1 min)")
    
    with open('configs/datasets.yaml', 'r') as f: config = yaml.safe_load(f)
    with open('configs/costs.yaml', 'r') as f: costs_cfg = yaml.safe_load(f)
    with open('configs/types.yaml', 'r') as f: types_cfg = yaml.safe_load(f)
        
    prior = [t['prior'] for t in types_cfg['types']]
    
    data_path = config['datasets'][dataset]['processed_path'].replace('.parquet', '.csv')
    df = pd.read_csv(data_path)
    X_full = df['body'].fillna('').values
    y_full = df['label'].values
    
    splits_dir = config['splits']['split_indices_dir']
    idx_test = np.load(os.path.join(splits_dir, f'{dataset}_test.npy'))
    
    np.random.seed(42)
    sample_idx = np.random.choice(idx_test, size=min(400, len(idx_test)), replace=False)
    X_test, y_test = X_full[sample_idx], y_full[sample_idx]
    
    email_splits = {'ham': X_test[y_test == 0].tolist(), 'threat': X_test[y_test == 1].tolist()}
    
    defenders = load_all_defenders(dataset)
    attacks = [
        NoAttack(), ObfuscationAttack(strength=0.15), SynonymSwapAttack(strength=0.5),
        SynonymSwapAttack(strength=1.0), HamInjectionAttack(strength=1), HamInjectionAttack(strength=3)
    ]
    
    # Baseline c_FP is 100. We test ratios 0.1:1, 1:1, 10:1 (default), 50:1
    cost_ratios = [0.1, 1.0, 10.0, 50.0] 
    
    print(f"\n{'Ratio (Miss:FP)':<15} | {'Value_D':<10}")
    print("-" * 30)
    
    # Suppress print statements from the engine temporarily to keep output clean
    import builtins
    original_print = builtins.print
    
    for ratio in cost_ratios:
        c_fp = 100.0
        c_miss = c_fp * ratio
        
        test_costs_cfg = costs_cfg.copy()
        test_costs_cfg['c_FP'] = c_fp
        test_costs_cfg['c_miss'] = c_miss
        
        builtins.print = lambda *args, **kwargs: None
        U_D, U_A = estimate_payoffs(defenders, attacks, types_cfg['types'], test_costs_cfg, email_splits)
        builtins.print = original_print
        
        res = solve_dobss(U_D, U_A, prior)
        print(f"{ratio:>5.1f}:1{'':<8} | {res.model_value_D:>10.4f}")

def main():
    dataset = 'enron'
    print(f"=== Stage 8: Robustness & Ablation on {dataset.capitalize()} ===")
    
    U_D = np.load(f'results/tensors/U_D_{dataset}.npy')
    U_A = np.load(f'results/tensors/U_A_{dataset}.npy')
    
    # 1. Sweep Prior (Fast, just resolves MILP on existing tensors)
    sweep_priors(U_D, U_A)
    
    # 2. Sweep Costs (Requires re-evaluating tensors)
    sweep_costs(dataset)
    
    print("\nStage 8 completed successfully!")

if __name__ == "__main__":
    main()
