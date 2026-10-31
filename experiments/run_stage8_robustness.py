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
from src.data.loaders import get_active_dataset, load_processed_data
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

def sweep_costs(dataset, U_D, U_A):
    print("\n--- Sweeping Cost Ratios (c_miss : c_FP) ---")
    
    with open('configs/costs.yaml', 'r') as f: costs_cfg = yaml.safe_load(f)
    with open('configs/types.yaml', 'r') as f: types_cfg = yaml.safe_load(f)
        
    prior = [t['prior'] for t in types_cfg['types']]
    
    T, D, A = U_D.shape
    dr = np.zeros((T, D, A))
    fpr = np.zeros((T, D, A))
    
    pi = costs_cfg['pi_threat']
    c_miss_base = costs_cfg['c_miss']
    c_fp_base = costs_cfg['c_FP']
    
    # Analytically recover empirical DR and FPR from base payoff tensors
    for t_idx, type_cfg in enumerate(types_cfg['types']):
        for d_idx in range(D):
            for a_idx in range(A):
                if a_idx not in type_cfg['available_attacks']:
                    continue
                cost_idx = type_cfg['available_attacks'].index(a_idx)
                a_cost = type_cfg['attack_costs'][cost_idx]
                d_cost = costs_cfg['cost_defender'][d_idx]
                one_minus_dr = (U_A[t_idx, d_idx, a_idx] + a_cost) / type_cfg['gain']
                dr[t_idx, d_idx, a_idx] = 1.0 - one_minus_dr
                fpr[t_idx, d_idx, a_idx] = (-U_D[t_idx, d_idx, a_idx] - c_miss_base * one_minus_dr * pi - d_cost) / (c_fp_base * (1.0 - pi))
                
    cost_ratios = [0.1, 1.0, 10.0, 50.0]
    print(f"\n{'Ratio (Miss:FP)':<15} | {'Value_D':<10}")
    print("-" * 30)
    
    for ratio in cost_ratios:
        c_fp = 100.0
        c_miss = c_fp * ratio
        U_D_new = np.full((T, D, A), -9999.0)
        for t_idx, type_cfg in enumerate(types_cfg['types']):
            for d_idx in range(D):
                for a_idx in range(A):
                    if a_idx not in type_cfg['available_attacks']:
                        continue
                    d_cost = costs_cfg['cost_defender'][d_idx]
                    U_D_new[t_idx, d_idx, a_idx] = -c_miss * (1.0 - dr[t_idx, d_idx, a_idx]) * pi - c_fp * fpr[t_idx, d_idx, a_idx] * (1.0 - pi) - d_cost
        res = solve_dobss(U_D_new, U_A, prior)
        print(f"{ratio:>5.1f}:1{'':<8} | {res.model_value_D:>10.4f}")

def main():
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = get_active_dataset(config)
    print(f"=== Stage 8: Robustness & Ablation on {dataset.capitalize()} ===")
    
    U_D = np.load(f'results/tensors/U_D_{dataset}.npy')
    U_A = np.load(f'results/tensors/U_A_{dataset}.npy')
    
    # 1. Sweep Prior (Fast, resolves MILP on existing tensors)
    sweep_priors(U_D, U_A)
    
    # 2. Sweep Costs (Fast analytic recalculation of utility tensors)
    sweep_costs(dataset, U_D, U_A)
    
    print("\nStage 8 completed successfully!")

if __name__ == "__main__":
    main()
