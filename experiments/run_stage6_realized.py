import os
import yaml
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.minimax_lp import solve as solve_minimax
from src.games.nash import solve as solve_nash
from src.games.bayesian_stackelberg import solve as solve_dobss
from src.evaluation.realized_utility import evaluate_realized_utility
from src.attacks.base import NoAttack
from src.attacks.obfuscation import ObfuscationAttack
from src.attacks.synonym_swap import SynonymSwapAttack
from src.attacks.ham_injection import HamInjectionAttack
from experiments.run_stage4_payoffs import load_all_defenders

def main():
    dataset = 'enron'
    print(f"=== Stage 6: Realized Utility vs Model Value on {dataset.capitalize()} ===")
    
    with open('configs/datasets.yaml', 'r') as f: config = yaml.safe_load(f)
    with open('configs/costs.yaml', 'r') as f: costs_cfg = yaml.safe_load(f)
    with open('configs/types.yaml', 'r') as f: types_cfg = yaml.safe_load(f)
        
    prior = [t['prior'] for t in types_cfg['types']]
    U_D = np.load(f'results/tensors/U_D_{dataset}.npy')
    U_A = np.load(f'results/tensors/U_A_{dataset}.npy')
    
    solvers = {
        "Minimax LP": solve_minimax,
        "Nash (General Sum)": solve_nash,
        "Bayesian Stackelberg": solve_dobss
    }
    
    data_path = config['datasets'][dataset]['processed_path'].replace('.parquet', '.csv')
    df = pd.read_csv(data_path)
    X_full = df['body'].fillna('').values
    y_full = df['label'].values
    
    splits_dir = config['splits']['split_indices_dir']
    idx_test = np.load(os.path.join(splits_dir, f'{dataset}_test.npy'))
    
    np.random.seed(42)
    sample_idx = np.random.choice(idx_test, size=min(1000, len(idx_test)), replace=False)
    X_test = X_full[sample_idx]
    y_test = y_full[sample_idx]
    
    email_splits = {
        'ham': X_test[y_test == 0].tolist(),
        'threat': X_test[y_test == 1].tolist()
    }
    
    defenders = load_all_defenders(dataset)
    attacks = [
        NoAttack(), ObfuscationAttack(strength=0.15), SynonymSwapAttack(strength=0.5),
        SynonymSwapAttack(strength=1.0), HamInjectionAttack(strength=1), HamInjectionAttack(strength=3)
    ]
    
    print(f"{'Game Model':<22} | {'Model Val':<9} | {'Realized Val (95% CI)':<25} | {'Gap'}")
    print("-" * 70)
    
    for name, solver_fn in solvers.items():
        res = solver_fn(U_D, U_A, prior)
        model_val = res.model_value_D
        
        realized_mean, (ci_low, ci_high) = evaluate_realized_utility(
            res, email_splits, attacks, defenders, types_cfg['types'], costs_cfg, n_trials=1000
        )
        
        gap = abs(model_val - realized_mean)
        
        print(f"{name:<22} | {model_val:>9.4f} | {realized_mean:>7.4f} [{ci_low:>7.4f}, {ci_high:>7.4f}] | {gap:.4f}")
        
    print("\nStage 6 completed successfully!")

if __name__ == "__main__":
    main()
