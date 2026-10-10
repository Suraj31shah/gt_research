import os
import yaml
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.budgeted_inspection import solve as solve_budgeted

def main():
    dataset = 'enron'
    print(f"=== Stage 9: Budgeted Inspection Game (Tiered Triage) on {dataset.capitalize()} ===")
    
    with open('configs/types.yaml', 'r') as f: types_cfg = yaml.safe_load(f)
    with open('configs/costs.yaml', 'r') as f: costs_cfg = yaml.safe_load(f)
        
    prior = [t['prior'] for t in types_cfg['types']]
    U_D = np.load(f'results/tensors/U_D_{dataset}.npy')
    U_A = np.load(f'results/tensors/U_A_{dataset}.npy')
    
    budgets = [0.01, 0.03, 0.05, 0.08, 0.15]
    
    defender_costs = costs_cfg['cost_defender']
    
    print(f"{'Budget':<10} | {'Value_D':<10} | {'Budget Used':<15} | {'High-Cost (SVM) %'}")
    print("-" * 65)
    
    for b in budgets:
        res = solve_budgeted(U_D, U_A, prior, budget=b, defender_costs=defender_costs)
        
        # Strategies 6, 7, 8 are the SVM models which have a cost of 0.1
        svm_usage = np.sum(res.defender_strategy[6:]) * 100
        
        print(f"{b:<10.3f} | {res.model_value_D:>10.4f} | {res.metadata['budget_used']:>15.4f} | {svm_usage:>14.1f}%")

    print("\nStage 9 completed successfully!")

if __name__ == "__main__":
    main()
