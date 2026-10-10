import os
import numpy as np
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.minimax_lp import solve as solve_minimax
from src.games.nash import solve as solve_nash
from src.games.strong_stackelberg import solve as solve_sse
from src.games.bayesian_stackelberg import solve as solve_dobss

def main():
    dataset = 'enron'
    print(f"=== Stage 5: Solving Games on {dataset.capitalize()} ===")
    
    with open('configs/types.yaml', 'r') as f:
        types_cfg = yaml.safe_load(f)
        
    prior = [t['prior'] for t in types_cfg['types']]
    
    U_D = np.load(f'results/tensors/U_D_{dataset}.npy')
    U_A = np.load(f'results/tensors/U_A_{dataset}.npy')
    
    solvers = {
        "Minimax LP": solve_minimax,
        "Nash (General Sum)": solve_nash,
        "Strong Stackelberg (SSE)": solve_sse,
        "Bayesian Stackelberg (DOBSS)": solve_dobss
    }
    
    print(f"{'Solver':<30} | {'Value_D':<10} | {'Solve Time (s)'}")
    print("-" * 60)
    
    for name, solver_fn in solvers.items():
        try:
            res = solver_fn(U_D, U_A, prior)
            print(f"{name:<30} | {res.model_value_D:>10.4f} | {res.solve_time_s:.4f}")
        except Exception as e:
            print(f"{name:<30} | ERROR: {str(e)}")
        
    print("\nStage 5 completed successfully!")

if __name__ == "__main__":
    main()
