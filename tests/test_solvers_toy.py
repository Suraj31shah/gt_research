import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.minimax_lp import solve as solve_minimax
from src.games.nash import solve as solve_nash
from src.games.strong_stackelberg import solve as solve_sse
from src.games.bayesian_stackelberg import solve as solve_dobss

def test_matching_pennies():
    U_D = np.array([[[1, -1], [-1, 1]]])
    U_A = np.array([[[-1, 1], [1, -1]]])
    prior = [1.0]
    
    res = solve_minimax(U_D, U_A, prior)
    assert np.allclose(res.defender_strategy, [0.5, 0.5])
    assert np.isclose(res.model_value_D, 0.0)

def test_paruchuri():
    U_D = np.array([
        [[5, -1],
         [-1, 5]],
        [[5, -1],
         [-1, 5]]
    ])
    U_A = np.array([
        [[-1, 5],
         [5, -1]],
        [[5, -1],
         [-1, 5]]
    ])
    prior = [0.5, 0.5]
    
    res = solve_dobss(U_D, U_A, prior)
    assert res.metadata["status"] == "Optimal"
    
if __name__ == "__main__":
    test_matching_pennies()
    test_paruchuri()
    print("Toy tests passed!")
