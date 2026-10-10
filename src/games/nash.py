import numpy as np
import nashpy as nash
import time
from .interface import SolverResult
import itertools

def solve(U_D, U_A, prior, **params) -> SolverResult:
    start = time.time()
    T, D, A = U_D.shape
    
    avail = []
    for t in range(T):
        valid = [a for a in range(A) if U_D[t, 0, a] != -9999]
        avail.append(valid)
        
    attacker_pure_strategies = list(itertools.product(*avail))
    num_A_strats = len(attacker_pure_strategies)
    
    mat_D = np.zeros((D, num_A_strats))
    mat_A = np.zeros((D, num_A_strats)) 
    
    for j, a_tuple in enumerate(attacker_pure_strategies):
        for d in range(D):
            u_d = sum(prior[t] * U_D[t, d, a_tuple[t]] for t in range(T))
            u_a = sum(prior[t] * U_A[t, d, a_tuple[t]] for t in range(T))
            mat_D[d, j] = u_d
            mat_A[d, j] = u_a
            
    game = nash.Game(mat_D, mat_A)
    
    best_eq = None
    best_val = -np.inf
    num_eqs = 0
    
    # Support enumeration can take hours if we exhaustively list all equilibria for a 9x16 matrix. 
    # Instead, we just grab the first one we find.
    try:
        for eq in game.support_enumeration():
            num_eqs += 1
            x, y = eq
            val = x @ mat_D @ y
            if val > best_val:
                best_val = val
                best_eq = (x, y)
            break # Just take the first one to save time
    except Exception:
        pass
        
    if best_eq is None:
        # Fallback to vertex enumeration (also grabbing first)
        for eq in game.vertex_enumeration():
            num_eqs += 1
            x, y = eq
            val = x @ mat_D @ y
            if val > best_val:
                best_val = val
                best_eq = (x, y)
            break
            
    x_opt, y_opt = best_eq
    
    attacker_responses = {t: np.zeros(A) for t in range(T)}
    for j, a_tuple in enumerate(attacker_pure_strategies):
        prob = y_opt[j]
        if prob > 0:
            for t in range(T):
                attacker_responses[t][a_tuple[t]] += prob
                
    val_A = {}
    for t in range(T):
        expected_u_a = x_opt @ U_A[t, :, :]
        val_A[t] = np.sum(expected_u_a * attacker_responses[t])
        
    return SolverResult(
        defender_strategy=x_opt,
        attacker_responses=attacker_responses,
        model_value_D=best_val,
        model_value_A=val_A,
        solve_time_s=time.time() - start,
        metadata={"num_equilibria": num_eqs}
    )
