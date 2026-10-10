import numpy as np
from scipy.optimize import linprog
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
    
    best_x = None
    best_a_tuple = None
    best_val_D = -np.inf
    
    for a_tuple in attacker_pure_strategies:
        c = -np.array([sum(prior[t] * U_D[t, d, a_tuple[t]] for t in range(T)) for d in range(D)])
        
        A_ub = []
        b_ub = []
        
        for t in range(T):
            a_best = a_tuple[t]
            for a_other in avail[t]:
                if a_other == a_best: continue
                row = -(U_A[t, :, a_best] - U_A[t, :, a_other])
                A_ub.append(row)
                b_ub.append(0.0)
                
        A_eq = np.ones((1, D))
        b_eq = [1.0]
        bounds = [(0, 1) for _ in range(D)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            val_D = -res.fun
            if val_D > best_val_D:
                best_val_D = val_D
                best_x = res.x
                best_a_tuple = a_tuple
                
    attacker_responses = {}
    val_A = {}
    for t in range(T):
        resp = np.zeros(A)
        resp[best_a_tuple[t]] = 1.0
        attacker_responses[t] = resp
        val_A[t] = best_x @ U_A[t, :, best_a_tuple[t]]
        
    return SolverResult(
        defender_strategy=best_x,
        attacker_responses=attacker_responses,
        model_value_D=best_val_D,
        model_value_A=val_A,
        solve_time_s=time.time() - start,
        metadata={"lps_solved": len(attacker_pure_strategies)}
    )
