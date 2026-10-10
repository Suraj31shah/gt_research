import numpy as np
from scipy.optimize import linprog
import time
from .interface import SolverResult

def solve(U_D, U_A, prior, **params) -> SolverResult:
    start = time.time()
    T, D, A = U_D.shape
    
    c = np.zeros(D + T)
    for t in range(T):
        c[D + t] = -prior[t]
        
    A_ub = []
    b_ub = []
    
    for t in range(T):
        for a in range(A):
            if U_D[t, 0, a] == -9999: continue
            row = np.zeros(D + T)
            row[:D] = -U_D[t, :, a]
            row[D + t] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_eq = np.zeros((1, D + T))
    A_eq[0, :D] = 1.0
    b_eq = [1.0]
    
    bounds = [(0, 1) for _ in range(D)] + [(None, None) for _ in range(T)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    x = res.x[:D]
    val_D = -res.fun
    
    attacker_responses = {}
    val_A = {}
    for t in range(T):
        expected_u_a = x @ U_A[t, :, :]
        best_a = np.argmax(expected_u_a)
        resp = np.zeros(A)
        resp[best_a] = 1.0
        attacker_responses[t] = resp
        val_A[t] = expected_u_a[best_a]
        
    true_val_D = sum(prior[t] * (x @ U_D[t, :, np.argmax(x @ U_A[t, :, :])]) for t in range(T))
        
    return SolverResult(
        defender_strategy=x,
        attacker_responses=attacker_responses,
        model_value_D=val_D,
        model_value_A=val_A,
        solve_time_s=time.time() - start,
        metadata={"true_val_D_against_BR": true_val_D}
    )
