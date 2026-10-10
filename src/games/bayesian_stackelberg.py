import numpy as np
import time
import pulp
from .interface import SolverResult

def solve(U_D, U_A, prior, **params) -> SolverResult:
    start = time.time()
    T, D, A = U_D.shape
    
    M = 100000 
    prob = pulp.LpProblem("DOBSS", pulp.LpMaximize)
    
    x = [pulp.LpVariable(f"x_{d}", 0, 1, pulp.LpContinuous) for d in range(D)]
    q = [[pulp.LpVariable(f"q_{t}_{a}", 0, 1, pulp.LpBinary) for a in range(A)] for t in range(T)]
    v = [pulp.LpVariable(f"v_{t}", None, None, pulp.LpContinuous) for t in range(T)]
    
    z = [[[pulp.LpVariable(f"z_{t}_{d}_{a}", 0, 1, pulp.LpContinuous) for a in range(A)] for d in range(D)] for t in range(T)]
    
    obj = pulp.lpSum([
        prior[t] * z[t][d][a] * U_D[t, d, a]
        for t in range(T) for d in range(D) for a in range(A) if U_D[t, d, a] != -9999
    ])
    
    eps = 1e-6
    obj += eps * pulp.lpSum([
        prior[t] * z[t][d][a] * U_A[t, d, a]
        for t in range(T) for d in range(D) for a in range(A) if U_A[t, d, a] != -9999
    ])
    
    prob += obj
    
    prob += pulp.lpSum(x) == 1
    
    for t in range(T):
        valid_attacks = [a for a in range(A) if U_D[t, 0, a] != -9999]
        
        prob += pulp.lpSum([q[t][a] for a in valid_attacks]) == 1
        
        for a in range(A):
            if a not in valid_attacks:
                prob += q[t][a] == 0
                
        for a in valid_attacks:
            expected_u_a = pulp.lpSum([x[d] * U_A[t, d, a] for d in range(D)])
            prob += v[t] - expected_u_a >= 0
            prob += v[t] - expected_u_a <= M * (1 - q[t][a])
            
            for d in range(D):
                prob += z[t][d][a] <= x[d]
                prob += z[t][d][a] <= q[t][a]
                prob += z[t][d][a] >= x[d] + q[t][a] - 1
                prob += z[t][d][a] >= 0
                
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    x_opt = np.array([pulp.value(var) for var in x])
    
    attacker_responses = {}
    val_A = {}
    for t in range(T):
        resp = np.array([pulp.value(q[t][a]) for a in range(A)])
        attacker_responses[t] = resp
        val_A[t] = pulp.value(v[t])
        
    val_D = pulp.value(prob.objective) 
    
    return SolverResult(
        defender_strategy=x_opt,
        attacker_responses=attacker_responses,
        model_value_D=val_D,
        model_value_A=val_A,
        solve_time_s=time.time() - start,
        metadata={"status": pulp.LpStatus[prob.status]}
    )
