import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attacks.similarity import similarity_check

def evaluate_realized_utility(solver_result, email_splits, attacks, defenders, types_cfg, costs_cfg, n_trials=5000):
    """
    Evaluates the REALIZED utility by simulating interactions on held-out emails.
    """
    true_prior = [t['prior'] for t in types_cfg]
    
    utilities_D = []
    
    ham_emails = email_splits['ham']
    threat_emails = email_splits['threat']
    
    pi = costs_cfg['pi_threat']
    c_miss = costs_cfg['c_miss']
    c_fp = costs_cfg['c_FP']
    
    for _ in range(n_trials):
        # Sample interaction type: threat or ham
        is_threat = random.random() < pi
        
        probs = np.array(solver_result.defender_strategy)
        probs = np.clip(probs, 0, None)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)
            
        d_idx = np.random.choice(len(defenders), p=probs)
        defender = defenders[d_idx]
        d_cost = costs_cfg['cost_defender'][d_idx]
        
        if not is_threat:
            email = random.choice(ham_emails)
            pred = defender.predict([email])[0]
            if pred == 1: 
                u_D = -c_fp - d_cost
            else:
                u_D = 0.0 - d_cost
        else:
            t_idx = np.random.choice(len(types_cfg), p=true_prior)
            
            a_probs = np.array(solver_result.attacker_responses[t_idx])
            a_probs = np.clip(a_probs, 0, None)
            if a_probs.sum() > 0:
                a_probs = a_probs / a_probs.sum()
            else:
                a_probs = np.ones(len(a_probs)) / len(a_probs)
                
            a_idx = np.random.choice(len(attacks), p=a_probs)
            attack = attacks[a_idx]
            
            email = random.choice(threat_emails)
            
            if a_idx > 0:
                attacked_email = attack.apply(email)
                if not similarity_check(email, attacked_email):
                    attacked_email = email
            else:
                attacked_email = email
                
            pred = defender.predict([attacked_email])[0]
            if pred == 0: 
                u_D = -c_miss - d_cost
            else:
                u_D = 0.0 - d_cost
                
        utilities_D.append(u_D)
        
    mean_u_D = np.mean(utilities_D)
    
    bootstraps = np.random.choice(utilities_D, size=(1000, len(utilities_D)), replace=True)
    means = np.mean(bootstraps, axis=1)
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    
    return mean_u_D, (lower, upper)
