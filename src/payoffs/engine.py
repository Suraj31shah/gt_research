import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.attacks.similarity import similarity_check

def measure_dr_and_fpr(defender, threat_emails, ham_emails, attack_fn=None):
    if attack_fn:
        attacked = []
        for e in threat_emails:
            a = attack_fn(e)
            if similarity_check(e, a):
                attacked.append(a)
            else:
                attacked.append(e) 
    else:
        attacked = threat_emails
        
    preds_threat = defender.predict(attacked)
    dr = np.mean(preds_threat == 1) if len(preds_threat) > 0 else 0
    
    preds_ham = defender.predict(ham_emails)
    fpr = np.mean(preds_ham == 1) if len(preds_ham) > 0 else 0
    
    return dr, fpr

def estimate_payoffs(defenders, attacks, types_cfg, costs_cfg, email_splits):
    T = len(types_cfg)
    D = len(defenders)
    A = len(attacks)
    
    U_D = np.zeros((T, D, A))
    U_A = np.zeros((T, D, A))
    
    ham_emails = email_splits['ham']
    
    for t_idx, type_cfg in enumerate(types_cfg):
        threat_emails = email_splits['threat']
        
        for d_idx, defender in enumerate(defenders):
            for a_idx, attack in enumerate(attacks):
                if a_idx not in type_cfg['available_attacks']:
                    U_D[t_idx, d_idx, a_idx] = -9999
                    U_A[t_idx, d_idx, a_idx] = -9999
                    continue
                
                print(f"Type {t_idx} | Defender {d_idx} | Attack {a_idx}")
                dr, fpr = measure_dr_and_fpr(defender, threat_emails, ham_emails, attack.apply if a_idx > 0 else None)
                
                cost_idx = type_cfg['available_attacks'].index(a_idx)
                a_cost = type_cfg['attack_costs'][cost_idx]
                d_cost = costs_cfg['cost_defender'][d_idx]
                
                pi = costs_cfg['pi_threat']
                c_miss = costs_cfg['c_miss']
                c_fp = costs_cfg['c_FP']
                
                U_D[t_idx, d_idx, a_idx] = -c_miss * (1 - dr) * pi - c_fp * fpr * (1 - pi) - d_cost
                U_A[t_idx, d_idx, a_idx] = type_cfg['gain'] * (1 - dr) - a_cost
                
    return U_D, U_A
