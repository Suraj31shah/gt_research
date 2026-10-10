from dataclasses import dataclass
import numpy as np

@dataclass
class SolverResult:
    defender_strategy: np.ndarray
    attacker_responses: dict
    model_value_D: float
    model_value_A: dict
    solve_time_s: float
    metadata: dict
