"""
End-to-End Pipeline Runner for Game-Theoretic Email Threat Detection.
Executes stages sequentially for the configured dataset (default: phishing).
"""
import sys
import os
import argparse
import subprocess
import time

STAGES = [
    (1, "Data Preparation & Splits", "experiments/run_stage1_data.py"),
    (2, "Train Defenders & Calibrate Thresholds", "experiments/run_stage2_defenders.py"),
    (3, "Evaluate Adversarial Attacks", "experiments/run_stage3_attacks.py"),
    (4, "Estimate Empirical Payoff Tensors", "experiments/run_stage4_payoffs.py"),
    (5, "Game Solvers (Minimax, Nash, SSE, DOBSS)", "experiments/run_stage5_solvers.py"),
    (6, "Realized Utility & Reality Gap", "experiments/run_stage6_realized.py"),
    (8, "Robustness & Ablation Sweeps", "experiments/run_stage8_robustness.py"),
    (9, "Budgeted Inspection Game (Tiered Triage)", "experiments/run_stage9_extensions.py"),
]

def run_stage(num, title, script_path, dataset):
    print("\n" + "=" * 70)
    print(f"  RUNNING STAGE {num}: {title.upper()} [{dataset.upper()}]")
    print("=" * 70)
    cmd = [sys.executable, script_path, "--dataset", dataset]
    start = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - start
    if ret.returncode != 0:
        print(f"\n[ERROR] Stage {num} failed with return code {ret.returncode} ({elapsed:.1f}s)")
        return False
    print(f"[SUCCESS] Stage {num} completed in {elapsed:.1f}s")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the Game Theory Pipeline")
    parser.add_argument("--dataset", default="phishing", choices=["phishing", "enron"],
                        help="Dataset to run pipeline on (default: phishing)")
    parser.add_argument("--stages", default="all",
                        help="Comma-separated stage numbers to run (e.g. '2,3,4,5,6,8,9' or 'all')")
    args = parser.parse_args()

    selected_stages = []
    if args.stages.strip().lower() == "all":
        selected_stages = [num for num, _, _ in STAGES]
    else:
        selected_stages = [int(s.strip()) for s in args.stages.split(",") if s.strip()]

    print("=" * 70)
    print(f"GAME THEORY PIPELINE EXECUTION")
    print(f"Target Dataset : {args.dataset.upper()}")
    print(f"Stages to Run  : {selected_stages}")
    print("=" * 70)

    for num, title, script_path in STAGES:
        if num in selected_stages:
            success = run_stage(num, title, script_path, args.dataset)
            if not success:
                print("\nPipeline execution halted due to failure.")
                sys.exit(1)

    print("\n" + "=" * 70)
    print(f"ALL SELECTED STAGES COMPLETED SUCCESSFULLY FOR {args.dataset.upper()}!")
    print("=" * 70)

if __name__ == "__main__":
    main()
