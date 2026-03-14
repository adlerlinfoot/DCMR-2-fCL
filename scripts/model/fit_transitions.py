import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

DATA_FINAL = PROJECT_ROOT / "data" / "final" / "CL_final.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TENSOR_NPY = OUTPUTS_DIR / "transition_matrix.npy"
SUMMARY_JSON = OUTPUTS_DIR / "transition_matrix_summary.json"

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

# Expand from 4 → 5 regimes
N_STATES = 5
SMOOTHING = 1.0   # Laplace smoothing (add-1)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def fit_transition_tensor():
    if TENSOR_NPY.exists():
        raise RuntimeError(
            f"{TENSOR_NPY} already exists. "
            "Refusing to overwrite a frozen model artifact."
        )

    if not DATA_FINAL.exists():
        raise FileNotFoundError(f"Final dataset not found: {DATA_FINAL}")

    # Load final dataset
    df = pd.read_parquet(DATA_FINAL)

    required = {"state", "state_t_minus_1", "state_t_minus_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in final dataset: {missing}")

    # Extract sequences
    s_t   = df["state"].astype(int).to_numpy()
    s_tm1 = df["state_t_minus_1"].astype(int).to_numpy()
    s_tm2 = df["state_t_minus_2"].astype(int).to_numpy()

    # Initialize count tensor
    counts = np.zeros((N_STATES, N_STATES, N_STATES), dtype=float)

    # Count transitions
    for i, j, k in zip(s_tm2, s_tm1, s_t):
        counts[i, j, k] += 1

    # Apply Laplace smoothing
    counts += SMOOTHING

    # Normalize along last axis to get probabilities
    T = counts / counts.sum(axis=2, keepdims=True)

    # Save tensor
    np.save(TENSOR_NPY, T)

    # Build summary for quick inspection
    summary = {}
    for i in range(N_STATES):
        for j in range(N_STATES):
            row = T[i, j].tolist()
            summary[f"{i}->{j}"] = {
                "to_state_probs": row,
                "max_prob_state": int(np.argmax(row)),
                "max_prob_value": float(np.max(row)),
                "min_prob_state": int(np.argmin(row)),
                "min_prob_value": float(np.min(row)),
            }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Transition tensor saved to: {TENSOR_NPY}")
    print(f"Summary saved to: {SUMMARY_JSON}")

if __name__ == "__main__":
    fit_transition_tensor()