import json
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.model.dcmr2 import DCMR2

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

DATA_FINAL = PROJECT_ROOT / "data" / "final" / "CL_final.parquet"
CONVICTION_PARQUET = PROJECT_ROOT / "outputs" / "conviction_neutral.parquet"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFUSION_CSV = OUTPUTS_DIR / "dcmr2_confusion_matrix.csv"
SUMMARY_JSON = OUTPUTS_DIR / "dcmr2_loglik_summary.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Direction map: +1 = up, -1 = down, 0 = chop
DIRECTION = {
    0: +1,   # low vol up
    1: -1,   # low vol down
    2: +1,   # high vol up
    3: -1,   # high vol down
    4:  0,   # chop
}

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def evaluate_dcmr2():
    if not DATA_FINAL.exists():
        raise FileNotFoundError(f"Final dataset not found: {DATA_FINAL}")
    if not CONVICTION_PARQUET.exists():
        raise FileNotFoundError(f"Conviction file not found: {CONVICTION_PARQUET}")

    # Load main dataset
    df = pd.read_parquet(DATA_FINAL)

    # Load conviction scores
    df_conv = pd.read_parquet(CONVICTION_PARQUET)

    # Merge conviction into main df
    df = df.merge(df_conv[["conviction"]], left_index=True, right_index=True)

    required = {"state", "state_t_minus_1", "state_t_minus_2", "age", "conviction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    engine = DCMR2()
    n_states = engine.n_states

    confusion = np.zeros((n_states, n_states), dtype=int)

    loglik = 0.0
    count = 0
    true_probs = []

    # For raw next-state prediction
    pred_next = []
    true_regimes = df["state"].to_numpy()

    # For flip-only tensor baseline
    flip_destinations = []

    # Store full next-step probability vectors for direction/regime scoring
    probs_next_list = []

    # Iterate through all rows
    for _, row in df.iterrows():
        s_t = int(row["state"])
        s_tm1 = int(row["state_t_minus_1"])
        s_tm2 = int(row["state_t_minus_2"])
        age = int(row["age"])
        conviction_t = float(row["conviction"])

        # Conviction-aware transition probabilities for S_t
        probs_t = engine.next_probs(s_tm2, s_tm1, age, conviction_t)

        # Log-likelihood of true state at t
        p_true = probs_t[s_t]
        true_probs.append(float(p_true))
        loglik += np.log(max(p_true, 1e-12))
        count += 1

        # Current-regime prediction and confusion
        pred_c = int(np.argmax(probs_t))
        confusion[s_t, pred_c] += 1

        # ---------- Raw next-state prediction ----------
        # Predict S_{t+1} using (S_{t-1}, S_t) as the new (t-2, t-1)
        next_age = max(1, age + 1)
        probs_tplus1 = engine.next_probs(s_tm1, s_t, next_age, conviction_t)
        pred_next.append(int(np.argmax(probs_tplus1)))

        # Store full next-step probabilities for later scoring
        probs_next_list.append(probs_tplus1.copy())

        # ---------- Flip-destination baseline (from transition tensor only) ----------
        # For a flip at t+1, the conditioning pair is (s_tm1, s_t)
        # Use the raw tensor T (no age/conviction) and pick the most likely non-self state.
        T_slice = engine.T[s_tm1, s_t, :].astype(float).copy()
        T_slice[s_t] = -1.0  # exclude self
        flip_dest = int(np.argmax(T_slice))
        flip_destinations.append(flip_dest)

    # Normalize confusion matrix rows
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)
    pd.DataFrame(confusion_norm).to_csv(CONFUSION_CSV, index=False)

    # ---------------- Raw next-state accuracy (all periods) ----------------
    pred_next = np.array(pred_next)
    if len(pred_next) > 1:
        next_state_acc = (pred_next[:-1] == true_regimes[1:]).mean()
    else:
        next_state_acc = float("nan")

    # ---------------- Flip mask (true regime changes) ----------------
    flip_mask = true_regimes[1:] != true_regimes[:-1]

    # ---------------- Tensor flip-only baseline accuracy ----------------
    flip_destinations = np.array(flip_destinations)
    if flip_mask.sum() > 0:
        tensor_flip_acc = (
            flip_destinations[:-1][flip_mask] == true_regimes[1:][flip_mask]
        ).mean()
    else:
        tensor_flip_acc = float("nan")

    # ---------------- Regime-probability and direction-probability on flips ----------------
    probs_next = np.vstack(probs_next_list)  # shape: (T, n_states)

    regime_prob_scores = []
    direction_prob_scores = []

    # t indexes the forecast made at time t for state at t+1
    for t in range(len(true_regimes) - 1):
        if not flip_mask[t]:
            continue  # only score when true regime actually changes

        # Predicted next regime (argmax of probs at t)
        pred_state = int(pred_next[t])
        pred_dir = DIRECTION[pred_state]

        # Probability of the predicted regime
        regime_prob = float(probs_next[t, pred_state])
        regime_prob_scores.append(regime_prob)

        # Probability mass on all regimes with the same direction as the predicted regime
        dir_prob = 0.0
        for r in range(n_states):
            if DIRECTION[r] == pred_dir:
                dir_prob += probs_next[t, r]
        direction_prob_scores.append(float(dir_prob))

    if len(regime_prob_scores) > 0:
        next_regime_accuracy = float(np.mean(regime_prob_scores))
    else:
        next_regime_accuracy = float("nan")

    if len(direction_prob_scores) > 0:
        next_direction_accuracy = float(np.mean(direction_prob_scores))
    else:
        next_direction_accuracy = float("nan")

    # Summary statistics
    summary = {
        "avg_log_likelihood": float(loglik / count),
        "avg_true_prob": float(np.mean(true_probs)),
        "min_true_prob": float(np.min(true_probs)),
        "max_true_prob": float(np.max(true_probs)),
        "n_observations": int(count),
        "next_state_accuracy": float(next_state_acc),
        "tensor_flip_accuracy": float(tensor_flip_acc),
        "next_regime_accuracy": float(next_regime_accuracy),
        "next_direction_accuracy": float(next_direction_accuracy),
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved confusion matrix to: {CONFUSION_CSV}")
    print(f"Saved summary to: {SUMMARY_JSON}")
    print(f"Next-state accuracy (all periods): {next_state_acc:.4f}")
    print(f"Tensor flip-only baseline accuracy: {tensor_flip_acc:.4f}")
    print(f"Flip-only next-regime probability (predicted bucket): {next_regime_accuracy:.4f}")
    print(f"Flip-only next-direction probability: {next_direction_accuracy:.4f}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_dcmr2()