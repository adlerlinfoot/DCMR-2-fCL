import numpy as np
import pandas as pd
from pathlib import Path
from scripts.model.dcmr2 import DCMR2

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

DATA_FINAL = PROJECT_ROOT / "data" / "final" / "CL_final.parquet"
# IMPORTANT: this is a separate conviction file, does NOT overwrite outputs/conviction.parquet
CONVICTION_PARQUET = PROJECT_ROOT / "outputs" / "conviction.parquet"


# ------------------------------------------------------------
# Step 1: Compute run-lengths
# ------------------------------------------------------------
def compute_run_lengths(states):
    # infer number of states dynamically
    n_states = int(states.max()) + 1
    run_lengths = {k: [] for k in range(n_states)}

    current_state = int(states[0])
    current_len = 1

    for s in states[1:]:
        s = int(s)
        if s == current_state:
            current_len += 1
        else:
            run_lengths[current_state].append(current_len)
            current_state = s
            current_len = 1

    run_lengths[current_state].append(current_len)
    return run_lengths


# ------------------------------------------------------------
# Step 2: Simulate DCMR-2 forward
# ------------------------------------------------------------
def simulate_dcmr2(df, engine):
    preds = []

    s_tm2 = int(df.iloc[0]["state_t_minus_2"])
    s_tm1 = int(df.iloc[0]["state_t_minus_1"])
    age = int(df.iloc[0]["age"])

    for t in range(len(df)):
        conviction = float(df.iloc[t]["conviction"])
        s_t = engine.step(s_tm2, s_tm1, age, conviction)
        preds.append(s_t)

        prev_state = s_tm1
        s_tm2, s_tm1 = s_tm1, s_t
        age = age + 1 if s_t == prev_state else 1

    return np.array(preds)


# ------------------------------------------------------------
# Step 3: Distance metric between empirical and predicted persistence
# ------------------------------------------------------------
def persistence_distance(emp, pred):
    # use whatever states appear in emp
    states = sorted(emp.keys())
    dist = 0.0
    for k in states:
        e = np.array(emp[k])
        p = np.array(pred.get(k, []))

        if len(p) == 0:
            dist += 1e6
            continue

        dist += abs(e.mean() - p.mean())
        dist += abs(np.median(e) - np.median(p))
        dist += abs(np.percentile(e, 90) - np.percentile(p, 90))

    return dist


# ------------------------------------------------------------
# Helper: build engine with full hyperparameter set
# ------------------------------------------------------------
def build_engine(A, alpha, baseline_scale=1.0, hazard_mult=None, conv_smooth=0.0):
    engine = DCMR2(
        conviction_alpha=alpha,
        conviction_age_full=A,
        baseline_scale=baseline_scale,
        hazard_mult=hazard_mult,
        conv_smooth=conv_smooth,
    )
    return engine


def evaluate_params(df, emp_run_lengths, A, alpha,
                    baseline_scale=1.0, hazard_mult=None, conv_smooth=0.0):
    # if hazard_mult not provided, default to all-ones of correct length
    if hazard_mult is None:
        n_states = int(df["state"].max()) + 1
        hazard_mult = [1.0] * n_states

    engine = build_engine(A, alpha, baseline_scale, hazard_mult, conv_smooth)
    preds = simulate_dcmr2(df, engine)
    pred_run_lengths = compute_run_lengths(preds)
    score = persistence_distance(emp_run_lengths, pred_run_lengths)
    return score


# ------------------------------------------------------------
# Layer 1: Coarse 2D sweep over A and alpha
# ------------------------------------------------------------
def coarse_tune_A_alpha(df, emp_run_lengths):
    A_values = [8, 12, 16, 20, 24]
    alpha_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    best_score = float("inf")
    best_params = None

    for A in A_values:
        for alpha in alpha_values:
            score = evaluate_params(df, emp_run_lengths, A, alpha)
            print(f"[COARSE] A={A}, alpha={alpha:.3f}, score={score:.4f}")
            if score < best_score:
                best_score = score
                best_params = (A, alpha)

    return best_params, best_score


# ------------------------------------------------------------
# Layer 2: Fine 1D sweep in alpha (0.01 steps) around coarse best
# ------------------------------------------------------------
def refine_alpha(df, emp_run_lengths, A_fixed, alpha_center, radius=0.05, step=0.01):
    alpha_min = max(0.01, alpha_center - radius)
    alpha_max = alpha_center + radius

    alphas = np.arange(alpha_min, alpha_max + 1e-9, step)

    best_score = float("inf")
    best_alpha = None

    for alpha in alphas:
        score = evaluate_params(df, emp_run_lengths, A_fixed, alpha)
        print(f"[ALPHA-REFINE] A={A_fixed}, alpha={alpha:.3f}, score={score:.4f}")
        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score


# ------------------------------------------------------------
# Layer 3: Local refinement in A around best alpha
# ------------------------------------------------------------
def refine_A(df, emp_run_lengths, A_center, alpha_fixed, deltas=(-4, -2, 0, 2, 4)):
    A_candidates = [A_center + d for d in deltas if A_center + d > 0]

    best_score = float("inf")
    best_A = None

    for A in A_candidates:
        score = evaluate_params(df, emp_run_lengths, A, alpha_fixed)
        print(f"[A-REFINE] A={A}, alpha={alpha_fixed:.3f}, score={score:.4f}")
        if score < best_score:
            best_score = score
            best_A = A

    return best_A, best_score


# ------------------------------------------------------------
# Layer 4: baseline_scale tuning (1D)
# ------------------------------------------------------------
def tune_baseline_scale(df, emp_run_lengths, A, alpha):
    scales = [0.7, 0.85, 1.0, 1.15, 1.3]

    best_score = float("inf")
    best_scale = None

    for s in scales:
        score = evaluate_params(df, emp_run_lengths, A, alpha, baseline_scale=s)
        print(f"[BASELINE] scale={s:.3f}, A={A}, alpha={alpha:.3f}, score={score:.4f}")
        if score < best_score:
            best_score = score
            best_scale = s

    return best_scale, best_score


# ------------------------------------------------------------
# Layer 5: hazard_mult per regime (independent 1D searches)
# ------------------------------------------------------------
def tune_hazard_mult(df, emp_run_lengths, A, alpha, baseline_scale):
    grid = [0.7, 0.85, 1.0, 1.15, 1.3]

    n_states = int(df["state"].max()) + 1
    best_mult = [1.0] * n_states

    for k in range(n_states):
        best_score_k = float("inf")
        best_mk = 1.0

        for m in grid:
            hazard_mult = best_mult.copy()
            hazard_mult[k] = m
            score = evaluate_params(df, emp_run_lengths, A, alpha,
                                    baseline_scale=baseline_scale,
                                    hazard_mult=hazard_mult)
            print(f"[HAZARD] k={k}, m={m:.3f}, score={score:.4f}")
            if score < best_score_k:
                best_score_k = score
                best_mk = m

        best_mult[k] = best_mk
        print(f"[HAZARD] best m[{k}]={best_mk:.3f}, score={best_score_k:.4f}")

    final_score = evaluate_params(df, emp_run_lengths, A, alpha,
                                  baseline_scale=baseline_scale,
                                  hazard_mult=best_mult)
    return best_mult, final_score


# ------------------------------------------------------------
# Layer 6: conv_smooth tuning (1D)
# ------------------------------------------------------------
def tune_conv_smooth(df, emp_run_lengths, A, alpha, baseline_scale, hazard_mult):
    lambdas = [0.0, 0.25, 0.5, 0.75]

    best_score = float("inf")
    best_lambda = None

    for lam in lambdas:
        score = evaluate_params(df, emp_run_lengths, A, alpha,
                                baseline_scale=baseline_scale,
                                hazard_mult=hazard_mult,
                                conv_smooth=lam)
        print(f"[CONV-SMOOTH] lambda={lam:.2f}, score={score:.4f}")
        if score < best_score:
            best_score = score
            best_lambda = lam

    return best_lambda, best_score


# ------------------------------------------------------------
# Orchestrator: full 5-layer tuning
# ------------------------------------------------------------
def full_tune(df, emp_run_lengths):
    (A_coarse, alpha_coarse), score_coarse = coarse_tune_A_alpha(df, emp_run_lengths)
    print(f"\n[COARSE BEST] A={A_coarse}, alpha={alpha_coarse:.3f}, score={score_coarse:.4f}")

    alpha_fine, score_alpha = refine_alpha(df, emp_run_lengths, A_coarse, alpha_coarse)
    print(f"[ALPHA BEST] A={A_coarse}, alpha={alpha_fine:.3f}, score={score_alpha:.4f}")

    A_fine, score_A = refine_A(df, emp_run_lengths, A_coarse, alpha_fine)
    print(f"[A BEST] A={A_fine}, alpha={alpha_fine:.3f}, score={score_A:.4f}")

    baseline_scale, score_base = tune_baseline_scale(df, emp_run_lengths, A_fine, alpha_fine)
    print(f"[BASELINE BEST] scale={baseline_scale:.3f}, score={score_base:.4f}")

    hazard_mult, score_hazard = tune_hazard_mult(df, emp_run_lengths, A_fine, alpha_fine, baseline_scale)
    print(f"[HAZARD BEST] mult={hazard_mult}, score={score_hazard:.4f}")

    conv_smooth, score_conv = tune_conv_smooth(df, emp_run_lengths, A_fine, alpha_fine,
                                               baseline_scale, hazard_mult)
    print(f"[CONV-SMOOTH BEST] lambda={conv_smooth:.2f}, score={score_conv:.4f}")

    best = {
        "A": A_fine,
        "alpha": alpha_fine,
        "baseline_scale": baseline_scale,
        "hazard_mult": hazard_mult,
        "conv_smooth": conv_smooth,
        "final_score": score_conv,
    }
    return best


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet(DATA_FINAL)
    conv = pd.read_parquet(CONVICTION_PARQUET)
    df["conviction"] = conv["conviction"].values

    emp_run_lengths = compute_run_lengths(df["state"].values)

    best_params = full_tune(df, emp_run_lengths)

    print("\n===== FINAL BEST PARAMS =====")
    for k, v in best_params.items():
        print(f"{k}: {v}")