import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.model.shmm2 import SHMM2

DATA_FINAL = Path("data/final/CL_final.parquet")
CONVICTION_PARQUET = Path("outputs/conviction_neutral.parquet")
AGE_FUNCTIONS_JSON = Path("outputs/age_functions.json")
OUTPUT_OVERLAY = Path("outputs/overlay.parquet")


def load_age_functions():
    """
    Load age deformation functions per regime.
    age_functions.json structure:
      {
        "0": {"max_age": ..., "values": [f_age(1), f_age(2), ...]},
        ...
      }
    We will use values[age-1] as the age factor, clamped at the last value.
    """
    with open(AGE_FUNCTIONS_JSON, "r") as f:
        age_funcs = json.load(f)

    age_factors = {}
    for k, v in age_funcs.items():
        k_int = int(k)
        age_factors[k_int] = np.array(v["values"], dtype=float)
    return age_factors


def age_factor_for(age_factors, regime, age):
    """
    Get age deformation factor for a given regime and age.
    If age exceeds the curve length, use the last value.
    """
    curve = age_factors.get(regime)
    if curve is None or len(curve) == 0:
        return 1.0
    a = max(1, int(age))
    if a > len(curve):
        return float(curve[-1])
    return float(curve[a - 1])


def compute_empirical_mean_run_lengths(regimes):
    """
    Compute empirical mean run length per regime from the full state sequence.
    regimes: 1D array of regime/state IDs.
    """
    run_lengths = {}
    current_state = int(regimes[0])
    length = 1

    for i in range(1, len(regimes)):
        s = int(regimes[i])
        if s == current_state:
            length += 1
        else:
            run_lengths.setdefault(current_state, []).append(length)
            current_state = s
            length = 1
    # last run
    run_lengths.setdefault(current_state, []).append(length)

    mean_lengths = {}
    for state, lengths in run_lengths.items():
        mean_lengths[state] = float(np.mean(lengths))
    return mean_lengths


def load_latest_row_with_time():
    df = pd.read_parquet(DATA_FINAL)
    df_conv = pd.read_parquet(CONVICTION_PARQUET)
    df = df.merge(df_conv[["conviction"]], left_index=True, right_index=True)

    ts = df.index[-1]
    row = df.iloc[-1]
    return ts, row


def compute_prev_run_length(regimes, i_start):
    """
    Given the index where the current run starts (i_start),
    walk backward to find how long the previous regime lasted.
    """
    if i_start == 0:
        return 0, int(regimes[0])  # no prior history; degenerate

    prev_regime = int(regimes[i_start - 1])
    length = 1
    j = i_start - 2
    while j >= 0 and int(regimes[j]) == prev_regime:
        length += 1
        j -= 1
    return length, prev_regime


def compute_next_regime(engine, r_tm1, r_t, age, conviction):
    """
    Run-level next regime:
      - use next_probs() to get next-bar probabilities
      - condition on flipping (exclude staying in r_t)
      - pick the most likely flip destination
    """
    probs = engine.next_probs(r_tm1, r_t, age, conviction)
    probs = np.asarray(probs, dtype=float)

    flip_probs = probs.copy()
    flip_probs[r_t] = 0.0
    total_flip = flip_probs.sum()

    if total_flip <= 0:
        # degenerate: no flip mass; say "stay" with full confidence
        return r_t, 1.0, probs

    flip_probs /= total_flip
    dest = int(np.argmax(flip_probs))
    confidence = float(flip_probs[dest])
    return dest, confidence, probs


def compute_time_to_flip(remaining_cur, conviction, age_prev, E_prev):
    """
    Time-to-flip:
      - base = expected remaining duration of current regime
      - conviction: higher conviction => longer remaining
      - previous-regime stability: short previous run => slightly shorter remaining,
        long previous run => slightly longer remaining
    """
    # Previous regime stability (empirical)
    if E_prev <= 0:
        stability_prev = 1.0
    else:
        stability_prev = age_prev / E_prev

    stability_prev_clamped = max(0.3, min(1.7, stability_prev))
    stab_factor = 1.0 + 0.3 * (stability_prev_clamped - 1.0)

    # Conviction factor: map [0,1] -> [0.7, 1.3]
    conv_factor = 0.7 + 0.6 * float(conviction)

    base = max(1.0, float(remaining_cur))
    ttf = base * conv_factor * stab_factor
    return int(max(1, round(ttf)))


def build_overlay(manual_current=None):
    engine = SHMM2()
    age_factors = load_age_functions()

    ts, row = load_latest_row_with_time()

    # Load full dataset
    df = pd.read_parquet(DATA_FINAL)
    regimes = df["state"].values

    # Empirical mean run lengths per regime
    mean_lengths = compute_empirical_mean_run_lengths(regimes)

    # Locate current row index
    i_cur = df.index.get_loc(ts)

    age = int(row["age"])
    conviction = float(row["conviction"])

    # Current regime from last row
    r_t = int(regimes[i_cur])

    # Manual override (regime-level)
    manual_override = manual_current is not None
    if manual_override:
        r_t = manual_current

    # Start index of current regime run
    i_start = max(0, i_cur - (age - 1))

    # Previous regime duration and identity
    age_prev, r_tm1 = compute_prev_run_length(regimes, i_start)

    # Empirical expected durations
    E_cur_emp = mean_lengths.get(r_t, float(age))
    E_prev_emp = mean_lengths.get(r_tm1, float(age_prev if age_prev > 0 else 1.0))

    # Age deformation factor for current regime
    age_fac = age_factor_for(age_factors, r_t, age)

    # Conviction factor for expected duration (same mapping as TTF)
    conv_fac = 0.7 + 0.6 * float(conviction)

    # Expected duration current: empirical mean * age factor * conviction factor
    E_cur = E_cur_emp * age_fac * conv_fac

    # Expected remaining duration: simple difference
    remaining_cur = max(1.0, E_cur - age)

    # Next regime (run-level)
    next_regime, next_conf, next_probs = compute_next_regime(
        engine, r_tm1, r_t, age, conviction
    )

    # Time-to-flip
    ttf = compute_time_to_flip(remaining_cur, conviction, age_prev, E_prev_emp)

    overlay = {
        "as_of_timestamp": ts,
        "current_regime": r_t,
        "previous_regime": r_tm1,
        "previous_regime_age": age_prev,
        "age": age,
        "conviction": conviction,
        "expected_duration_current": E_cur,
        "expected_duration_previous": E_prev_emp,
        "expected_remaining_current": remaining_cur,
        "next_regime": next_regime,
        "next_regime_confidence": next_conf,
        "next_regime_probs": [float(x) for x in next_probs.tolist()],
        "predicted_time_to_flip": ttf,
    }

    df_overlay = pd.DataFrame([overlay])
    OUTPUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    df_overlay.to_parquet(OUTPUT_OVERLAY, index=False)

    # Pretty print
    print(f"\n=== OVERLAY (as of {ts}) ===")
    print(f"as_of_timestamp:           {ts}")
    print(f"current_regime:            {r_t}")
    print(f"previous_regime:           {r_tm1}")
    print(f"previous_regime_age:       {age_prev}")
    print(f"age:                       {age}")
    print(f"conviction:                {conviction:.3f}")
    print(f"expected_duration_current: {E_cur:.3f}")
    print(f"expected_duration_previous:{E_prev_emp:.3f}")
    print(f"expected_remaining_current:{remaining_cur:.3f}")
    print(f"next_regime:               {next_regime}")
    print(f"next_regime_confidence:    {next_conf:.3f}")
    probs_str = ", ".join(f"{p:.4f}" for p in next_probs.tolist())
    print(f"next_regime_probs:         [{probs_str}]")
    print(f"predicted_time_to_flip:    {ttf}")
    print(f"\nSaved overlay to {OUTPUT_OVERLAY}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_current", type=int, default=None)
    args = parser.parse_args()

    build_overlay(manual_current=args.manual_current)


if __name__ == "__main__":
    main()