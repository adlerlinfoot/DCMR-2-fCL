import os
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
FINAL_DIR = DATA_DIR / "final"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = CLEANED_DIR / "CL.csv"
FINAL_CSV = FINAL_DIR / "CL_final.csv"
FINAL_PARQUET = FINAL_DIR / "CL_final.parquet"
RUN_LENGTHS_JSON = INTERMEDIATE_DIR / "run_lengths_by_state.json"


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

# Volatility: 24h realized vol from hourly data, threshold at 1.5%
VOL_WINDOW_HOURS = 24
VOL_THRESHOLD = 0.015  # 1.5% daily-equivalent

# Trend: 5-day (120h) window
TREND_WINDOW_HOURS = 24 * 5  # 120

# Normalized drift: window and threshold
DRIFT_WINDOW_HOURS = 24 * 3  # e.g., 72h window
DRIFT_THRESHOLD = 0.3        # |D| <= 0.3 => chop (tune later)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    return np.log(close / close.shift(1))


def compute_realized_vol_24h(log_ret: pd.Series) -> pd.Series:
    # rolling std over 24 hourly returns, scaled to daily-equivalent
    return log_ret.rolling(VOL_WINDOW_HOURS).std() * np.sqrt(VOL_WINDOW_HOURS)


def compute_trend_5d(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    return close.pct_change(TREND_WINDOW_HOURS)

def compute_normalized_drift(df: pd.DataFrame) -> pd.Series:
    """
    Normalized drift D_t = cumulative log return over window / realized vol over same window.
    This is a Sharpe-like signal: trend strength relative to noise.
    """
    log_ret = df["log_ret"]

    # cumulative log return over DRIFT_WINDOW_HOURS
    cum_log_ret = log_ret.rolling(DRIFT_WINDOW_HOURS).sum()

    # realized vol over same window (std * sqrt(window))
    vol_window = log_ret.rolling(DRIFT_WINDOW_HOURS).std() * np.sqrt(DRIFT_WINDOW_HOURS)

    D = cum_log_ret / vol_window
    return D

def assign_states_5reg(vol: pd.Series,
                       trend: pd.Series,
                       norm_drift: pd.Series) -> pd.Series:
    """
    Map (vol, trend, normalized drift) -> 5 semantic regimes:

        0 = low vol, up trend
        1 = low vol, down trend
        2 = high vol, up trend
        3 = high vol, down trend
        4 = chop (low normalized drift, regardless of vol)
    """
    low_vol = vol < VOL_THRESHOLD
    up_trend = trend > 0

    state = pd.Series(index=vol.index, dtype="float64")

    # chop first: low normalized drift in absolute value
    is_chop = norm_drift.abs() <= DRIFT_THRESHOLD
    state[is_chop] = 4

    # directional regimes only where not chop
    directional = ~is_chop

    # low vol, up
    state[directional & low_vol & up_trend] = 0
    # low vol, down
    state[directional & low_vol & ~up_trend] = 1
    # high vol, up
    state[directional & ~low_vol & up_trend] = 2
    # high vol, down
    state[directional & ~low_vol & ~up_trend] = 3

    return state.astype("Int64")


def compute_age(state: pd.Series) -> pd.Series:
    """
    Age_t = number of consecutive time steps spent in the current regime.
    """
    age = np.zeros(len(state), dtype=int)
    prev_state = None
    current_age = 0

    for i, s in enumerate(state.to_numpy()):
        if pd.isna(s):
            age[i] = 0
            prev_state = None
            current_age = 0
            continue

        if prev_state is None or s != prev_state:
            current_age = 1
        else:
            current_age += 1

        age[i] = current_age
        prev_state = s

    return pd.Series(age, index=state.index, dtype="Int64")


def compute_run_lengths(state: pd.Series, age: pd.Series) -> pd.Series:
    """
    A run ends when state_t != state_{t+1} (or at the last observation).
    At run ends, run_length = age_t.
    """
    next_state = state.shift(-1)
    is_run_end = (state != next_state) | next_state.isna()
    run_length = pd.Series(index=state.index, dtype="Int64")
    run_length[is_run_end] = age[is_run_end]
    return is_run_end, run_length


def collect_run_lengths_by_state(state: pd.Series,
                                 run_length: pd.Series) -> dict:
    """
    Return a dict: {state_int: [run_length_1, run_length_2, ...]}
    Only includes completed runs (non-NaN run_length).
    """
    result = {}
    mask_valid = run_length.notna()
    for s in sorted(state.dropna().unique()):
        s_int = int(s)
        lengths = run_length[mask_valid & (state == s)].astype(int).tolist()
        result[s_int] = lengths
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def build_final_dataset():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input cleaned file not found: {INPUT_CSV}")

    # Load cleaned hourly data
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic sanity: ensure required columns
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleaned data: {missing}")

    # Compute log returns
    df["log_ret"] = compute_log_returns(df)

    # Compute normalized drift (for chop regime)
    df["norm_drift"] = compute_normalized_drift(df)

    # Compute 24h realized volatility
    df["realized_vol_24h"] = compute_realized_vol_24h(df["log_ret"])

    # Compute 5-day trend
    df["trend_5d"] = compute_trend_5d(df)

        # Assign semantic 5-state regimes (with normalized-drift chop)
    df["state"] = assign_states_5reg(
        df["realized_vol_24h"],
        df["trend_5d"],
        df["norm_drift"],
    )

    # Drop initial rows where features are NaN due to rolling windows
    df = df.dropna(subset=["log_ret", "realized_vol_24h", "trend_5d", "norm_drift", "state"]).reset_index(drop=True)

    # Compute lagged states for 2nd-order Markov
    df["state_t_minus_1"] = df["state"].shift(1)
    df["state_t_minus_2"] = df["state"].shift(2)

    # Drop rows where we don't have full 2nd-order context
    df = df.dropna(subset=["state_t_minus_1", "state_t_minus_2"]).reset_index(drop=True)
    df["state_t_minus_1"] = df["state_t_minus_1"].astype("Int64")
    df["state_t_minus_2"] = df["state_t_minus_2"].astype("Int64")

    # Compute age (run-length so far)
    df["age"] = compute_age(df["state"])

    # Compute run ends and run_length at run end
    df["is_run_end"], df["run_length"] = compute_run_lengths(df["state"], df["age"])

    # Collect run lengths per state and save to JSON
    run_lengths_by_state = collect_run_lengths_by_state(df["state"], df["run_length"])
    with open(RUN_LENGTHS_JSON, "w") as f:
        json.dump(run_lengths_by_state, f, indent=2)

    # Save final dataset
    df.to_csv(FINAL_CSV, index=False)
    df.to_parquet(FINAL_PARQUET, index=False)

    print(f"Final dataset written to:\n  {FINAL_CSV}\n  {FINAL_PARQUET}")
    print(f"Run lengths by state written to:\n  {RUN_LENGTHS_JSON}")


if __name__ == "__main__":
    build_final_dataset()