import json
import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RUN_LENGTHS_JSON = INTERMEDIATE_DIR / "run_lengths_by_state.json"
AGE_FUNCS_JSON = OUTPUTS_DIR / "age_functions.json"
AGE_SUMMARY_JSON = OUTPUTS_DIR / "age_functions_summary.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def fit_age_functions():
    if AGE_FUNCS_JSON.exists():
        raise RuntimeError(
            f"{AGE_FUNCS_JSON} already exists. "
            "Refusing to overwrite frozen model artifact."
        )

    if not RUN_LENGTHS_JSON.exists():
        raise FileNotFoundError(
            f"Run-length file not found: {RUN_LENGTHS_JSON}"
        )

    with open(RUN_LENGTHS_JSON, "r") as f:
        run_lengths = json.load(f)

    run_lengths = {int(k): np.array(v, dtype=int) for k, v in run_lengths.items()}

    age_functions = {}
    summary = {}

    for state, lengths in run_lengths.items():
        max_len = lengths.max()

        # Survival curve S(a) = P(run_length >= a)
        S = np.zeros(max_len + 2)
        for a in range(1, max_len + 1):
            S[a] = np.mean(lengths >= a)

        # Hazard ratio f(a) = S(a+1) / S(a)
        f = np.zeros(max_len + 1)
        for a in range(1, max_len):
            if S[a] > 0:
                f[a] = S[a + 1] / S[a]
            else:
                f[a] = 0.0

        # Clip to stable range
        f = np.clip(f, 0.5, 2.0)

        age_functions[state] = {
            "max_age": int(max_len),
            "values": f[1:max_len + 1].tolist()
        }

        summary[state] = {
            "max_age": int(max_len),
            "mean_f": float(np.mean(f[1:max_len])),
            "min_f": float(np.min(f[1:max_len])),
            "max_f": float(np.max(f[1:max_len]))
        }

    with open(AGE_FUNCS_JSON, "w") as f:
        json.dump(age_functions, f, indent=2)

    with open(AGE_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Age functions saved to: {AGE_FUNCS_JSON}")
    print(f"Summary saved to: {AGE_SUMMARY_JSON}")


if __name__ == "__main__":
    fit_age_functions()