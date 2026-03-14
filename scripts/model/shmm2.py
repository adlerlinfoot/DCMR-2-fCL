import json
import numpy as np
from pathlib import Path
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRANSITION_NPY = OUTPUTS_DIR / "transition_matrix.npy"
AGE_FUNCS_JSON = OUTPUTS_DIR / "age_functions.json"

DATA_FINAL = PROJECT_ROOT / "data" / "final" / "CL_final.parquet"


class SHMM2:
    def __init__(self,
                 transition_path: Path = TRANSITION_NPY,
                 age_funcs_path: Path = AGE_FUNCS_JSON,
                 conviction_alpha: float = 0.27,
                 conviction_age_full: int = 16,
                 baseline_scale: float = 1,
                 hazard_mult=[1.0, 1.3, 1.15, 1.0, 1.3],
                 conv_smooth: float = 0.25):

        if not transition_path.exists():
            raise FileNotFoundError(f"Transition tensor not found: {transition_path}")
        if not age_funcs_path.exists():
            raise FileNotFoundError(f"Age functions file not found: {age_funcs_path}")

        # Load transition tensor
        self.T = np.load(transition_path)
        self.n_states = self.T.shape[0]

        # Hyperparameters
        self.conviction_alpha = conviction_alpha
        self.conviction_age_full = conviction_age_full
        self.baseline_scale = float(baseline_scale)

        # Expand hazard multipliers to match number of states
        if hazard_mult is None:
            # default: directional tuned, chop neutral
            self.hazard_mult = [1.0, 0.85, 1.0, 1.0, 1.0]
        else:
            if len(hazard_mult) != self.n_states:
                raise ValueError(
                    f"hazard_mult length {len(hazard_mult)} does not match n_states {self.n_states}"
                )
            self.hazard_mult = hazard_mult

        self.conv_smooth = float(conv_smooth)

        # Load age functions
        with open(age_funcs_path, "r") as f:
            age_data = json.load(f)

        self.age_functions = {}
        for k, v in age_data.items():
            state = int(k)
            max_age = int(v["max_age"])
            values = np.array(v["values"], dtype=float)
            if len(values) != max_age:
                raise ValueError(
                    f"Age function length mismatch for state {state}: "
                    f"len(values)={len(values)}, max_age={max_age}"
                )
            self.age_functions[state] = {
                "max_age": max_age,
                "values": values
            }

    def _get_age_factor(self, state: int, age: int) -> float:
        info = self.age_functions[state]
        max_age = info["max_age"]
        values = info["values"]

        if age <= 0:
            age = 1
        if age > max_age:
            return float(values[-1])
        return float(values[age - 1])

    def next_probs(self, s_tm2: int, s_tm1: int, age: int, conviction: float = 0.5) -> np.ndarray:
        i = int(s_tm2)
        j = int(s_tm1)

        base_row = self.T[i, j, :].astype(float).copy()

        # 1. GLOBAL BASELINE SCALE (NEW)
        base_row *= self.baseline_scale

        # 2. AGE EFFECT
        age_factor = self._get_age_factor(j, age)
        base_row[j] *= age_factor

        # 3. REGIME-SPECIFIC HAZARD MULTIPLIER (NEW)
        base_row[j] *= self.hazard_mult[j]

        # 4. CONVICTION SMOOTHING (NEW)
        if self.conv_smooth > 0:
            conviction = (1 - self.conv_smooth) * conviction + self.conv_smooth * 0.5

        # 5. CONVICTION EFFECT
        A = self.conviction_age_full
        boundary_weight = min(1.0, age / A)
        alpha = self.conviction_alpha
        phi = 1.0 + alpha * boundary_weight * (conviction - 0.5)
        base_row[j] *= phi

        # Normalize
        row_sum = base_row.sum()
        if row_sum <= 0:
            return np.ones(self.n_states) / self.n_states

        return base_row / row_sum

    def step(self, s_tm2: int, s_tm1: int, age: int,
             conviction: float = 0.5,
             rng: np.random.Generator = None) -> int:

        if rng is None:
            rng = np.random.default_rng()

        probs = self.next_probs(s_tm2, s_tm1, age, conviction)
        return int(rng.choice(self.n_states, p=probs))


# ----------------------------------------------------------------------
# CLI: generate neutral conviction file for evaluation
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if not DATA_FINAL.exists():
        raise FileNotFoundError(f"Final dataset not found: {DATA_FINAL}")

    df = pd.read_parquet(DATA_FINAL)

    df_conv = df[[]].copy()
    df_conv["conviction"] = 0.5

    neutral_path = OUTPUTS_DIR / "conviction_neutral.parquet"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_conv.to_parquet(neutral_path, index=True)

    print(f"Neutral conviction file written to: {neutral_path}")