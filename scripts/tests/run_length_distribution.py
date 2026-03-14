import pandas as pd
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent  
DATA_FINAL = PROJECT_ROOT / "data" / "final" / "CL_final.parquet"

df = pd.read_parquet(DATA_FINAL)

states = df["state"].astype(int).values

run_lengths = {0: [], 1: [], 2: [], 3: []}

current_state = states[0]
current_len = 1

for s in states[1:]:
    if s == current_state:
        current_len += 1
    else:
        run_lengths[current_state].append(current_len)
        current_state = s
        current_len = 1

# Add the final run
run_lengths[current_state].append(current_len)

# Convert to DataFrame for inspection
import numpy as np

summary = {}
for k, v in run_lengths.items():
    arr = np.array(v)
    summary[k] = {
        "count": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }

print(summary)