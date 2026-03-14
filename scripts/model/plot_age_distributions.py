import json
from pathlib import Path
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
RUN_LENGTHS_JSON = INTERMEDIATE_DIR / "run_lengths_by_state.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Individual plots
PLOT_DIR = OUTPUTS_DIR / "age_distributions"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Load run-lengths
# ----------------------------------------------------------------------

with open(RUN_LENGTHS_JSON, "r") as f:
    run_lengths = json.load(f)

# Convert keys to ints
run_lengths = {int(k): v for k, v in run_lengths.items()}


# ----------------------------------------------------------------------
# Plot individual histograms
# ----------------------------------------------------------------------

def plot_individual_histograms():
    for state, lengths in run_lengths.items():
        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=50, alpha=0.75, color="steelblue")
        plt.title(f"Run-Length Distribution — State {state}")
        plt.xlabel("Run Length (hours)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)

        out_path = PLOT_DIR / f"run_length_state_{state}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved individual run-length histograms to: {PLOT_DIR}")


# ----------------------------------------------------------------------
# Plot combined overlay histogram
# ----------------------------------------------------------------------

def plot_overlay_histogram():
    plt.figure(figsize=(12, 6))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for state, color in zip(sorted(run_lengths.keys()), colors):
        plt.hist(
            run_lengths[state],
            bins=60,
            alpha=0.4,
            label=f"State {state}",
            color=color
        )

    plt.title("Run-Length Distributions — All States")
    plt.xlabel("Run Length (hours)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = PLOT_DIR / "run_length_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved overlay run-length histogram to: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    plot_individual_histograms()
    plot_overlay_histogram()