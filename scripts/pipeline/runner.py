import os
import subprocess
import shutil
from pathlib import Path

RAW_TXT = Path("data/raw/CL.txt")
RAW_CSV = Path("data/cleaned/CL.csv")     # this is where your converter writes
OUTPUT_DIR = Path("outputs")

def run(cmd):
    subprocess.run(cmd, check=True)

def ensure_csv():
    # Only convert if CSV does NOT exist
    if RAW_CSV.exists():
        print("CSV already exists → skipping TXT conversion.")
        return

    # If CSV missing but TXT exists → convert
    if RAW_TXT.exists():
        print("Converting TXT → CSV...")
        run(["python", "-m", "scripts.data.convert_txt_to_csv"])
    else:
        raise FileNotFoundError(f"Neither {RAW_CSV} nor {RAW_TXT} exists.")

def clear_output():
    print("Clearing output directory...")
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            f.unlink()

def run_pipeline_steps():
    print("Running stitch_yf_updates...")
    run(["python", "-m", "scripts.data.stitch_yf_updates"])

    print("Running final_dataset...")
    run(["python", "-m", "scripts.data.final_dataset"])

    print("Running fit_transitions...")
    run(["python", "-m", "scripts.model.fit_transitions"])

    print("Running fit_age_functions...")
    run(["python", "-m", "scripts.model.fit_age_functions"])

    print("Plotting age distributions...")
    run(["python", "-m", "scripts.model.plot_age_distributions"])

    print("Running DCMR-2 model...")
    run(["python", "-m", "scripts.model.dcmr2"])

    print("Evaluating DCMR-2...")
    run(["python", "-m", "scripts.model.eval_dcmr2"])

def manual_regime_prompt():
    ans = input("Manual CURRENT regime override? (Y/N): ").strip().upper()
    if ans == "Y":
        r = input("Enter CURRENT regime (0,1,2,3,4): ").strip()
        if r not in {"0", "1", "2", "3", "4"}:
            raise ValueError("Invalid regime.")
        return int(r)
    return None

def run_overlay(manual_regime):
    print("Running overlay...")
    args = ["python", "-m", "scripts.overlay.build_overlay"]

    if manual_regime is not None:
        args += ["--manual_current", str(manual_regime)]

    run(args)

def main():
    ensure_csv()
    clear_output()
    run_pipeline_steps()
    manual_current = manual_regime_prompt()
    run_overlay(manual_current)

if __name__ == "__main__":
    main()
