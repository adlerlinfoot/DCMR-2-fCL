import os
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "cleaned")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Column names
# ---------------------------------------------------------
COLS = ["date", "time", "open", "high", "low", "close", "volume"]

# ---------------------------------------------------------
# Convert + aggregate to hourly bars
# ---------------------------------------------------------
def convert_and_aggregate(txt_path, csv_path):
    df = pd.read_csv(
        txt_path,
        header=None,
        names=COLS,
        delimiter=",",
        dtype={
            "date": str,
            "time": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": int,
        }
    )

    # Combine into timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.set_index("timestamp")

    # Hourly OHLCV aggregation
    hourly = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    # Drop hours with no data
    hourly = hourly.dropna(subset=["open", "close"])

    hourly.to_csv(csv_path)
    print(f"Converted + aggregated: {os.path.basename(txt_path)} → {os.path.basename(csv_path)}")

# ---------------------------------------------------------
# Run for all files
# ---------------------------------------------------------
def main():
    files = ["CL.txt"]

    for fname in files:
        txt_path = os.path.join(RAW_DIR, fname)
        csv_path = os.path.join(OUT_DIR, fname.replace(".txt", ".csv"))

        if not os.path.exists(txt_path):
            print(f"WARNING: {fname} not found — skipping.")
            continue

        convert_and_aggregate(txt_path, csv_path)

    print("\nAll hourly aggregations complete.")

if __name__ == "__main__":
    main()