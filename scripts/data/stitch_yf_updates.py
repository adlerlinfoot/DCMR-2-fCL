import os
import pandas as pd
import yfinance as yf

CLEAN_DIR = "data/cleaned"

# Map your filenames to YF tickers
TICKERS = {
    "CL.csv": "CL=F",
}

def stitch_file(fname, ticker):
    path = os.path.join(CLEAN_DIR, fname)

    # Load existing cleaned hourly data
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index.name = "timestamp"  # ensure index name is stable

    last_ts = df.index.max()
    print(f"{fname}: last local timestamp = {last_ts}")

    # Pull new hourly data from YF
    yf_data = yf.download(
        ticker,
        interval="60m",
        start=last_ts,
        progress=False
    )

    if yf_data.empty:
        print(f"{fname}: no new YF data.")
        return

    # Flatten MultiIndex columns if needed
    yf_data.columns = [c[0] if isinstance(c, tuple) else c for c in yf_data.columns]

    # Strip timezone to match your tz-naive index
    if yf_data.index.tz is not None:
        yf_data.index = yf_data.index.tz_localize(None)

    # Keep only OHLCV
    yf_data = yf_data[["Open", "High", "Low", "Close", "Volume"]]

    # Rename to match your schema
    yf_data = yf_data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Make sure the new data index is named 'timestamp' too
    yf_data.index.name = "timestamp"

    # Drop rows that already exist
    yf_data = yf_data[yf_data.index > last_ts]

    if yf_data.empty:
        print(f"{fname}: no rows newer than {last_ts}.")
        return

    # Apply rounding policy (same as your cleaned CSVs)
    yf_data["open"] = yf_data["open"].round(2)
    yf_data["high"] = yf_data["high"].round(2)
    yf_data["low"] = yf_data["low"].round(2)
    yf_data["close"] = yf_data["close"].round(2)
    yf_data["volume"] = yf_data["volume"].astype(int)

    # Append to existing data
    updated = pd.concat([df, yf_data])

    # CRITICAL: preserve 'timestamp' as the column name
    updated.index.name = "timestamp"
    updated.reset_index().to_csv(path, index=False)

    print(f"{fname}: appended {len(yf_data)} new rows.")


def main():
    for fname, ticker in TICKERS.items():
        try:
            stitch_file(fname, ticker)
        except Exception as e:
            print(f"ERROR processing {fname}: {e}")


if __name__ == "__main__":
    main()