import json
import numpy as np
import pandas as pd
from pathlib import Path
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit


def trace(msg):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

DATA_FINAL = PROJECT_ROOT / "data" / "final" / "CL_final.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RF_CONFUSION_CSV = OUTPUTS_DIR / "rf_confusion_matrix.csv"
GB_CONFUSION_CSV = OUTPUTS_DIR / "gb_confusion_matrix.csv"

RF_IMPORTANCE_CSV = OUTPUTS_DIR / "rf_feature_importances.csv"
GB_IMPORTANCE_CSV = OUTPUTS_DIR / "gb_feature_importances.csv"

RF_SUMMARY_JSON = OUTPUTS_DIR / "rf_summary.json"
GB_SUMMARY_JSON = OUTPUTS_DIR / "gb_summary.json"

CONVICTION_PARQUET = OUTPUTS_DIR / "conviction.parquet"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    trace("Loading CL_final.parquet...")
    df = pd.read_parquet(DATA_FINAL)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    if len(datetime_cols) > 0:
        print("Dropping datetime columns:", list(datetime_cols))
        df = df.drop(columns=datetime_cols)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()

    exclude = {"state", "state_t_minus_1", "state_t_minus_2", "age"}
    feature_cols = [c for c in df.columns if c not in exclude]

    trace("Feature diagnostics (nunique):")
    trace(str(df[feature_cols].nunique()))

    trace("Feature diagnostics (std):")
    trace(str(df[feature_cols].std()))

    X = df[feature_cols].values
    y = df["state"].astype(int).values

    trace(f"Loaded dataset with {len(df)} rows and {len(feature_cols)} features.")
    return df, X, y, feature_cols


def time_series_split(X, y, test_frac=0.2):
    trace("Performing time-series split...")
    n = len(X)
    split = int(n * (1 - test_frac))
    trace(f"Train size: {split}, Test size: {n - split}")
    return X[:split], X[split:], y[:split], y[split:]


def compute_oof_conviction(df, X, y, feature_cols):
    """
    Time-respecting out-of-fold conviction:
    For each time block, train on the past, predict on the future block.
    No row is ever scored by a model that saw it during training.
    """
    trace("Computing out-of-fold conviction (time-series)...")

    n_splits = 5  # you can bump this if you want finer granularity
    tscv = TimeSeriesSplit(n_splits=n_splits)

    n_states = int(df["state"].max()) + 1
    oof_rf = np.zeros((len(df), n_states), dtype=float)
    oof_gb = np.zeros((len(df), n_states), dtype=float)
    seen = np.zeros(len(df), dtype=bool)

    split_idx = 0
    for train_idx, val_idx in tscv.split(X):
        split_idx += 1
        trace(f"TimeSeriesSplit {split_idx}/{n_splits}: "
              f"train={train_idx[0]}..{train_idx[-1]}, val={val_idx[0]}..{val_idx[-1]}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

                # RF – deliberately weak, shallow, regularized
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            n_jobs=-1,
            class_weight=None,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        oof_rf[val_idx] = rf.predict_proba(X_val)

        # GB – small, shallow, subsampled
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=2,
            subsample=0.5,
            random_state=42,
        )
        gb.fit(X_train, y_train)
        oof_gb[val_idx] = gb.predict_proba(X_val)

        seen[val_idx] = True

    # For any rows not covered by splits (e.g., very early rows), fall back to a final model on earliest chunk
    if not seen.all():
        trace("Some rows not covered by TimeSeriesSplit; fitting fallback models on initial segment.")
        last_seen = np.where(seen)[0].min() if seen.any() else len(df) // 2
        fallback_train_idx = np.arange(0, last_seen + 1)
        fallback_val_idx = np.where(~seen)[0]

        X_train, y_train = X[fallback_train_idx], y[fallback_train_idx]
        X_val = X[fallback_val_idx]

        rf_fb = RandomForestClassifier(
            n_estimators=600,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=123,
        )
        rf_fb.fit(X_train, y_train)
        oof_rf[fallback_val_idx] = rf_fb.predict_proba(X_val)

        gb_fb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            random_state=123,
        )
        gb_fb.fit(X_train, y_train)
        oof_gb[fallback_val_idx] = gb_fb.predict_proba(X_val)

    state_idx = df["state"].astype(int).values
    rf_confidence = oof_rf[np.arange(len(df)), state_idx]
    gb_confidence = oof_gb[np.arange(len(df)), state_idx]

    conviction = 0.5 * (rf_confidence + gb_confidence)
    shrink = 0.5  # 0 = no shrink, 1 = all 0.5
    conviction = (1 - shrink) * conviction + shrink * 0.5

    df_conv = df[[]].copy()
    df_conv["conviction"] = conviction
    df_conv.to_parquet(CONVICTION_PARQUET, index=True)
    trace(f"Saved OUT-OF-FOLD conviction scores to: {CONVICTION_PARQUET}")


def tune_rf_gb():
    start_time = time.time()
    trace("Starting RF+GB tuner...")

    df, X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = time_series_split(X, y)

    # ---------------- Random Forest ----------------
    trace("Fitting Random Forest (train/test evaluation)...")
    rf_start = time.time()

    rf = RandomForestClassifier(
        n_estimators=1200,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf.fit(X_train, y_train)

    trace(f"RF fit completed in {time.time() - rf_start:.1f} seconds.")

    trace("Evaluating RF...")
    y_proba_rf_test = rf.predict_proba(X_test)
    y_pred_rf = np.argmax(y_proba_rf_test, axis=1)

    rf_conf = confusion_matrix(y_test, y_pred_rf, normalize="true")
    pd.DataFrame(rf_conf).to_csv(RF_CONFUSION_CSV, index=False)

    rf_importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    rf_importances.to_csv(RF_IMPORTANCE_CSV, index=False)

    rf_summary = {
        "log_loss": float(log_loss(y_test, y_proba_rf_test)),
        "classification_report": classification_report(y_test, y_pred_rf, output_dict=True),
    }
    with open(RF_SUMMARY_JSON, "w") as f:
        json.dump(rf_summary, f, indent=2)

    trace("RF evaluation complete.")

    # ---------------- Gradient Boosting ----------------
    trace("Fitting Gradient Boosting (train/test evaluation)...")
    gb_start = time.time()

    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    trace(f"GB fit completed in {time.time() - gb_start:.1f} seconds.")

    trace("Evaluating GB...")
    y_proba_gb_test = gb.predict_proba(X_test)
    y_pred_gb = np.argmax(y_proba_gb_test, axis=1)

    gb_conf = confusion_matrix(y_test, y_pred_gb, normalize="true")
    pd.DataFrame(gb_conf).to_csv(GB_CONFUSION_CSV, index=False)

    gb_importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": gb.feature_importances_
    }).sort_values("importance", ascending=False)
    gb_importances.to_csv(GB_IMPORTANCE_CSV, index=False)

    gb_summary = {
        "log_loss": float(log_loss(y_test, y_proba_gb_test)),
        "classification_report": classification_report(y_test, y_pred_gb, output_dict=True),
    }
    with open(GB_SUMMARY_JSON, "w") as f:
        json.dump(gb_summary, f, indent=2)

    trace("GB evaluation complete.")

    # -------- OUT-OF-FOLD, TIME-RESPECTING CONVICTION (NO LEAKAGE) --------
    compute_oof_conviction(df, X, y, feature_cols)

    total_time = time.time() - start_time
    trace(f"RF+GB tuner finished in {total_time/3600:.2f} hours.")


if __name__ == "__main__":
    tune_rf_gb()