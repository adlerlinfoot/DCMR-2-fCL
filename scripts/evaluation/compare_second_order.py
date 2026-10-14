"""Compare frozen DCMR weights to the same unadjusted second-order tensor.

Private input paths are explicit. Outputs are aggregate metrics only.
See report/oos-2026-10-14/README.md for the recorded protocol and limitations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.data import final_dataset as features
from scripts.model.dcmr2 import DCMR2


def build_features(raw):
    d = raw.rename(columns=str.lower).copy()
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(d.columns):
        raise ValueError("Yahoo CSV must contain Open, High, Low, Close and Volume")
    d = d[required].sort_index()
    if d.index.duplicated().any():
        raise ValueError("Duplicate input timestamps")
    d = d.dropna()
    if not (d[["open", "high", "low", "close"]] > 0).all().all():
        raise ValueError("Log-return features require positive prices")
    d[["open", "high", "low", "close"]] = d[["open", "high", "low", "close"]].round(2)
    d["log_ret"] = features.compute_log_returns(d)
    d["norm_drift"] = features.compute_normalized_drift(d)
    d["realized_vol_24h"] = features.compute_realized_vol_24h(d.log_ret)
    d["trend_5d"] = features.compute_trend_5d(d)
    d = d.dropna()
    d["state"] = features.assign_states_5reg(d.realized_vol_24h, d.trend_5d, d.norm_drift).astype(int)
    d["age"] = features.compute_age(d.state).astype(int)
    d["previous"] = d.state.shift(1)
    return d


def make_pairs(d, start):
    # Targets are added only after all causal features have been constructed.
    d = d.copy()
    d["target"] = d.state.shift(-1)
    d["target_timestamp"] = d.index.to_series().shift(-1)
    eligible = d.loc[(d.index >= start) & d.target.notna() & d.previous.notna()]
    one_hour = eligible.target_timestamp - eligible.index.to_series() == pd.Timedelta(hours=1)
    return eligible.loc[one_hour].copy(), int((~one_hour).sum())


def score(p, target, current):
    rows = np.arange(len(target))
    changed = target != current
    prediction = p.argmax(axis=1)
    risk = 1 - p[rows, current]
    dest = p.copy()
    dest[rows, current] = 0
    return {
        "n": len(target), "observed_flips": int(changed.sum()),
        "accuracy": float((prediction == target).mean()),
        "predicted_flips": int((prediction != current).sum()),
        "destination_accuracy_given_observed_flip": float((dest.argmax(axis=1)[changed] == target[changed]).mean()),
        "log_loss": float(-np.log(p[rows, target]).mean()),
        "multiclass_brier": float(np.square(p - np.eye(5)[target]).sum(axis=1).mean()),
        "flip_risk_roc_auc": float(roc_auc_score(changed, risk)),
        "flip_risk_average_precision": float(average_precision_score(changed, risk)),
        "change_brier": float(np.square(risk - changed).mean()),
        "mean_predicted_flip_probability": float(risk.mean()),
        "actual_flip_rate": float(changed.mean()),
    }


def bootstrap(adjusted, baseline, target, current, days, resamples=2000, seed=42):
    rows = np.arange(len(target))
    changed = target != current
    risk_a, risk_b = 1 - adjusted[rows, current], 1 - baseline[rows, current]
    losses = {}
    for name, p in [("DCMR", adjusted), ("second_order", baseline)]:
        losses[name] = {
            "log_loss": -np.log(p[rows, target]),
            "multiclass_brier": np.square(p - np.eye(5)[target]).sum(axis=1),
            "change_brier": np.square(1 - p[rows, current] - changed),
        }
    delta = {key: float((losses["DCMR"][key] - losses["second_order"][key]).mean()) for key in losses["DCMR"]}
    delta["change_roc_auc"] = float(roc_auc_score(changed, risk_a) - roc_auc_score(changed, risk_b))
    delta["change_average_precision"] = float(average_precision_score(changed, risk_a) - average_precision_score(changed, risk_b))
    groups = [np.flatnonzero(days == day) for day in sorted(np.unique(days))]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in delta}
    for _ in range(resamples):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(changed[ix])) < 2:
            continue
        for key in losses["DCMR"]:
            samples[key].append(float((losses["DCMR"][key][ix] - losses["second_order"][key][ix]).mean()))
        samples["change_roc_auc"].append(roc_auc_score(changed[ix], risk_a[ix]) - roc_auc_score(changed[ix], risk_b[ix]))
        samples["change_average_precision"].append(average_precision_score(changed[ix], risk_a[ix]) - average_precision_score(changed[ix], risk_b[ix]))
    if not samples["log_loss"]:
        raise ValueError("No bootstrap sample contains both changes and nonchanges")
    intervals = {key: np.quantile(value, [.025, .975]).tolist() for key, value in samples.items()}
    return delta, intervals, len(samples["log_loss"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--age-functions", type=Path, required=True)
    parser.add_argument("--yahoo-csv", type=Path, required=True)
    parser.add_argument("--test-start", default="2026-03-16T00:00:00Z")
    parser.add_argument("--as-of", default="2026-10-14T18:00:00Z")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    start, as_of = pd.Timestamp(args.test_start), pd.Timestamp(args.as_of)
    if start.tzinfo is None or as_of.tzinfo is None:
        parser.error("test-start and as-of must include a timezone")
    start, as_of = start.tz_convert("UTC"), as_of.tz_convert("UTC")
    train = pd.read_parquet(args.train_parquet).sort_values("timestamp")
    # Historical timestamps are naive; the multi-day boundary avoids the
    # ambiguous archive timezone. Do not stitch archive and Yahoo prices.
    if pd.Timestamp(train.timestamp.max()).tz_localize(None) >= start.tz_localize(None) - pd.Timedelta(days=2):
        raise ValueError("Training cutoff must precede OOS start by at least two days")
    model = DCMR2(transition_path=args.transitions, age_funcs_path=args.age_functions)
    idx = train.state_t_minus_2.to_numpy(dtype=int) * 25 + train.state_t_minus_1.to_numpy(dtype=int) * 5 + train.state.to_numpy(dtype=int)
    counts = np.bincount(idx, minlength=125).reshape(5, 5, 5) + 1
    if not np.array_equal(model.T, counts / counts.sum(axis=2, keepdims=True)):
        raise ValueError("Saved tensor does not match training-only add-one counts")
    raw = pd.read_csv(args.yahoo_csv, index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.loc[raw.index < as_of.floor("h")]
    pairs, excluded = make_pairs(build_features(raw), start)
    if pairs.empty or pairs.target.nunique() < 2:
        raise ValueError("Insufficient OOS observations")
    previous, current, target = (pairs[col].to_numpy(dtype=int) for col in ["previous", "state", "target"])
    baseline = model.T[previous, current]
    adjusted = np.vstack([model.next_probs(i, j, int(a) + 1, .5) for i, j, a in zip(previous, current, pairs.age)])
    for p in [baseline, adjusted]:
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("Invalid forecast probabilities")
    delta, intervals, accepted = bootstrap(adjusted, baseline, target, current, pairs.index.date, args.resamples)
    result = {
        "DCMR": score(adjusted, target, current), "second_order": score(baseline, target, current),
        "DCMR_minus_second_order": delta, "paired_day_bootstrap_95pct_intervals": intervals,
        "bootstrap_resamples": args.resamples, "accepted_bootstrap_resamples": accepted, "bootstrap_seed": 42,
        "different_top_predictions": int((adjusted.argmax(axis=1) != baseline.argmax(axis=1)).sum()),
        "training_rows": len(train), "training_end": str(train.timestamp.max()),
        "test_first_origin": str(pairs.index.min()), "test_last_target": str(pairs.target_timestamp.max()),
        "excluded_non_hourly_pairs": excluded,
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            "training": args.train_parquet, "transitions": args.transitions,
            "age_functions": args.age_functions, "yahoo_snapshot": args.yahoo_csv}.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    monthly = []
    for month in sorted(pairs.index.strftime("%Y-%m").unique()):
        ix = pairs.index.strftime("%Y-%m") == month
        for name, p in [("DCMR", adjusted), ("second_order", baseline)]:
            monthly.append({"month": month, "model": name, **score(p[ix], target[ix], current[ix])})
    pd.DataFrame(monthly).to_csv(args.output_dir / "monthly.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
