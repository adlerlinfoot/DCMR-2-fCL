# DCMR-2-fCL

**Duration- and Conviction-Aware Markov Regime Model, Second Order, for front-month WTI crude oil futures.**

This research pipeline models transitions and simulates paths between five
explicitly labeled price regimes. It combines an empirical second-order
transition tensor with age-dependent and conviction-dependent adjustments to
the probability of remaining in the current regime.

## Model definition

The base tensor estimates `P(S_t | S_(t-2), S_(t-1))` from counts of labeled
historical transitions, with add-one smoothing. The engine then adjusts the
current regime's self-transition weight using its age, a regime-specific
multiplier, and a smoothed conviction score before normalizing the row.
Its effective transition law therefore also depends on age and conviction.

This is a duration-aware Markov regime model, not a hidden Markov model:
states are assigned directly from observed price features. There is no
state-conditioned emission model, latent-state filtering, or hidden-state
sequence inference. Duration dependence is semi-Markov-style, but the code
does not implement a canonical hidden semi-Markov model with explicitly
inferred latent durations.

## Regime segmentation

Regimes are **rule-based**, using manually specified windows and thresholds in
`scripts/data/final_dataset.py`. No k-means, mixture model, geometric cluster
search, or other unsupervised regime discovery is implemented.

| State | Rule |
| --- | --- |
| 0 | Low volatility, positive trend, outside chop |
| 1 | Low volatility, nonpositive trend, outside chop |
| 2 | High volatility, positive trend, outside chop |
| 3 | High volatility, nonpositive trend, outside chop |
| 4 | Absolute normalized drift at most 0.3, overriding volatility/trend |

Volatility is the rolling standard deviation of 24 hourly log returns,
scaled by `sqrt(24)`, with a 0.015 threshold. Trend is the 120-hour percentage
price change. Normalized drift uses cumulative log returns over 72 hours
divided by realized volatility over the same window.

These are researcher-defined semantic categories, not statistically discovered
clusters. Random Forest and Gradient Boosting classify the existing labels
to construct a conviction signal; they do not determine regime boundaries.
Results depend on the selected feature windows and thresholds.

## Duration and conviction adjustments

Age functions are estimated from observed regime run lengths. The ratio
`S(a+1) / S(a)` is a conditional survival probability, not an exit hazard.
The implementation clips this ratio and uses it as a multiplier on an existing
self-transition weight. The normalized result is not the empirical duration
hazard itself; an exit hazard would be `1 - P(stay)`.

The legacy parameter `hazard_mult` multiplies the stay weight: larger positive
values increase persistence, all else equal. Positive `baseline_scale` values
multiply the entire transition row and cancel on normalization, so that
parameter does not change the resulting probabilities in the usual valid case.
These parameter names remain in the API to preserve existing behavior.

## Execution and evaluation

From the repository root:

```bash
python -m scripts.pipeline.runner
```

The pipeline requires local input data and the scientific Python dependencies
imported by the scripts. Data and fitted artifacts are not supplied in this
repository. The runner updates input data, clears output files, builds labels,
fits transitions and age functions, generates neutral conviction, evaluates
the model, and creates an overlay. Learned conviction and parameter tuning are
separate entry points:

```bash
python -m scripts.model.tune_rf_gb
python -m scripts.model.tune_dcmr2
```

Outputs are written to `data/final/` and `outputs/`. Forward simulation is
stochastic unless an explicit seeded generator is supplied to `DCMR2.step`;
the current tuning helper does not supply one. The workflow is not guaranteed
to be deterministic.

The evaluation script reports next-state accuracy and a tensor flip-destination
accuracy. Its legacy `next_regime_accuracy` and `next_direction_accuracy` fields
actually measure mean confidence in the predicted regime/direction on observed
flips, not the fraction of correct predictions. Historical percentages should
not be interpreted as validated out-of-sample accuracy: transition fitting uses
the full supplied dataset, and current evaluation/feature construction needs
an independent temporal leakage audit. In particular, the retrospective
current-state score uses current-state age, and classifier feature selection
does not exclude run-end/run-length information.

## Historical report and naming correction

The [historical research report](report/historical-research-report.pdf) is
preserved unchanged as a record of the original work. Its former “semi-hidden”
terminology and performance/reproducibility claims are superseded by the model
definition and limitations above. Its PDF text has not been revised.

The current class is `DCMR2` in `scripts/model/dcmr2.py`; evaluation and tuning
modules use `eval_dcmr2` and `tune_dcmr2`. Evaluation artifact prefixes are now
`dcmr2_`. Update external imports and commands accordingly.

This project is part of an ongoing research series exploring regime dynamics
and market structure in crude oil. Licensed under MIT.
