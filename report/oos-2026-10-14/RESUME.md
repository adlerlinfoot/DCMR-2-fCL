**DCMR-2-fCL: Duration- and Conviction-Aware Markov Regime Model**

- Engineered a modular Python research pipeline for front-month WTI crude oil futures, modeling five volatility/trend regimes with a second-order Markov transition tensor and empirical duration adjustments.
- Implemented age-dependent persistence scaling and configurable conviction adjustments to generate regime-transition probabilities and stochastic forward simulations.
- Evaluated frozen historical parameters across 3,241 out-of-sample hourly forecasts and 329 regime changes, improving transition-risk ROC AUC from 0.701 to 0.743 over a simple second-order Markov baseline.
- Benchmarked forecast probabilities using log loss, Brier score, and average precision; quantified the transition-risk ranking gain with a 2,000-resample paired day-block bootstrap (AUC improvement 0.042; 95% interval 0.028–0.058).

Supported by [the evaluation report](README.md) and [aggregate results](evidence/comparison.json).
The reported gain is for neutral-conviction exit-risk ranking. It does not claim
improved top-state predictions, better overall probability scores, validated
learned conviction, or profitable trading decisions.
