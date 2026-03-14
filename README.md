SHMM-2-fCL: Semi Hidden Markov Model, 2nd Order, for Crude Oil

---

Reproducible pipeline for modeling regime dynamics and forward regime simulation for front‑month WTI Crude Oil (CL) using a second‑order Semi‑Hidden Markov Model with age and conviction‑aware hazard deformation.

Of note, this project is part of an ongoing research series exploring regime-based modeling and market structure in crude oil.

OVERVIEW
This project implements a duration-aware, second-order regime engine designed to model NYMEX WTI Crude Oil (CL). The architecture combines:
- a second-order transition tensor estimated from labeled historical regimes
- age functions encoding empirical run-length behavior
- a conviction field that modulates self-transition probability
- scalar previous-regime encoding for smoother hazard deformation
- a five-regime topology including a dedicated chop regime
Moreover, the model produces both next-state probabilities and next-flip probabilities enabling run-level forecasting rather than bar-level prediction. 


The SHMM-2 forecast takes two primary forms:

A next-state probability forecast that attempts to predict the next 1-hour bar

...and...

A next-regime probability forecast that attempts to predict the next regime.

The model was evaluated using a leakage‑controlled validation harness, including next‑regime accuracy, flip‑only accuracy, and empirical vs simulated persistence comparison. The static transition structure achieves 87.54% next‑state accuracy and 55.61% next-regime accuracy, validating the underlying labeling and tensor estimation. A full five‑layer hyperparameter tuning pipeline calibrates the hazard engine to match empirical run‑length distributions.



---

METHOD SUMMARY
- **Second-Order Transition Tensor:** empirical estimation of P(St|St-1,St-2) across four regimes.  
- **Age Functions:** nonparametric mappings from regime age to baseline hazard.
- **Conviction Field:** learned directional confidence signal used to modulate self-transition strength.  
- **Hazard Deformation:** multiplicative adjustments via calibrated parameters (A, alpha, baseline_scale, hazard_mult, conv_smooth). 
- **Scalar Previous-Regime Encoding:** continuous representation of prior regime identity.
- **Forward Simulation:** full path generation using age and conviction-aware transition probabilities.
- **Validation:** next-state accuracy, next-regime accuracy, empircal vs simulated persistence.
- **Outcome:** calibrated hazard engine producing realistic regime persistence and flip timing.

---

REPRODUCIBILITY
The entire workflow is deterministic and leakage-controlled.  
Run the full pipeline with:
python scripts/pipeline/runner.py 

Outputs (transition tensor, age functions, conviction field, tuning diagnostics, evaluation reports) are saved to: `data/final/` `outputs/`.

---

LICENSE  
MIT License.
