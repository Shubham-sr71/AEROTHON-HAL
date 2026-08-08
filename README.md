# Combustor Digital Twin

A physics-informed machine-learning twin of the combustor section of a
turbojet. Given operating/flight conditions it predicts **T3 (turbine inlet
temperature)** — the interface variable your friend's **turbine digital
twin** needs — along with combustor health and degradation diagnostics.

## Why hybrid physics + ML

A pure lookup/ML model fits the training data well but doesn't generalize
outside it and gives no physical insight. A pure physics model is
interpretable but can't capture everything a real combustor does. So this
twin does both:

```
T3_hybrid = T3_physics(energy balance + Lefebvre loading + health) + ML_residual_correction
```

| Model                      | RMSE (K) | MAE (K) | R²     | MAPE  |
|-----------------------------|---------:|--------:|-------:|------:|
| Physics-only (white box)    |   102.6  |   56.2  | 0.9907 | 2.41% |
| Pure ML (no physics)        |    47.6  |   33.1  | 0.9980 | 1.61% |
| **Hybrid (physics + ML)**   | **44.6** | **29.8**| **0.9982** | **1.41%** |

The hybrid model wins on every metric: the physics baseline anchors the
prediction in a real energy balance (so it stays sane outside the training
envelope), and the ML residual mops up everything the simplified physics
can't represent (imperfect flow proxy, sensor noise, higher-order effects).

## Physics inside the twin

**Station numbering** (matches the CSVs): 2 = combustor inlet (compressor
discharge), 3 = combustor exit / turbine inlet (**the predicted quantity**),
4 = turbine exit (your friend's twin's output).

1. **Energy balance**
   `T3 = T2 + FAR · η_b · LHV / (cp_gas·(1+FAR))`, FAR = fuel/air ratio.

2. **Air mass flow** isn't measured directly, so it's estimated with a
   calibrated corrected-flow power law `m_air = C · P2^a · RPM^b · T2^c`,
   fit by solving the energy balance on the labelled training data
   (R² ≈ 0.99 against the implied flow).

3. **Lefebvre loading parameter** (Lefebvre & Ballal):
   `θ = m_air · exp(T2/b) / P3^1.75`, with combustion efficiency saturating
   as `η_lefebvre(θ) = 1 − exp(−c·θ^k)` — the classic aerothermal effect of
   flying lean/high-altitude (low pressure, high loading) on combustion
   efficiency and weak-extinction risk.

4. **Degradation**: combustion efficiency is decomposed as
   `η_b = η_max · CombustorHealth · η_lefebvre(θ)` — a slow-varying
   **health index** (fouling, injector coking, liner wear) multiplied by
   the fast, reversible loading effect. A `RandomForestRegressor`
   (`HealthEstimator`) learns `CombustorHealth` from Cycle count + sensors
   (R² ≈ 0.77 on held-out engines vs the ground-truth health label),
   giving the twin a real-time degradation estimate — useful on its own for
   maintenance/RUL, and used internally to condition the physics baseline.

5. **Residual ML correction**: a `GradientBoostingRegressor` learns
   `T3_true − T3_physics` from the physics intermediates (θ, η, FAR,
   estimated health, m_air) plus raw operating conditions.

All physics constants are *calibrated*, not guessed — see
`physics.calibrate()`, run automatically by `train.py`.

## Files

| File | Purpose |
|---|---|
| `physics.py` | White-box combustor model: energy balance, Lefebvre loading parameter, degradation, calibration routine. No ML. |
| `model.py` | `CombustorHybridModel`: health estimator + residual ML corrector wrapped around the physics core. |
| `train.py` | Runs calibration + training + evaluation on `train.csv`/`test.csv`/`ground_truth.csv`; writes `metrics.json`, `test_predictions.csv`, and the saved model files. |
| `twin.py` | **The file to hand to your friend.** `CombustorTwin` class with `.predict(**conditions)` and `.predict_batch(df)`. |
| `combustor_twin_physics.json` | Calibrated physics constants. |
| `combustor_twin_health_model.joblib` | Trained health estimator. |
| `combustor_twin_residual_model.joblib` | Trained residual corrector. |
| `metrics.json` | All evaluation numbers + data for the dashboard. |
| `test_predictions.csv` | Row-level predictions on the test set for auditing. |
| `requirements.txt` | Python dependencies. |

## Usage

```bash
pip install -r requirements.txt
python3 train.py          # re-calibrates physics + retrains models, writes all artifacts
```

```python
from twin import CombustorTwin

twin = CombustorTwin()  # loads the pre-trained model from this folder

# Single point (e.g. one mission timestep) — this is what your friend's
# turbine twin should call:
out = twin.predict(
    Altitude_m=6000, Mach=0.5, Tamb_K=245, Pamb_Pa=45000,
    RPM_rev_min=55000, FuelFlow_kg_s=1.1,
    P2_Pa=230000, T2_K=400, P3_Pa=220000, Cycle=120,
)
# {'T3_K': 1851.3, 'CombustorHealth_estimated': 0.976,
#  'CombustionEfficiency': 0.975, 'LefebvreLoadingParameter': 4.4e-08,
#  'FuelAirRatio': 0.042, 'AirMassFlowProxy': 25.99, ...}

# Batch mode (a whole flight profile or fleet snapshot):
import pandas as pd
mission = pd.read_csv("mission_profile.csv")
result_df = twin.predict_batch(mission)
```

`result["T3_K"]` (or `result_df["T3_K_predicted"]`) is the value to hand
downstream to the turbine twin as turbine-inlet boundary condition; the
other fields (`CombustorHealth_estimated`, `LefebvreLoadingParameter`,
`CombustionEfficiency`) are exposed so the turbine twin (or a shared
prognostics layer) can also reason about combustor-driven degradation
propagating into the turbine.

## Honest caveats

- `m_air`, `FAR`, and `θ` are engineering **proxies** derived purely from
  sensor data (no combustor geometry/volume was provided) — they carry the
  right physical trends and directionality but are not absolute-unit
  measurements.
- In this dataset, the Lefebvre loading parameter has only weak marginal
  correlation with combustion efficiency once `CombustorHealth` is
  accounted for (`corr(θ, η/Health) ≈ 0.04` — see `metrics.json →
  calibration_diagnostics`); degradation vs. cycle count is the dominant
  driver of T3 drift here. The loading-parameter term is kept because it's
  the mechanistically correct term to carry into the energy balance, and it
  will matter more on data with wider low-pressure/lean excursions.
- The health estimator is trained with `ground_truth.csv` labels available
  in this exercise; in a real deployment without that label, it would need
  to be trained against maintenance records or replaced with an
  unsupervised drift/anomaly signal.
