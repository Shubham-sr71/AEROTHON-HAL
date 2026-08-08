"""
model.py
========
Learned components of the Combustor Digital Twin, built on top of physics.py.

Two models are trained:

1. HealthEstimator (RandomForestRegressor)
   Estimates CombustorHealth (0-1 degradation index) from operating
   conditions + cycle count. Used (a) as a standalone prognostics output for
   maintenance / RUL purposes, and (b) internally, to feed a realistic health
   value into the physics energy-balance baseline (instead of assuming a
   perfect population-average trend).

2. ResidualCorrector (GradientBoostingRegressor)
   Learns the residual between the physics baseline T3_phys and the true
   T3_K, using engineered physics features (loading parameter, efficiency,
   FAR, estimated health) plus raw operating conditions. This is what turns
   a decent white-box model into an accurate hybrid twin.

Final prediction:
    T3_hybrid = T3_phys(estimated health) + residual_model(features)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import physics

HEALTH_FEATURES = [
    "Cycle", "Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min",
    "FuelFlow_kg_s", "P2_Pa", "T2_K", "P3_Pa",
]

RESIDUAL_FEATURES = [
    "Cycle", "Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min",
    "FuelFlow_kg_s", "P2_Pa", "T2_K", "P3_Pa",
    "m_air_proxy", "theta_loading_param", "eta_lefebvre", "health_used",
    "eta_combustion", "FAR", "T3_phys",
]


def add_physics_features(df: pd.DataFrame, params: physics.CombustorPhysicsParams,
                          health_series=None) -> pd.DataFrame:
    """Attach all physics-derived columns (health-aware) to a raw operating dataframe."""
    out = df.copy()
    m_air = physics.air_mass_flow(out.P2_Pa, out.RPM_rev_min, out.T2_K, params)
    theta = physics.loading_parameter(m_air, out.T2_K, out.P3_Pa)
    if health_series is None:
        health = physics.population_health_prior(out.Cycle, params)
    else:
        health = health_series
    eta_b = physics.combustion_efficiency(theta, health, params)
    T3_phys, FAR = physics.t3_energy_balance(out.T2_K, out.FuelFlow_kg_s, m_air, eta_b)

    out["m_air_proxy"] = m_air
    out["theta_loading_param"] = theta
    out["eta_lefebvre"] = physics.eta_lefebvre(theta, params)
    out["health_used"] = health
    out["eta_combustion"] = eta_b
    out["FAR"] = FAR
    out["T3_phys"] = T3_phys
    return out


class CombustorHybridModel:
    """Container for the fitted health estimator, residual corrector, and physics params."""

    def __init__(self, physics_params: physics.CombustorPhysicsParams,
                 health_model: RandomForestRegressor,
                 residual_model: GradientBoostingRegressor):
        self.physics_params = physics_params
        self.health_model = health_model
        self.residual_model = residual_model

    # -- training -----------------------------------------------------
    @classmethod
    def fit(cls, train_df: pd.DataFrame, gt_df: pd.DataFrame):
        merged = train_df.merge(gt_df[["EngineID", "Cycle", "CombustorHealth"]],
                                 on=["EngineID", "Cycle"])

        physics_params, diag = physics.calibrate(merged)

        # --- health estimator: learns CombustorHealth from sensors+cycle ---
        health_model = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        health_model.fit(merged[HEALTH_FEATURES], merged["CombustorHealth"])

        # --- physics features using the estimated (not ground-truth) health,
        #     exactly as the model will operate at deployment time ---
        health_hat_train = health_model.predict(merged[HEALTH_FEATURES])
        feat_df = add_physics_features(merged, physics_params, health_series=health_hat_train)
        residual = merged["T3_K"].values - feat_df["T3_phys"].values

        residual_model = GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        residual_model.fit(feat_df[RESIDUAL_FEATURES], residual)

        model = cls(physics_params, health_model, residual_model)
        return model, diag

    # -- inference ------------------------------------------------------
    def estimate_health(self, df: pd.DataFrame) -> np.ndarray:
        return self.health_model.predict(df[HEALTH_FEATURES])

    def predict(self, df: pd.DataFrame, use_estimated_health=True,
                health_override=None) -> pd.DataFrame:
        """
        Returns a dataframe with physics intermediates, T3_phys, residual
        correction, and final T3_hybrid prediction.
        """
        if health_override is not None:
            health_series = health_override
        elif use_estimated_health:
            health_series = self.estimate_health(df)
        else:
            health_series = None  # falls back to population prior inside add_physics_features

        feat_df = add_physics_features(df, self.physics_params, health_series=health_series)
        resid_pred = self.residual_model.predict(feat_df[RESIDUAL_FEATURES])
        feat_df["T3_residual_correction"] = resid_pred
        feat_df["T3_hybrid_pred"] = feat_df["T3_phys"] + resid_pred
        return feat_df

    # -- persistence ------------------------------------------------------
    def save(self, path_prefix: str):
        self.physics_params.to_json(f"{path_prefix}_physics.json")
        joblib.dump(self.health_model, f"{path_prefix}_health_model.joblib")
        joblib.dump(self.residual_model, f"{path_prefix}_residual_model.joblib")

    @classmethod
    def load(cls, path_prefix: str) -> "CombustorHybridModel":
        pp = physics.CombustorPhysicsParams.from_json(f"{path_prefix}_physics.json")
        hm = joblib.load(f"{path_prefix}_health_model.joblib")
        rm = joblib.load(f"{path_prefix}_residual_model.joblib")
        return cls(pp, hm, rm)


def eval_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100),
    }
