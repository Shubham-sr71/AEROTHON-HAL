"""
train.py
========
End-to-end pipeline:
  1. Load train/test/ground_truth CSVs
  2. Calibrate combustor physics (physics.calibrate)
  3. Fit HealthEstimator + ResidualCorrector (model.CombustorHybridModel.fit)
  4. Evaluate: physics-only vs pure-ML-only vs hybrid, on the held-out test set
  5. Evaluate the health estimator against ground truth
  6. Save the model, metrics.json, and predictions for the dashboard

Run:  python3 train.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import physics
import model as M

RAW_FEATURES = [
    "Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min",
    "FuelFlow_kg_s", "P2_Pa", "T2_K", "P3_Pa", "Cycle",
]


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    gt = pd.read_csv("ground_truth.csv")

    print(f"train: {train.shape}, test: {test.shape}, ground_truth: {gt.shape}")

    # ---------------- 1) fit the hybrid physics+ML twin ----------------
    hybrid, calib_diag = M.CombustorHybridModel.fit(train, gt)
    print("Calibration diagnostics:", json.dumps(calib_diag, indent=2))

    # ---------------- 2) benchmark: pure-ML baseline (no physics) ------
    pure_ml = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    pure_ml.fit(train[RAW_FEATURES], train["T3_K"])

    # ---------------- 3) evaluate on test.csv ---------------------------
    test_gt = test.merge(gt[["EngineID", "Cycle", "CombustorHealth"]], on=["EngineID", "Cycle"], how="left")

    hybrid_out = hybrid.predict(test)
    physics_only_pred = hybrid_out["T3_phys"].values
    hybrid_pred = hybrid_out["T3_hybrid_pred"].values
    pure_ml_pred = pure_ml.predict(test[RAW_FEATURES])

    y_true = test["T3_K"].values
    metrics = {
        "physics_only": M.eval_metrics(y_true, physics_only_pred),
        "pure_ml": M.eval_metrics(y_true, pure_ml_pred),
        "hybrid_physics_ml": M.eval_metrics(y_true, hybrid_pred),
    }
    print(json.dumps(metrics, indent=2))

    # ---------------- 4) health estimator quality ------------------------
    health_pred_test = hybrid.estimate_health(test)
    health_metrics = M.eval_metrics(test_gt["CombustorHealth"].values, health_pred_test)
    print("Health estimator metrics:", json.dumps(health_metrics, indent=2))

    # ---------------- 5) feature importances (residual model) ------------
    fi = dict(zip(M.RESIDUAL_FEATURES, hybrid.residual_model.feature_importances_.round(4)))
    fi = dict(sorted(fi.items(), key=lambda kv: -kv[1]))

    fi_health = dict(zip(M.HEALTH_FEATURES, hybrid.health_model.feature_importances_.round(4)))
    fi_health = dict(sorted(fi_health.items(), key=lambda kv: -kv[1]))

    # ---------------- 6) save model + outputs -----------------------------
    hybrid.save("combustor_twin")

    out = test.copy()
    out["CombustorHealth_true"] = test_gt["CombustorHealth"]
    out["CombustorHealth_est"] = health_pred_test
    out["T3_true"] = y_true
    out["T3_physics_only"] = physics_only_pred
    out["T3_pure_ml"] = pure_ml_pred
    out["T3_hybrid"] = hybrid_pred
    out["theta_loading_param"] = hybrid_out["theta_loading_param"].values
    out["eta_lefebvre"] = hybrid_out["eta_lefebvre"].values
    out["eta_combustion"] = hybrid_out["eta_combustion"].values
    out["m_air_proxy"] = hybrid_out["m_air_proxy"].values
    out.to_csv("test_predictions.csv", index=False)

    # degradation trend: population avg CombustorHealth vs Cycle (true vs est)
    trend = out.groupby("Cycle").agg(
        health_true_mean=("CombustorHealth_true", "mean"),
        health_est_mean=("CombustorHealth_est", "mean"),
    ).reset_index()

    # engine-level example trajectories for dashboard (pick a few engines)
    example_engines = [1, 25, 50, 75, 100]
    traj = out[out.EngineID.isin(example_engines)][
        ["EngineID", "Cycle", "CombustorHealth_true", "CombustorHealth_est",
         "T3_true", "T3_hybrid", "T3_physics_only"]
    ].sort_values(["EngineID", "Cycle"])

    # Lefebvre efficiency curve, sorted by theta, for plotting
    curve_df = out[["theta_loading_param", "eta_lefebvre", "eta_combustion"]].sort_values("theta_loading_param")
    curve_sample = curve_df.iloc[::max(1, len(curve_df) // 300)]  # thin to ~300 points

    metrics_payload = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_engines": int(train.EngineID.nunique()),
        "calibration_diagnostics": calib_diag,
        "t3_metrics": metrics,
        "health_metrics": health_metrics,
        "residual_feature_importance": fi,
        "health_feature_importance": fi_health,
        "physics_params": {
            "mair_a": hybrid.physics_params.mair_a,
            "mair_b": hybrid.physics_params.mair_b,
            "mair_c": hybrid.physics_params.mair_c,
            "mair_logC": hybrid.physics_params.mair_logC,
            "lef_c": hybrid.physics_params.lef_c,
            "lef_k": hybrid.physics_params.lef_k,
            "health_slope": hybrid.physics_params.health_slope,
            "health_intercept": hybrid.physics_params.health_intercept,
        },
        "degradation_trend": trend.to_dict(orient="list"),
        "example_trajectories": traj.to_dict(orient="records"),
        "lefebvre_curve_sample": curve_sample.to_dict(orient="list"),
        "scatter_sample": out.sample(min(600, len(out)), random_state=1)[
            ["T3_true", "T3_hybrid", "T3_physics_only", "T3_pure_ml", "CombustorHealth_true"]
        ].to_dict(orient="list"),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved: combustor_twin_physics.json, combustor_twin_health_model.joblib, "
          "combustor_twin_residual_model.joblib, metrics.json, test_predictions.csv")


if __name__ == "__main__":
    main()
