import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from twin import CombustorTwin

# ------------------------------------------------------------
# Test your trained Combustor Digital Twin on test (1).csv
# Put this file in the same folder as:
#   twin.py
#   model.py
#   physics.py
#   combustor_twin_physics.json
#   combustor_twin_health_model.joblib
#   combustor_twin_residual_model.joblib
#   test (1).csv
# ------------------------------------------------------------

INPUT_FILE = "test (1).csv"
OUTPUT_FILE = "test_1_predictions.csv"

required = [
    "Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min",
    "FuelFlow_kg_s", "P2_Pa", "T2_K", "P3_Pa", "Cycle"
]

print(f"Reading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Rows: {len(df)}")
print("Columns:", list(df.columns))

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Load the already-trained hybrid physics + ML model
twin = CombustorTwin()

# Predict all rows
result = twin.predict_batch(df)

# If the CSV contains the true T3_K, evaluate the model
if "T3_K" in result.columns:
    y_true = result["T3_K"].to_numpy()
    y_pred = result["T3_K_predicted"].to_numpy()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\n===== MODEL TEST RESULTS =====")
    print(f"RMSE : {rmse:.4f} K")
    print(f"MAE  : {mae:.4f} K")
    print(f"R²   : {r2:.6f}")
    print(f"MAPE : {mape:.4f}%")

    result["T3_error_K"] = result["T3_K_predicted"] - result["T3_K"]
    result["T3_abs_error_K"] = result["T3_error_K"].abs()

# Save predictions
result.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved predictions to: {OUTPUT_FILE}")
print("\nFirst 10 predictions:")
cols = [c for c in [
    "EngineID", "Cycle", "T3_K", "T3_K_predicted",
    "T3_error_K", "T3_abs_error_K", "CombustorHealth_estimated"
] if c in result.columns]
print(result[cols].head(10).to_string(index=False))
