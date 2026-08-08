"""
twin.py
=======
Deployment-facing API for the Combustor Digital Twin.

This is the file your friend's turbine digital twin should import. It loads
the trained hybrid physics+ML model and exposes a simple predict() call that
takes combustor operating conditions and returns T3 (turbine inlet
temperature) plus the diagnostic quantities a turbine twin (or a maintenance
dashboard) would want: combustor health, combustion efficiency, and the
Lefebvre loading parameter.

Example
-------
    from twin import CombustorTwin

    twin = CombustorTwin()  # loads combustor_twin_*.joblib/json from this folder

    result = twin.predict(
        Altitude_m=6000, Mach=0.5, Tamb_K=245, Pamb_Pa=45000,
        RPM_rev_min=55000, FuelFlow_kg_s=1.1,
        P2_Pa=230000, T2_K=400, P3_Pa=220000, Cycle=120,
    )
    print(result["T3_K"], result["CombustorHealth_estimated"])

    # Batch / dataframe mode (e.g. a whole flight mission profile):
    import pandas as pd
    mission_df = pd.read_csv("mission_profile.csv")
    out_df = twin.predict_batch(mission_df)
"""
from __future__ import annotations
import pandas as pd

import model as M

REQUIRED_INPUTS = [
    "Altitude_m", "Mach", "Tamb_K", "Pamb_Pa", "RPM_rev_min",
    "FuelFlow_kg_s", "P2_Pa", "T2_K", "P3_Pa", "Cycle",
]


class CombustorTwin:
    def __init__(self, path_prefix: str = "combustor_twin"):
        self.model = M.CombustorHybridModel.load(path_prefix)

    def predict(self, **kwargs) -> dict:
        """
        Single-point prediction. Pass keyword arguments matching
        REQUIRED_INPUTS (EngineID is optional/unused by the model itself).
        Returns a flat dict — ready to hand to a turbine digital twin.
        """
        missing = [k for k in REQUIRED_INPUTS if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        df = pd.DataFrame([{k: kwargs[k] for k in REQUIRED_INPUTS}])
        out = self.model.predict(df)
        row = out.iloc[0]
        return {
            "T3_K": float(row["T3_hybrid_pred"]),
            "T3_physics_only_K": float(row["T3_phys"]),
            "CombustorHealth_estimated": float(row["health_used"]),
            "CombustionEfficiency": float(row["eta_combustion"]),
            "LefebvreLoadingParameter": float(row["theta_loading_param"]),
            "FuelAirRatio": float(row["FAR"]),
            "AirMassFlowProxy": float(row["m_air_proxy"]),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorised prediction for a dataframe of operating points (e.g. a
        full mission or fleet snapshot). `df` must contain REQUIRED_INPUTS
        columns; any extra columns (EngineID, timestamps, ...) are preserved.
        """
        missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        out = self.model.predict(df)
        result = df.copy()
        result["T3_K_predicted"] = out["T3_hybrid_pred"].values
        result["T3_physics_only_K"] = out["T3_phys"].values
        result["CombustorHealth_estimated"] = out["health_used"].values
        result["CombustionEfficiency"] = out["eta_combustion"].values
        result["LefebvreLoadingParameter"] = out["theta_loading_param"].values
        result["FuelAirRatio"] = out["FAR"].values
        return result


if __name__ == "__main__":
    # quick smoke test
    twin = CombustorTwin()
    demo = twin.predict(
        Altitude_m=6000, Mach=0.5, Tamb_K=245, Pamb_Pa=45000,
        RPM_rev_min=55000, FuelFlow_kg_s=1.1,
        P2_Pa=230000, T2_K=400, P3_Pa=220000, Cycle=120,
    )
    print(demo)
