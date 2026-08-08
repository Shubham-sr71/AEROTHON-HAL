"""
====================================================================
dataset.py

Dataset for Physics-Informed Turbine Transformer

Input
-----
Sequence of 50 cycles

Output
------
Turbine Health
Measured T4

====================================================================
"""

import pandas as pd
import torch

from torch.utils.data import Dataset

from physics_model import build_physics_model


class TurbineDataset(Dataset):

    def __init__(

        self,

        csv_file,

        window_size=50,

        stride=1,

        flight_hours_per_cycle=1.5,

        max_engines=None,

        max_rows_per_engine=None,

    ):

        self.window_size = window_size
        self.stride = stride
        self.samples = []
        self.max_engines = max_engines
        self.max_rows_per_engine = max_rows_per_engine

        df = pd.read_csv(csv_file)

        ############################################################
        # Process every engine separately
        ############################################################

        engine_ids = df["EngineID"].unique()

        if self.max_engines is not None:

            engine_ids = engine_ids[: self.max_engines]

        for engine_id in engine_ids:

            engine_df = (

                df[df["EngineID"] == engine_id]

                .sort_values("Cycle")

                .reset_index(drop=True)

            )

            physics = build_physics_model()

            history = []

            flight_hours = 0.0

            ########################################################

            for row_idx, (_, row) in enumerate(engine_df.iterrows()):

                if self.max_rows_per_engine is not None and row_idx >= self.max_rows_per_engine:

                    break

                flight_hours += flight_hours_per_cycle

                ####################################################
                # Physics model input
                ####################################################

                physics_input = {

                    "RPM": torch.tensor(row["RPM_rev_min"], dtype=torch.float32),

                    "FuelFlow": torch.tensor(row["FuelFlow_kg_s"], dtype=torch.float32),

                    "P3": torch.tensor(row["P3_Pa"], dtype=torch.float32),

                    "T3": torch.tensor(row["T3_K"], dtype=torch.float32),

                    "P4": torch.tensor(row["P4_Pa"], dtype=torch.float32),

                    "P5": torch.tensor(row["P4_Pa"] * 0.9, dtype=torch.float32),

                    "T4": torch.tensor(row["T4_K"], dtype=torch.float32),

                    "Tcool": torch.tensor(row["Tamb_K"], dtype=torch.float32),

                    "Pcool": torch.tensor(row["Pamb_Pa"], dtype=torch.float32),

                    "mdot_core": torch.tensor(row["FuelFlow_kg_s"], dtype=torch.float32),

                    "mdot_cooling": torch.tensor(row["FuelFlow_kg_s"] * 0.05, dtype=torch.float32),

                    "flight_hours": torch.tensor(flight_hours_per_cycle, dtype=torch.float32),

                }

                ####################################################
                # Run physics model
                ####################################################

                state, outputs = physics.step(physics_input)

                ####################################################
                # Physics residual
                ####################################################

                residual = (

                    torch.tensor(float(row["T4_K"]), dtype=torch.float32)

                    -

                    outputs["predicted_T5"].detach()

                )

                ####################################################
                # Feature vector
                ####################################################

                feature = [

                    # Measured engine data
                    float(row["RPM_rev_min"]),
                    float(row["FuelFlow_kg_s"]),
                    float(row["P3_Pa"]),
                    float(row["T3_K"]),
                    float(row["P4_Pa"]),
                    float(row["Tamb_K"]),
                    float(row["Pamb_Pa"]),

                    # Physics outputs
                    float(outputs["blade_temperature"].detach().item()),
                    float(outputs["tbc_thickness"].detach().item()),
                    float(outputs["tgo_thickness"].detach().item()),
                    float(outputs["creep_strain"].detach().item()),
                    float(outputs["tmf_damage"].detach().item()),
                    float(outputs["tip_clearance"].detach().item()),
                    float(outputs["turbine_efficiency"].detach().item()),
                    float(outputs["predicted_T5"].detach().item()),

                    # Residual
                    float(residual.detach().item()),

                    # Time
                    float(row["Cycle"] / 100.0),
                    float(flight_hours / 150.0),

                ]

                history.append(feature)

            ########################################################
            # Build windows
            ########################################################

            for start in range(

                0,

                len(history) - window_size + 1,

                stride,

            ):

                end = start + window_size

                window = history[start:end]

                target_row = engine_df.iloc[end - 1]

                self.samples.append(

                    {

                        "sequence": torch.tensor(

                            window,

                            dtype=torch.float32,

                        ),

                        "health": torch.tensor(

                            target_row["TurbineHealth"],

                            dtype=torch.float32,

                        ),

                        "measured_T4": torch.tensor(

                            target_row["T4_K"],

                            dtype=torch.float32,

                        ),

                        "flight_hours": torch.tensor(

                            float(flight_hours),

                            dtype=torch.float32,

                        ),

                        "engine_id": int(engine_id),

                        "cycle": int(target_row["Cycle"]),

                    }

                )

        print(f"Created {len(self.samples)} samples.")

    ############################################################

    def __len__(self):

        return len(self.samples)

    ############################################################

    def __getitem__(self, idx):

        return self.samples[idx]