import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapping import PhysicsCorrections, PhysicsMapping
from physics_model import build_physics_model
from transformer import TurbineTransformer

FEATURE_COLUMNS = [
    "Altitude_m",
    "Mach",
    "Tamb_K",
    "Pamb_Pa",
    "RPM_rev_min",
    "FuelFlow_kg_s",
    "P2_Pa",
    "T2_K",
    "P3_Pa",
    "T3_K",
    "P4_Pa",
    "T4_K",
]


class SequenceDataset(Dataset):
    def __init__(self, df, feature_columns, target_column, window_size, max_engines=None, max_rows_per_engine=None):
        self.window_size = window_size
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.samples = []

        engine_ids = sorted(df["EngineID"].astype(int).unique())
        if max_engines is not None:
            engine_ids = engine_ids[:max_engines]

        for engine_id in engine_ids:
            engine_df = df[df["EngineID"] == engine_id].sort_values("Cycle").reset_index(drop=True)
            if max_rows_per_engine is not None:
                engine_df = engine_df.iloc[:max_rows_per_engine]
            if len(engine_df) < window_size:
                continue

            for start in range(0, len(engine_df) - window_size + 1):
                window = engine_df.iloc[start:start + window_size][feature_columns].to_numpy(dtype="float32")
                target = float(engine_df.iloc[start + window_size - 1][target_column])
                self.samples.append((window, target))

        print(f"Built {len(self.samples)} training windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, target = self.samples[idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the turbine transformer with a physics-informed loss")
    parser.add_argument("--train-csv", type=str, default=str(base_dir.parent / "PS2_dataset" / "train.csv"))
    parser.add_argument("--ground-truth-csv", type=str, default=str(base_dir.parent / "PS2_dataset" / "ground_truth.csv"))
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-engines", type=int, default=None)
    parser.add_argument("--max-rows-per-engine", type=int, default=None)
    return parser.parse_args()


def feature_vector_to_inputs(vec):
    vec = vec.to(dtype=torch.float32)
    return {
        "RPM": vec[4],
        "FuelFlow": vec[5],
        "P3": vec[6],
        "T3": vec[8],
        "P4": vec[10],
        "P5": vec[10] * 0.9,
        "T4": vec[11],
        "Tcool": vec[2],
        "Pcool": vec[3],
        "mdot_core": vec[5],
        "mdot_cooling": vec[5] * 0.05,
        "flight_hours": torch.tensor(1.5, dtype=torch.float32, device=vec.device),
    }


def build_corrections_from_tensor(corrections, idx):
    return PhysicsCorrections(
        gas_htc_factor=corrections.gas_htc_factor[idx],
        oxidation_factor=corrections.oxidation_factor[idx],
        creep_factor=corrections.creep_factor[idx],
        clearance_offset=corrections.clearance_offset[idx],
        efficiency_offset=corrections.efficiency_offset[idx],
        latent_delta=corrections.latent_delta[idx],
    )


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_csv)
    ground_truth = pd.read_csv(args.ground_truth_csv)

    merged = train_df.merge(
        ground_truth[["EngineID", "Cycle", "TurbineHealth"]],
        on=["EngineID", "Cycle"],
        how="inner",
    )

    dataset = SequenceDataset(
        merged,
        feature_columns=FEATURE_COLUMNS,
        target_column="TurbineHealth",
        window_size=args.window_size,
        max_engines=args.max_engines,
        max_rows_per_engine=args.max_rows_per_engine,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Transformer will receive an extra residual channel appended to sensor features
    augmented_input_dim = len(FEATURE_COLUMNS) + 1
    model = TurbineTransformer(input_dim=augmented_input_dim, sequence_length=args.window_size).to(device)
    mapper = PhysicsMapping()
    physics_model = build_physics_model().to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(physics_model.parameters()), lr=args.lr)

    loss_history = []
    model.train()
    physics_model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # ============================================================
            # BUILD PHYSICS RESIDUAL CHANNEL
            # ============================================================

            batch_size = sequences.size(0)
            seq_len = sequences.size(1)

            residual_windows = []

            for b in range(batch_size):

                physics_model.reset()

                sample_residuals = []

                for t in range(seq_len):

                    # ----------------------------------------------------
                    # Convert sensor vector -> physics inputs
                    # ----------------------------------------------------

                    step_inputs = feature_vector_to_inputs(
                        sequences[b, t]
                    )

                    # ----------------------------------------------------
                    # Run nominal physics model
                    # ----------------------------------------------------

                    _, outputs = physics_model.step(
                        step_inputs,
                        corrections=None
                    )

                    # ----------------------------------------------------
                    # Physics residual
                    #
                    # IMPORTANT:
                    # This assumes the physics model returns predicted_T5.
                    # ----------------------------------------------------

                    measured_T4 = sequences[b, t, 11]

                    predicted_T5 = outputs["predicted_T5"]

                    residual = measured_T4 - predicted_T5

                    # Make absolutely sure residual is scalar
                    residual = residual.reshape(())

                    sample_residuals.append(residual)

                # --------------------------------------------------------
                # (sequence,) for this sample
                # --------------------------------------------------------

                sample_residuals = torch.stack(
                    sample_residuals,
                    dim=0
                )

                residual_windows.append(sample_residuals)


            # ============================================================
            # STACK BATCH
            # ============================================================

            # Expected:
            #
            # residual_tensor = (batch, sequence)

            residual_tensor = torch.stack(
                residual_windows,
                dim=0
            )


            # ============================================================
            # ADD CHANNEL DIMENSION
            # ============================================================

            # (batch, sequence)
            #
            # becomes
            #
            # (batch, sequence, 1)

            residual_tensor = residual_tensor.unsqueeze(2)


            # ============================================================
            # SAFETY CHECK
            # ============================================================

            print(
                "DEBUG:",
                "sequences =", sequences.shape,
                "residual =", residual_tensor.shape
            )


            # ============================================================
            # AUGMENT INPUT
            # ============================================================

            sequences_aug = torch.cat(
                [
                    sequences,
                    residual_tensor
                ],
                dim=2
            )


            print(
                "DEBUG:",
                "augmented input =",
                sequences_aug.shape
            )

            # Transformer forward on augmented input
            transformer_out = model(sequences_aug)
            predictions = transformer_out.turbine_health.squeeze(-1)
            corrections = mapper.map(transformer_out.delta_z)

            health_loss = nn.functional.mse_loss(predictions, targets)

            physics_losses = []
            residual_losses = []
            for b in range(batch_size):
                physics_model.reset()
                sample_corrections = build_corrections_from_tensor(corrections, b)
                # after applying corrections through the window, check final physics outputs
                for t in range(seq_len):
                    step_inputs = feature_vector_to_inputs(sequences[b, t])
                    _, outputs = physics_model.step(step_inputs, corrections=sample_corrections)
                # supervise efficiency, creep, clearance as before
                health_scalar = targets[b].clamp(0.0, 1.0)
                eff_target = 0.82 + 0.12 * (1.0 - health_scalar)
                creep_target = 1e-5 * (1.0 - health_scalar) + 1e-6
                clear_target = 5e-4 + 2e-4 * (1.0 - health_scalar)

                physics_losses.append(
                    nn.functional.mse_loss(outputs["turbine_efficiency"], eff_target)
                    + 0.5 * nn.functional.mse_loss(outputs["creep_strain"], creep_target)
                    + 0.5 * nn.functional.mse_loss(outputs["tip_clearance"], clear_target)
                )

                # residual supervision: minimize corrected residual at last timestep
                measured_T4_last = sequences[b, -1, 11]
                pred_T5_corrected = outputs.get("predicted_T5", outputs.get("predicted_T5s", torch.tensor(0.0, device=device)))
                residual_final = measured_T4_last - pred_T5_corrected
                residual_losses.append(nn.functional.mse_loss(residual_final, torch.zeros_like(residual_final)))

            physics_loss = torch.stack(physics_losses).mean() if physics_losses else torch.tensor(0.0, device=device)
            residual_loss = torch.stack(residual_losses).mean() if residual_losses else torch.tensor(0.0, device=device)
            loss = health_loss + 0.25 * physics_loss + 0.5 * residual_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequences.size(0)

        avg_loss = running_loss / len(dataset)
        loss_history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"epoch {epoch + 1}/{args.epochs} loss={avg_loss:.6f}")

    pd.DataFrame(loss_history).to_csv("training_history.csv", index=False)
    torch.save(model.state_dict(), "turbine_transformer.pt")
    print("Saved model to turbine_transformer.pt")


if __name__ == "__main__":
    main()
