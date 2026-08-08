import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapping import PhysicsMapping
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


def load_transformer(weights_path=None):
    # transformer expects an extra residual channel appended
    model = TurbineTransformer(input_dim=len(FEATURE_COLUMNS) + 1, sequence_length=8)
    if weights_path is None:
        weights_path = Path(__file__).resolve().parent / "turbine_transformer.pt"
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        model_state = model.state_dict()
        adapted = {}
        for k, v in ckpt.items():
            if k in model_state and v.shape != model_state[k].shape:
                # attempt safe resize by copying overlapping slices
                target = model_state[k]
                try:
                    new = target.clone()
                    # determine overlapping dims
                    if v.ndim == 2:
                        r = min(v.shape[0], new.shape[0])
                        c = min(v.shape[1], new.shape[1])
                        new[:r, :c] = v[:r, :c]
                    elif v.ndim == 3:
                        d0 = min(v.shape[0], new.shape[0])
                        d1 = min(v.shape[1], new.shape[1])
                        d2 = min(v.shape[2], new.shape[2])
                        new[:d0, :d1, :d2] = v[:d0, :d1, :d2]
                    elif v.ndim == 1:
                        n = min(v.shape[0], new.shape[0])
                        new[:n] = v[:n]
                    else:
                        # fallback: skip copying and use target
                        new = target
                    adapted[k] = new
                    print(f"Adapted checkpoint param {k}: {v.shape} -> {tuple(new.shape)}")
                except Exception:
                    adapted[k] = model_state[k]
                    print(f"Warning: could not adapt param {k}, using default shape {tuple(model_state[k].shape)}")
            else:
                adapted[k] = v

        # load adapted state dict non-strictly so new params remain initialized
        model.load_state_dict(adapted, strict=False)
    model.eval()
    return model


def build_inputs_from_row(row):
    return {
        "RPM": torch.tensor(float(row["RPM_rev_min"]), dtype=torch.float32),
        "FuelFlow": torch.tensor(float(row["FuelFlow_kg_s"]), dtype=torch.float32),
        "P3": torch.tensor(float(row["P3_Pa"]), dtype=torch.float32),
        "T3": torch.tensor(float(row["T3_K"]), dtype=torch.float32),
        "P4": torch.tensor(float(row["P4_Pa"]), dtype=torch.float32),
        "P5": torch.tensor(float(row["P4_Pa"] * 0.9), dtype=torch.float32),
        "T4": torch.tensor(float(row["T4_K"]), dtype=torch.float32),
        "Tcool": torch.tensor(float(row["Tamb_K"]), dtype=torch.float32),
        "Pcool": torch.tensor(float(row["Pamb_Pa"]), dtype=torch.float32),
        "mdot_core": torch.tensor(float(row["FuelFlow_kg_s"]), dtype=torch.float32),
        "mdot_cooling": torch.tensor(float(row["FuelFlow_kg_s"] * 0.05), dtype=torch.float32),
        "flight_hours": torch.tensor(1.5, dtype=torch.float32),
    }


def trace_engine(engine_df, model, mapper, window_size=8):
    physics = build_physics_model()
    rows = []

    for idx in range(window_size - 1, len(engine_df)):
        window_df = engine_df.iloc[idx - window_size + 1:idx + 1]
        # first run nominal physics to build residual history (measured T4 - predicted_T5)
        x_feat = torch.tensor(window_df[FEATURE_COLUMNS].to_numpy(dtype="float32"), dtype=torch.float32)
        physics.reset = getattr(physics, "reset", lambda: None)
        residuals = []
        for t in range(window_size):
            inputs = build_inputs_from_row(window_df.iloc[t])
            _, out_nom = physics.step(inputs, corrections=None)
            measured_T4 = torch.tensor(float(window_df.iloc[t]["T4_K"]), dtype=torch.float32)
            pred_T5 = out_nom.get("predicted_T5", out_nom.get("predicted_T5s", torch.tensor(0.0)))
            residuals.append((measured_T4 - pred_T5).unsqueeze(-1))

        residuals = torch.cat(residuals, dim=0).unsqueeze(-1)
        x = torch.cat([x_feat, residuals], dim=1).unsqueeze(0)

        # MC-dropout: sample transformer multiple times to estimate correction uncertainty
        mc_samples = 16
        corr_samples = []
        model.train()
        with torch.no_grad():
            for _ in range(mc_samples):
                out = model(x)
                corr = mapper.map(out.delta_z)
                corr_samples.append(corr)
        model.eval()

        # stack samples and compute mean corrections
        def stack_field(field_name):
            vals = torch.stack([getattr(c, field_name).squeeze() for c in corr_samples], dim=0)
            return vals.mean(dim=0), vals.std(dim=0)

        gas_mean, gas_std = stack_field("gas_htc_factor")
        ox_mean, ox_std = stack_field("oxidation_factor")
        creep_mean, creep_std = stack_field("creep_factor")
        clear_mean, clear_std = stack_field("clearance_offset")
        eta_mean, eta_std = stack_field("efficiency_offset")

        # build a PhysicsCorrections container using the mean corrections
        from mapping import PhysicsCorrections
        corrections = PhysicsCorrections(
            gas_htc_factor=gas_mean,
            oxidation_factor=ox_mean,
            creep_factor=creep_mean,
            clearance_offset=clear_mean,
            efficiency_offset=eta_mean,
            latent_delta=torch.zeros(4),
        )

        row = engine_df.iloc[idx]
        inputs = build_inputs_from_row(row)
        _, outputs = physics.step(inputs, corrections=corrections)

        rows.append(
            {
                "cycle": int(row["Cycle"]),
                "health": float(row.get("TurbineHealth", 0.0)),
                "efficiency": float(outputs["turbine_efficiency"].detach().cpu().item()),
                "creep": float(outputs["creep_strain"].detach().cpu().item()),
                "clearance": float(outputs["tip_clearance"].detach().cpu().item()),
                "blade_temp": float(outputs["blade_temperature"].detach().cpu().item()),
                "latent_thermal": float(corrections.gas_htc_factor.detach().cpu().item()),
                "latent_material": float(corrections.oxidation_factor.detach().cpu().item()),
                "latent_geometry": float(corrections.creep_factor.detach().cpu().item()),
                "latent_aero": float(corrections.efficiency_offset.detach().cpu().item()),
            }
        )

    return pd.DataFrame(rows)


def make_plots():
    base_dir = Path(__file__).resolve().parent.parent / "PS2_dataset"
    train_df = pd.read_csv(base_dir / "train.csv")
    gt_df = pd.read_csv(base_dir / "ground_truth.csv")
    merged = train_df.merge(gt_df[["EngineID", "Cycle", "TurbineHealth"]], on=["EngineID", "Cycle"], how="inner")

    model = load_transformer()
    mapper = PhysicsMapping()

    engine_ids = sorted(merged["EngineID"].astype(int).unique())[:2]
    traces = []
    for engine_id in engine_ids:
        engine_df = merged[merged["EngineID"] == engine_id].sort_values("Cycle").reset_index(drop=True)
        if len(engine_df) >= 8:
            traces.append(trace_engine(engine_df, model, mapper, window_size=8))

    if not traces:
        raise RuntimeError("No valid engine traces were created")

    trace_df = pd.concat(traces, ignore_index=True)
    trace_df = trace_df.groupby("cycle", as_index=False).mean()

    loss_history = pd.read_csv(Path(__file__).resolve().parent / "training_history.csv") if (Path(__file__).resolve().parent / "training_history.csv").exists() else None

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if loss_history is not None and not loss_history.empty:
        axes[0].plot(loss_history["epoch"], loss_history["loss"], marker="o", color="tab:blue")
        axes[0].set_title("Training Loss")
        axes[0].set_ylabel("MSE")
        axes[0].grid(True, alpha=0.3)

    axes[1].plot(trace_df["cycle"], trace_df["efficiency"], marker="o", color="tab:green", label="Efficiency")
    axes[1].plot(trace_df["cycle"], trace_df["creep"], marker="s", color="tab:red", label="Creep")
    axes[1].set_title("Physics State Trends")
    axes[1].set_ylabel("Value")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(trace_df["cycle"], trace_df["clearance"], marker="^", color="tab:orange", label="Clearance")
    axes[2].plot(trace_df["cycle"], trace_df["health"], marker="D", color="tab:purple", label="Health")
    axes[2].set_title("Clearance and Health")
    axes[2].set_xlabel("Cycle")
    axes[2].set_ylabel("Value")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "digital_twin_state_plot.png")
    print("Saved digital_twin_state_plot.png")

    # Latent-signal bar chart
    latent_values = trace_df[["latent_thermal", "latent_material", "latent_geometry", "latent_aero"]].mean().to_dict()
    plt.figure(figsize=(8, 4))
    plt.bar(latent_values.keys(), latent_values.values())
    plt.title("Average Transformer Latent Corrections")
    plt.ylabel("Correction value")
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "transformer_latent_plot.png")
    print("Saved transformer_latent_plot.png")

    # Direct relation of latent correction to physics states
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(trace_df["latent_thermal"], trace_df["efficiency"], alpha=0.7)
    axes[0, 0].set_title("Efficiency vs Thermal Latent")
    axes[0, 0].set_xlabel("Thermal latent")
    axes[0, 0].set_ylabel("Efficiency")

    axes[0, 1].scatter(trace_df["latent_material"], trace_df["creep"], alpha=0.7)
    axes[0, 1].set_title("Creep vs Material Latent")
    axes[0, 1].set_xlabel("Material latent")
    axes[0, 1].set_ylabel("Creep")

    axes[1, 0].scatter(trace_df["latent_geometry"], trace_df["clearance"], alpha=0.7)
    axes[1, 0].set_title("Clearance vs Geometry Latent")
    axes[1, 0].set_xlabel("Geometry latent")
    axes[1, 0].set_ylabel("Clearance")

    axes[1, 1].scatter(trace_df["latent_aero"], trace_df["efficiency"], alpha=0.7)
    axes[1, 1].set_title("Efficiency vs Aerodynamic Latent")
    axes[1, 1].set_xlabel("Aerodynamic latent")
    axes[1, 1].set_ylabel("Efficiency")

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "latent_physics_relationships.png")
    print("Saved latent_physics_relationships.png")

    # Predicted vs actual health comparison
    health_pred = []
    health_true = []
    physics = build_physics_model()
    for engine_id in sorted(merged["EngineID"].astype(int).unique())[:2]:
        engine_df = merged[merged["EngineID"] == engine_id].sort_values("Cycle").reset_index(drop=True)
        if len(engine_df) < 8:
            continue
        for idx in range(7, len(engine_df)):
            window_df = engine_df.iloc[idx - 7:idx + 1]
            x_feat = torch.tensor(window_df[FEATURE_COLUMNS].to_numpy(dtype="float32"), dtype=torch.float32)
            
            # Compute nominal physics residuals to build residual channel
            physics.reset = getattr(physics, "reset", lambda: None)
            physics.reset()
            residuals = []
            for t in range(8):
                inputs = build_inputs_from_row(window_df.iloc[t])
                _, out_nom = physics.step(inputs, corrections=None)
                measured_T4 = torch.tensor(float(window_df.iloc[t]["T4_K"]), dtype=torch.float32)
                pred_T5 = out_nom.get("predicted_T5", out_nom.get("predicted_T5s", torch.tensor(0.0)))
                residuals.append((measured_T4 - pred_T5).unsqueeze(-1))
            
            residuals = torch.cat(residuals, dim=0).unsqueeze(-1)
            x = torch.cat([x_feat, residuals], dim=1).unsqueeze(0)
            
            with torch.no_grad():
                out = model(x)
                pred = float(out.turbine_health.squeeze(-1).detach().cpu().item())
            true = float(engine_df.iloc[idx]["TurbineHealth"])
            health_pred.append(pred)
            health_true.append(true)

    plt.figure(figsize=(6, 6))
    plt.scatter(health_true, health_pred, alpha=0.7)
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("Actual health")
    plt.ylabel("Predicted health")
    plt.title("Predicted vs Actual Health")
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "predicted_vs_actual_health.png")
    print("Saved predicted_vs_actual_health.png")


if __name__ == "__main__":
    make_plots()
