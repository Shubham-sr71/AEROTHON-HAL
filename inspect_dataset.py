from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent

candidate_paths = [
    base_dir / "data" / "engine_dataset.csv",
    base_dir.parent / "PS2_dataset" / "engine_dataset.csv",
    base_dir.parent / "engine_dataset.csv",
    Path("PS2_dataset/engine_dataset.csv"),
    Path("engine_dataset.csv"),
]

data_path = next((path for path in candidate_paths if path.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        "Could not find engine_dataset.csv. Checked: "
        + ", ".join(str(path) for path in candidate_paths)
    )

df = pd.read_csv(data_path)

print(df.head())
print(df.shape)
print(df.columns)
