import pandas as pd
import torch
import yaml
from pathlib import Path
from ctgan import CTGAN

# --------------------------
# Load CONFIG (portable)
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "params.yaml"

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)

INPUT_CSV = Path(params["generate_icd"]["input"])
OUTPUT_PATH = Path(params["generate_icd"]["output"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_PATH / "synthetic_icd.csv"

EPOCHS = params["generate_icd"].get("epochs", 10)
BATCH_SIZE = params["generate_icd"].get("batch_size", 100)

# --------------------------
# Load Data
# --------------------------
df = pd.read_csv(INPUT_CSV, dtype=str)
print(f"📂 Loaded data: {df.shape}")

# --------------------------
# Features
# --------------------------
features = [
    "SEX_CODE",
    "PAT_AGE",
    "RACE",
    "ETHNICITY",
    "PAT_ZIP",
    "PAT_COUNTY",
    "PUBLIC_HEALTH_REGION",
    "APR_MDC"   # ✅ included as covariate
]

target_col = "PRINC_DIAG_CODE"
columns = features + [target_col]

df = df[columns].dropna()

# Convert to categorical
for col in columns:
    df[col] = df[col].astype("category")

# --------------------------
# Train CTGAN
# --------------------------
print("🚀 Training CTGAN...")

ctgan = CTGAN(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    embedding_dim=64,
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    verbose=True
)

ctgan.fit(df, discrete_columns=columns)

print("✅ Training complete!")

# --------------------------
# Generate Synthetic Data
# --------------------------
N_SAMPLES = params["generate_icd"].get("n_samples", 5000)

synthetic = ctgan.sample(N_SAMPLES)

print(f"✅ Generated synthetic data: {synthetic.shape}")

# --------------------------
# Save
# --------------------------
synthetic.to_csv(OUTPUT_CSV, index=False)

print(f"💾 Saved to: {OUTPUT_CSV}")
