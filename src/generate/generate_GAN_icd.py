import pandas as pd
import yaml
from pathlib import Path
from ctgan import CTGAN

# --------------------------
# Load CONFIG
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "params.yaml"

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)

# Train dataset [0]
TRAIN_PATH = Path(params["generate_icd"]["input"])

# Synthetic population [1]
POP_PATH = Path(params["generate_icd"]["input_population"])

OUTPUT_PATH = Path(params["generate_icd"]["output"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_PATH / "synthetic_population_with_icd.csv"

# --------------------------
# Load Data
# --------------------------
df_train = pd.read_csv(TRAIN_PATH, dtype=str)
df_pop = pd.read_csv(POP_PATH, dtype=str)

print(f"Train: {df_train.shape}")
print(f"Population: {df_pop.shape}")

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
    "APR_MDC"
]

target_col = "PRINC_DIAG_CODE"
columns = features + [target_col]

# --------------------------
# Prepare training data
# --------------------------
df_train = df_train[columns].dropna()

for col in columns:
    df_train[col] = df_train[col].astype("category")

# --------------------------
# Train CTGAN
# --------------------------
print("🚀 Training CTGAN...")

ctgan = CTGAN(
    epochs=10,
    batch_size=100,
    verbose=True
)

ctgan.fit(df_train, discrete_columns=columns)

print("✅ Training done")

# --------------------------
# Generate LARGE synthetic pool
# --------------------------
print("🔹 Generating large ICD pool...")

synthetic_pool = ctgan.sample(len(df_pop) * 5)

# --------------------------
# MATCHING STEP (KEY PART)
# --------------------------
print("🔹 Matching ICD to each row...")

# Keep only needed columns
synthetic_pool = synthetic_pool[columns]

# Merge on key covariates
merge_cols = ["APR_MDC", "SEX_CODE"]

df_merged = df_pop.merge(
    synthetic_pool,
    on=merge_cols,
    how="left"
)

# --------------------------
# Handle missing matches
# --------------------------
missing = df_merged[target_col].isna().sum()
print(f"Missing ICD after merge: {missing}")

if missing > 0:
    print("🔁 Filling missing with random samples...")
    fallback = synthetic_pool[target_col].sample(missing, replace=True).values
    df_merged.loc[df_merged[target_col].isna(), target_col] = fallback

# --------------------------
# Final dataset
# --------------------------
df_final = df_merged

print(f"✅ Final dataset: {df_final.shape}")

# --------------------------
# Save
# --------------------------
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"💾 Saved to {OUTPUT_CSV}")
