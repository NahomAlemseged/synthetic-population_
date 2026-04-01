import argparse
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
from joblib import parallel_backend
from ctgan import CTGAN
import time

# --------------------------
# Command line arguments
# --------------------------
parser = argparse.ArgumentParser(description="Synthetic population generation with CTGAN (no APR_MDC from IPF)")
parser.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate")
parser.add_argument("--epochs", type=int, default=5, help="Number of CTGAN training epochs")
parser.add_argument("--num_processes", type=int, default=1, help="Number of Torch CPU threads")
parser.add_argument("--sample_rows", type=int, default=None, help="Optional: use subset of rows for faster testing")
args = parser.parse_args()

n_samples = args.n_samples
epochs = args.epochs
num_processes = args.num_processes
sample_rows = args.sample_rows

# --------------------------
# Load YAML config
# --------------------------
CONFIG_PATH = Path("/mnt/c/Users/nahomw/Desktop/from_mac/nahomworku/Desktop/uthealth/gra_project/synthetic-population/config/params.yaml")
with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

INPUT_CSV = Path(params_["generate"]["input"])
OUTPUT_PATH = Path(params_["generate"]["output"])
OUTPUT_CSV = OUTPUT_PATH / "synthetic_emergency.csv"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(num_processes)

# --------------------------
# Synthetic Generator Class
# --------------------------
class SyntheticGenerator:
    def __init__(self, input_csv):
        self.input_csv = input_csv

    def generate_ipf(self, df, features, target_marginals, tol=1e-5, max_iter=100):
        """
        Generates only demographic features using IPF — excludes APR_MDC.
        """
        df = df.copy()
        df["weight"] = 1.0

        for iteration in range(max_iter):
            old_weights = df["weight"].copy()
            for feat in features:
                current = df.groupby(feat)["weight"].sum()
                desired = pd.Series(target_marginals[feat])
                ratios = desired / current
                df["weight"] *= df[feat].map(ratios)
            if np.allclose(df["weight"], old_weights, atol=tol):
                print(f"✅ IPF converged at iteration {iteration}")
                break

        synthetic_demographics = df.sample(
            n=min(n_samples, len(df)),
            weights="weight",
            replace=True,
            random_state=42
        ).drop(columns=["weight"])

        # 🚫 Ensure APR_MDC is not included
        if "APR_MDC" in synthetic_demographics.columns:
            synthetic_demographics = synthetic_demographics.drop(columns=["APR_MDC"])
            print("⚠️ Removed APR_MDC column from IPF-generated demographics")

        return synthetic_demographics

    def learn_ctgan(self, df_real, features, target_col="APR_MDC", epochs=5):
        """
        Train CTGAN on features + APR_MDC.
        """
        columns_to_use = features + [target_col]
        df_ctgan = df_real[columns_to_use].copy()

        for col in columns_to_use:
            df_ctgan[col] = df_ctgan[col].astype("category")

        print(f"🚀 Training CTGAN on {len(df_ctgan):,} rows and {len(columns_to_use)} columns...")

        # Ensure batch size is divisible by pac (default pac=10)
        batch_size = 120
        pac = 10
        if batch_size % pac != 0:
            adjusted = batch_size - (batch_size % pac)
            print(f"[INFO] Adjusting batch size from {batch_size} to {adjusted} (must be divisible by pac={pac})")
            batch_size = adjusted
        assert batch_size % pac == 0, f"Batch size {batch_size} must be divisible by pac={pac}"

        ctgan = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=64,
            generator_dim=(128, 128),
            discriminator_dim=(128, 128),
            verbose=True
        )

        with parallel_backend("threading", n_jobs=4):
            ctgan.fit(df_ctgan, discrete_columns=columns_to_use)

        print("✅ CTGAN training complete!")
        return ctgan

    def generate_gan(self, ctgan, synthetic_demographics, target_col="APR_MDC"):
        """
        Generate APR_MDC via CTGAN and merge with IPF demographics.
        """
        df_demo = synthetic_demographics.copy()
        for col in df_demo.columns:
            df_demo[col] = df_demo[col].astype("category")

        synthetic_target = ctgan.sample(len(df_demo))

        # Keep only APR_MDC from GAN, drop duplicates
        if target_col in df_demo.columns:
            df_demo = df_demo.drop(columns=[target_col])

        df_final = pd.concat(
            [df_demo.reset_index(drop=True),
             synthetic_target[[target_col]].reset_index(drop=True)],
            axis=1
        )

        print(f"✅ Final synthetic dataset shape: {df_final.shape}")
        return df_final

# --------------------------
# Main Execution
# --------------------------
# if __name__ == "__main__":
def main():
    print("⚙️ Starting synthetic data generation pipeline...")

    df_real = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"📂 Loaded real dataset with {len(df_real):,} rows")

    # clear nad 

    start_time = time.time()
    if sample_rows:
        df_real = df_real.sample(sample_rows, random_state=42)
        print(f"⚡ Using subset of {len(df_real):,} rows for faster training")

    features = [
        "SEX_CODE", "PAT_AGE", "RACE", "ETHNICITY",
        "PAT_ZIP", "PAT_COUNTY", "PUBLIC_HEALTH_REGION"
    ]
    target_col = "APR_MDC"

    target_marginals = {col: df_real[col].value_counts().to_dict() for col in features}

    synth_gen = SyntheticGenerator(INPUT_CSV)

    print("🔹 Step 1: Generating synthetic demographics (IPF, no APR_MDC)...")
    synthetic_demographics = synth_gen.generate_ipf(df_real, features, target_marginals)

    print("🔹 Step 2: Training CTGAN model (for APR_MDC)...")
    ctgan_model = synth_gen.learn_ctgan(df_real, features, target_col=target_col, epochs=epochs)

    print("🔹 Step 3: Generating APR_MDC via GAN...")
    synthetic_dataset = synth_gen.generate_gan(ctgan_model, synthetic_demographics, target_col=target_col)
    end_time = time.time()
    print("🔹 Step 4: Saving results...")
    synthetic_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"Elapsed time to generate synthetic data is {start_time - end_time}")

    print(f"💾 Synthetic dataset saved at: {OUTPUT_CSV}") 

