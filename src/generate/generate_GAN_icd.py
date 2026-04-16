import pandas as pd
import yaml
from pathlib import Path
from ctgan import CTGAN
from joblib import parallel_backend
import time


# --------------------------
# CONFIG
# --------------------------
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)

train_path = params["generate_icd"]["input"][0]
pop_path = params["generate_icd"]["input"][1]

TRAIN_PATH = Path(train_path)
POP_PATH = Path(pop_path)

OUTPUT_PATH = Path(params["generate_icd"]["output"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_PATH / "synthetic_population_with_icd.csv"

# 🔥 SAMPLE SIZE
N_SAMPLES = 50000


# --------------------------
# ICD Generator Class
# --------------------------
class ICDGenerator:
    def __init__(self, train_path, pop_path):
        self.train_path = train_path
        self.pop_path = pop_path

    def load_data(self):
        print(f"📂 Loading train: {self.train_path}")
        print(f"📂 Loading population: {self.pop_path}")

        if not self.train_path.exists():
            raise FileNotFoundError(f"❌ Train file not found: {self.train_path}")

        if not self.pop_path.exists():
            raise FileNotFoundError(f"❌ Population file not found: {self.pop_path}")

        df_train = pd.read_csv(self.train_path, dtype=str)
        df_pop = pd.read_csv(self.pop_path, dtype=str)

        print(f"📊 Train shape: {df_train.shape}")
        print(f"📊 Population shape: {df_pop.shape}")

        return df_train, df_pop

    def prepare_training_data(self, df_train, features, target_col):
        columns = features + [target_col]

        df_train = df_train[columns].dropna()

        for col in columns:
            df_train[col] = df_train[col].astype("category")

        return df_train

    def train_ctgan(self, df_train, columns, epochs=10):
        print(f"🚀 Training CTGAN on {len(df_train):,} rows...")

        batch_size = 100
        pac = 10

        if batch_size % pac != 0:
            batch_size = batch_size - (batch_size % pac)

        ctgan = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )

        with parallel_backend("threading", n_jobs=4):
            ctgan.fit(df_train, discrete_columns=columns)

        print("✅ CTGAN training complete")
        return ctgan

    def generate_pool(self, ctgan, n_samples):
        print(f"🔹 Generating synthetic ICD pool ({n_samples:,} rows)...")
        return ctgan.sample(n_samples)

    def match_icd(self, df_pop, synthetic_pool, target_col):
        print("🔹 Matching ICD codes...")

        merge_cols = [
            "APR_MDC",
            "SEX_CODE",
            "PAT_AGE",
            "RACE",
            "ETHNICITY"
        ]

        df_merged = df_pop.merge(
            synthetic_pool,
            on=merge_cols,
            how="left"
        )

        missing = df_merged[target_col].isna().sum()
        print(f"⚠️ Missing ICD after merge: {missing}")

        if missing > 0:
            print("🔁 Filling missing with fallback sampling...")
            fallback = synthetic_pool[target_col].sample(
                missing, replace=True, random_state=42
            ).values
            df_merged.loc[df_merged[target_col].isna(), target_col] = fallback

        return df_merged


# --------------------------
# Main Execution
# --------------------------
def main():
    print("⚙️ Starting ICD generation pipeline...")

    start_time = time.time()

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

    generator = ICDGenerator(TRAIN_PATH, POP_PATH)

    # Step 1: Load data
    df_train, df_pop = generator.load_data()

    # 🔥 Step 1.5: SAMPLE 50K POPULATION
    N = min(N_SAMPLES, len(df_pop))
    df_pop = df_pop.sample(n=N, random_state=42)

    print(f"⚡ Using {N:,} population rows")

    # Step 2: Prepare training data
    df_train_prepared = generator.prepare_training_data(
        df_train, features, target_col
    )

    columns = features + [target_col]

    # Step 3: Train CTGAN
    ctgan_model = generator.train_ctgan(
        df_train_prepared,
        columns,
        epochs=10
    )

    # Step 4: Generate synthetic pool (slightly larger than needed)
    synthetic_pool = generator.generate_pool(
        ctgan_model,
        int(N * 2.5)
    )

    synthetic_pool = synthetic_pool[columns]

    # Step 5: Match ICDs
    df_final = generator.match_icd(
        df_pop,
        synthetic_pool,
        target_col
    )

    print(f"✅ Final dataset shape: {df_final.shape}")

    # Step 6: Save
    df_final.to_csv(OUTPUT_CSV, index=False)

    end_time = time.time()

    print(f"💾 Saved to: {OUTPUT_CSV}")
    print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
