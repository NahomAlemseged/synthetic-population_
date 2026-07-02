import pandas as pd
import yaml
import numpy as np
import time
from pathlib import Path
from ctgan import CTGAN
from joblib import parallel_backend


# --------------------------
# CONFIG
# --------------------------
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)

TRAIN_PATH = Path(params["evaluate"]["input"][1])   # evaluate[1]
TEST_PATH  = Path(params["evaluate"]["input"][2])   # evaluate[2]
SYNTH_PATH = Path(params["generate_icd"]["input"][1])

OUTPUT_PATH = Path(params["generate"]["output"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_PATH / "synthetic_with_apr_drg_gan.csv"

SAMPLE_SIZE = 50000


# --------------------------
# CTGAN GENERATOR
# --------------------------
class ICDGenerator:

    def __init__(self, train_path, test_path, synth_path):
        self.train_path = train_path
        self.test_path = test_path
        self.synth_path = synth_path

    # --------------------------
    # LOAD DATA
    # --------------------------
    def load_data(self):
        df_train = pd.read_csv(self.train_path, low_memory=False)
        df_test = pd.read_csv(self.test_path, low_memory=False)
        df_syn = pd.read_csv(self.synth_path, low_memory=False)

        for df in [df_train, df_test, df_syn]:
            df.drop(columns=["PRINC_DIAG_CODE"], errors="ignore", inplace=True)

        print(f"📊 Train: {df_train.shape}")
        print(f"📊 Test : {df_test.shape}")
        print(f"📊 Synth: {df_syn.shape}")

        return df_train, df_test, df_syn

    # --------------------------
    # SAFE SAMPLING
    # --------------------------
    def sample(self, df, n):
        n = min(n, len(df))
        return df.sample(n=n, random_state=42)

    # --------------------------
    # PREPARE DATA
    # --------------------------
    def prepare(self, df, features, target_col):
        cols = features + [target_col]

        df = df[cols].dropna().copy()

        for c in cols:
            df[c] = df[c].astype(str)

        return df

    # --------------------------
    # TRAIN CTGAN (FIXED PAC ISSUE)
    # --------------------------
    def train_ctgan(self, df, discrete_cols, epochs=10):

        print(f"🚀 Training CTGAN on {len(df):,} rows")

        pac = 10

        # default batch size
        batch_size = 512

        # 🔥 FIX 1: ensure divisibility by pac
        batch_size = (batch_size // pac) * pac

        # fallback safety
        if batch_size == 0:
            batch_size = pac * 10

        print(f"🧠 Using batch_size={batch_size}, pac={pac}")

        model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )

        with parallel_backend("threading", n_jobs=4):
            model.fit(df, discrete_columns=discrete_cols)

        print("✅ CTGAN training complete")

        return model

    # --------------------------
    # GENERATE DATA
    # --------------------------
    def generate(self, model, n_samples, columns):

        print(f"🔹 Generating {n_samples:,} synthetic rows")

        synth = model.sample(n_samples)

        return synth[columns]


# --------------------------
# MAIN PIPELINE
# --------------------------
def main():

    print("⚙️ CTGAN pipeline started...")
    start_time = time.time()

    # --------------------------
    # FEATURES
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

    target_col = "APR_DRG"
    columns = features + [target_col]

    # --------------------------
    # INIT
    # --------------------------
    gen = ICDGenerator(TRAIN_PATH, TEST_PATH, SYNTH_PATH)

    df_train, df_test, df_syn = gen.load_data()

    # --------------------------
    # SAMPLE TRAINING DATA
    # --------------------------
    df_train = gen.sample(df_train, SAMPLE_SIZE)

    # --------------------------
    # PREPARE DATA
    # --------------------------
    df_train_ctgan = gen.prepare(df_train, features, target_col)

    # CTGAN treats everything as categorical
    discrete_cols = columns

    # --------------------------
    # TRAIN MODEL
    # --------------------------
    ctgan = gen.train_ctgan(
        df_train_ctgan,
        discrete_cols,
        epochs=10
    )

    # --------------------------
    # GENERATE SYNTHETIC POPULATION
    # --------------------------
    pool_size = min(len(df_syn), SAMPLE_SIZE)

    synthetic = gen.generate(
        ctgan,
        pool_size,
        columns
    )

    synthetic = synthetic.dropna().reset_index(drop=True)

    print(f"📊 Final synthetic shape: {synthetic.shape}")

    # --------------------------
    # SAVE OUTPUT
    # --------------------------
    synthetic.to_csv(OUTPUT_CSV, index=False)

    print(f"💾 Saved: {OUTPUT_CSV}")
    print(f"⏱️ Time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()
