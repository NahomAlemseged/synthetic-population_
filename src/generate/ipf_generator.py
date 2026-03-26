import argparse
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
from joblib import parallel_backend

# from ctgan import CTGAN   # 🔒 Uncomment if using CTGAN
import time

import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier   # 🔒 optional
from sklearn.linear_model import LogisticRegression   # 🔒 optional


# --------------------------
# Command line arguments
# --------------------------
parser = argparse.ArgumentParser(description="Synthetic population generation")
parser.add_argument("--n_samples", type=int, required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--num_processes", type=int, default=1)
parser.add_argument("--sample_rows", type=int, default=None)
args = parser.parse_args()

n_samples = args.n_samples
epochs = args.epochs
num_processes = args.num_processes
sample_rows = args.sample_rows


# --------------------------
# Load YAML config
# --------------------------
CONFIG_PATH = Path("config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

INPUT_CSV = Path(params_["generate"]["input"])
OUTPUT_PATH = Path(params_["generate"]["output"])
OUTPUT_CSV = OUTPUT_PATH / "synthetic_emergency.csv"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(num_processes)


# --------------------------
# Synthetic Generator (IPF)
# --------------------------
class SyntheticGenerator:
    def __init__(self, input_csv):
        self.input_csv = input_csv

    def generate_ipf(self, df, features, target_marginals, tol=1e-5, max_iter=100):
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

        synthetic = df.sample(
            n=min(n_samples, len(df)),
            weights="weight",
            replace=True,
            random_state=42
        ).drop(columns=["weight"])

        if "APR_MDC" in synthetic.columns:
            synthetic = synthetic.drop(columns=["APR_MDC"])

        return synthetic


# --------------------------
# ML Generator (ACTIVE)
# --------------------------
class GenerateML:
    def __init__(self, synthetic_demographics, target_col="APR_MDC"):
        self.input_path = Path(params_['generate']['input'])
        self.output_path = Path(params_['generate']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_file = self.output_path / "model.pkl"

    def train_and_generate(self, synthetic_demographics, target_col="APR_MDC"):
        # Load training CSV
        df = pd.read_csv(self.input_path, low_memory=False)
        print(f">>> Loaded {len(df)} rows for training")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include='object').columns
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        n_classes = len(np.unique(y))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Define pipeline
        pipeline = Pipeline([
            ("model", XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                enable_categorical=True,
                random_state=42
            ))
        ])
        params = {"model__n_estimators": [200, 400]}

        # Grid search
        grid = GridSearchCV(
            pipeline, params, scoring="f1_weighted", cv=3, n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)

        # Evaluate
        y_pred = grid.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Predict APR_MDC for synthetic demographics
        df_generated = synthetic_demographics.copy()
        
        # Encode categorical columns in synthetic data using training encoders
        for col in categorical_cols:
            if col in df_generated.columns:
                df_generated[col] = df_generated[col].astype(str)
                df_generated[col] = encoders[col].transform(df_generated[col])

        # Convert remaining object columns to category (safe for XGB)
        for col in df_generated.select_dtypes(include='object').columns:
            df_generated[col] = df_generated[col].astype('category')

        df_generated[target_col] = grid.predict(df_generated[X_train.columns])

        return df_generated


# --------------------------
# CTGAN (COMMENTED FOR LATER)
# --------------------------
"""
class CTGANGenerator:

    def train(self, df, features, target_col="APR_MDC"):
        columns = features + [target_col]

        for col in columns:
            df[col] = df[col].astype("category")

        ctgan = CTGAN(epochs=10)
        ctgan.fit(df[columns], discrete_columns=columns)

        return ctgan

    def generate(self, ctgan, n):
        return ctgan.sample(n)
"""


# --------------------------
# Main Execution
# --------------------------
def main():
    print("⚙️ Starting synthetic data generation pipeline...")

    df_real = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"📂 Loaded real dataset with {len(df_real):,} rows")

    if sample_rows:
        df_real = df_real.sample(sample_rows, random_state=42)
        print(f"⚡ Using subset of {len(df_real):,} rows for faster generation")

    features = [
        "SEX_CODE", "PAT_AGE", "RACE", "ETHNICITY",
        "PAT_ZIP", "PAT_COUNTY", "PUBLIC_HEALTH_REGION"
    ]
    target_col = "APR_MDC"
    target_marginals = {col: df_real[col].value_counts().to_dict() for col in features}

    # Step 1: IPF demographics
    synth_gen = SyntheticGenerator(INPUT_CSV)
    synthetic_demographics = synth_gen.generate_ipf(df_real, features, target_marginals)

    # Step 2: ML generation of APR_MDC
    ml_gen = GenerateML(synthetic_demographics)
    synthetic_dataset = ml_gen.train_and_generate(synthetic_demographics, target_col=target_col)

    # Step 3: Save results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    synthetic_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Synthetic dataset saved at: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
