import argparse
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
from joblib import parallel_backend
import os

# from ctgan import CTGAN   # 🔒 Uncomment if using CTGAN
import time

import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

# Pick train and test CSV separately
TRAIN_CSV = Path(params_["generate"]["input"][0])
TEST_CSV = Path(params_["generate"]["input"][1])

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
    def __init__(self, train_path, test_path):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)

    def train_and_generate(self, synthetic_demographics, target_col="APR_MDC"):
        # Load data
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        # Encode features
        categorical_cols = X_train.select_dtypes(include="object").columns
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = X_test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            encoders[col] = le

        # Encode target
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train.astype(str))
        y_test = target_encoder.transform(y_test.astype(str))

        # =========================
        # Models
        # =========================
        experiments = {
            "xgboost": {
                "model": XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42
                ),
                "params": {"n_estimators": [200, 400]}
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {"n_estimators": [200, 400]}
            },
            "logistic_regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {"C": [0.1, 1, 10]}
            }
        }

        best_model = None
        best_score = 0

        mlflow.set_experiment("APR_MDC_multiclass")

        for name, exp in experiments.items():
            with mlflow.start_run(run_name=name):
                print(f"\n🚀 Training {name}")
                grid = GridSearchCV(exp["model"], exp["params"], scoring="accuracy", cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(classification_report(y_test, y_pred))
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("accuracy", acc)

                if acc > best_score:
                    best_score = acc
                    best_model = grid.best_estimator_

        # =========================
        # Generate synthetic target
        # =========================
        df_demo = synthetic_demographics.copy()
        df_demo = df_demo[X_train.columns]

        for col, le in encoders.items():
            df_demo[col] = df_demo[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        probs = best_model.predict_proba(df_demo)

        preds = [np.random.choice(target_encoder.classes_, p=p) for p in probs]
        df_demo[target_col] = target_encoder.inverse_transform(preds)

        return df_demo

# --------------------------
# Main Execution
# --------------------------
def main():
    print("⚙️ Starting pipeline...")

    df_real = pd.read_csv(TRAIN_CSV, dtype=str)

    if sample_rows:
        df_real = df_real.sample(sample_rows, random_state=42)

    features = [
        "SEX_CODE", "PAT_AGE", "RACE", "ETHNICITY",
        "PAT_ZIP", "PAT_COUNTY", "PUBLIC_HEALTH_REGION"
    ]

    target_col = "APR_MDC"

    target_marginals = {col: df_real[col].value_counts().to_dict() for col in features}

    synth = SyntheticGenerator(TRAIN_CSV)

    print("🔹 Step 1: IPF")
    synthetic_demographics = synth.generate_ipf(df_real, features, target_marginals)

    print("🔹 Step 2: ML Generation")
    ml_gen = GenerateML(TRAIN_CSV, TEST_CSV)

    synthetic_dataset = ml_gen.train_and_generate(synthetic_demographics, target_col)

    print("🔹 Saving...")
    synthetic_dataset.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
    
