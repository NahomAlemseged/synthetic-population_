import os
import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import torch

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBClassifier

# =========================================================
# DEVICE SETUP
# =========================================================

GPU_AVAILABLE = torch.cuda.is_available()
device = "GPU" if GPU_AVAILABLE else "CPU"
print(f"⚡ Using device: {device}")

# =========================================================
# LOAD CONFIG
# =========================================================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as file:
    params = yaml.safe_load(file)

# =========================================================
# PRIVACY EVALUATION CLASS
# =========================================================

class PrivacyEval:

    def __init__(self):

        self.input_path_train = (
            params["evaluate"]["input"][2]
            if isinstance(params["evaluate"]["input"], list)
            else params["evaluate"]["input"]
        )

        self.input_path_test = (
            params["evaluate"]["input"][1]
            if isinstance(params["evaluate"]["input"], list)
            else params["evaluate"]["input"]
        )

        self.input_path_synth = params["generate_icd"]["input"][1]

        self.target_col = "APR_MDC"

    # =====================================================
    # ENCODING (same logic as your training pipeline)
    # =====================================================

    def split_encode(self, df, target_col):

        df = df.copy()

        # Drop rows with missing values
        df.dropna(inplace=True)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if 'PRINC_DIAG_CODE' in X.columns:
            X = X.drop(columns=['PRINC_DIAG_CODE'])

        encoders = {}

        categorical_cols = X.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        return X, y, encoders

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_data(self):

        df_train = pd.read_csv(self.input_path_train, low_memory=False)
        df_test = pd.read_csv(self.input_path_test, low_memory=False)
        df_synth = pd.read_csv(self.input_path_synth, low_memory=False)

        print(f"Train: {len(df_train)}")
        print(f"Test: {len(df_test)}")
        print(f"Synth: {len(df_synth)}")

        return df_train, df_test, df_synth

    # =====================================================
    # MIA (MEMBERSHIP INFERENCE ATTACK)
    # =====================================================

    def membership_inference_attack(self, X_train, X_test):

        X_in = X_train
        X_out = X_test

        y_in = np.ones(len(X_in))
        y_out = np.zeros(len(X_out))

        X_mia = np.vstack([X_in, X_out])
        y_mia = np.hstack([y_in, y_out])

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_mia, y_mia,
            test_size=0.3,
            random_state=42
        )

        attack_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        attack_model.fit(X_tr, y_tr)

        probs = attack_model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, probs)
        acc = accuracy_score(y_val, (probs > 0.5).astype(int))

        return {
            "mia_auc": float(auc),
            "mia_acc": float(acc)
        }

    # =====================================================
    # NEAREST NEIGHBOR LEAKAGE (NND)
    # =====================================================

    def nearest_neighbor_leakage(self, X_real_train, X_synth):

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X_real_train)

        distances, _ = nn.kneighbors(X_synth)

        distances = distances.flatten()

        leak_threshold = np.percentile(distances, 5)
        leak_rate = np.mean(distances < leak_threshold)

        return {
            "mean_nnd": float(np.mean(distances)),
            "min_nnd": float(np.min(distances)),
            "median_nnd": float(np.median(distances)),
            "leak_rate": float(leak_rate)
        }

    # =====================================================
    # SYNTH QUALITY CHECK (OPTIONAL AUXILIARY)
    # =====================================================

    def synth_quality_check(self, X_synth, y_synth, X_test, y_test):

        model = XGBClassifier(
            objective="multi:softmax",
            num_class=24,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if GPU_AVAILABLE else "cpu",
            random_state=42
        )

        model.fit(X_synth, y_synth)
        preds = model.predict(X_test)

        return {
            "synth_to_real_acc": accuracy_score(y_test, preds),
            "synth_to_real_f1": f1_score(y_test, preds, average="weighted")
        }

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def run(self):

        df_train, df_test, df_synth = self.load_data()

        # encode
        X_train, y_train, _ = self.split_encode(df_train, self.target_col)
        X_test, y_test, _ = self.split_encode(df_test, self.target_col)
        X_synth, y_synth, _ = self.split_encode(df_synth, self.target_col)

        # align columns (important safety step)
        X_synth = X_synth.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # =====================================================
        # 1. MIA
        # =====================================================
        print("\n🔐 Running Membership Inference Attack...")
        mia_results = self.membership_inference_attack(
            X_train.values,
            X_test.values
        )

        # =====================================================
        # 2. NND
        # =====================================================
        print("\n📏 Running Nearest Neighbor Leakage...")
        nnd_results = self.nearest_neighbor_leakage(
            X_train.values,
            X_synth.values
        )

        # =====================================================
        # 3. SYNTH UTILITY CHECK
        # =====================================================
        print("\n🧪 Running synthetic utility check...")
        utility_results = self.synth_quality_check(
            X_synth, y_synth,
            X_test, y_test
        )

        # =====================================================
        # FINAL REPORT
        # =====================================================

        report = {
            **mia_results,
            **nnd_results,
            **utility_results
        }

        print("\n==============================")
        print("🔐 PRIVACY REPORT")
        print("==============================")

        for k, v in report.items():
            print(f"{k}: {v:.4f}")

        # =====================================================
        # MLflow logging (optional but recommended)
        # =====================================================

        try:
            mlflow.log_metrics(report)
        except Exception as e:
            print("MLflow logging skipped:", e)

        return report


# =========================================================
# MAIN
# =========================================================

def main():
    evaluator = PrivacyEval()
    evaluator.run()


if __name__ == "__main__":
    main()
