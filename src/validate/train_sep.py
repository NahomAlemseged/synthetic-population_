import os
import pandas as pd
import numpy as np
import joblib
import yaml
import mlflow
import mlflow.sklearn
import torch

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)

# GPU / CPU detection
GPU_AVAILABLE = torch.cuda.is_available()
device = "GPU" if GPU_AVAILABLE else "CPU"
print(f"⚡ Using device: {device}")

# ML model
from xgboost import XGBClassifier


# =========================================================
# LOAD CONFIG
# =========================================================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as file:
    params = yaml.safe_load(file)


# =========================================================
# TRAIN CLASS
# =========================================================

class TrainSynth:

    def __init__(self):

        # INPUT PATHS
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

        # OUTPUT PATH
        self.output_path = Path(
            params["train"]["output"][0]
            if isinstance(params["train"]["output"], list)
            else params["train"]["output"]
        )

        # self.output_path.mkdir(parents=True, exist_ok=True)

        # self.model_file = self.output_path / "model.pkl"

        # TARGET COLUMN
        self.target_col = 'APR_MDC'

    # =====================================================
    # ENCODE FUNCTION
    # =====================================================

    def split_encode(self, df, target_col):

        df = df.copy()

        # Split X and y
        X = df.drop(columns=[target_col])
        if 'PRINC_DIAG_CODE' in X.columns:
          X.drop(columns = 'PRINC_DIAG_CODE')
        y = df[target_col]

        df.dropna(inplace = True)


        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=["object"]).columns

        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        return X, y, encoders

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    def train_model(self):

        # -----------------------------
        # LOAD DATASETS
        # -----------------------------

        df_train = pd.read_csv(self.input_path_train, low_memory=False)
        print(f">>> Loaded TRAIN dataset: {len(df_train)} rows")
        if 'PRINC_DIAG_CODE' in df_train.columns:
          df_train.drop(columns = 'PRINC_DIAG_CODE')

        df_test = pd.read_csv(self.input_path_test, low_memory=False)
        print(f">>> Loaded TEST dataset: {len(df_test)} rows")
        if 'PRINC_DIAG_CODE' in df_test.columns:
          df_test.drop(columns = 'PRINC_DIAG_CODE')

        df_synth = pd.read_csv(self.input_path_synth, low_memory=False)
        print(f">>> Loaded SYNTH dataset: {len(df_synth)} rows")

        if 'PRINC_DIAG_CODE' in df_synth.columns:
          df_synth.drop(columns = 'PRINC_DIAG_CODE')

        df_train = df_train.sample(frac=0.5, random_state=42)
        print(f">>> Sampled TRAIN dataset: {len(df_train)} rows")

        df_test = df_test.sample(frac=0.5, random_state=42)
        print(f">>> Sampled TEST dataset: {len(df_test)} rows")

        df_synth = df_synth.sample(frac=0.5, random_state=42)
        print(f">>> Sampled SYNTH dataset: {len(df_synth)} rows")

        # -----------------------------
        # ENCODE REAL DATA
        # -----------------------------

        X_train, y_train, _ = self.split_encode(
            df_train,
            self.target_col
        )

        X_test, y_test, _ = self.split_encode(
            df_test,
            self.target_col
        )

        # -----------------------------
        # ENCODE SYNTHETIC DATA
        # -----------------------------

        X_synth, y_synth, _ = self.split_encode(
            df_synth,
            self.target_col
        )

        # -----------------------------
        # XGBOOST MODEL
        # -----------------------------

        model = XGBClassifier(
            objective="multi:softmax",
            num_class=24,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            device="cuda" if GPU_AVAILABLE else "cpu"
        )

        # =================================================
        # REAL → REAL
        # =================================================

        print("\n==============================")
        print("REAL → REAL")
        print("==============================")

        xgb_real = model.fit(X_train, y_train)

        y_pred_real = xgb_real.predict(X_test)

        real_acc = accuracy_score(y_test, y_pred_real)
        real_f1 = f1_score(y_test, y_pred_real, average="weighted")

        print(f"Accuracy on Real Dataset : {real_acc:.4f}")
        print(f"F1 Score on Real Dataset : {real_f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_real))

        # =================================================
        # SYNTH → REAL
        # =================================================

        print("\n==============================")
        print("SYNTH → REAL")
        print("==============================")

        xgb_synth = model.fit(X_synth, y_synth)

        y_pred_synth = xgb_synth.predict(X_test)

        synth_acc = accuracy_score(y_test, y_pred_synth)
        synth_f1 = f1_score(y_test, y_pred_synth, average="weighted")

        print(f"Accuracy on Synthetic Dataset : {synth_acc:.4f}")
        print(f"F1 Score on Synthetic Dataset : {synth_f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_synth))



        # =================================================
        # SYNTH → SYNTH
        # =================================================
# =================================================
# SYNTH → SYNTH
# =================================================

        print("\n==============================")
        print("SYNTH → SYNTH")
        print("==============================")

        (
            X_train_synth,
            X_test_synth,
            y_train_synth,
            y_test_synth
        ) = train_test_split(
            X_synth,
            y_synth,
            test_size=0.2,
            random_state=42
        )

        print(f"X_train_synth shape: {X_train_synth.shape}")
        print(f"X_test_synth shape: {X_test_synth.shape}")
        print(f"y_train_synth shape: {y_train_synth.shape}")
        print(f"y_test_synth shape: {y_test_synth.shape}")

        xgb_synth_s = model.fit(
            X_train_synth,
            y_train_synth
        )

        y_pred_synth_s = xgb_synth_s.predict(
            X_test_synth
        )

        synth_s_acc = accuracy_score(
            y_test_synth,
            y_pred_synth_s
        )

        synth_s_f1 = f1_score(
            y_test_synth,
            y_pred_synth_s,
            average="weighted"
        )

        print(f"Accuracy on Synthetic Dataset : {synth_s_acc:.4f}")
        print(f"F1 Score on Synthetic Dataset : {synth_s_f1:.4f}")

        print("\nClassification Report:")

        print(
            classification_report(
                y_test_synth,
                y_pred_synth_s,
                zero_division=0
            )
        )
        # =================================================
        # SAVE MODEL
        # =================================================

        # joblib.dump(xgb_synth, self.model_file)

        # print(f"\n✅ Model saved to: {self.model_file}")


# =========================================================
# MAIN
# =========================================================

def main():

    trainer = TrainSynth()
    trainer.train_model()


if __name__ == "__main__":
    main()
