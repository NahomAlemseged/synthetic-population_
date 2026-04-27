import pandas as pd
import numpy as np
import joblib
import yaml
import torch

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier

# =========================
# Device
# =========================
DEVICE = "GPU" if torch.cuda.is_available() else "CPU"
print(f"⚡ Using device: {DEVICE}")

# =========================
# Config
# =========================
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")
with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)


class TwoModelTrainer:

    def __init__(self):
        self.syn_path = params['train']['input']   # synthetic dataset
        self.real_path = params['test']['input']   # real dataset
        self.output_path = Path(params['train']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)

    # =========================
    # Preprocessing
    # =========================
    def preprocess(self, df, encoders=None, fit=True):
        if encoders is None:
            encoders = {}

        for col in df.select_dtypes(include='object').columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                if col in encoders:
                    le = encoders[col]
                    df[col] = le.transform(df[col].astype(str))
                else:
                    # unseen column safety
                    df[col] = df[col].astype(str)

        return df, encoders

    # =========================
    # Model
    # =========================
    def get_model(self, n_classes):
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="gpu_hist",
            random_state=42,
            n_estimators=300
        )

    # =========================
    # Train
    # =========================
    def train(self):

        # -------------------------
        # Load data
        # -------------------------
        df_syn = pd.read_csv(self.syn_path, low_memory=False)
        df_real = pd.read_csv(self.real_path, low_memory=False)

        print(f">>> Synthetic rows: {len(df_syn)}")
        print(f">>> Real rows: {len(df_real)}")

        target_mdc = "APR_MDC"
        target_icd = "PRINC_DIAG_CODE"

        # -------------------------
        # Preprocess
        # -------------------------
        df_syn, encoders = self.preprocess(df_syn, fit=True)
        df_real, _ = self.preprocess(df_real, encoders=encoders, fit=False)

        # -------------------------
        # Define feature sets
        # -------------------------
        base_cols = [
            c for c in df_syn.columns
            if c not in [target_mdc, target_icd]
        ]

        # =========================
        # MODEL 1: APR_MDC
        # =========================
        print("\n🚀 Training APR_MDC model (synthetic)")

        X_syn_mdc = df_syn[base_cols]
        y_mdc_syn = df_syn[target_mdc]

        X_real_mdc = df_real[base_cols]
        y_mdc_real = df_real[target_mdc]

        mdc_model = self.get_model(len(np.unique(y_mdc_syn)))
        mdc_model.fit(X_syn_mdc, y_mdc_syn)

        print("\n📊 Evaluating APR_MDC on REAL data")

        y_mdc_pred = mdc_model.predict(X_real_mdc)

        print(classification_report(y_mdc_real, y_mdc_pred))
        print("Accuracy:", accuracy_score(y_mdc_real, y_mdc_pred))
        print("F1:", f1_score(y_mdc_real, y_mdc_pred, average="weighted"))

        # =========================
        # MODEL 2: ICD
        # =========================
        print("\n🚀 Training ICD model (synthetic)")

        # IMPORTANT: APR is part of X
        X_syn_icd = df_syn[base_cols + [target_mdc]]
        y_icd_syn = df_syn[target_icd]

        X_real_icd = df_real[base_cols + [target_mdc]]
        y_icd_real = df_real[target_icd]

        icd_model = self.get_model(len(np.unique(y_icd_syn)))
        icd_model.fit(X_syn_icd, y_icd_syn)

        print("\n📊 Evaluating ICD on REAL data (using REAL APR)")

        y_icd_pred = icd_model.predict(X_real_icd)

        print(classification_report(y_icd_real, y_icd_pred))
        print("Accuracy:", accuracy_score(y_icd_real, y_icd_pred))
        print("F1:", f1_score(y_icd_real, y_icd_pred, average="weighted"))

       

        # =========================
        # Save
        # =========================
        joblib.dump({
            "mdc_model": mdc_model,
            "icd_model": icd_model,
            "encoders": encoders,
            "base_features": base_cols
        }, self.output_path / "two_model_pipeline.pkl")

        print("\n✅ Models saved successfully")


def main():
    trainer = TwoModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
