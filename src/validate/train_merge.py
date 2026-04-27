import pandas as pd
import numpy as np
import joblib
import yaml
import torch

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
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


# =========================
# SAFE FEATURE SET (NO LEAKAGE)
# =========================
SAFE_FEATURES_FOR_APR = [
    "AGE",
    "SEX",
    "RACE",
    "ADMISSION_TYPE",
    "PAYOR_TYPE",
    "ER_FLAG",
    "ICU_FLAG",
    "HOSPITAL_ID"
]


# =========================
# Trainer
# =========================
class TwoModelTrainer:

    def __init__(self):
        self.syn_path = params['train']['input'][0]
        self.output_path = Path(params['train']['output'][0])
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.model_path = self.output_path / "two_model_pipeline.pkl"

    # =========================
    # Preprocess
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
                    df[col] = encoders[col].transform(df[col].astype(str))
                else:
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
            tree_method="gpu_hist" if DEVICE == "GPU" else "hist",
            n_estimators=300,
            random_state=42
        )

    # =========================
    # Train
    # =========================
    def train(self):

        df = pd.read_csv(self.syn_path, low_memory=False)

        print(f">>> Synthetic rows: {len(df)}")

        target_mdc = "APR_MDC"
        target_icd = "PRINC_DIAG_CODE"

        # -------------------------
        # Encode all categorical
        # -------------------------
        df, encoders = self.preprocess(df, fit=True)

        # -------------------------
        # SAFETY CHECK: ensure required features exist
        # -------------------------
        missing = [c for c in SAFE_FEATURES_FOR_APR if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing safe features: {missing}")

        # =========================
        # SPLIT SYNTHETIC DATA
        # =========================
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # =========================
        # MODEL 1: APR_MDC (NO ICD FEATURES)
        # =========================
        print("\n🚀 Training APR_MDC model (leakage-safe)")

        X_train = train_df[SAFE_FEATURES_FOR_APR]
        y_train = train_df[target_mdc]

        X_test = test_df[SAFE_FEATURES_FOR_APR]
        y_test = test_df[target_mdc]

        mdc_model = self.get_model(len(np.unique(y_train)))
        mdc_model.fit(X_train, y_train)

        y_pred = mdc_model.predict(X_test)

        print("\n📊 APR_MDC Evaluation (synthetic)")
        print(classification_report(y_test, y_pred))

        # =========================
        # ADD APR PREDICTION FOR ICD MODEL
        # =========================
        df["APR_MDC_PRED"] = mdc_model.predict(df[SAFE_FEATURES_FOR_APR])

        # =========================
        # MODEL 2: ICD (USES APR ONLY, NOT RAW ICD FEATURES)
        # =========================
        print("\n🚀 Training ICD model")

        icd_features = SAFE_FEATURES_FOR_APR + ["APR_MDC_PRED"]

        X_train_icd = df.loc[train_df.index, icd_features]
        y_train_icd = train_df[target_icd]

        X_test_icd = df.loc[test_df.index, icd_features]
        y_test_icd = test_df[target_icd]

        icd_model = self.get_model(len(np.unique(y_train_icd)))
        icd_model.fit(X_train_icd, y_train_icd)

        y_pred_icd = icd_model.predict(X_test_icd)

        print("\n📊 ICD Evaluation (synthetic)")
        print(classification_report(y_test_icd, y_pred_icd))

        # =========================
        # SAVE MODELS
        # =========================
        joblib.dump({
            "mdc_model": mdc_model,
            "icd_model": icd_model,
            "encoders": encoders,
            "safe_features": SAFE_FEATURES_FOR_APR
        }, self.model_path)

        print(f"\n✅ Saved pipeline → {self.model_path}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    trainer = TwoModelTrainer()
    trainer.train()
