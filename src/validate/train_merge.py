import pandas as pd
import numpy as np
import joblib
import yaml
import torch

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
# SAFE FEATURES (NO LEAKAGE)
# =========================
FEATURES = [
    "SEX_CODE",
    "PAT_AGE",
    "RACE",
    "ETHNICITY",
    "PAT_ZIP",
    "PAT_COUNTY",
    "PUBLIC_HEALTH_REGION",
    "FIRST_PAYMENT_SRC",
    "EMERGENCY_DEPT_FLAG"
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

    # -------------------------
    # Encoding
    # -------------------------
    def preprocess(self, df, encoders=None, fit=True):

        if encoders is None:
            encoders = {}

        for col in df.columns:
            if df[col].dtype == "object":
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

    # -------------------------
    # Model
    # -------------------------
    def get_model(self, n_classes):
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="gpu_hist" if DEVICE == "GPU" else "hist",
            n_estimators=300,
            random_state=42
        )

    # -------------------------
    # Train
    # -------------------------
    def train(self):

        df = pd.read_csv(self.syn_path, low_memory=False)
        print(f">>> Synthetic rows: {len(df)}")

        target_mdc = "APR_MDC"
        target_icd = "PRINC_DIAG_CODE"

        # Encode
        df, encoders = self.preprocess(df, fit=True)

        # safety check
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # =========================
        # ICD COVERAGE CHECK (NEW)
        # =========================
        train_icd_set = set(train_df[target_icd])
        test_icd_set = set(test_df[target_icd])

        unseen_icd = test_icd_set - train_icd_set

        print("\n🔍 ICD Coverage Check")
        print(f"Unique ICDs in TEST not in TRAIN: {len(unseen_icd)}")
        print(f"Fraction of TEST ICDs unseen: {len(unseen_icd) / len(test_icd_set):.2%}")
        print("Examples:", list(unseen_icd)[:20])

        # =========================
        # MODEL 1: APR_MDC
        # =========================
        print("\n🚀 Training APR_MDC model")

        X_train = train_df[FEATURES]
        y_train = train_df[target_mdc]

        X_test = test_df[FEATURES]
        y_test = test_df[target_mdc]

        mdc_model = self.get_model(len(np.unique(y_train)))
        mdc_model.fit(X_train, y_train)

        print("\n📊 APR_MDC Evaluation")
        print(classification_report(y_test, mdc_model.predict(X_test)))

        # add prediction for ICD stage
        df["APR_MDC_PRED"] = mdc_model.predict(df[FEATURES])

        # =========================
        # MODEL 2: ICD
        # =========================
        print("\n🚀 Training ICD model")

        icd_features = FEATURES + ["APR_MDC_PRED"]

        X_train_icd = df.loc[train_df.index, icd_features]
        y_train_icd = train_df[target_icd]

        X_test_icd = df.loc[test_df.index, icd_features]
        y_test_icd = test_df[target_icd]

        icd_model = self.get_model(len(np.unique(y_train_icd)))
        icd_model.fit(X_train_icd, y_train_icd)

        print("\n📊 ICD Evaluation")
        print(classification_report(y_test_icd, icd_model.predict(X_test_icd)))

        # =========================
        # SAVE
        # =========================
        joblib.dump({
            "mdc_model": mdc_model,
            "icd_model": icd_model,
            "encoders": encoders,
            "features": FEATURES
        }, self.model_path)

        print(f"\n✅ Saved pipeline → {self.model_path}")


# =========================
# MAIN ENTRY POINT
# =========================
def main():
    trainer = TwoModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
