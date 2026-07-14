import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import mlflow

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# --------------------------
# CONFIG
# --------------------------
CONFIG_PATH = Path("config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

TRAIN_PATH = Path(params_["evaluate"]["input"][1])   # evaluate[1]
TEST_PATH  = Path(params_["evaluate"]["input"][2])   # evaluate[2]
SYNTH_PATH = Path(params_["generate_icd"]["input"][1])

OUTPUT_PATH = Path(params_["generate_icd"]["output"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_PATH / "synthetic_with_apr_icd_ml.csv"
AUDIT_CSV = OUTPUT_PATH / "ICD_removed_audit.csv"

SAMPLE_SIZE = 100000


# --------------------------
# PIPELINE
# --------------------------
class GenerateML:

    def __init__(self, train_path, test_path, synth_path):
        self.train_path = train_path
        self.test_path = test_path
        self.synth_path = synth_path

    def _load(self, path):
        df = pd.read_csv(path, low_memory=False)
        return df

    # --------------------------
    # FAST VECTOR ENCODING
    # --------------------------
    def _encode(self, X_train, X_test, X_syn):
        encoders = {}
        cat_cols = X_train.select_dtypes(include="object").columns

        for col in cat_cols:

            all_vals = pd.concat(
                [X_train[col], X_test[col], X_syn[col]],
                axis=0
            ).astype(str).unique()

            mapping = {v: i for i, v in enumerate(all_vals)}
            encoders[col] = mapping

            X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
            X_test[col]  = X_test[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
            X_syn[col]   = X_syn[col].astype(str).map(mapping).fillna(-1).astype(np.int32)

        return X_train, X_test, X_syn

    # --------------------------
    # MAIN
    # --------------------------
    def train_and_generate(self, target_col="PRINC_DIAG_CODE"):

        print("⚙️ Loading data...")

        df_train = self._load(self.train_path)
        df_test  = self._load(self.test_path)
        df_syn   = self._load(self.synth_path)

        df_syn = df_syn.drop(columns=[target_col], errors="ignore")

        # --------------------------
        # sampling (fast mode)
        # --------------------------
        df_train = df_train.sample(min(SAMPLE_SIZE, len(df_train)), random_state=42)
        df_test  = df_test.sample(min(SAMPLE_SIZE, len(df_test)), random_state=42)
        df_syn   = df_syn.sample(min(SAMPLE_SIZE, len(df_syn)), random_state=42)

        # --------------------------
        # feature alignment
        # --------------------------
        feature_cols = sorted(list(set(df_train.columns) & set(df_test.columns)))
        feature_cols = [c for c in feature_cols if c != target_col]

        X_train = df_train[feature_cols].copy()
        y_train = df_train[target_col].astype(str)

        X_test = df_test[feature_cols].copy()
        y_test = df_test[target_col].astype(str)

        X_syn = df_syn.reindex(columns=feature_cols).copy()

        # --------------------------
        # remove unseen labels (AUDIT)
        # --------------------------
        train_labels = set(y_train.unique())

        mask = y_test.isin(train_labels)
        removed = df_test.loc[~mask, [target_col]].copy()
        removed["reason"] = "unseen_PRINC_DIAG_CODE_in_train"

        print(f"❌ Removed test rows: {len(removed):,}")

        if len(removed) > 0:
            print(removed[target_col].value_counts().head(15))

        removed.to_csv(AUDIT_CSV, index=False)

        X_test = X_test.loc[mask].reset_index(drop=True)
        y_test = y_test.loc[mask].reset_index(drop=True)

        # --------------------------
        # encoding (FAST)
        # --------------------------
        X_train, X_test, X_syn = self._encode(X_train, X_test, X_syn)

        # --------------------------
        # encode target
        # --------------------------
        target_encoder = LabelEncoder()
        y_train_enc = target_encoder.fit_transform(y_train)
        y_test_enc = target_encoder.transform(y_test)

        # --------------------------
        # MODEL (FAST)
        # --------------------------
        print("🚀 Training model...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )

        mlflow.set_experiment("ICD_FAST")

        with mlflow.start_run(run_name="rf_fast"):

            model.fit(X_train, y_train_enc)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test_enc, preds)

            print("\n📊 Classification Report:\n")
            print(classification_report(y_test_enc, preds))

            print(f"\n✅ Accuracy: {acc:.4f}")

            mlflow.log_metric("test_accuracy", acc)

        # --------------------------
        # SYNTHETIC GENERATION
        # --------------------------
        print("🧠 Generating APR_DRG for synthetic data...")

        probs = model.predict_proba(X_syn)

        df_syn = df_syn.reindex(columns=feature_cols)

        df_syn[target_col] = [
            np.random.choice(target_encoder.classes_, p=p)
            for p in probs
        ]

        # --------------------------
        # SAVE OUTPUT
        # --------------------------
        df_syn.to_csv(OUTPUT_CSV, index=False)

        print(f"✅ Saved synthetic dataset: {OUTPUT_CSV}")
        print(f"📁 Audit log: {AUDIT_CSV}")

        return df_syn


# --------------------------
# MAIN
# --------------------------
def main():

    print("⚙️ Pipeline started")

    ml_gen = GenerateML(TRAIN_PATH, TEST_PATH, SYNTH_PATH)

    ml_gen.train_and_generate()

    print("🎉 Done")


if __name__ == "__main__":
    main()



