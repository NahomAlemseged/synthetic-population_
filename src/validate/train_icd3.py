import yaml
import joblib
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from xgboost import XGBClassifier


# ======================================================
# CONFIG
# ======================================================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)

with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)


INPUTS = params["evaluate"]["input"]

OUTPUT = Path(
    params["train"]["output"][1]
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = OUTPUT / "ctgan_icd3.pkl"


# ======================================================
# ICD TRAINER
# ======================================================

class ICDTrainer:

    def __init__(self):

        # Synthetic dataset
        self.synthetic_path = INPUTS[0]

        # Real test dataset
        self.test_path = (
            "/content/drive/MyDrive/data_THCIC/gold/test.csv"
        )

        self.target = "ICD3"

        self.features = [
            "SEX_CODE",
            "PAT_AGE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC",
            "APR_DRG"
        ]

        self.cat_features = [
            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC",
            "APR_DRG"
        ]

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )


    # ==================================================
    # LOAD
    # ==================================================

    def load(self, path):

        print("Loading:", path)

        return pd.read_csv(
            path,
            low_memory=False
        )


    # ==================================================
    # EXTRACT ICD3
    # ==================================================

    @staticmethod
    def extract_icd3(df):

        df = df.copy()

        # If ICD3 already exists, keep it
        if "ICD3" in df.columns:
            return df

        if "PRINC_DIAG_CODE" not in df.columns:
            raise ValueError(
                "PRINC_DIAG_CODE missing from dataset"
            )

        df["ICD3"] = (
            df["PRINC_DIAG_CODE"]
            .astype(str)
            .str.upper()
            .str.strip()
            .str.extract(
                r"([A-Z]\d{2})",
                expand=False
            )
        )

        df.drop(
            columns=["PRINC_DIAG_CODE"],
            inplace=True
        )

        return df


    # ==================================================
    # CLEAN
    # ==================================================

    def clean(self, df):

        df = df.copy()

        df = df[
            df[self.target].notna()
        ]

        df[self.target] = (
            df[self.target]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df = df[
            df[self.target] != "UNKNOWN"
        ]

        df["PAT_AGE"] = (
            df["PAT_AGE"]
            .astype(str)
            .str.replace(
                r"\D",
                "",
                regex=True
            )
        )

        df["PAT_AGE"] = pd.to_numeric(
            df["PAT_AGE"],
            errors="coerce"
        )

        required = self.features + [self.target]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        df = df.dropna(
            subset=required
        )

        return df


    # ==================================================
    # ENCODE FEATURES
    # ==================================================

    def encode(self, train, test):

        train = train.copy()
        test = test.copy()

        self.encoder.fit(
            train[self.cat_features]
            .astype(str)
        )

        train[self.cat_features] = (
            self.encoder.transform(
                train[self.cat_features]
                .astype(str)
            )
        )

        test[self.cat_features] = (
            self.encoder.transform(
                test[self.cat_features]
                .astype(str)
            )
        )

        return train, test


    # ==================================================
    # TRAIN
    # ==================================================

    def run(self):

        print("\n==============================")
        print(" ICD3 TSTR TRAINING ")
        print("==============================\n")


        # ------------------------------------------------
        # Load
        # ------------------------------------------------

        synthetic = self.load(
            self.synthetic_path
        )

        test = self.load(
            self.test_path
        )


        # ------------------------------------------------
        # Extract ICD3
        # ------------------------------------------------

        synthetic = self.extract_icd3(
            synthetic
        )

        test = self.extract_icd3(
            test
        )


        # ------------------------------------------------
        # Clean
        # ------------------------------------------------

        synthetic = self.clean(
            synthetic
        )

        test = self.clean(
            test
        )


        print(
            "\nSynthetic:",
            synthetic.shape
        )

        print(
            "Test:",
            test.shape
        )


        # ------------------------------------------------
        # Keep common ICD3 classes
        # ------------------------------------------------

        common = (
            set(synthetic["ICD3"])
            &
            set(test["ICD3"])
        )

        print(
            "Common ICD3 classes:",
            len(common)
        )

        if len(common) == 0:
            raise ValueError(
                "No common ICD3 classes found."
            )

        synthetic = synthetic[
            synthetic["ICD3"].isin(common)
        ].copy()

        test = test[
            test["ICD3"].isin(common)
        ].copy()


        # ------------------------------------------------
        # Encode features
        # ------------------------------------------------

        synthetic, test = self.encode(
            synthetic,
            test
        )


        # ------------------------------------------------
        # X / Y
        # ------------------------------------------------

        X_train = synthetic[
            self.features
        ]

        y_train = synthetic[
            self.target
        ]

        X_test = test[
            self.features
        ]

        y_test = test[
            self.target
        ]


        # ------------------------------------------------
        # Encode ICD3 target
        # ------------------------------------------------

        labels = sorted(
            y_train.unique()
        )

        label_map = {
            label: i
            for i, label in enumerate(labels)
        }

        reverse_map = {
            i: label
            for label, i in label_map.items()
        }

        y_train = y_train.map(
            label_map
        )

        y_test = y_test.map(
            label_map
        )

        valid = y_test.notna()

        X_test = X_test.loc[
            valid
        ]

        y_test = y_test.loc[
            valid
        ]

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)


        # ------------------------------------------------
        # Model
        # ------------------------------------------------

        print(
            "\nTraining rows:",
            len(X_train)
        )

        print(
            "Testing rows:",
            len(X_test)
        )

        print(
            "ICD3 classes:",
            len(labels)
        )

        model = XGBClassifier(
            objective="multi:softmax",
            num_class=len(labels),
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42
        )


        print(
            "\nTraining XGBoost..."
        )

        model.fit(
            X_train,
            y_train
        )


        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------

        pred = model.predict(
            X_test
        )


        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            pred
        )

        weighted_f1 = f1_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            pred,
            average="macro",
            zero_division=0
        )


        print("\n==============================")
        print(" ICD3 RESULTS ")
        print("==============================")

        print(
            "\nAccuracy:",
            accuracy
        )

        print(
            "Weighted F1:",
            weighted_f1
        )

        print(
            "Macro F1:",
            macro_f1
        )

        print(
            "\nClassification Report:\n"
        )

        print(
            classification_report(
                y_test,
                pred,
                zero_division=0
            )
        )


        # ------------------------------------------------
        # Save model
        # ------------------------------------------------

        bundle = {
            "model": model,
            "encoder": self.encoder,
            "features": self.features,
            "mapping": reverse_map,
            "target": "ICD3",
            "target_source": "PRINC_DIAG_CODE"
        }

        joblib.dump(
            bundle,
            MODEL_PATH
        )

        print(
            "\nModel saved:",
            MODEL_PATH
        )


# ======================================================
# MAIN
# ======================================================

def main():

    trainer = ICDTrainer()

    trainer.run()


if __name__ == "__main__":
    main()
