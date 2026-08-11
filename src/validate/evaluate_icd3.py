import yaml
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from scipy.spatial.distance import jensenshannon


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)


# ============================================================
# EVALUATION CLASS
# ============================================================

class EvaluateICD3:

    def __init__(self):

        print("Starting ICD3 evaluation")

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        self.output_path = Path(
            params["evaluate"]["output"][0]
        )

        self.report_path = (
            self.output_path / "reports"
        )

        self.report_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Input files
        # ----------------------------------------------------

        files = params["evaluate"]["input"]

        # ----------------------------------------------------
        # Load datasets
        # ----------------------------------------------------

        print("\nLoading datasets")

        self.synthetic = pd.read_csv(
            files[0],
            low_memory=False
        )

        self.test = pd.read_csv(
            files[1],
            low_memory=False
        )

        self.train = pd.read_csv(
            files[2],
            low_memory=False
        )

        print(
            "Synthetic:",
            self.synthetic.shape
        )

        print(
            "Test:",
            self.test.shape
        )

        print(
            "Train:",
            self.train.shape
        )

        # ----------------------------------------------------
        # Extract ICD3
        # ----------------------------------------------------

        self.synthetic = self.extract_icd3(
            self.synthetic
        )

        self.test = self.extract_icd3(
            self.test
        )

        self.train = self.extract_icd3(
            self.train
        )

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        print("\nLoading model")

        bundle = joblib.load(files[3])

        print(
            "Bundle keys:",
            bundle.keys()
        )

        self.model = bundle["model"]

        self.features = bundle["features"]

        self.mapping = bundle.get(
            "mapping",
            {}
        )

        self.encoder = bundle.get(
            "encoder",
            None
        )

        self.target = bundle.get(
            "target",
            "ICD3"
        )


    # ========================================================
    # EXTRACT ICD3
    # ========================================================

    @staticmethod
    def extract_icd3(df):

        df = df.copy()

        # ----------------------------------------------------
        # ICD3 already exists
        # ----------------------------------------------------

        if "ICD3" in df.columns:

            df["ICD3"] = (
                df["ICD3"]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            return df

        # ----------------------------------------------------
        # Extract ICD3 from PRINC_DIAG_CODE
        # ----------------------------------------------------

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


    # ========================================================
    # CLEAN DATA
    # ========================================================

    def clean_data(self, df):

        df = df.copy()

        # ----------------------------------------------------
        # ICD3
        # ----------------------------------------------------

        df = df[
            df["ICD3"].notna()
        ]

        df["ICD3"] = (
            df["ICD3"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        df = df[
            df["ICD3"] != "UNKNOWN"
        ]

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

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

        df = df[
            df["PAT_AGE"].notna()
        ]

        return df


    # ========================================================
    # FEATURE ENCODING
    # ========================================================

    def encode_features(self, df):

        df = df.copy()

        if self.encoder is None:

            raise ValueError(
                "Model bundle does not contain an encoder."
            )

        # ----------------------------------------------------
        # Get encoder feature columns
        # ----------------------------------------------------

        cols = list(
            self.encoder.feature_names_in_
        )

        temp = df[cols].copy()

        # Convert to string for encoder
        temp = temp.astype(str)

        # Encode
        encoded = self.encoder.transform(
            temp
        )

        encoded = pd.DataFrame(
            encoded,
            columns=cols,
            index=df.index
        )

        # Replace original columns
        for col in cols:

            df[col] = encoded[col]

        return df


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(self, df):

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        df = self.clean_data(df)

        # ----------------------------------------------------
        # Encode features
        # ----------------------------------------------------

        df = self.encode_features(df)

        # ----------------------------------------------------
        # Select model features
        # ----------------------------------------------------

        X = df[
            self.features
        ].copy()

        # ----------------------------------------------------
        # Ensure numeric
        # ----------------------------------------------------

        for col in X.columns:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        # Missing values
        X = X.fillna(-1)

        # ----------------------------------------------------
        # ICD3 target
        # ----------------------------------------------------

        y = df["ICD3"].map(
            self.mapping
        )

        # Keep only mapped ICD3 classes
        valid = y.notna()

        X = X.loc[valid]

        y = y.loc[valid]

        y = y.astype(int)

        return X, y


    # ========================================================
    # JENSEN-SHANNON SIMILARITY
    # ========================================================

    def js_similarity(
        self,
        a,
        b,
        name
    ):

        # Remove missing values
        a = a.dropna()

        b = b.dropna()

        # Probability distributions
        p = a.value_counts(
            normalize=True
        )

        q = b.value_counts(
            normalize=True
        )

        # Union of categories
        idx = p.index.union(
            q.index
        )

        p = p.reindex(
            idx,
            fill_value=0
        )

        q = q.reindex(
            idx,
            fill_value=0
        )

        # Jensen-Shannon distance
        js = jensenshannon(
            p.values,
            q.values
        )

        # Convert distance to similarity
        score = (
            1 - js
        ) * 100

        print(
            name,
            ":",
            round(score, 2)
        )

        return score


    # ========================================================
    # DISTRIBUTION EVALUATION
    # ========================================================

    def evaluate_distribution(self):

        print(
            "\nDistribution evaluation"
        )

        results = {}

        # ----------------------------------------------------
        # Combine synthetic + test
        # ----------------------------------------------------

        eval_df = pd.concat(
            [
                self.synthetic,
                self.test
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # Features to evaluate
        # ----------------------------------------------------

        evaluation_columns = [
            "ICD3",
            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "APR_MDC",
        ]

        for col in evaluation_columns:

            results[col] = self.js_similarity(
                eval_df[col],
                self.train[col],
                col
            )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            {
                "Feature": results.keys(),
                "Similarity": results.values(),
            }
        )

        output_file = (
            self.report_path
            / "distribution_similarity.csv"
        )

        results_df.to_csv(
            output_file,
            index=False
        )

        print(
            "\nSaved:",
            output_file
        )


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    def evaluate_accuracy(self):

        print(
            "\nModel evaluation"
        )

        # ----------------------------------------------------
        # Combine synthetic + test
        # ----------------------------------------------------

        eval_df = pd.concat(
            [
                self.synthetic,
                self.test
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        X, y = self.preprocess(
            eval_df
        )

        print(
            "Evaluation X:",
            X.shape
        )

        print(
            "Classes:",
            len(
                np.unique(y)
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred = self.model.predict(
            X
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y,
            pred
        )

        weighted_f1 = f1_score(
            y,
            pred,
            average="weighted",
            zero_division=0
        )

        macro_f1 = f1_score(
            y,
            pred,
            average="macro",
            zero_division=0
        )

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

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report = classification_report(
            y,
            pred,
            zero_division=0
        )

        print(
            "\nClassification Report:"
        )

        print(report)

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        report_file = (
            self.report_path
            / "model_report.txt"
        )

        with open(
            report_file,
            "w"
        ) as f:

            f.write(
                f"Accuracy: {accuracy}\n"
            )

            f.write(
                f"Weighted F1: {weighted_f1}\n"
            )

            f.write(
                f"Macro F1: {macro_f1}\n\n"
            )

            f.write(report)

        print(
            "\nSaved:",
            report_file
        )


    # ========================================================
    # RUN EVALUATION
    # ========================================================

    def run(self):

        self.evaluate_distribution()

        self.evaluate_accuracy()


# ============================================================
# MAIN
# ============================================================

def main():

    evaluator = EvaluateICD3()

    evaluator.run()


if __name__ == "__main__":

    main()
