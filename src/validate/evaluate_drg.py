import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import jensenshannon


# ======================================================
# CONFIG
# ======================================================
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH) as file:
    params_ = yaml.safe_load(file)


# ======================================================
# EVALUATION CLASS
# ======================================================
class Evaluate:
    def __init__(self):

        self.output_path = Path(
            params_["evaluate_icd"]["output"][0]
            if isinstance(params_["evaluate_icd"]["output"], list)
            else params_["evaluate_icd"]["output"]
        )

        self.reports_path = self.output_path / "reports"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        self.input_files = params_["evaluate"]["input"]

        # -----------------------------
        # LOAD DATA (STRICT CLEANING)
        # -----------------------------
        self.df_train = pd.read_csv(self.input_files[2], low_memory=False)
        self.df_test = pd.read_csv(self.input_files[1], low_memory=False)
        self.df_train2 = pd.read_csv(self.input_files[0], low_memory=False)

        # ❌ HARD DROP: remove PRINC_DIAG_CODE everywhere
        for df in [self.df_train, self.df_test, self.df_train2]:
            if "PRINC_DIAG_CODE" in df.columns:
                df.drop(columns=["PRINC_DIAG_CODE"], inplace=True)

        # Evaluation dataset
        self.df_eval = pd.concat(
            [self.df_train2, self.df_test],
            ignore_index=True
        )

        # Load model
        bundle = joblib.load(self.input_files[3])
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.encoders = bundle.get("encoders", {})

    # ==================================================
    # PREPROCESS
    # ==================================================
    def preprocess(self, df):

        df_ = df.copy()

        # Apply saved encoders
        for col, le in self.encoders.items():
            if col in df_.columns:
                df_[col] = pd.Categorical(df_[col], categories=le.classes_)
                df_[col] = df_[col].cat.codes

        # Convert remaining object columns
        obj_cols = df_[self.features].select_dtypes(include="object").columns
        for col in obj_cols:
            df_[col] = df_[col].astype("category").cat.codes

        X = df_[self.features]

        if X.select_dtypes(include="object").shape[1] > 0:
            raise ValueError(
                f"Unprocessed object columns: "
                f"{X.select_dtypes(include='object').columns.tolist()}"
            )

        # ✅ ONLY TARGET LEFT
        y = df_["APR_DRG"]

        return X, y

    # ==================================================
    # DISTRIBUTION MATCH
    # ==================================================
    def dist_match(self, p, q, label_name):

        output_file = self.reports_path / f"{label_name}_dist_match.png"

        idx = p.index.union(q.index)
        p = p.reindex(idx, fill_value=0)
        q = q.reindex(idx, fill_value=0)

        js = jensenshannon(p.values, q.values)
        similarity = (1 - js) * 100

        plt.figure(figsize=(8, 5))
        plt.bar(idx.astype(str), p, alpha=0.6, label="Eval")
        plt.bar(idx.astype(str), q, alpha=0.6, label="Train")
        plt.xlabel(label_name)
        plt.ylabel("Probability")
        plt.title(f"{label_name} Distribution\nSimilarity = {similarity:.2f}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"✅ Saved {label_name} distribution — Similarity: {similarity:.2f}%")
        return similarity

    # ==================================================
    # DATA DISTRIBUTION
    # ==================================================
    def evaluate_data_distribution(self):

        print("\n>>> Evaluating APR_DRG distribution (STRICT MODE)")

        results = {}

        # ONLY APR_DRG (no PRINC_DIAG_CODE anywhere)
        results["APR_DRG"] = self.dist_match(
            self.df_eval["APR_DRG"].value_counts(normalize=True),
            self.df_train["APR_DRG"].value_counts(normalize=True),
            "APR_DRG"
        )

        results["SEX_CODE"] = self.dist_match(
            self.df_eval["SEX_CODE"].value_counts(normalize=True),
            self.df_train["SEX_CODE"].value_counts(normalize=True),
            "SEX_CODE"
        )

        results["RACE"] = self.dist_match(
            self.df_eval["RACE"].value_counts(normalize=True),
            self.df_train["RACE"].value_counts(normalize=True),
            "RACE"
        )

        summary = pd.DataFrame({
            "Feature": results.keys(),
            "Jensen_Shannon": results.values()
        })

        summary_path = self.reports_path / "data_dist_match_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\n✅ Saved summary → {summary_path}")
        print(summary)

    # ==================================================
    # MODEL ACCURACY
    # ==================================================
    def evaluate_accuracy(self):

        print("\n>>> Evaluating model accuracy (APR_DRG ONLY)")

        X, y = self.preprocess(self.df_eval)
        preds = self.model.predict(X)

        acc = accuracy_score(y, preds)

        with open(self.reports_path / "accuracy.txt", "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")

        print(f"✅ Model accuracy = {acc:.4f}")
        return acc


# ======================================================
# MAIN
# ======================================================
def main():
    print("⚙️ Starting STRICT APR_DRG evaluation pipeline...")
    evaluator = Evaluate()
    evaluator.evaluate_data_distribution()
    evaluator.evaluate_accuracy()


if __name__ == "__main__":
    main()
