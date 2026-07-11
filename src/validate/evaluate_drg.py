import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import jensenshannon


# ======================================================
# CONFIG
# ======================================================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)


with open(CONFIG_PATH) as file:
    params_ = yaml.safe_load(file)



# ======================================================
# EVALUATION CLASS
# ======================================================

class Evaluate:

    def __init__(self):

        self.output_path = Path(
            params_["evaluate"]["output"][0]
        )


        self.reports_path = (
            self.output_path / "reports"
        )


        self.reports_path.mkdir(
            parents=True,
            exist_ok=True
        )


        self.input_files = (
            params_["evaluate"]["input"]
        )


        # -----------------------------
        # LOAD DATA
        # -----------------------------

        # synthetic dataset
        self.df_synthetic = pd.read_csv(
            self.input_files[0],
            low_memory=False
        )


        # real test dataset
        self.df_test = pd.read_csv(
            self.input_files[1],
            low_memory=False
        )


        # real train dataset (for distribution)
        self.df_train = pd.read_csv(
            self.input_files[2],
            low_memory=False
        )


        for df in [
            self.df_synthetic,
            self.df_test,
            self.df_train
        ]:

            df.drop(
                columns=[
                    "PRINC_DIAG_CODE"
                ],
                errors="ignore",
                inplace=True
            )


        # -----------------------------
        # FEATURES
        # -----------------------------

        self.features = [

            "SEX_CODE",
            "PAT_AGE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC"

        ]


        self.target = "APR_DRG"



        # -----------------------------
        # LOAD MODEL
        # -----------------------------

        obj = joblib.load(
            self.input_files[3]
        )


        # bundle model

        if isinstance(obj, dict):

            print(
                "Loading model bundle"
            )

            self.model = obj["model"]

            self.features = obj.get(
                "features",
                self.features
            )


            self.encoder = (
                obj.get(
                    "encoder",
                    None
                )
            )


        # standalone model

        else:

            print(
                "Loading standalone XGBClassifier"
            )

            self.model = obj

            self.encoder = None



    # ==================================================
    # PREPROCESS
    # ==================================================

    def preprocess(self, df):

        df = df.copy()


        # PAT AGE

        if "PAT_AGE" in df.columns:

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



        required = (
            self.features
            +
            [self.target]
        )


        df = df.dropna(
            subset=required
        )



        # categorical encoding

        cat_cols = [

            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC"

        ]


        if self.encoder is not None:


            df[cat_cols] = (
                self.encoder.transform(
                    df[cat_cols]
                    .astype(str)
                )
            )


        else:

            for c in cat_cols:

                if c in df.columns:

                    df[c] = (
                        df[c]
                        .astype("category")
                        .cat.codes
                    )



        X = df[self.features]

        y = df[self.target].astype(int)


        return X, y



    # ==================================================
    # KEEP COMMON CLASSES
    # ==================================================

    def keep_common_classes(self):

        common = (

            set(
                self.df_synthetic[self.target]
            )

            &

            set(
                self.df_test[self.target]
            )

        )


        print(
            f"Keeping {len(common)} APR_DRG classes"
        )


        self.df_synthetic = (
            self.df_synthetic[
                self.df_synthetic[self.target]
                .isin(common)
            ]
        )


        self.df_test = (
            self.df_test[
                self.df_test[self.target]
                .isin(common)
            ]
        )



    # ==================================================
    # DISTRIBUTION MATCH
    # ==================================================

    def dist_match(
            self,
            p,
            q,
            label_name
    ):


        output_file = (
            self.reports_path
            /
            f"{label_name}_dist_match.png"
        )


        idx = (
            p.index
            .union(q.index)
        )


        p = p.reindex(
            idx,
            fill_value=0
        )


        q = q.reindex(
            idx,
            fill_value=0
        )



        js = jensenshannon(
            p.values,
            q.values
        )


        similarity = (
            1-js
        )*100



        plt.figure(
            figsize=(10,5)
        )


        plt.bar(
            idx.astype(str),
            p,
            alpha=0.6,
            label="Synthetic"
        )


        plt.bar(
            idx.astype(str),
            q,
            alpha=0.6,
            label="Real"
        )


        plt.title(
            f"{label_name} Similarity={similarity:.2f}%"
        )


        plt.xlabel(
            label_name
        )


        plt.ylabel(
            "Probability"
        )


        plt.legend()


        plt.tight_layout()


        plt.savefig(
            output_file,
            dpi=300
        )


        plt.close()



        print(
            f"{label_name}: {similarity:.2f}%"
        )


        return similarity



    # ==================================================
    # DISTRIBUTION EVALUATION
    # ==================================================

    def evaluate_distribution(self):

        print(
            "\n>>> Distribution similarity"
        )


        results = {}


        for col in [

            "APR_DRG",
            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "APR_MDC"

        ]:


            if col in self.df_synthetic.columns:


                results[col] = self.dist_match(

                    self.df_synthetic[col]
                    .value_counts(normalize=True),

                    self.df_test[col]
                    .value_counts(normalize=True),

                    col
                )



        pd.DataFrame({

            "Feature": results.keys(),

            "Similarity": results.values()

        }).to_csv(

            self.reports_path /
            "distribution_similarity.csv",

            index=False

        )



    # ==================================================
    # TSTR
    # ==================================================

    def evaluate_tstr(self):


        print(
            "\n=============================="
        )

        print(
            "TSTR: SYNTHETIC TRAIN → REAL TEST"
        )

        print(
            "=============================="
        )



        X_train, y_train = self.preprocess(
            self.df_synthetic
        )


        X_test, y_test = self.preprocess(
            self.df_test
        )


        print(
            "Synthetic train:",
            X_train.shape
        )


        print(
            "Real test:",
            X_test.shape
        )


        pred = self.model.predict(
            X_test
        )


        acc = accuracy_score(
            y_test,
            pred
        )


        print(
            f"TSTR Accuracy = {acc:.4f}"
        )


        print(
            classification_report(
                y_test,
                pred,
                zero_division=0
            )
        )


        with open(
            self.reports_path /
            "tstr_accuracy.txt",
            "w"
        ) as f:

            f.write(
                f"TSTR Accuracy: {acc:.4f}\n\n"
            )

            f.write(
                classification_report(
                    y_test,
                    pred,
                    zero_division=0
                )
            )


        return acc




# ======================================================
# MAIN
# ======================================================

def main():

    print(
        "🚀 Starting Evaluation"
    )


    evaluator = Evaluate()


    evaluator.keep_common_classes()


    evaluator.evaluate_distribution()


    evaluator.evaluate_tstr()



if __name__ == "__main__":
    main()
