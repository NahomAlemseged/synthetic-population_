import pandas as pd
import yaml
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier
import joblib


# ==============================
# CONFIG
# ==============================

CONFIG_PATH = Path(
    "/content/synthetic-population_/config/params.yaml"
)

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)


data_dir = params.get(
    "data_dir",
    "/content/drive/MyDrive/data_THCIC/gold"
)



# ==============================
# DRG TRAINER
# ==============================

class DRGTrainer:


    def __init__(self, sample_rows=100000):

        self.train_path = os.path.join(
            data_dir,
            "train.csv"
        )

        self.test_path = os.path.join(
            data_dir,
            "test.csv"
        )

        self.synth_path = os.path.join(
            data_dir,
            "synthetic_inpatient",
            "synthetic_with_apr_drg_gan.csv"
        )


        self.model_dir = Path(
            "/content/synthetic-population_/model"
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        self.sample_rows = sample_rows


        self.target_column = "APR_DRG"


        # SAME FEATURES USED BY CTGAN
        self.feature_columns = [

            "SEX_CODE",
            "PAT_AGE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC"

        ]


        self.categorical_features = [

            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "PAT_ZIP",
            "PAT_COUNTY",
            "PUBLIC_HEALTH_REGION",
            "APR_MDC"

        ]


        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )



    # ==============================
    # LOAD DATA
    # ==============================

    def load_data(self, path):

        print(
            f"Loading {path}"
        )

        return pd.read_csv(
            path,
            low_memory=False
        )



    # ==============================
    # PREPROCESS
    # ==============================

    def preprocess(self, df):

        df = df.copy()


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
            self.feature_columns
            +
            [self.target_column]
        )


        df = df.dropna(
            subset=required
        )


        if "PAT_AGE" in df.columns:

            df["PAT_AGE"] = (
                df["PAT_AGE"]
                .astype(int)
            )


        for col in self.categorical_features:

            df[col] = (
                df[col]
                .astype(str)
            )


        return df



    # ==============================
    # COMMON APR DRG CLASSES
    # ==============================

    def harmonize_classes(
            self,
            synth,
            test,
            min_count=2
    ):


        synth_counts = (
            synth[self.target_column]
            .value_counts()
        )


        test_counts = (
            test[self.target_column]
            .value_counts()
        )


        common = (

            set(
                synth_counts[
                    synth_counts >= min_count
                ].index
            )

            &

            set(
                test_counts[
                    test_counts >= min_count
                ].index
            )
        )


        print(
            f"Keeping {len(common)} common APR_DRG classes"
        )


        synth = synth[
            synth[self.target_column]
            .isin(common)
        ].copy()


        test = test[
            test[self.target_column]
            .isin(common)
        ].copy()



        # Encode DRG labels 0...N-1

        mapping = {

            old:new

            for new, old in enumerate(
                sorted(common)
            )

        }


        synth[self.target_column] = (
            synth[self.target_column]
            .map(mapping)
            .astype(int)
        )


        test[self.target_column] = (
            test[self.target_column]
            .map(mapping)
            .astype(int)
        )


        return synth, test, mapping



    # ==============================
    # ENCODING
    # ==============================

    def fit_encoder(self, df):

        self.encoder.fit(
            df[self.categorical_features]
        )


    def transform_features(self, df):

        df = df.copy()


        df[
            self.categorical_features
        ] = self.encoder.transform(
            df[self.categorical_features]
        )


        return df



    # ==============================
    # TSTR PIPELINE
    # ==============================

    def train_and_evaluate(self):


        print("\n==============================")
        print("TSTR: SYNTHETIC TRAIN → REAL TEST")
        print("==============================")


        # LOAD

        synth = self.load_data(
            self.synth_path
        )

        test = self.load_data(
            self.test_path
        )


        # PREPROCESS

        synth = self.preprocess(
            synth
        )

        test = self.preprocess(
            test
        )


        # COMMON CLASSES

        synth, test, mapping = self.harmonize_classes(
            synth,
            test
        )


        print(
            "Synthetic:",
            synth.shape
        )

        print(
            "Real:",
            test.shape
        )



        # ENCODE

        self.fit_encoder(
            synth
        )


        synth = self.transform_features(
            synth
        )

        test = self.transform_features(
            test
        )



        # FEATURES

        X_train = synth[
            self.feature_columns
        ]

        y_train = synth[
            self.target_column
        ]


        X_test = test[
            self.feature_columns
        ]

        y_test = test[
            self.target_column
        ]



        # SAFE SAMPLE SYNTHETIC DATA

        if (
            self.sample_rows
            and
            len(X_train) > self.sample_rows
        ):

            X_train, _, y_train, _ = train_test_split(

                X_train,

                y_train,

                train_size=self.sample_rows,

                random_state=42,

                stratify=y_train
            )


        print(
            f"Synthetic training rows: {len(X_train)}"
        )

        print(
            f"Real testing rows: {len(X_test)}"
        )



        # MODEL

        model = XGBClassifier(

            objective="multi:softmax",

            num_class=len(mapping),

            eval_metric="mlogloss",

            random_state=42,

            n_jobs=-1

        )


        print(
            "Training XGBoost..."
        )


        model.fit(
            X_train,
            y_train
        )



        # TEST REAL

        pred = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            pred
        )


        print(
            f"\nTSTR Accuracy: {accuracy:.4f}"
        )


        print(
            classification_report(
                y_test,
                pred,
                zero_division=0
            )
        )



        joblib.dump(
            model,
            self.model_dir /
            "xgb_tstr_drg.pkl"
        )


        print(
            "Model saved."
        )



# ==============================
# MAIN
# ==============================

def main():

    trainer = DRGTrainer(
        sample_rows=100000
    )

    trainer.train_and_evaluate()



if __name__ == "__main__":

    main()
