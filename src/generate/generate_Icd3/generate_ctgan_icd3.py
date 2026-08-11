import time
from pathlib import Path

import pandas as pd
import yaml
from ctgan import CTGAN
from joblib import parallel_backend


# ==========================================================
# CONFIG
# ==========================================================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)


TRAIN_PATH = Path(params["generate_icd"]["input"][0])
POP_PATH = Path(params["generate_icd"]["input"][1])

OUTPUT_PATH = Path(params["generate_icd"]["output"])
OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = OUTPUT_PATH / "synthetic_population_with_ctgan_icd3.csv"


POOL_MULTIPLIER = 3
EPOCHS = 10



# ==========================================================
# ICD GENERATOR
# ==========================================================

class ICDGenerator:


    def __init__(self, train_path, pop_path):

        self.train_path = train_path
        self.pop_path = pop_path



    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    def load_data(self):

        df_train = pd.read_csv(
            self.train_path,
            dtype=str
        )

        df_pop = pd.read_csv(
            self.pop_path,
            dtype=str
        )


        # clean headers

        df_train.columns = (
            df_train.columns
            .str.strip()
        )

        df_pop.columns = (
            df_pop.columns
            .str.strip()
        )


        print(
            f"Training shape   : {df_train.shape}"
        )

        print(
            f"Population shape : {df_pop.shape}"
        )


        return df_train, df_pop



    # ------------------------------------------------------
    # CREATE ICD3 ONLY FROM TRAINING DATA
    # ------------------------------------------------------

    @staticmethod
    def extract_icd3(df):

        df = df.copy()


        if "PRINC_DIAG_CODE" not in df.columns:

            raise ValueError(
                "PRINC_DIAG_CODE missing from training data"
            )


        df["ICD3"] = (
            df["PRINC_DIAG_CODE"]
            .astype(str)
            .str.upper()
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



    # ------------------------------------------------------
    # SAMPLE TRAINING
    # ------------------------------------------------------

    @staticmethod
    def sample_training(df_train, n):

        n = min(
            n,
            len(df_train)
        )


        print(
            f"Training sample size: {n:,}"
        )


        return df_train.sample(
            n=n,
            random_state=42
        )



    # ------------------------------------------------------
    # PREPARE CTGAN DATA
    # ------------------------------------------------------

    @staticmethod
    def prepare_training(
            df,
            features,
            target):


        cols = features + [target]


        df = (
            df[cols]
            .dropna()
            .copy()
        )


        for col in cols:

            df[col] = (
                df[col]
                .astype(str)
            )


        print(
            "CTGAN training data:",
            df.shape
        )


        return df



    # ------------------------------------------------------
    # TRAIN CTGAN
    # ------------------------------------------------------

    @staticmethod
    def train_ctgan(
            df,
            discrete_columns,
            epochs):


        print(
            "Training CTGAN..."
        )


        model = CTGAN(
            epochs=epochs,
            batch_size=100,
            verbose=True
        )


        with parallel_backend(
            "threading",
            n_jobs=4
        ):

            model.fit(
                df,
                discrete_columns=discrete_columns
            )


        print(
            "CTGAN training finished"
        )


        return model



    # ------------------------------------------------------
    # GENERATE ICD POOL
    # ------------------------------------------------------

    @staticmethod
    def generate_pool(
            model,
            n):


        print(
            f"Generating synthetic ICD pool: {n:,}"
        )


        return model.sample(n)



    # ------------------------------------------------------
    # MATCH ICD TO POPULATION
    # ------------------------------------------------------

    @staticmethod
    def assign_icd(
            df_pop,
            synthetic_pool,
            merge_cols):


        print(
            "Assigning ICD3..."
        )


        df = df_pop.merge(
            synthetic_pool,
            on=merge_cols,
            how="left"
        )


        missing = (
            df["ICD3"]
            .isna()
            .sum()
        )


        print(
            f"Missing ICD3 after match: {missing:,}"
        )


        # fallback

        if missing > 0:

            fallback = (
                synthetic_pool["ICD3"]
                .sample(
                    missing,
                    replace=True,
                    random_state=42
                )
                .values
            )


            df.loc[
                df["ICD3"].isna(),
                "ICD3"
            ] = fallback



        df["ICD_SOURCE"] = "CTGAN"


        return df




# ==========================================================
# MAIN
# ==========================================================


def main():


    start = time.time()



    features = [

        "APR_MDC",
        "SEX_CODE",
        "PAT_AGE",
        "RACE",
        "ETHNICITY",
        "PAT_ZIP",
        "PAT_COUNTY",
        "PUBLIC_HEALTH_REGION",
        "APR_DRG"

    ]


    generator = ICDGenerator(
        TRAIN_PATH,
        POP_PATH
    )



    # -----------------------------
    # Load
    # -----------------------------

    df_train, df_pop = (
        generator.load_data()
    )



    # -----------------------------
    # ICD3 only training
    # -----------------------------

    df_train = (
        generator.extract_icd3(
            df_train
        )
    )



    # -----------------------------
    # Train sample
    # -----------------------------

    df_train = generator.sample_training(
        df_train,
        len(df_pop)
    )



    # -----------------------------
    # Prepare CTGAN
    # -----------------------------

    df_train = (
        generator.prepare_training(
            df_train,
            features,
            "ICD3"
        )
    )



    # -----------------------------
    # Train
    # -----------------------------

    model = generator.train_ctgan(
        df_train,
        features + ["ICD3"],
        EPOCHS
    )



    # -----------------------------
    # Generate ICD pool
    # -----------------------------

    synthetic_pool = (
        generator.generate_pool(
            model,
            len(df_pop) * POOL_MULTIPLIER
        )
    )


    synthetic_pool = synthetic_pool[
        features + ["ICD3"]
    ]



    # -----------------------------
    # Assign ICD
    # -----------------------------

    df_final = generator.assign_icd(
        df_pop,
        synthetic_pool,
        features
    )



    # -----------------------------
    # Save
    # -----------------------------

    df_final.to_csv(
        OUTPUT_CSV,
        index=False
    )


    elapsed = time.time() - start


    print("\n========================")
    print("DONE")
    print("========================")
    print(
        "Saved:",
        OUTPUT_CSV
    )

    print(
        "Final shape:",
        df_final.shape
    )

    print(
        f"Time: {elapsed:.2f}s"
    )



if __name__ == "__main__":

    main()
