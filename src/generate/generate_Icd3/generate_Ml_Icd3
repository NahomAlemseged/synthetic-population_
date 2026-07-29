import pandas as pd
import numpy as np
import yaml
import mlflow
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
with open("config/params.yaml") as f:
    params = yaml.safe_load(f)

TRAIN = Path(params["evaluate"]["input"][1])
TEST = Path(params["evaluate"]["input"][2])
SYNTH = Path(params["generate_icd"]["input"][1])

OUT_DIR = Path(params["generate_icd"]["output"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "synthetic_population_with_ICD3_RF.csv"

SAMPLE_SIZE = 100000

FEATURES = [
    "APR_MDC",
    "SEX_CODE",
    "PAT_AGE",
    "RACE",
    "ETHNICITY",
    "PAT_ZIP",
    "PAT_COUNTY",
    "PUBLIC_HEALTH_REGION",
    # "FIRST_PAYMENT_SRC",
    # "EMERGENCY_DEPT_FLAG",
]

TARGET = "ICD3"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def extract_icd3(df):
    df = df.copy()

    if "PRINC_DIAG_CODE" not in df.columns:
        raise ValueError("PRINC_DIAG_CODE missing.")

    df[TARGET] = (
        df["PRINC_DIAG_CODE"]
        .astype(str)
        .str.upper()
        .str.extract(r"([A-Z]\d{2})", expand=False)
    )

    return df.drop(columns="PRINC_DIAG_CODE")


def encode(train, test, synth):

    for c in FEATURES:

        vals = pd.concat([
            train[c],
            test[c],
            synth[c]
        ]).fillna("UNK").astype(str).unique()

        m = {v: i for i, v in enumerate(vals)}

        train[c] = train[c].fillna("UNK").astype(str).map(m)
        test[c] = test[c].fillna("UNK").astype(str).map(m)
        synth[c] = synth[c].fillna("UNK").astype(str).map(m)

    return train, test, synth


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    print("Loading data...")

    train = extract_icd3(load(TRAIN))
    test = extract_icd3(load(TEST))
    synth = load(SYNTH)

    train = train.dropna(subset=[TARGET]).sample(min(SAMPLE_SIZE, len(train)), random_state=42)
    test = test.dropna(subset=[TARGET]).sample(min(SAMPLE_SIZE, len(test)), random_state=42)
    synth = synth.sample(min(SAMPLE_SIZE, len(synth)), random_state=42)

    train, test, synth = encode(train, test, synth)

    X_train = train[FEATURES]
    X_test = test[FEATURES]
    X_syn = synth[FEATURES]

    y_train = train[TARGET].astype(str)
    y_test = test[TARGET].astype(str)

    keep = y_test.isin(y_train.unique())

    X_test = X_test.loc[keep]
    y_test = y_test.loc[keep]

    enc = LabelEncoder()

    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    mlflow.set_experiment("ICD3_RF")

    with mlflow.start_run():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        print(classification_report(y_test, pred))

        print(f"Accuracy = {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

    synth[TARGET] = enc.inverse_transform(
        model.predict(X_syn)
    )

    synth["ICD_SOURCE"] = "RandomForest"

    synth.to_csv(OUTPUT, index=False)

    print(f"\nSaved to {OUTPUT}")
    print(synth.head())


if __name__ == "__main__":
    main()
