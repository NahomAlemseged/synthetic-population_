
import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================
parser = argparse.ArgumentParser(
    description="Synthetic population generation"
)
parser.add_argument("--n_samples", type=int, required=True)
parser.add_argument("--epochs", type=int, default=5)  # Retained for compatibility
parser.add_argument("--num_processes", type=int, default=2)
parser.add_argument("--sample_rows", type=int, default=None)
parser.add_argument("--model", choices=["xgboost", "random_forest",
                                        "logistic_regression", "all"],
                    default="xgboost")
parser.add_argument("--max_estimators", type=int, default=300)
parser.add_argument("--patience", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=50000)
args = parser.parse_args()

N_SAMPLES = args.n_samples
N_JOBS = max(1, args.num_processes)
RANDOM_STATE = 42

torch.set_num_threads(N_JOBS)
np.random.seed(RANDOM_STATE)

with open("config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

TRAIN_CSV = Path(params["generate"]["input"][0])
TEST_CSV = Path(params["generate"]["input"][1])
OUTPUT_DIR = Path(params["generate"]["output"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "synthetic_inpatient_mdc_xgb.csv"


# ============================================================
# Shared feature configuration
# ============================================================
TARGET = "APR_MDC"

FEATURES = [
    "SEX_CODE",
    "PAT_AGE",
    "RACE",
    "ETHNICITY",
    "PAT_ZIP",
    "PAT_COUNTY",
    "PUBLIC_HEALTH_REGION",
]

# These variables represent categories even if CSV inference
# interprets their values as integers.
CATEGORICAL_HINTS = {
    "SEX_CODE",
    "RACE",
    "ETHNICITY",
    "PAT_ZIP",
    "PAT_COUNTY",
    "PUBLIC_HEALTH_REGION",
}


def load_csv(path, sample_rows=None):
    """Load a CSV and optionally sample rows reproducibly."""
    df = pd.read_csv(path, low_memory=False)

    if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=RANDOM_STATE)

    return df


# ============================================================
# Step 1: IPF-style demographic resampling
# ============================================================
class SyntheticGenerator:
    def __init__(self, input_csv=None):
        self.input_csv = input_csv

    def generate_ipf(
        self,
        df,
        features,
        target_marginals=None,
        n_samples=None,
        tol=1e-5,
        max_iter=100,
    ):
        """
        Resample demographic records using IPF weights.

        IMPORTANT:
        If target_marginals are calculated from the same input
        dataset, the resulting weights will generally remain 1.
        Use independent population marginals for actual calibration.
        """
        n_samples = n_samples or N_SAMPLES
        work = df.copy()

        missing = set(features) - set(work.columns)
        if missing:
            raise ValueError(f"Missing IPF columns: {sorted(missing)}")

        weights = np.ones(len(work), dtype=np.float64)

        if target_marginals is not None:
            for iteration in range(max_iter):
                old_weights = weights.copy()

                for feature in features:
                    desired = pd.Series(
                        target_marginals[feature], dtype=float
                    )

                    codes, levels = pd.factorize(
                        work[feature].fillna("__MISSING__"),
                        sort=False,
                    )

                    current = np.bincount(
                        codes,
                        weights=weights,
                        minlength=len(levels),
                    )

                    desired_values = np.array(
                        [desired.get(level, 0.0) for level in levels],
                        dtype=float,
                    )

                    ratios = np.ones(len(levels), dtype=float)
                    valid = current > 0
                    ratios[valid] = (
                        desired_values[valid] / current[valid]
                    )

                    weights *= ratios[codes]

                    # Prevent numerical overflow during repeated updates.
                    weights /= max(weights.mean(), 1e-12)

                relative_change = (
                    np.linalg.norm(weights - old_weights)
                    / max(np.linalg.norm(old_weights), 1e-12)
                )

                if relative_change < tol:
                    print(f"IPF converged at iteration {iteration + 1}")
                    break

        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("IPF generated invalid sampling weights.")

        # Weighted bootstrap preserves whole demographic records.
        rng = np.random.default_rng(RANDOM_STATE)
        probabilities = weights / weights.sum()

        sampled_indices = rng.choice(
            len(work),
            size=n_samples,
            replace=True,
            p=probabilities,
        )

        return work.iloc[sampled_indices].drop(
            columns=["weight"], errors="ignore"
        ).reset_index(drop=True)


# ============================================================
# Step 2: ML-based target generation
# ============================================================
class GenerateML:
    def __init__(self, train_path, test_path):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)

    @staticmethod
    def make_preprocessor(X):
        # Explicit hints avoid treating ZIP codes and other
        # integer-coded categories as continuous numeric variables.
        categorical = [
            c for c in X.columns
            if c in CATEGORICAL_HINTS
            or not pd.api.types.is_numeric_dtype(X[c])
        ]
        numeric = [c for c in X.columns if c not in categorical]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipe, numeric),
                ("categorical", categorical_pipe, categorical),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return preprocessor

    def train_and_generate(
        self,
        synthetic_demographics,
        target_col=TARGET,
    ):
        start = time.time()

        # Read without forcing all columns to strings.
        train_df = load_csv(self.train_path, args.sample_rows)
        test_df = load_csv(self.test_path)

        required = set(FEATURES + [target_col])
        for label, df in [("training", train_df), ("test", test_df)]:
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"{label} CSV is missing columns: {sorted(missing)}"
                )

        # The model uses only the specified predictors.
        X = train_df[FEATURES].copy()
        y_raw = train_df[target_col].astype("string").fillna("__MISSING__")

        # Avoid missing/invalid target labels.
        valid_target = y_raw.ne("__MISSING__")
        X = X.loc[valid_target].reset_index(drop=True)
        y_raw = y_raw.loc[valid_target].reset_index(drop=True)

        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))

        if len(target_encoder.classes_) < 2:
            raise ValueError("At least two target classes are required.")

        # Keep a validation set for early stopping and model selection.
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        preprocessor = self.make_preprocessor(X_train)

        X_train_t = preprocessor.fit_transform(X_train)
        X_val_t = preprocessor.transform(X_val)

        # Ensure arrays are numeric and contiguous. This also avoids
        # the pandas dtype error encountered during predict_proba().
        X_train_t = np.asarray(X_train_t, dtype=np.float32, order="C")
        X_val_t = np.asarray(X_val_t, dtype=np.float32, order="C")

        # XGBoost multiclass class IDs must be contiguous integers.
        n_classes = len(target_encoder.classes_)
        if n_classes > 2:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        else:
            objective = "binary:logistic"
            eval_metric = "logloss"

        models = {}

        if args.model in ("xgboost", "all"):
            xgb_params = dict(
                objective=objective,
                eval_metric=eval_metric,
                n_estimators=args.max_estimators,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=5.0,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",
                early_stopping_rounds=args.patience,
            )
            if n_classes > 2:
                xgb_params["num_class"] = n_classes

            models["xgboost"] = XGBClassifier(**xgb_params)

        if args.model in ("random_forest", "all"):
            models["random_forest"] = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )

        if args.model in ("logistic_regression", "all"):
            models["logistic_regression"] = LogisticRegression(
                max_iter=500,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )

        mlflow.set_experiment("APR_MDC_multiclass")

        best_model = None
        best_name = None
        best_macro_f1 = -1.0

        for name, model in models.items():
            print(f"\nTraining {name} ...")
            with mlflow.start_run(run_name=name):
                t0 = time.time()

                if name == "xgboost":
                    model.fit(
                        X_train_t,
                        y_train,
                        eval_set=[(X_val_t, y_val)],
                        verbose=False,
                    )
                else:
                    model.fit(X_train_t, y_train)

                val_pred = model.predict(X_val_t)
                val_acc = accuracy_score(y_val, val_pred)
                val_macro_f1 = f1_score(
                    y_val, val_pred, average="macro", zero_division=0
                )

                elapsed = time.time() - t0
                print(
                    f"{name}: validation accuracy={val_acc:.4f}, "
                    f"macro-F1={val_macro_f1:.4f}, "
                    f"fit time={elapsed:.1f}s"
                )

                mlflow.log_param("model", name)
                mlflow.log_param("n_features", len(FEATURES))
                mlflow.log_param("n_training_rows", len(X_train))
                mlflow.log_metric("validation_accuracy", val_acc)
                mlflow.log_metric("validation_macro_f1", val_macro_f1)
                mlflow.log_metric("fit_seconds", elapsed)

                if name == "xgboost":
                    mlflow.log_param(
                        "best_iteration",
                        int(getattr(model, "best_iteration", 0)),
                    )

                # Do not select models using test-set performance.
                if val_macro_f1 > best_macro_f1:
                    best_macro_f1 = val_macro_f1
                    best_model = model
                    best_name = name

        print(f"\nSelected model: {best_name}")
        print(f"Validation macro-F1: {best_macro_f1:.4f}")

        # Evaluate once on the held-out test set.
        X_test = test_df[FEATURES].copy()
        y_test_raw = (
            test_df[target_col].astype("string").fillna("__MISSING__")
        )
        known_test = y_test_raw.isin(target_encoder.classes_)
        X_test = X_test.loc[known_test]
        y_test_raw = y_test_raw.loc[known_test]

        if len(X_test):
            X_test_t = np.asarray(
                preprocessor.transform(X_test),
                dtype=np.float32,
                order="C",
            )
            y_test = target_encoder.transform(y_test_raw.astype(str))
            test_pred = best_model.predict(X_test_t)

            print("\nHeld-out test report:")
            print(classification_report(
                y_test,
                test_pred,
                labels=np.arange(n_classes),
                target_names=target_encoder.classes_.astype(str),
                zero_division=0,
            ))
            print(
                "Test accuracy:",
                round(accuracy_score(y_test, test_pred), 4),
            )
            print(
                "Test macro-F1:",
                round(f1_score(
                    y_test, test_pred, average="macro",
                    zero_division=0,
                ), 4),
            )
            if (~known_test).any():
                print(
                    f"Warning: {(~known_test).sum()} test rows had target "
                    "classes absent from training and were excluded."
                )

        # Transform the synthetic demographic data using the SAME
        # fitted preprocessor and the SAME feature order.
        df_demo = synthetic_demographics.reindex(columns=FEATURES).copy()

        if df_demo.isnull().all(axis=0).any():
            bad = df_demo.columns[df_demo.isnull().all(axis=0)].tolist()
            raise ValueError(
                f"Synthetic input is missing feature values/columns: {bad}"
            )

        X_demo_t = preprocessor.transform(df_demo)
        X_demo_t = np.asarray(
            X_demo_t, dtype=np.float32, order="C"
        )

        # Batch prediction helps limit memory use for large populations.
        rng = np.random.default_rng(RANDOM_STATE)
        generated_targets = []

        for start_idx in range(0, len(X_demo_t), args.batch_size):
            batch = X_demo_t[start_idx:start_idx + args.batch_size]
            probs = best_model.predict_proba(batch)

            # Align model probability columns to LabelEncoder classes.
            model_classes = np.asarray(best_model.classes_, dtype=int)
            full_probs = np.zeros(
                (len(batch), n_classes), dtype=np.float64
            )
            full_probs[:, model_classes] = probs

            # Guard against floating-point issues and invalid rows.
            full_probs = np.clip(full_probs, 0, None)
            row_sums = full_probs.sum(axis=1, keepdims=True)
            full_probs = np.divide(
                full_probs,
                row_sums,
                out=np.full_like(full_probs, 1 / n_classes),
                where=row_sums > 0,
            )

            batch_targets = [
                rng.choice(n_classes, p=row_probs)
                for row_probs in full_probs
            ]
            generated_targets.extend(batch_targets)

        output = df_demo.copy()
        output[target_col] = target_encoder.inverse_transform(
            np.asarray(generated_targets, dtype=int)
        )

        print(f"Total ML stage time: {time.time() - start:.1f}s")
        return output


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("Starting synthetic population pipeline ...")

    df_real = load_csv(TRAIN_CSV, args.sample_rows)

    # Preserve the existing IPF demographic generation workflow.
    target_marginals = {
        col: df_real[col].value_counts(dropna=False).to_dict()
        for col in FEATURES
    }

    synth = SyntheticGenerator(TRAIN_CSV)

    print("Step 1: demographic resampling")
    synthetic_demographics = synth.generate_ipf(
        df_real,
        FEATURES,
        target_marginals=target_marginals,
        n_samples=N_SAMPLES,
    )

    print("Step 2: ML target generation")
    ml_gen = GenerateML(TRAIN_CSV, TEST_CSV)
    synthetic_dataset = ml_gen.train_and_generate(
        synthetic_demographics,
        target_col=TARGET,
    )

    print("Saving synthetic dataset ...")
    synthetic_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
