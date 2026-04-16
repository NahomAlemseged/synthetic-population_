import os
import pandas as pd
import numpy as np
import joblib
import yaml
import mlflow
import mlflow.sklearn
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# GPU / CPU detection
GPU_AVAILABLE = torch.cuda.is_available()
device = "GPU" if GPU_AVAILABLE else "CPU"
print(f"⚡ Using device: {device}")

# ML models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load config
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")
with open(CONFIG_PATH) as file:
    params = yaml.safe_load(file)

class TrainSynth:
    def __init__(self):
        self.input_path = params['train']['input'][0] if isinstance(params['train']['input'], list) else params['train_icd']['input']
        self.output_path = Path(params['train']['output'][0] if isinstance(params['train']['output'], list) else params['train_icd']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_file = self.output_path / "model.pkl"

    def train_model(self):
        # Load dataset
        df = pd.read_csv(self.input_path, low_memory=False)
        print(f">>> Loaded {len(df)} rows")

        target_col = "PRINC_DIAG_CODE"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical columns
        categorical_cols = X.select_dtypes(include='object').columns
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        n_classes = len(np.unique(y))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # =========================
        # Define experiments
        # =========================
        experiments = {}

        # --- XGBoost (GPU A100 ready) ---
        experiments["xgboost"] = {
            "pipeline": Pipeline([
                ("model", XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    random_state=42,
                    tree_method="gpu_hist",  # GPU acceleration
                    n_jobs=-1
                ))
            ]),
            "params": {
                "model__n_estimators": [200, 400]
            }
        }

        # --- Random Forest ---
        experiments["random_forest"] = {
            "pipeline": Pipeline([
                ("model", RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 20],
                "model__min_samples_split": [2, 5]
            }
        }

        # --- Logistic Regression ---
        experiments["logistic_regression"] = {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1000,
                    n_jobs=-1,
                    solver="saga"
                ))
            ]),
            "params": {
                "model__C": [0.1, 1, 10]
            }
        }

        # =========================
        # MLflow Experiment
        # =========================
        mlflow.set_experiment("APR_MDC_multiclass")
        best_score = 0

        for name, exp in experiments.items():
            with mlflow.start_run(run_name=name):
                print(f"\n🚀 Training {name} on {device}")

                grid = GridSearchCV(
                    exp["pipeline"],
                    exp["params"],
                    scoring="f1_weighted",
                    cv=3,
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise'  # will stop immediately if something fails
                )

                grid.fit(X_train, y_train)
                y_pred = grid.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                print(classification_report(y_test, y_pred))

                # MLflow logging
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_weighted", f1)
                mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

                if f1 > best_score:
                    best_score = f1
                    best_overall = grid.best_estimator_

        # Save best model
        joblib.dump({
            "model": best_overall,
            "features": X.columns.tolist(),
            "encoders": encoders
        }, self.model_file)

        print(f"\n✅ Best model saved to {self.model_file}")
        print(f"🏆 Best weighted F1: {best_score:.4f}")


def main():
    trainer = TrainSynth()
    trainer.train_model()


if __name__ == "__main__":
    main()
