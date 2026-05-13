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
        self.input_path_train = params['evaluate']['input'][0] if isinstance(params['train']['input'], list) else params['train']['input'][0]
        self.input_path_test = params['evaluate']['input'][1] if isinstance(params['train']['input'], list) else params['train']['input'][1]
        self.input_path_synth = params['generate_icd']['input'] if isinstance(params['train']['input'], list) else params['generate_icd']['input']
        
      
        self.output_path = Path(params['train']['output'][0] if isinstance(params['train']['output'], list) else params['train']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_file = self.output_path / "model.pkl"

  
    def split_encode(df, target_col):
      X = df.drop(column = target_col)
      y = df[target_col]
      categorical_cols = X.select_dtypes(include='object').columns
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        return X,y
      
    def train_model(self):
        # Load dataset
        df_train = pd.read_csv(self.input_path_train, low_memory=False)
        print(f">>> Loaded {len(df_train)} rows of train dataset")

        df_test = pd.read_csv(self.input_path_test, low_memory=False)
        print(f">>> Loaded {len(df_test)} rows of test dataset")

        df_synth = pd.read_csv(self.input_path_synth, low_memory=False)
        print(f">>> Loaded {len(df_synth)} rows of synthetic dataset")


        X_train, y_train = split_encode(df_train)
        X_test, y_test = split_encode(df_test)


       # TRAIN-TEST ON REAL DATASET
       model = xgb.XGBClassifier(
          objective='binary:logistic',
          n_estimators=100,
          learning_rate=0.1,
          max_depth=5,
          random_state=42
      )

      xgb_real = model.fit(X_train, y_train)
      printf("Accuracy on Real Dataset : {accuracy_score(xgb_real.predict(X_test), y_test)}")
      printf("F1 score on Real Dataset : {f1_score(xgb_real.predict(X_test), y_test)}")
      

        # TRAIN TEST ON SYNHTETIC DATASET
        # Split data
        X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
            X_synth, y_synth, test_size=0.2, stratify=y, random_state=42
        )
      xgb_synth = model.fit(X_train, y_train)
      
      printf("Accuracy on Synthetic Dataset : {accuracy_score(xgb_synth.predict(X_test_synth), y_test_synth)}")
      printf("F1 score on Synthetic Dataset : {f1_score(xgb_synth.predict(X_test_synth), y_test_synth)}")


def main():
    trainer = TrainSynth()
    trainer.train_model()


if __name__ == "__main__":
    main()
