import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ============================== #
# Configuration and Paths        #
# ============================== #

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")
with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)

data_dir = params.get('data_dir', '/content/drive/MyDrive/data_THCIC/gold')

class DRGTrainer:
    def __init__(self, n_samples=None, sample_rows=None):
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.synth_path = os.path.join(data_dir, 'synthetic_inpatient/synthetic_with_apr_drg_ml.csv')
        self.model_dir = Path('/content/synthetic-population_/model')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.n_samples = n_samples
        self.sample_rows = sample_rows

        self.target_column = 'APR_DRG'
        self.feature_columns = [
            'DISCHARGE', 'EMERGENCY_DEPT_FLAG', 'TYPE_OF_ADMISSION',
            'SOURCE_OF_ADMISSION', 'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION',
            'PAT_STATUS', 'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY',
            'LENGTH_OF_STAY', 'PAT_AGE', 'FIRST_PAYMENT_SRC'
        ]

        self.numerical_features = ['LENGTH_OF_STAY', 'PAT_AGE', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION']
        self.categorical_features = [
            'DISCHARGE', 'EMERGENCY_DEPT_FLAG', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
            'PAT_ZIP', 'PAT_STATUS', 'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY',
            'FIRST_PAYMENT_SRC'
        ]

        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def load_data(self, path):
        df = pd.read_csv(path, low_memory=False)
        return df

    def preprocess(self, df):
        # Clean PAT_AGE
        if 'PAT_AGE' in df.columns:
            df['PAT_AGE'] = df['PAT_AGE'].astype(str).str.replace(r'\\D', '', regex=True)
            df['PAT_AGE'] = pd.to_numeric(df['PAT_AGE'], errors='coerce')

        # Drop rows with NaN in target or features
        df = df.dropna(subset=[self.target_column] + self.feature_columns).copy()

        # Ensure PAT_AGE is integer after cleaning
        if 'PAT_AGE' in df.columns:
            df['PAT_AGE'] = df['PAT_AGE'].astype(int)

        # Convert categorical features to string type for encoding
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def fit_encoder(self, df_train):
        # Fit encoder on training data for categorical features
        self.encoder.fit(df_train[self.categorical_features])

    def transform_features(self, df):
        df_encoded = df.copy()
        df_encoded[self.categorical_features] = self.encoder.transform(df_encoded[self.categorical_features])
        return df_encoded

    def train_and_evaluate(self):
        print("\n==============================")
        print("REAL \u2192 REAL")
        print("==============================")
        df_train = self.load_data(self.train_path)
        df_test = self.load_data(self.test_path)

        df_train = self.preprocess(df_train)
        df_test = self.preprocess(df_test)

        self.fit_encoder(df_train)

        df_train_encoded = self.transform_features(df_train)
        df_test_encoded = self.transform_features(df_test)

        X_train = df_train_encoded[self.feature_columns]
        y_train = df_train_encoded[self.target_column].astype(int)
        X_test = df_test_encoded[self.feature_columns]
        y_test = df_test_encoded[self.target_column].astype(int)

        # Sample if n_samples is specified
        if self.sample_rows and len(X_train) > self.sample_rows:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=self.sample_rows, random_state=42, stratify=y_train)
            X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=self.sample_rows // 4, random_state=42, stratify=y_test)

        print(f">>> Loaded TRAIN dataset: {len(X_train)} rows")
        print(f">>> Loaded TEST dataset: {len(X_test)} rows")

        xgb_real = XGBClassifier(objective='multi:softmax', num_class=len(y_train.unique()), eval_metric='mlogloss', use_label_encoder=False, random_state=42, n_jobs=-1)
        xgb_real.fit(X_train, y_train)

        y_pred_real = xgb_real.predict(X_test)
        print(f"Accuracy on Real Dataset : {accuracy_score(y_test, y_pred_real):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_real, zero_division=0))
        joblib.dump(xgb_real, self.model_dir / 'xgb_real_drg.pkl')

        # Synthetic data evaluation
        print("\n==============================")
        print("SYNTH \u2192 SYNTH")
        print("==============================")
        df_synth = self.load_data(self.synth_path)

        # Ensure synthetic data has the target column and features, drop if not
        if self.target_column not in df_synth.columns:
            print(f"Warning: Target column '{self.target_column}' not found in synthetic data. Skipping synthetic evaluation.")
            return
        if not all(col in df_synth.columns for col in self.feature_columns):
            missing_cols = [col for col in self.feature_columns if col not in df_synth.columns]
            print(f"Warning: Missing features {missing_cols} in synthetic data. Skipping synthetic evaluation.")
            return

        df_synth = self.preprocess(df_synth)
        df_synth_encoded = self.transform_features(df_synth)

        X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
            df_synth_encoded[self.feature_columns],
            df_synth_encoded[self.target_column].astype(int),
            test_size=0.2, random_state=42, stratify=df_synth_encoded[self.target_column].astype(int)
        )

        print(f">>> Loaded SYNTH dataset: {len(X_synth_train) + len(X_synth_test)} rows")
        if self.sample_rows and len(X_synth_train) > self.sample_rows:
             X_synth_train, _, y_synth_train, _ = train_test_split(X_synth_train, y_synth_train, train_size=self.sample_rows, random_state=42, stratify=y_synth_train)

        xgb_synth = XGBClassifier(objective='multi:softmax', num_class=len(y_synth_train.unique()), eval_metric='mlogloss', use_label_encoder=False, random_state=42, n_jobs=-1)
        xgb_synth.fit(X_synth_train, y_synth_train)

        y_pred_synth = xgb_synth.predict(X_synth_test)
        print(f"Accuracy on Synthetic Dataset : {accuracy_score(y_synth_test, y_pred_synth):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_synth_test, y_pred_synth, zero_division=0))
        joblib.dump(xgb_synth, self.model_dir / 'xgb_synth_drg.pkl')

def main():
    trainer = DRGTrainer(n_samples=500000, sample_rows=500000)
    trainer.train_and_evaluate()

if __name__ == "__main__":
    main()
