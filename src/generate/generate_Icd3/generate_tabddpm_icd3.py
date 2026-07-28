import pandas as pd
import numpy as np
import os
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder

def load_config(config_path="config/params.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def main():
  def extract_icd3(df_): 
      df = df_.copy()
      df["ICD3"] = df["PRINC_DIAG_CODE"].str.extract(r"([A-Z]\d{2})")
      df.drop(columns = ["PRINC_DIAG_CODE"], inplace = True)
      print("ICD three digit created: PRINC_DIAG_CODE dropped")
      return df
    config = load_config()
    # Fallbacks to known paths in case config doesn't specify them
    data_dir = config.get('data_dir', '/content/drive/MyDrive/data_THCIC/gold')
    train_path = os.path.join(data_dir, 'train.csv')

    generated_dir = os.path.join(data_dir, 'generated')
    synth_path = os.path.join(generated_dir, 'synthetic_inpatient_with_tabddpm.csv')
    out_path = os.path.join(generated_dir, 'synthetic_inpatient_icd__with_tabddpm.csv')
    
    print(f"Loading real data from {train_path}...")
    df_real = pd.read_csv(train_path, low_memory=False)
    df_real = extract_icd3(df_real)
    
    print(f"Loading synthetic data from {synth_path}...")
    df_synth = pd.read_csv(synth_path, low_memory=False)
    df_synth = extract_icd(df_synth)
    # Features to condition on for generating ICD3
    features = ['APR_MDC', 'SEX_CODE', 'PAT_AGE', 'LENGTH_OF_STAY']
    target = 'ICD3'

    # Clean PAT_AGE
    for df in [df_real, df_synth]:
        if 'PAT_AGE' in df.columns:
            df['PAT_AGE'] = df['PAT_AGE'].astype(str).str.replace(r'\D', '', regex=True)
            df['PAT_AGE'] = pd.to_numeric(df['PAT_AGE'], errors='coerce').fillna(0)

    # Drop missing targets in real data for training the mapping model
    df_real_clean = df_real.dropna(subset=[target] + features).copy()

    # Sample real data for KNN training to keep memory/time reasonable
    if len(df_real_clean) > 100000:
        df_real_clean = df_real_clean.sample(100000, random_state=42)

    print("Encoding categorical variables...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # Prepare training features
    X_real = df_real_clean[features].copy()
    for col in features:
        X_real[col] = X_real[col].astype(str)

    X_real_encoded = encoder.fit_transform(X_real)
    y_real = df_real_clean[target].astype(str)

    print("Training Nearest Neighbors to map ICD3...")
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
    knn.fit(X_real_encoded, y_real)

    print("Preparing synthetic data features...")
    X_synth = df_synth[features].copy()
    # Ensure no NaNs before transformation
    for col in features:
        X_synth[col] = X_synth[col].fillna(X_real[col].mode()[0]).astype(str)

    X_synth_encoded = encoder.transform(X_synth)

    print("Predicting ICD3 for synthetic data...")
    synth_preds = knn.predict(X_synth_encoded)

    # Add the generated codes
    df_synth['ICD3'] = synth_preds

    print(f"Saving generated data to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_synth.to_csv(out_path, index=False)
    print("Done!")
    print(f"Dataset shape: {df_synth.shape}")

if __name__ == "__main__":
    main()
