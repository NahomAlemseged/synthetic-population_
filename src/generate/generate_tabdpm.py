import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
warnings.filterwarnings('ignore')

def main():
    train_path = '/content/drive/MyDrive/data_THCIC/gold/train.csv'
    synth_path = '/content/drive/MyDrive/data_THCIC/gold/synthetic_inpatient/synthetic_inpatient.csv'
    out_dir = '/content/drive/MyDrive/data_THCIC/gold/generated'
    out_path = Path(out_dir) / 'synthetic_inpatient_with_apr_mdc.csv'

    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    synth_df = pd.read_csv(synth_path, low_memory=False)

    features = [
        'SEX_CODE', 'PAT_AGE', 'RACE', 'ETHNICITY', 'PAT_ZIP', 
        'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'FIRST_PAYMENT_SRC', 
        'EMERGENCY_DEPT_FLAG'
    ]
    target = 'APR_MDC'

    print("Cleaning and preparing training data...")
    train_subset = train_df[features + [target]].dropna().copy()
    
    # Treat PAT_AGE as ordinal (int)
    train_subset['PAT_AGE'] = train_subset['PAT_AGE'].astype(str).str.replace(r'\D', '', regex=True)
    train_subset = train_subset[train_subset['PAT_AGE'] != '']
    train_subset['PAT_AGE'] = train_subset['PAT_AGE'].astype(int)

    # Treat rest as categorical (strings)
    cat_features = [f for f in features if f != 'PAT_AGE']
    for col in cat_features:
        train_subset[col] = train_subset[col].astype(str)
    
    # Encode target to string as well for DDPM generation stability if needed
    train_subset[target] = train_subset[target].astype(str)

    print("Encoding categorical variables for TabDDPM & Nearest Neighbors...")
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        train_subset[col] = le.fit_transform(train_subset[col])
        encoders[col] = le

    # Sample for DDPM training to prevent memory crashes
    sample_size = min(50000, len(train_subset))
    train_sample = train_subset.sample(n=sample_size, random_state=42).copy()

    print(f"Training TabDDPM model on {len(train_sample)} real records...")
    loader = GenericDataLoader(train_sample)
    # Initialize DDPM
    syn_model = Plugins().get("ddpm", n_iter=100, batch_size=2048)
    syn_model.fit(loader)

    print("Generating a TabDDPM mapping pool...")
    pool_size = min(200000, len(synth_df))
    ddpm_pool = syn_model.generate(pool_size).dataframe()

    print("Preparing synthetic demographic data...")
    synth_features = synth_df[features].copy()
    
    # Clean PAT_AGE
    synth_features['PAT_AGE'] = synth_features['PAT_AGE'].astype(str).str.replace(r'\D', '', regex=True)
    synth_features.loc[synth_features['PAT_AGE'] == '', 'PAT_AGE'] = '0'
    synth_features['PAT_AGE'] = synth_features['PAT_AGE'].astype(int)

    # Encode categorical features in synth data
    for col in cat_features:
        synth_features[col] = synth_features[col].astype(str)
        classes = encoders[col].classes_
        synth_features[col] = synth_features[col].apply(lambda x: x if x in classes else classes[0])
        synth_features[col] = encoders[col].transform(synth_features[col])

    print("Fitting Nearest Neighbors on TabDDPM pool to conditionally map APR_MDC...")
    # We use NN to find the closest demographic profile in the DDPM generated pool
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='minkowski')
    X_pool = ddpm_pool[features]
    y_pool = ddpm_pool[target]
    nn.fit(X_pool)

    print("Mapping APR_MDC for synthetic data...")
    # Process in chunks to save memory
    chunk_size = 50000
    preds = []
    for i in range(0, len(synth_features), chunk_size):
        chunk = synth_features.iloc[i:i+chunk_size]
        distances, indices = nn.kneighbors(chunk)
        # Map the indices to the APR_MDC from the pool
        chunk_preds = y_pool.iloc[indices.flatten()].values
        preds.extend(chunk_preds)

    print("Adding predictions to synthetic dataset...")
    synth_df['APR_MDC'] = preds

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)
    
    print(f"Successfully generated {len(synth_df)} records and saved to {out_path}")

if __name__ == '__main__':
    main()
