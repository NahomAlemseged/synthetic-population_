import os
import pandas as pd
import numpy as np
from be_great import GReaT
from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = "config/params.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)
    data_dir = config.get('data_dir', '/content/drive/MyDrive/data_THCIC/gold')
    train_path = Path(data_dir) / 'train.csv'

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        return

    print("Loading training data...")
    df = pd.read_csv(train_path, low_memory=False)

    # Clean PAT_AGE by stripping non-numeric characters
    if 'PAT_AGE' in df.columns:
        df['PAT_AGE'] = df['PAT_AGE'].astype(str).str.replace(r'\D', '', regex=True)
        # Filter out empty strings
        df = df[df['PAT_AGE'] != '']

    # Select features needed for ICD generation
    # Modify this list based on what features you want to condition the generation on
    features = ['RECORD_ID', 'PRINC_DIAG_CODE', 'APR_MDC', 'SEX_CODE', 'PAT_AGE', 'LENGTH_OF_STAY']
    df_subset = df[features].dropna().astype(str).sample(n=10000, random_state=42) # Sample for training speed

    print(f"Training GReaT model on {len(df_subset)} records...")
    # Initialize GReaT model
    model = GReaT(llm='distilgpt2', batch_size=32, epochs=5, save_steps=400000)

    # Train the model
    model.fit(df_subset)

    n_samples = 10000
    print(f"Generating {n_samples} synthetic records...")
    synthetic_data = model.sample(n_samples=n_samples, k=50)

    out_dir = Path(data_dir) / 'generated'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'synthetic_icd_great.csv'

    synthetic_data.to_csv(out_path, index=False)
    print(f"Successfully saved generated data to {out_path}")

if __name__ == '__main__':
    main()
