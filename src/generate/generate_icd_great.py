import pandas as pd
from be_great import GReaT
import os
from pathlib import Path
import random

def main():
    # Paths
    train_path = '/content/drive/MyDrive/data_THCIC/gold/train.csv'
    synth_path = '/content/drive/MyDrive/data_THCIC/gold/synthetic_inpatient/synthetic_inpatient.csv'
    out_dir = '/content/drive/MyDrive/data_THCIC/gold/generated'
    out_path = f'{out_dir}/synthetic_inpatient_with_gan_icd.csv'

    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    synth_df = pd.read_csv(synth_path, low_memory=False)

    # Define features to learn the relationship (context + target)
    features = [
        'APR_MDC',
        'SEX_CODE',
        'PAT_AGE',
        'RACE',
        'ETHNICITY',
        'PAT_ZIP',
        'PAT_COUNTY',
        'PUBLIC_HEALTH_REGION',
        'FIRST_PAYMENT_SRC',
        'EMERGENCY_DEPT_FLAG',
        'PRINC_DIAG_CODE'
    ]

    # Clean train data
    print("Cleaning and sampling training data...")
    train_subset = train_df[features].dropna().copy()
    train_subset['PAT_AGE'] = train_subset['PAT_AGE'].astype(str).str.replace(r'\D', '', regex=True)
    train_subset = train_subset[train_subset['PAT_AGE'] != '']

    # Convert all to strings for text-based LLM generation
    for col in features:
        train_subset[col] = train_subset[col].astype(str)

    print("Data types after cleaning:")
    print(train_subset.dtypes)

    # Sample for training speed (GReaT can take a while)
    train_sample = train_subset.sample(n=5000, random_state=42)

    print(f"Training GReaT model on {len(train_sample)} records...")
    # Increased epochs to 10 for better convergence
    model = GReaT(llm='distilgpt2', batch_size=32, epochs=10, save_steps=400000)
    model.fit(train_sample)

    # Generate a pool of synthetic records to match conditions
    pool_size = 100000
    print(f"Generating {pool_size} synthetic records for the mapping pool...")
    # Added max_length=2000 to prevent premature breaking loop errors
    generated_pool = model.sample(n_samples=pool_size, k=50, max_length=2000)

    # Create a mapping of APR_MDC to a list of generated ICD codes
    print("Creating MDC to ICD mapping...")
    icd_mapping = generated_pool.groupby('APR_MDC')['PRINC_DIAG_CODE'].apply(list).to_dict()

    # Function to sample from the generated pool based on MDC
    def get_icd(mdc):
        mdc_str = str(mdc)
        if mdc_str in icd_mapping and len(icd_mapping[mdc_str]) > 0:
            return random.choice(icd_mapping[mdc_str])
        return "UNKNOWN" # Fallback if MDC not in generated pool

    print("Applying generated ICD codes to synthetic inpatient dataset...")
    # If PRINC_DIAG_CODE already exists in synth_df, we overwrite or create a new column
    synth_df['GENERATED_PRINC_DIAG_CODE'] = synth_df['APR_MDC'].apply(get_icd)

    # Overwrite the original with the GReaT generated one to match requested schema
    synth_df['PRINC_DIAG_CODE'] = synth_df['GENERATED_PRINC_DIAG_CODE']
    synth_df.drop(columns=['GENERATED_PRINC_DIAG_CODE'], inplace=True, errors='ignore')

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {out_path}...")
    synth_df.to_csv(out_path, index=False)
    print("Done! Output schema:")
    print(synth_df.columns.tolist())

if __name__ == '__main__':
    main()
