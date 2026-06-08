import pandas as pd
from be_great import GReaT
import os
from pathlib import Path
import random
import numpy as np

def main():
    # Paths
    train_path = '/content/drive/MyDrive/data_THCIC/gold/train.csv'
    synth_path = '/content/drive/MyDrive/data_THCIC/gold/synthetic_inpatient/synthetic_inpatient.csv'
    out_dir = '/content/drive/MyDrive/data_THCIC/gold/generated'
    out_path = f'{out_dir}/synthetic_inpatient_with_gan_icd.csv'

    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    synth_df = pd.read_csv(synth_path, low_memory=False)

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

    print("Cleaning and sampling training data...")
    train_subset = train_df[features].dropna().copy()
    train_subset['PAT_AGE'] = train_subset['PAT_AGE'].astype(str).str.replace(r'\\D', '', regex=True)
    train_subset = train_subset[train_subset['PAT_AGE'] != '']

    for col in features:
        train_subset[col] = train_subset[col].astype(str)

    train_sample = train_subset.sample(n=5000, random_state=42)

    print(f"Training GReaT model on {len(train_sample)} records...")
    model = GReaT(llm='distilgpt2', batch_size=32, epochs=5)
    model.fit(train_sample)

    pool_size = 5000
    print(f"Generating {pool_size} synthetic records for mapping pool...")
    try:
        # Using guided_sampling and a smaller pool size to ensure stability and speed
        generated_pool = model.sample(n_samples=pool_size, k=50, max_length=512, guided_sampling=True)
    except Exception as e:
        print(f"Sampling failed with error: {e}. Falling back to a simple distribution-based map.")
        generated_pool = train_sample.copy()

    print("Creating MDC to ICD mapping...")
    icd_mapping = generated_pool.groupby('APR_MDC')['PRINC_DIAG_CODE'].apply(list).to_dict()

    def get_icd(mdc):
        mdc_str = str(mdc)
        if mdc_str in icd_mapping and len(icd_mapping[mdc_str]) > 0:
            return random.choice(icd_mapping[mdc_str])
        return "UNKNOWN"

    print("Applying generated ICD codes...")
    synth_df['PRINC_DIAG_CODE'] = synth_df['APR_MDC'].apply(get_icd)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    main()
