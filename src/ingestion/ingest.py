import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# ==============================
# Load Config
# ==============================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")
with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

input_dir = Path(params_["ingestion"]["input_dir"])
output_path = Path(params_["ingestion"]["output"])
test_size = params_["ingestion"].get("test_size", 0.2)
random_state = params_["ingestion"].get("random_state", 42)

# ==============================
# Columns
# ==============================

BASE_COLUMNS = [
    'RECORD_ID', 'DISCHARGE','EMERGENCY_DEPT_FLAG', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
    'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS',
    'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY',
    'PAT_AGE', 'FIRST_PAYMENT_SRC','PRINC_DIAG_CODE'
]


#     BASE_COLUMNS = [
#     'RECORD_ID', 'EMERGENCY_DEPT_FLAG','DISCHARGE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
#     'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS',
#     'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY',
#     'PAT_AGE', 'FIRST_PAYMENT_SRC','PRINC_DIAG_CODE'
# ]

GROUPER_COLUMNS = ['RECORD_ID', 'APR_MDC']

# ==============================
# Helper
# ==============================

def part_number(path):
    m = re.search(r"part\.(\d+)", path.name)
    return int(m.group(1)) if m else -1

# ==============================
# Ingestion Class
# ==============================

class Ingestion:

    def __init__(self, input_dir, output_path):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # READ & COMBINE PARTITIONS
    # --------------------------------------------------

    def read_full_base(self):
        base_folders = sorted(self.input_dir.glob("df_base_1_*"))
        all_base_dfs = []

        for base_folder in base_folders:
            base_files = sorted(base_folder.glob("*.parquet"), key=part_number)
            for bf in base_files:
                df = pd.read_parquet(bf)[BASE_COLUMNS]
                df.columns = df.columns.str.upper()
                all_base_dfs.append(df)

        if not all_base_dfs:
            raise FileNotFoundError("No BASE parquet files found.")
        df_base_full = pd.concat(all_base_dfs, ignore_index=True)
        return df_base_full

    def read_full_grouper(self):
        grouper_folders = sorted(self.input_dir.glob("df_grouper_*"))
        all_grouper_dfs = []

        for gf in grouper_folders:
            grouper_files = sorted(gf.glob("*.parquet"), key=part_number)
            for pf in grouper_files:
                df = pd.read_parquet(pf)[GROUPER_COLUMNS]
                df.columns = df.columns.str.upper()
                all_grouper_dfs.append(df)

        if not all_grouper_dfs:
            raise FileNotFoundError("No GROUPER parquet files found.")
        df_grouper_full = pd.concat(all_grouper_dfs, ignore_index=True)
        return df_grouper_full

    # --------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------

    def validate_dataframes(self, df_base, df_grouper):
        print(f"BASE total rows: {len(df_base):,}")
        print(f"GROUPER total rows: {len(df_grouper):,}")

        base_dups = df_base['RECORD_ID'].duplicated().sum()
        grouper_dups = df_grouper['RECORD_ID'].duplicated().sum()
        print(f"BASE duplicate RECORD_ID: {base_dups}")
        print(f"GROUPER duplicate RECORD_ID: {grouper_dups}")

        if base_dups > 0:
            print("⚠️ Duplicate RECORD_ID detected in BASE")
        if grouper_dups > 0:
            print("⚠️ Duplicate RECORD_ID detected in GROUPER")

        missing_in_grouper = set(df_base['RECORD_ID']) - set(df_grouper['RECORD_ID'])
        missing_in_base = set(df_grouper['RECORD_ID']) - set(df_base['RECORD_ID'])
        print(f"Missing RECORD_ID in GROUPER: {len(missing_in_grouper):,}")
        print(f"Missing RECORD_ID in BASE: {len(missing_in_base):,}")
        if len(missing_in_grouper) > 0:
            print("Example missing in GROUPER:", list(missing_in_grouper)[:5])

    # --------------------------------------------------
    # MERGE FULL DATAFRAMES
    # --------------------------------------------------

    def merge_full_dataframes(self, df_base, df_grouper):
        df_base = df_base.set_index("RECORD_ID")
        df_grouper = df_grouper.set_index("RECORD_ID")
        df_merged = df_base.join(df_grouper, how="left").reset_index()
        return df_merged

    # --------------------------------------------------
    # INGEST
    # --------------------------------------------------

    def ingest_data(self):
        df_base_full = self.read_full_base()
        df_grouper_full = self.read_full_grouper()

        print("\n🔹 Validating full DataFrames before merge...")
        self.validate_dataframes(df_base_full, df_grouper_full)

        print("\n🔹 Merging full DataFrames...")
        df_merged = self.merge_full_dataframes(df_base_full, df_grouper_full)

        output_file = self.output_path / "final_data.csv"
        df_merged.to_csv(output_file, index=False)
        print(f"✅ Final merged dataset saved: {output_file}")
        print(f"✅ Total rows: {len(df_merged):,}")

        return output_file

    # --------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------

    def save_splits(self, csv_file):
        print("\n✂️ Splitting train/test")
        df = pd.read_csv(csv_file)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_file = self.output_path / "train.csv"
        test_file = self.output_path / "test.csv"
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        print(f"✅ Train rows: {len(train_df):,}")
        print(f"✅ Test rows: {len(test_df):,}")

# ==============================
# Main
# ==============================

def main():
    ingest = Ingestion(input_dir, output_path)
    final_csv = ingest.ingest_data()
    ingest.save_splits(final_csv)
    print("\n🎯 Pipeline complete")

if __name__ == "__main__":
    main()
