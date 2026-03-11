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
# Columns to keep
# ==============================

BASE_COLUMNS = [
    'RECORD_ID', 'DISCHARGE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
    'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS',
    'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY',
    'PAT_AGE', 'FIRST_PAYMENT_SRC'
]

GROUPER_COLUMNS = ['RECORD_ID', 'APR_MDC']


# ==============================
# Helper: natural sorting
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
    # INGEST + MERGE
    # --------------------------------------------------

    def ingest_data(self):

        output_file = self.output_path / "final_data.csv"
        first_file = True
        total_rows = 0

        base_folders = sorted(self.input_dir.glob("df_base_1_*"))

        for base_folder in base_folders:

            suffix = base_folder.name.replace("df_base_1_", "")
            grouper_folder = self.input_dir / f"df_grouper_{suffix}"

            if not grouper_folder.exists():
                print(f"⚠️ No matching grouper folder for {base_folder.name}")
                continue

            print(f"\n📂 Processing dataset: {suffix}")

            # Sort partitions correctly
            base_files = sorted(base_folder.glob("*.parquet"), key=part_number)
            grouper_files = sorted(grouper_folder.glob("*.parquet"), key=part_number)

            print(f"BASE partitions: {len(base_files)}")
            print(f"GROUPER partitions: {len(grouper_files)}")

            # Map grouper partitions
            grouper_map = {part_number(p): p for p in grouper_files}

            for bf in base_files:

                part = part_number(bf)

                gf = grouper_map.get(part)

                print(f"\n📄 BASE: {bf.name}")

                df_base = pd.read_parquet(bf)[BASE_COLUMNS]

                if gf:

                    print(f"📄 GROUPER: {gf.name}")

                    df_grouper = pd.read_parquet(gf)[GROUPER_COLUMNS]

                    df_merged = df_base.merge(
                        df_grouper,
                        on="RECORD_ID",
                        how="left"
                    )

                else:

                    print("⚠️ Missing grouper partition")

                    df_base["APR_MDC"] = None
                    df_merged = df_base

                df_merged.drop(columns=["RECORD_ID"], inplace=True)

                rows = len(df_merged)
                total_rows += rows

                print(f"✅ Rows merged: {rows:,}")

                # Save incrementally
                if first_file:

                    df_merged.to_csv(output_file, index=False, mode="w")
                    first_file = False

                else:

                    df_merged.to_csv(output_file, index=False, mode="a", header=False)

        if first_file:
            raise FileNotFoundError("❌ No data merged.")

        print(f"\n✅ Final dataset rows: {total_rows:,}")
        print(f"Saved → {output_file}")

        return output_file

    # --------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------

    def save_splits(self, csv_file):

        print("\n✂️ Splitting train/test")

        df = pd.read_csv(csv_file)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

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

