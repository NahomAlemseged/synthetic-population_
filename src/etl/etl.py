import yaml
import torch
from pathlib import Path

# =============================
# GPU DETECTION
# =============================

GPU_AVAILABLE = torch.cuda.is_available()

# if GPU_AVAILABLE:
#     print("🚀 GPU detected — using RAPIDS")
#     import dask_cudf as dd
# else:
#     print("💻 Using CPU Dask")
import dask.dataframe as dd


# =============================
# LOAD CONFIG
# =============================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH) as f:
    params_ = yaml.safe_load(f)


# =============================
# ETL CLASS
# =============================

class ETL:

    def __init__(self):

        self.input_dirs = [Path(p) for p in params_["etl"]["input"]]
        self.output_dir = Path(params_["etl"]["output"][0])

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # SAFE CSV LOADER
    # --------------------------------------------------

    def load_txt(self, files):

        ddf = dd.read_csv(
            files,
            sep="\t",
            dtype=str,
            encoding="latin1",
            assume_missing=True,
            blocksize="128MB",
            on_bad_lines="warn"
        )

        return ddf

    # --------------------------------------------------
    # EXTRACT + TRANSFORM
    # --------------------------------------------------

    def extract_transform(self):

        datasets = {}

        for directory in self.input_dirs:

            print(f"\n📂 Scanning {directory}")

            if not directory.exists():
                print("❌ directory missing")
                continue

            txt_files = list(directory.rglob("*.txt"))

            if not txt_files:
                print("⚠️ no txt files found")
                continue

            print(f"Found {len(txt_files)} files")

            # THCIC naming
            if "outpatient" in directory.name.lower():

                base_files = [str(p) for p in txt_files if "BASE" in p.name]
                grouper_files = [str(p) for p in txt_files if "GROUPER" in p.name]

            else:

              # Filter BASE_1 ED datasets (IP_ED and OP_ED), all quarters
              base_files = [
                  str(p) for p in txt_files
                  if ("IP_ED_BASE_DATA_1" in p.name)
              ]

              # Filter corresponding grouper files for ED datasets
              grouper_files = [
                  str(p) for p in txt_files
                  if ("IP_ED_GROUPER" in p.name)
              ]

            # -------------------------
            # BASE DATA
            # -------------------------

            if base_files:

                print(f"Loading {len(base_files)} BASE files")

                df_base = self.load_txt(base_files)

                df_base["DATASET_TYPE"] = directory.name

                base_rows = df_base.shape[0].compute()

                print(f"BASE rows: {base_rows:,}")

                datasets[f"df_base_1_{directory.name}"] = df_base

            else:
                print("⚠️ no BASE files found")

            # -------------------------
            # GROUPER DATA
            # -------------------------

            if grouper_files:

                print(f"Loading {len(grouper_files)} GROUPER files")

                df_grouper = self.load_txt(grouper_files)

                df_grouper["DATASET_TYPE"] = directory.name

                grouper_rows = df_grouper.shape[0].compute()

                print(f"GROUPER rows: {grouper_rows:,}")

                datasets[f"df_grouper_{directory.name}"] = df_grouper

            else:
                print("⚠️ no GROUPER files found")

        if not datasets:
            raise ValueError("❌ No datasets extracted")

        return datasets

    # --------------------------------------------------
    # LOAD → PARQUET
    # --------------------------------------------------

    def load(self, datasets):

        for name, ddf in datasets.items():

            out_dir = self.output_dir / name

            print(f"\n💾 Saving {name}")

            out_dir.mkdir(parents=True, exist_ok=True)

            # deterministic partition size
            ddf = ddf.repartition(partition_size="200MB")

            ddf.to_parquet(
                out_dir,
                write_index=False,
                compression="snappy"
            )

            print(f"✅ saved → {out_dir}")

        print("\n🎯 ETL COMPLETE")


# =============================
# MAIN
# =============================

def main():

    etl = ETL()

    datasets = etl.extract_transform()

    etl.load(datasets)

    print("\n🎯 Pipeline finished successfully")


if __name__ == "__main__":
    main()
