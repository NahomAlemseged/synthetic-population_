import subprocess
from datetime import datetime
import os

from src.etl.etl import main as etl_main
from src.ingestion.ingest import main as ingest_main
from src.generate.generate_apr_mdc.generate_ctgan import main as generate_main_mdc
from src.generate.generate_drg.generate_drg_ctgan import main as generate_main_drg
from src.generate.generate_icd.generate_GAN_icd import main as generate_main_icd

from src.validate.train import main as train_main
from src.validate.evaluate import main as evaluate_main

BASE_DIR = os.path.abspath(os.getcwd())


def extract():
    print("============================================")
    print("STARTING EXTRACT STEP")
    print("============================================")

    subprocess.run(
        ["bash", "/content/drive/MyDrive/data_THCIC/extract_load.sh"],
        check=True
    )

    print("============================================")
    print("EXTRACT STEP COMPLETED")
    print("============================================")


def run_stage(func, name):
    print(f"\n🚀 Starting {name}")
    func()
    print(f"✅ Finished {name}")


def main():
    start_time = datetime.now()
    print(f"PIPELINE STARTED AT {start_time}")

    # 1️⃣ Extract (blocking)
    # extract()  # uncomment if you want to run extraction

    # 2️⃣ ETL (GPU-safe) — sequential
    # run_stage(etl_main, "ETL")

    # 3️⃣ Ingest (CPU) — sequential or parallel
    # run_stage(ingest_main, "INGEST")

    # 4️⃣ Generate APR-MDC (GPU)
    run_stage(generate_main_mdc, "GENERATE")

    # 4️⃣ Generate APR-DRG (GPU)
    run_stage(generate_main_icd, "GENERATE ICD")

     # 4️⃣ Generate ICD-10 (GPU)
    run_stage(generate_main_icd, "GENERATE ICD")

    # 5️⃣ Train (GPU)
    run_stage(train_main, "TRAIN")

    # 6️⃣ Evaluate (CPU)
    run_stage(evaluate_main, "EVALUATE")

    end_time = datetime.now()
    print(f"\nPIPELINE FINISHED AT {end_time}")
    print(f"TOTAL RUNTIME: {end_time - start_time}")


if __name__ == "__main__":
    main()
