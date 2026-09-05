"""
Ingestion layer (Task 1 of the pipeline).

Reads the raw source CSVs (UCI Student Performance Dataset), stamps them with
an extraction timestamp, copies an immutable raw snapshot, and writes an
ingestion log recording source, status, and row count for every run --
so every extraction is auditable and repeatable.
"""
import os
import sys
import shutil
import logging
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SOURCE_FILES, RAW_DIR, STAGING_DIR, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ingestion.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ingest")

INGESTION_LOG_CSV = os.path.join(LOG_DIR, "ingestion_log.csv")


def _append_ingestion_record(record: dict):
    df = pd.DataFrame([record])
    header = not os.path.exists(INGESTION_LOG_CSV)
    df.to_csv(INGESTION_LOG_CSV, mode="a", header=header, index=False)


def ingest_source(subject_name: str, source_path: str) -> pd.DataFrame:
    """Extract one source file, snapshot it into staging, and log the result."""
    extraction_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "extraction_timestamp": datetime.now().isoformat(),
        "source_name": f"UCI_student_performance_{subject_name}",
        "source_path": source_path,
        "status": None,
        "row_count": 0,
        "error": "",
    }
    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)

        df = pd.read_csv(source_path, sep=";")
        df["subject"] = "Mathematics" if subject_name == "mathematics" else "Portuguese"
        df["source_file"] = os.path.basename(source_path)
        df["extraction_timestamp"] = extraction_ts

        staged_name = f"{subject_name}_{extraction_ts}.csv"
        staged_path = os.path.join(STAGING_DIR, staged_name)
        df.to_csv(staged_path, index=False)

        record["status"] = "SUCCESS"
        record["row_count"] = len(df)
        log.info(f"Ingested {subject_name}: {len(df)} rows -> {staged_path}")
        _append_ingestion_record(record)
        return df

    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = str(exc)
        log.error(f"Failed to ingest {subject_name}: {exc}")
        _append_ingestion_record(record)
        raise


def run_ingestion() -> dict:
    """Ingest all configured source files. Returns {subject_name: DataFrame}."""
    log.info("=== Starting ingestion task ===")
    results = {}
    for subject_name, path in SOURCE_FILES.items():
        try:
            results[subject_name] = ingest_source(subject_name, path)
        except Exception:
            # Errors are logged; ingestion continues with remaining sources
            # so one bad/missing file doesn't kill the whole run.
            continue
    log.info(f"=== Ingestion complete: {len(results)}/{len(SOURCE_FILES)} sources succeeded ===")
    return results


if __name__ == "__main__":
    run_ingestion()
