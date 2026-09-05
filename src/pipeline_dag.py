"""
DAG-style orchestrator (approved equivalent ETL workflow).

Runs the full pipeline as an ordered set of tasks with dependencies,
per-task logging, timing, and retry-on-failure -- mirroring the structure
of the Apache Airflow DAG defined in airflow_dag/student_pipeline_dag.py
(which encodes the same four tasks as native Airflow operators for
deployment on an Airflow instance; see README for how to run it under
`astro dev` / a local Airflow install).

Usage:
    python src/pipeline_dag.py
"""
import os
import sys
import time
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LOG_DIR
from src.ingest import run_ingestion
from src.validate_clean import run_validation_cleaning
from src.transform import build_analytics_tables
from src.load_db import load_to_warehouse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline_run.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline_dag")

RUN_LOG_CSV = os.path.join(LOG_DIR, "pipeline_run_history.csv")


def _run_task(name, func, max_retries=2):
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            log.info(f"--- Task '{name}' started (attempt {attempt}/{max_retries}) ---")
            result = func()
            elapsed = round(time.time() - start, 2)
            log.info(f"--- Task '{name}' SUCCEEDED in {elapsed}s ---")
            return result, "SUCCESS", elapsed
        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            log.error(f"--- Task '{name}' FAILED on attempt {attempt} after {elapsed}s: {exc} ---")
            if attempt == max_retries:
                return None, "FAILED", elapsed
            time.sleep(1)


def run_pipeline():
    run_start = datetime.now()
    log.info(f"##### PIPELINE RUN START: {run_start.isoformat()} #####")

    task_status = {}

    # Task 1: Ingestion (extract raw -> staging)
    _, status, elapsed = _run_task("ingest_extract", run_ingestion)
    task_status["ingest_extract"] = (status, elapsed)
    if status == "FAILED":
        log.error("Pipeline halted: ingestion failed.")
        _log_run(run_start, task_status, "FAILED")
        return task_status

    # Task 2: Validation & cleaning (staging -> cleaned, depends on Task 1)
    _, status, elapsed = _run_task("validate_and_clean", run_validation_cleaning)
    task_status["validate_and_clean"] = (status, elapsed)
    if status == "FAILED":
        log.error("Pipeline halted: validation/cleaning failed.")
        _log_run(run_start, task_status, "FAILED")
        return task_status

    # Task 3: Transformation (cleaned -> analytics, depends on Task 2)
    _, status, elapsed = _run_task("transform_features", build_analytics_tables)
    task_status["transform_features"] = (status, elapsed)
    if status == "FAILED":
        log.error("Pipeline halted: transformation failed.")
        _log_run(run_start, task_status, "FAILED")
        return task_status

    # Task 4: Load to warehouse (analytics -> SQL DB, depends on Task 3)
    _, status, elapsed = _run_task("load_to_warehouse", load_to_warehouse)
    task_status["load_to_warehouse"] = (status, elapsed)

    overall = "SUCCESS" if all(s == "SUCCESS" for s, _ in task_status.values()) else "FAILED"
    log.info(f"##### PIPELINE RUN END: overall={overall} #####")
    _log_run(run_start, task_status, overall)
    return task_status


def _log_run(run_start, task_status, overall):
    import pandas as pd
    row = {"run_timestamp": run_start.isoformat(), "overall_status": overall}
    for task, (status, elapsed) in task_status.items():
        row[f"{task}_status"] = status
        row[f"{task}_seconds"] = elapsed
    df = pd.DataFrame([row])
    header = not os.path.exists(RUN_LOG_CSV)
    df.to_csv(RUN_LOG_CSV, mode="a", header=header, index=False)


if __name__ == "__main__":
    run_pipeline()
