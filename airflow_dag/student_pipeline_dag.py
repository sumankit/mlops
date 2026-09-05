"""
Apache Airflow DAG for the Student Performance and Dropout-Risk pipeline.

Drop this file into your Airflow `dags/` folder (or `astro dev start` DAGs
folder) to schedule and orchestrate the same four tasks that
src/pipeline_dag.py runs synchronously for local demo/grading purposes.

Schedule: daily at 02:00 (adjust `schedule` as needed for your institution's
data-refresh cadence).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import run_ingestion
from src.validate_clean import run_validation_cleaning
from src.transform import build_analytics_tables
from src.load_db import load_to_warehouse

default_args = {
    "owner": "student_performance_pipeline",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="student_performance_dropout_risk_pipeline",
    description="Ingest -> validate -> transform -> load student performance data",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",   # daily at 2 AM
    catchup=False,
    tags=["mlops", "student-performance", "etl"],
) as dag:

    t1_ingest = PythonOperator(
        task_id="ingest_extract",
        python_callable=run_ingestion,
    )

    t2_validate = PythonOperator(
        task_id="validate_and_clean",
        python_callable=run_validation_cleaning,
    )

    t3_transform = PythonOperator(
        task_id="transform_features",
        python_callable=build_analytics_tables,
    )

    t4_load = PythonOperator(
        task_id="load_to_warehouse",
        python_callable=load_to_warehouse,
    )

    t1_ingest >> t2_validate >> t3_transform >> t4_load
