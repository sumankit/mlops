"""
Load layer (Task 4): loads the cleaned + analytics tables into the SQL
warehouse (SQLite for the demo; schema is written to be Postgres-portable --
see docs/data_dictionary.md).

Schema (star-ish design):
  dim_student        -- one row per student_id (demographics)
  fact_assessment     -- one row per student-subject record (grades, risk)
  agg_subject_summary  -- analytics mart: subject-wise pass/fail
  agg_department_summary -- analytics mart: school x subject performance
"""
import os
import sys
import logging

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ANALYTICS_DIR, LOG_DIR, DB_URI, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "load.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("load_db")

DEMOGRAPHIC_COLS = [
    "student_id", "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "guardian", "internet",
]

FACT_COLS = [
    "student_id", "subject", "G1", "G2", "G3", "avg_internal_marks",
    "previous_semester_performance", "attendance_percentage",
    "assignment_completion_rate", "final_grade_percent", "pass_fail",
    "risk_flag", "risk_score", "absences", "studytime", "failures",
    "extraction_timestamp",
]


def load_to_warehouse():
    log.info("=== Starting load task ===")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(DB_URI)

    master = pd.read_csv(os.path.join(ANALYTICS_DIR, "student_master.csv"))
    subject_summary = pd.read_csv(os.path.join(ANALYTICS_DIR, "subject_summary.csv"))
    dept_summary = pd.read_csv(os.path.join(ANALYTICS_DIR, "department_summary.csv"))

    dim_student = master[DEMOGRAPHIC_COLS].drop_duplicates(subset=["student_id"])
    fact_assessment = master[FACT_COLS]

    with engine.begin() as conn:
        dim_student.to_sql("dim_student", conn, if_exists="replace", index=False)
        fact_assessment.to_sql("fact_assessment", conn, if_exists="replace", index=False)
        subject_summary.to_sql("agg_subject_summary", conn, if_exists="replace", index=False)
        dept_summary.to_sql("agg_department_summary", conn, if_exists="replace", index=False)

        # Primary key / indexes for query performance (SQLite syntax; the
        # Postgres DDL equivalent is in docs/data_dictionary.md)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_student ON fact_assessment(student_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_subject ON fact_assessment(subject)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_risk ON fact_assessment(risk_flag)"))

    log.info(f"Loaded dim_student: {len(dim_student)} rows")
    log.info(f"Loaded fact_assessment: {len(fact_assessment)} rows")
    log.info(f"Loaded agg_subject_summary: {len(subject_summary)} rows")
    log.info(f"Loaded agg_department_summary: {len(dept_summary)} rows")
    log.info(f"=== Load complete -> {DB_PATH} ===")


if __name__ == "__main__":
    load_to_warehouse()
