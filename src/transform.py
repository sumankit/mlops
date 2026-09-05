"""
Transformation layer (Task 3): builds the analytical (feature) tables from
the cleaned data layer.

Produces:
  - data/analytics/student_master.csv     : one row per student-subject record
                                             with engineered features + risk flag
  - data/analytics/subject_summary.csv    : subject-wise pass/fail aggregate
  - data/analytics/department_summary.csv : school x subject aggregate
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLEANED_DIR, ANALYTICS_DIR, LOG_DIR,
    MAX_ABSENCES, RISK_ATTENDANCE_THRESHOLD, RISK_FINAL_GRADE_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "transform.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("transform")


def _load_cleaned() -> pd.DataFrame:
    frames = []
    for subject_key in ["mathematics", "portuguese"]:
        path = os.path.join(CLEANED_DIR, f"{subject_key}_cleaned.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing. Run validate_clean.py first.")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Attendance percentage, derived from absences over an assumed
    # MAX_ABSENCES-day academic year (documented assumption; the UCI dataset
    # records absence counts, not a raw attendance percentage).
    df["attendance_percentage"] = (
        100 - (df["absences"].clip(0, MAX_ABSENCES) / MAX_ABSENCES * 100)
    ).round(2)

    # Average internal marks: mean of the two internal periodic assessments G1, G2
    df["avg_internal_marks"] = df[["G1", "G2"]].mean(axis=1).round(2)

    # Previous-semester performance: G2 acts as the "previous" mark relative to
    # the final grade G3 (the dataset's three periodic grades approximate three
    # checkpoints across the term).
    df["previous_semester_performance"] = df["G2"]

    # Assignment completion rate proxy: no direct LMS/assignment field exists in
    # the UCI dataset, so `studytime` (1-4 scale) and `failures` (past class
    # failures) are combined into a 0-100 completion proxy: more weekly study
    # time and fewer past failures -> higher assumed completion rate.
    df["assignment_completion_rate"] = (
        ((df["studytime"] / 4) * 0.7 + (1 - df["failures"].clip(0, 3) / 3) * 0.3) * 100
    ).round(2)

    # Final performance category
    df["final_grade_percent"] = (df["G3"] / 20 * 100).round(2)
    df["pass_fail"] = np.where(df["G3"] >= RISK_FINAL_GRADE_THRESHOLD, "Pass", "Fail")

    # Dropout / intervention risk flag: fails OR low attendance OR repeated
    # past failures OR sharp grade decline are treated as risk signals.
    grade_decline = (df["G1"] - df["G3"]) >= 4
    risk_condition = (
        (df["G3"] < RISK_FINAL_GRADE_THRESHOLD)
        | (df["attendance_percentage"] < RISK_ATTENDANCE_THRESHOLD)
        | (df["failures"] >= 1)
        | grade_decline
    )
    df["risk_flag"] = np.where(risk_condition, "High Risk", "Low Risk")

    # Simple 0-100 risk score for ranking the high-risk list on the dashboard
    df["risk_score"] = (
        (100 - df["final_grade_percent"]) * 0.4
        + (100 - df["attendance_percentage"]) * 0.3
        + (df["failures"].clip(0, 3) / 3 * 100) * 0.2
        + (grade_decline.astype(int) * 100) * 0.1
    ).round(1)

    return df


def build_analytics_tables():
    log.info("=== Starting transformation task ===")
    df = _load_cleaned()
    df = engineer_features(df)

    master_path = os.path.join(ANALYTICS_DIR, "student_master.csv")
    df.to_csv(master_path, index=False)
    log.info(f"student_master: {len(df)} rows -> {master_path}")

    subject_summary = (
        df.groupby("subject")
        .agg(
            total_students=("student_id", "count"),
            pass_count=("pass_fail", lambda s: (s == "Pass").sum()),
            fail_count=("pass_fail", lambda s: (s == "Fail").sum()),
            avg_final_grade_percent=("final_grade_percent", "mean"),
            avg_attendance_percentage=("attendance_percentage", "mean"),
            high_risk_count=("risk_flag", lambda s: (s == "High Risk").sum()),
        )
        .reset_index()
    )
    subject_summary["pass_rate_percent"] = (
        subject_summary["pass_count"] / subject_summary["total_students"] * 100
    ).round(2)
    subject_summary["avg_final_grade_percent"] = subject_summary["avg_final_grade_percent"].round(2)
    subject_summary["avg_attendance_percentage"] = subject_summary["avg_attendance_percentage"].round(2)
    subject_summary_path = os.path.join(ANALYTICS_DIR, "subject_summary.csv")
    subject_summary.to_csv(subject_summary_path, index=False)
    log.info(f"subject_summary -> {subject_summary_path}")

    dept_summary = (
        df.groupby(["school", "subject"])
        .agg(
            total_students=("student_id", "count"),
            avg_final_grade_percent=("final_grade_percent", "mean"),
            avg_attendance_percentage=("attendance_percentage", "mean"),
            high_risk_count=("risk_flag", lambda s: (s == "High Risk").sum()),
        )
        .reset_index()
    )
    dept_summary["avg_final_grade_percent"] = dept_summary["avg_final_grade_percent"].round(2)
    dept_summary["avg_attendance_percentage"] = dept_summary["avg_attendance_percentage"].round(2)
    dept_summary_path = os.path.join(ANALYTICS_DIR, "department_summary.csv")
    dept_summary.to_csv(dept_summary_path, index=False)
    log.info(f"department_summary -> {dept_summary_path}")

    log.info("=== Transformation complete ===")
    return df, subject_summary, dept_summary


if __name__ == "__main__":
    build_analytics_tables()
