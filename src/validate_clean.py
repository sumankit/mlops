"""
Validation & Cleaning layer (Task 2).

Reads the most recent staged file per subject, applies data-quality rules,
assigns a standardized student_id, removes duplicates/invalid records, and
writes:
  - data/cleaned/<subject>_cleaned.csv   (records that pass all checks)
  - data/rejected/rejected_records.csv   (records that fail, with a reason)
"""
import os
import sys
import glob
import logging
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    STAGING_DIR, CLEANED_DIR, REJECTED_DIR, LOG_DIR,
    VALID_GRADE_RANGE, VALID_AGE_RANGE, MAX_ABSENCES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "validation.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("validate_clean")

REJECTED_CSV = os.path.join(REJECTED_DIR, "rejected_records.csv")


def _latest_staged_file(subject_key: str) -> str:
    pattern = os.path.join(STAGING_DIR, f"{subject_key}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No staged file found for {subject_key}. Run ingest.py first.")
    return files[-1]


def _standardize_ids(df: pd.DataFrame, subject_key: str) -> pd.DataFrame:
    """UCI dataset has no native student ID -> synthesize a stable, unique one
    (standardized identifier scheme: <SUBJECT_PREFIX>-<zero_padded_row>)."""
    prefix = "MAT" if subject_key == "mathematics" else "POR"
    df = df.reset_index(drop=True)
    df["student_id"] = [f"{prefix}-{i+1:04d}" for i in range(len(df))]
    return df


def _validate(df: pd.DataFrame):
    """Split df into (valid_df, rejected_df) applying data-quality rules."""
    reasons = pd.Series([""] * len(df), index=df.index)

    # Rule 1: grades must be within the valid 0-20 scale
    for col in ["G1", "G2", "G3"]:
        bad = ~df[col].between(*VALID_GRADE_RANGE)
        reasons.loc[bad] += f"invalid_{col};"

    # Rule 2: age within a plausible school-age range
    bad_age = ~df["age"].between(*VALID_AGE_RANGE)
    reasons.loc[bad_age] += "invalid_age;"

    # Rule 3: absences must be non-negative and within a school-year bound
    bad_abs = (df["absences"] < 0) | (df["absences"] > MAX_ABSENCES)
    reasons.loc[bad_abs] += "invalid_absences;"

    # Rule 4: required fields must not be null
    required_cols = ["school", "sex", "age", "G1", "G2", "G3", "absences"]
    bad_null = df[required_cols].isnull().any(axis=1)
    reasons.loc[bad_null] += "missing_required_field;"

    is_bad = reasons != ""
    rejected = df.loc[is_bad].copy()
    rejected["rejection_reason"] = reasons.loc[is_bad]
    valid = df.loc[~is_bad].copy()

    return valid, rejected


def clean_subject(subject_key: str) -> pd.DataFrame:
    staged_path = _latest_staged_file(subject_key)
    df = pd.read_csv(staged_path)
    log.info(f"Validating {subject_key}: {len(df)} staged rows from {staged_path}")

    df = _standardize_ids(df, subject_key)

    # Duplicate removal (full-row duplicates, and duplicate student_id)
    before = len(df)
    df = df.drop_duplicates(subset=["student_id"])
    dup_dropped = before - len(df)
    if dup_dropped:
        log.info(f"Dropped {dup_dropped} duplicate rows for {subject_key}")

    valid, rejected = _validate(df)

    if len(rejected):
        header = not os.path.exists(REJECTED_CSV)
        rejected["rejected_at"] = datetime.now().isoformat()
        rejected["subject_source"] = subject_key
        rejected.to_csv(REJECTED_CSV, mode="a", header=header, index=False)
        log.warning(f"Rejected {len(rejected)} rows for {subject_key} -> {REJECTED_CSV}")

    out_path = os.path.join(CLEANED_DIR, f"{subject_key}_cleaned.csv")
    valid.to_csv(out_path, index=False)
    log.info(f"Cleaned {subject_key}: {len(valid)} valid rows -> {out_path}")
    return valid


def run_validation_cleaning() -> dict:
    log.info("=== Starting validation & cleaning task ===")
    results = {}
    for subject_key in ["mathematics", "portuguese"]:
        results[subject_key] = clean_subject(subject_key)
    log.info("=== Validation & cleaning complete ===")
    return results


if __name__ == "__main__":
    run_validation_cleaning()
