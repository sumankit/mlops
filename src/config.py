"""
Central configuration for the Student Performance and Dropout-Risk pipeline.
All paths are relative to the project root so the pipeline is reproducible
on any machine after `git clone` + `pip install -r requirements.txt`.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data layers -----------------------------------------------------------
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
ANALYTICS_DIR = os.path.join(PROJECT_ROOT, "data", "analytics")
REJECTED_DIR = os.path.join(PROJECT_ROOT, "data", "rejected")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

for _d in (RAW_DIR, STAGING_DIR, CLEANED_DIR, ANALYTICS_DIR, REJECTED_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Source files ------------------------------------------------------
# UCI Student Performance Dataset (Cortez, 2008)
# Source: https://archive.ics.uci.edu/dataset/320/student+performance
# License: free for academic/research use, publicly downloadable, no auth required.
SOURCE_FILES = {
    "mathematics": os.path.join(RAW_DIR, "student_performance_uci", "student-mat.csv"),
    "portuguese": os.path.join(RAW_DIR, "student_performance_uci", "student-por.csv"),
}

# --- Database ---------------------------------------------------------
# SQLite is used for the local demo/submission. The schema is written in
# plain ANSI SQL and is portable to PostgreSQL (see docs/data_dictionary.md
# for the equivalent `CREATE TABLE` statements and the note on swapping the
# SQLAlchemy connection string to a postgresql+psycopg2:// URL).
DB_PATH = os.path.join(PROJECT_ROOT, "db", "student_performance.db")
DB_URI = f"sqlite:///{DB_PATH}"

# --- Business rules used during validation -----------------------------
VALID_GRADE_RANGE = (0, 20)          # UCI grades G1/G2/G3 are on a 0-20 scale
VALID_AGE_RANGE = (14, 25)
MAX_ABSENCES = 93                     # ~full academic year of school days, used to derive attendance %
RISK_ATTENDANCE_THRESHOLD = 75.0      # % below this is flagged low-attendance
RISK_FINAL_GRADE_THRESHOLD = 10.0     # G3 < 10 (out of 20) is a fail in the source study
