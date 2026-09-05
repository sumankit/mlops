# Data Dictionary & Validation Rules

## Source dataset
**UCI Student Performance Dataset** (Cortez & Silva, 2008)
- URL: https://archive.ics.uci.edu/dataset/320/student+performance
- Access: free, public, no authentication required — direct ZIP download.
- Files used: `student-mat.csv` (395 records, Mathematics course), `student-por.csv` (649 records, Portuguese language course)
- License: made available by UCI Machine Learning Repository for research/academic use.

## Raw source columns (as provided by UCI)

| Column | Type | Description |
|---|---|---|
| school | string | Student's school ('GP' or 'MS') |
| sex | string | 'F' or 'M' |
| age | int | Student age (15–22) |
| address | string | 'U' (urban) / 'R' (rural) |
| famsize | string | 'LE3' (≤3) / 'GT3' (>3) |
| Pstatus | string | Parent cohabitation status |
| Medu, Fedu | int | Mother's / Father's education (0–4) |
| Mjob, Fjob | string | Mother's / Father's job |
| reason | string | Reason to choose school |
| guardian | string | Student's guardian |
| traveltime | int | Home-to-school travel time (1–4) |
| studytime | int | Weekly study time (1–4) |
| failures | int | Number of past class failures |
| schoolsup, famsup, paid, activities, nursery, higher, internet, romantic | string | yes/no flags |
| famrel | int | Quality of family relationships (1–5) |
| freetime, goout | int | Free time / going out (1–5) |
| Dalc, Walc | int | Workday / weekend alcohol consumption (1–5) |
| health | int | Health status (1–5) |
| absences | int | Number of school absences (0–93) |
| G1, G2, G3 | int | Period 1, Period 2, and Final grade (0–20 scale) |

## Engineered / analytical columns (added by the pipeline)

| Column | Derivation | Notes |
|---|---|---|
| student_id | `<MAT/POR>-<zero-padded row number>` | Synthesized standardized identifier — the UCI dataset has no native ID |
| subject | Set at ingestion from source filename | 'Mathematics' or 'Portuguese' |
| attendance_percentage | `100 - (absences / 93 * 100)` | 93 = assumed academic-year school-day count (documented assumption; source only provides an absence count) |
| avg_internal_marks | mean(G1, G2) | Average internal periodic assessment mark |
| previous_semester_performance | = G2 | Proxy for "previous checkpoint" performance relative to final grade G3 |
| assignment_completion_rate | `(studytime/4)*0.7 + (1 - failures/3)*0.3`, scaled to % | Proxy — the UCI dataset has no direct assignment-submission field |
| final_grade_percent | `G3 / 20 * 100` | |
| pass_fail | `Pass` if G3 ≥ 10 else `Fail` | Threshold from the original UCI study |
| risk_flag | `High Risk` if (G3<10) OR (attendance%<75) OR (failures≥1) OR (G1−G3≥4) | Combines fail risk, attendance risk, historical failure, and grade decline |
| risk_score | weighted 0–100 composite of the above four signals | Used to rank the high-risk student list |

## Validation rules (applied in `src/validate_clean.py`)

| Rule | Rejects when |
|---|---|
| Grade range | G1, G2, or G3 not in [0, 20] |
| Age range | age not in [14, 25] |
| Absence range | absences < 0 or > 93 |
| Required fields | school, sex, age, G1, G2, G3, or absences is null |
| Duplicate student_id | exact duplicate rows dropped before validation |

Rejected records are written with a `rejection_reason` column to `data/rejected/rejected_records.csv`.

## Database schema (SQLite demo; Postgres-portable)

```sql
CREATE TABLE dim_student (
    student_id      TEXT PRIMARY KEY,
    school          TEXT,
    sex             TEXT,
    age             INTEGER,
    address         TEXT,
    famsize         TEXT,
    "Pstatus"       TEXT,
    "Medu"          INTEGER,
    "Fedu"          INTEGER,
    "Mjob"          TEXT,
    "Fjob"          TEXT,
    guardian        TEXT,
    internet        TEXT
);

CREATE TABLE fact_assessment (
    student_id                      TEXT REFERENCES dim_student(student_id),
    subject                         TEXT,
    "G1" INTEGER, "G2" INTEGER, "G3" INTEGER,
    avg_internal_marks              NUMERIC,
    previous_semester_performance   NUMERIC,
    attendance_percentage           NUMERIC,
    assignment_completion_rate      NUMERIC,
    final_grade_percent             NUMERIC,
    pass_fail                       TEXT,
    risk_flag                       TEXT,
    risk_score                      NUMERIC,
    absences                        INTEGER,
    studytime                       INTEGER,
    failures                        INTEGER,
    extraction_timestamp            TEXT
);

CREATE TABLE agg_subject_summary (
    subject TEXT, total_students INTEGER, pass_count INTEGER, fail_count INTEGER,
    avg_final_grade_percent NUMERIC, avg_attendance_percentage NUMERIC,
    high_risk_count INTEGER, pass_rate_percent NUMERIC
);

CREATE TABLE agg_department_summary (
    school TEXT, subject TEXT, total_students INTEGER,
    avg_final_grade_percent NUMERIC, avg_attendance_percentage NUMERIC,
    high_risk_count INTEGER
);
```

To point the pipeline at PostgreSQL instead of SQLite, set:
```
DB_URI = "postgresql+psycopg2://<user>:<password>@<host>:5432/<dbname>"
```
in `src/config.py` (read from an environment variable in production — see README's "Environment variables" section) and install `psycopg2-binary`. No other code changes are required since `load_db.py` uses SQLAlchemy.
