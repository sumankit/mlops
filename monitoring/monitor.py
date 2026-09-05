"""
Monitoring module (Part 2): feature drift, class-distribution shift,
prediction-performance, and latency/failure monitoring for the deployed
dropout-risk model.

Compares the live prediction log (logs/prediction_log.csv, written by
api/main.py on every /predict call) against the training reference
distribution (data/analytics/student_master.csv) and writes a timestamped
JSON report to monitoring/reports/.

Usage:
    python monitoring/monitor.py
"""
import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ANALYTICS_DIR, PROJECT_ROOT, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "monitoring.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("monitor")

PREDICTION_LOG_CSV = os.path.join(LOG_DIR, "prediction_log.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "monitoring", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

DRIFT_NUMERIC_FEATURES = ["attendance_percentage", "avg_internal_marks", "G1", "G2", "absences", "failures"]
PSI_WARN_THRESHOLD = 0.10   # PSI 0.1-0.25 -> moderate drift, monitor closely
PSI_ALERT_THRESHOLD = 0.25  # PSI > 0.25 -> significant drift, trigger retraining review
LATENCY_ALERT_MS = 200      # p95 latency budget for the inference endpoint


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Standard PSI: bins the reference distribution into `bins` quantile
    buckets, then compares the % of reference vs current observations in
    each bucket. PSI < 0.1 = stable, 0.1-0.25 = moderate drift, > 0.25 =
    significant drift requiring investigation/retraining."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(reference, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(len(reference), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(len(current), 1))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def check_feature_drift(reference_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    results = {}
    for col in DRIFT_NUMERIC_FEATURES:
        if col not in live_df.columns:
            continue
        psi = population_stability_index(
            reference_df[col].astype(float).values, live_df[col].astype(float).values
        )
        status = "STABLE"
        if psi > PSI_ALERT_THRESHOLD:
            status = "SIGNIFICANT_DRIFT"
        elif psi > PSI_WARN_THRESHOLD:
            status = "MODERATE_DRIFT"
        results[col] = {"psi": psi, "status": status}
    return results


def check_class_distribution(reference_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    ref_rate = (reference_df["risk_flag"] == "High Risk").mean()
    if "prediction" in live_df.columns and len(live_df):
        live_rate = (live_df["prediction"] == "High Risk").mean()
    else:
        live_rate = None
    shift = None if live_rate is None else round(abs(live_rate - ref_rate), 4)
    return {
        "reference_high_risk_rate": round(float(ref_rate), 4),
        "live_high_risk_rate": None if live_rate is None else round(float(live_rate), 4),
        "absolute_shift": shift,
        "status": "OK" if (shift is None or shift < 0.15) else "ALERT_CLASS_SHIFT",
    }


def check_latency_and_failures(live_df: pd.DataFrame) -> dict:
    if "latency_ms" not in live_df.columns or len(live_df) == 0:
        return {"status": "NO_DATA", "requests_logged": 0}
    p95 = float(np.percentile(live_df["latency_ms"], 95))
    return {
        "requests_logged": int(len(live_df)),
        "avg_latency_ms": round(float(live_df["latency_ms"].mean()), 2),
        "p95_latency_ms": round(p95, 2),
        "status": "OK" if p95 <= LATENCY_ALERT_MS else "ALERT_HIGH_LATENCY",
    }


def check_input_quality(live_df: pd.DataFrame) -> dict:
    """Flags out-of-range inputs that reached the API (schema validation
    already rejects most of these, so this mainly catches boundary/edge
    values worth reviewing)."""
    issues = 0
    if "attendance_percentage" in live_df.columns:
        issues += int(((live_df["attendance_percentage"] < 0) | (live_df["attendance_percentage"] > 100)).sum())
    if "G1" in live_df.columns:
        issues += int(((live_df["G1"] < 0) | (live_df["G1"] > 20)).sum())
    return {"out_of_range_inputs": issues, "status": "OK" if issues == 0 else "ALERT_BAD_INPUT"}


def run_monitoring():
    log.info("=== Starting monitoring run ===")
    reference_df = pd.read_csv(os.path.join(ANALYTICS_DIR, "student_master.csv"))

    if not os.path.exists(PREDICTION_LOG_CSV):
        log.warning("No prediction_log.csv found yet -- serve some /predict requests first.")
        live_df = pd.DataFrame()
    else:
        live_df = pd.read_csv(PREDICTION_LOG_CSV)

    report = {
        "generated_at": datetime.now().isoformat(),
        "live_requests_analyzed": int(len(live_df)),
        "feature_drift": check_feature_drift(reference_df, live_df) if len(live_df) else {},
        "class_distribution": check_class_distribution(reference_df, live_df),
        "latency_and_reliability": check_latency_and_failures(live_df),
        "input_data_quality": check_input_quality(live_df) if len(live_df) else {"status": "NO_DATA"},
    }

    alerts = []
    for feat, res in report["feature_drift"].items():
        if res["status"] != "STABLE":
            alerts.append(f"Feature drift on '{feat}': PSI={res['psi']} ({res['status']})")
    if report["class_distribution"]["status"] != "OK":
        alerts.append(f"Class distribution shift: {report['class_distribution']}")
    if report["latency_and_reliability"].get("status") == "ALERT_HIGH_LATENCY":
        alerts.append(f"High latency: p95={report['latency_and_reliability']['p95_latency_ms']}ms")
    if report["input_data_quality"].get("status") == "ALERT_BAD_INPUT":
        alerts.append(f"Bad input data: {report['input_data_quality']}")

    report["alerts"] = alerts
    report["retraining_recommended"] = len(alerts) > 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"drift_report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"Report written to {report_path}")
    if alerts:
        log.warning(f"{len(alerts)} alert(s) raised -- retraining review recommended:")
        for a in alerts:
            log.warning(f"  - {a}")
    else:
        log.info("No drift/quality alerts. Model is healthy.")

    log.info("=== Monitoring run complete ===")
    return report


if __name__ == "__main__":
    run_monitoring()
