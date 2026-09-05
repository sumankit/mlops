# Model Lifecycle & Retraining Criteria

## 1. Prediction target
Binary classification: `is_high_risk` (1 = High Risk / needs early academic
intervention, 0 = Low Risk), derived in Part 1 from final grade, attendance,
past failures, and grade decline (see `docs/data_dictionary.md`).

## 2. Model lifecycle stages

| Stage | Tool | Artifact |
|---|---|---|
| Feature engineering | `src/transform.py` (Part 1) | `data/analytics/student_master.csv` |
| Data/model versioning | DVC + Git | `.dvc` pointer files, `dvc.yaml`-tracked data, Git commit history |
| Training & experimentation | `ml/train.py` | 3 candidate runs (Logistic Regression, Random Forest, XGBoost) |
| Experiment tracking | MLflow (SQLite backend: `mlflow.db`, artifacts in `mlruns/`) | params, metrics, confusion matrices, model artifacts per run |
| Model selection | Best validation ROC-AUC | `random_forest` selected in the reference run (val ROC-AUC 0.98) |
| Registration | MLflow Model Registry | `student_dropout_risk_model`, aliased `production` |
| Export for serving | joblib | `ml/models/best_model.joblib` + `ml/models/model_info.json` |
| Serving | FastAPI | `api/main.py` → `POST /predict`, `GET /health` |
| Containerization | Docker | `Dockerfile` → image `student-risk-api` |
| Monitoring | `monitoring/monitor.py` | JSON reports in `monitoring/reports/` |

## 3. Retraining criteria

Retraining is triggered when **any** of the following hold, as flagged by
`monitoring/monitor.py`:

1. **Feature drift** — Population Stability Index (PSI) for any of the
   monitored features (`attendance_percentage`, `avg_internal_marks`, `G1`,
   `G2`, `absences`, `failures`) exceeds **0.25** (significant drift). A PSI
   between 0.10–0.25 is logged as a moderate-drift warning to watch, not an
   automatic trigger.
2. **Class-distribution shift** — the live high-risk prediction rate departs
   from the training-set high-risk rate by more than **15 percentage
   points**, suggesting the incoming student population no longer resembles
   the training population.
3. **Latency/reliability degradation** — p95 inference latency exceeds
   **200ms**, or the API's failure rate rises materially (tracked via
   `/health` and API logs).
4. **Input data-quality issues** — a rising share of out-of-range or
   malformed inputs reaching the model, indicating an upstream ETL/schema
   problem that should be fixed before retraining is even considered.
5. **Scheduled review** — independent of alerts, a retraining review is
   scheduled every academic term (aligned with new G1/G2/G3 grade cycles) so
   the model incorporates the latest cohort.

## 4. Retraining procedure

1. Re-run the Part 1 pipeline (`python src/pipeline_dag.py`) to refresh
   `data/analytics/student_master.csv` with the latest records.
2. Re-run `python ml/train.py` — this re-trains all three candidate models,
   logs a new MLflow experiment run, and re-selects the best model by
   validation ROC-AUC.
3. Compare the new model's test metrics against the currently registered
   `production` alias's metrics (both available in MLflow and
   `ml/models/model_info.json`). Promote only if the new model matches or
   improves ROC-AUC/F1 without a material recall drop on the High-Risk class
   (false negatives there are the costlier error — a missed at-risk
   student).
4. If promoted, MLflow registers a new model version; move the `production`
   alias to it (`client.set_registered_model_alias(...)`) and rebuild/redeploy
   the Docker image with the refreshed `ml/models/best_model.joblib`.
5. Tag the corresponding dataset and model versions in DVC/Git (see
   `docs/versioning.md`) so every deployed model is traceable back to the
   exact training data snapshot that produced it.

## 5. Rollback

Because MLflow's Model Registry keeps every prior version, rolling back is a
single alias move: point `production` back to the previous version number
and redeploy the previously built Docker image (image tags follow the
registered model version, e.g. `student-risk-api:v1`).
