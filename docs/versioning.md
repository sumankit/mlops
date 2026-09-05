# Dataset & Model Versioning (DVC + Git)

Datasets and trained models are content-addressed and version-tracked with
**DVC**, layered on top of a **Git** repository for code + metadata history.

## What is tracked

| Artifact | DVC pointer file | Why |
|---|---|---|
| `data/analytics/student_master.csv` | `data/analytics/student_master.csv.dvc` | the exact feature snapshot used to train a given model version |
| `ml/models/best_model.joblib` | `ml/models/best_model.joblib.dvc` | the exact serialized model deployed by the API/Docker image |

DVC stores a content hash (MD5) of each file in the small `.dvc` pointer
file, which **is** committed to Git; the actual (potentially large) data/model
file itself is kept out of Git history and cached by DVC (locally in
`.dvc/cache/` for this project — a remote such as S3/GCS/DagsHub can be
added later with `dvc remote add`).

## How it fits the lifecycle

1. `python src/pipeline_dag.py` regenerates `student_master.csv`.
2. `dvc add data/analytics/student_master.csv` snapshots the new version;
   `git commit` the updated `.dvc` pointer + code changes.
3. `python ml/train.py` trains against that exact snapshot, logs the
   dataset's SHA-256 checksum as an MLflow parameter (`dataset_checksum`),
   and exports `ml/models/best_model.joblib`.
4. `dvc add ml/models/best_model.joblib` snapshots the new model version;
   `git commit` (and optionally `git tag v1.1`) ties a specific Git commit to
   a specific data version and model version.

## Reproducing / rolling back a version

```bash
git checkout <commit-or-tag>   # restores code + .dvc pointer files
dvc checkout                   # restores the matching data/model files from the DVC cache
```

This guarantees that any past model version can be traced back to the exact
training data that produced it, and vice versa — satisfying reproducibility
without needing a hosted DVC remote for the class submission (one can be
added trivially later: `dvc remote add -d storage <url>` then `dvc push`).
