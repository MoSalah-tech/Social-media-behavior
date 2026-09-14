
# Social Media Behavior — ML Pipeline

An end-to-end machine learning pipeline built on the [Social Media User Behavior](https://www.kaggle.com/datasets/hmcodes27/social-media-user-behavior) dataset. The project's real value is not a single model score — it is a **complete, reproducible pipeline** plus an honest, diagnostic investigation of the dataset's targets.

---

## TL;DR

- Built a production-style pipeline: ingest → validate → engineer → preprocess → train → evaluate → serve.
- Trained XGBoost with 5-fold cross-validation and MLflow experiment tracking.
- Screened every categorical target in the dataset to check for learnable signal.
- Found that `primary_platform` has **no signal** (F1 ≈ random baseline).
- Found that `influencer_status` has **trivial signal** (F1 = 1.00 — a deterministic rule on one feature).
- Documented both findings with diagnostic scripts.
- The pipeline itself is correct, portable, and reusable for any tabular classification target.

---

## Why this project exists

Most beginner ML projects pick a target, train a model, and report the accuracy. That approach hides the questions that matter in real work:

1. Does the target have a learnable relationship with the features?
2. Is a high score a sign of skill, or of leakage / triviality?
3. What does the pipeline look like when the target changes?

This project answers all three. It went through three phases:

**Phase 1 — Build the pipeline.**
Target: `primary_platform` (8 classes). Result: F1-macro ≈ 0.12, essentially random. Diagnosis: the target is randomly assigned in this synthetic dataset.

**Phase 2 — Screen every candidate target.**
Wrote a screening script (`notebooks/02_screen_targets.py`) that trains a quick model on every categorical column and compares against a majority-class baseline. Ranked targets by signal strength.

**Phase 3 — Switch to a target with signal.**
Target: `influencer_status` (binary). Result: F1 = 1.0000. Diagnosis: `influencer_status` is a **deterministic rule** — `Yes` if `followers_count >= 10,000`, `No` otherwise. Perfect class separation on a single feature. Confirmed with `notebooks/03_diagnose_perfect.py`.

The final deliverable is not a "successful model." It is a **working pipeline plus evidence-based understanding of the data** — including the fact that this particular dataset is not suitable for demonstrating realistic ML performance.

---

## What the pipeline does

| Stage | File | Purpose |
| :--- | :--- | :--- |
| Load | `src/data/load.py` | Read raw CSV |
| Validate | `src/data/validate.py` | Check target exists, class balance |
| Engineer | `src/features/build.py` | Derive 8 features, convert date column |
| Preprocess | `src/data/preprocess.py` | Median/mode imputation, scaling, one-hot |
| Train | `src/models/train.py` | Stratified splits, 5-fold CV, MLflow logging, XGBoost |
| Evaluate | `src/models/evaluate.py` | Confusion matrix, ROC, PR, feature importance |
| Predict | `src/models/predict.py` | Inference on new inputs with defaults |
| Serve | `src/service/api.py` | FastAPI endpoint |

The pipeline **auto-detects** binary vs multi-class and configures XGBoost accordingly. Changing the target requires editing one line in `configs/config.yaml` — no code changes.

---





---

## Setup

### 1. Create a virtual environment

```shell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
### 2. Install dependencies

```shell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


### 3. Download the dataset
Get social_media_user_behavior.csv from Kaggle and place it at:
data/raw/social_media_user_behavior.csv

### 4. Verify
```shell
python notebooks\01_inspect.py
```

### Finding — influencer_status is trivially predictable

- Target: binary (Yes / No), 7% positive class.

Result:

```text
CV F1:        0.9951 ± 0.0098
Test F1:      0.9589
Test ROC AUC: 1.0000
```

Diagnostic (notebooks/03_diagnose_perfect.py):

A single-feature decision stump on followers_count achieves F1 = 0.9931. Range analysis shows perfect class separation:

```text
followers_count
  Yes range: [10,035, 2,110,323]
  No  range: [0, 9,879]
```
Interpretation: the dataset's creator generated the label with a threshold rule:

```python
influencer_status = "Yes" if followers_count >= 10_000 else "No"
```

The 100% score is not a modeling achievement — the target is a deterministic function of one feature. No generalizable pattern exists.

### What these findings mean

- The pipeline is correct. It faithfully learns whatever signal exists.
- The dataset is not suitable for demonstrating realistic ML performance on either target.
- A production-quality project must be able to detect these issues, not just report metrics.

### Design decisions

1. Why XGBoost?
 - Best-in-class performance on tabular data. Handles mixed numeric/categorical features after encoding. Fast, robust to outliers, interpretable via feature importance.


2. Why stratified splits?
 - Preserves class balance across train/val/test. Critical for imbalanced binary targets like influencer_status (7% positive).

3. Why median/mode defaults at inference?
 - The model expects a fixed 33-column schema. Users may supply only a subset of fields. Medians (numeric) and modes (categorical) provide a "typical user" baseline that user inputs override.

4. Why device: cpu?
 - The GTX 1650 has CUDA compatibility issues with the installed XGBoost wheel. For 2,000 rows, CPU is faster than GPU anyway (launch overhead dominates). 



