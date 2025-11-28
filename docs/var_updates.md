# Var Updates – Objective 2 Additions

## Context
These updates introduce the entire Objective 2 (OKR 2) algorithm pipeline so you and your partner can experiment immediately, even before the real golden dataset is ready. Everything plugs into the existing annotation/OKR work and prepares us for Objective 3 analysis.

## New/Updated Files

### scripts/generate_sample_golden_dataset.py
- **What**: Synthetic golden-dataset generator with heuristics + train/val/test splits (1k record test set for KR2).
- **Why**: Lets us prototype algorithms now; later, swap in the real labels without changing downstream code.
- **Codebase context**: Complements Objective‑1 annotation tools by faking their output until the real ground truth lands.

### scripts/baseline_heuristics.py
- **What**: Four rule-based selectors (Most Recent, Confidence, Completeness, Hybrid) plus evaluation helper.
- **Why**: Establishes baselines; “Most Recent” is the KR1 comparison point (must beat by 15%+ F1).
- **Codebase context**: Lives beside annotation scripts; first automation layer that consumes the conflation dataset.

### scripts/extract_features.py
- **What**: Feature-engineering pipeline (string similarity, formatting, metadata) for current vs base attributes.
- **Why**: Supplies ML-ready data for Objective 2 models.
- **Codebase context**: Bridges the raw data/annotations with the ML training stack.

### scripts/train_models.py
- **What**: Trains Logistic Regression, Random Forest, Gradient Boosting; saves best model + scaler.
- **Why**: Core modeling entry point; outputs artifacts for evaluation/inference.
- **Codebase context**: Adds ML training capability alongside rule-based methods.

### scripts/evaluate_models.py
- **What**: Shared evaluation utilities (accuracy, precision, recall, F1, coverage, KR checks, comparison tables).
- **Why**: Standardizes how we measure KR1–KR3 progress.
- **Codebase context**: Sits between the modeling scripts and future Objective‑3 reporting.

### scripts/run_algorithm_pipeline.py
- **What**: Orchestrates the full pipeline (generate sample labels → extract features → baselines → ML training).
- **Why**: Single command to reproduce the Objective‑2 workflow on any machine.
- **Codebase context**: End-to-end driver that ties new scripts to existing data assets.

### docs/algorithm_development_guide.md
- **What**: Step-by-step instructions (commands, targets, troubleshooting, folder map).
- **Why**: Onboards anyone to the new tooling quickly.
- **Codebase context**: Companion to README/OKRs, focusing specifically on Objective 2.

### ALGORITHM_SETUP_SUMMARY.md
- **What**: High-level recap + quick start cheat sheet.
- **Why**: Gives leadership/teammates a fast overview of what’s ready.
- **Codebase context**: Root-level status doc summarizing the Objective‑2 deliverables.

### requirements.txt (updated)
- **What**: Added `scikit-learn`, `joblib`, and `numpy`.
- **Why**: Needed for feature extraction and ML models.
- **Codebase context**: Ensures both machines share identical dependencies.

## Folder Structure Changes
- Created `models/` (subdirs ready for rule-based/ML artifacts) and `data/processed` + `data/results` for pipeline outputs.

## How It Fits Overall
1. **Objective 1 assets** (annotation scripts, guidelines) still produce/validate real labels.
2. **Objective 2 additions** (above) consume the conflation dataset, simulate labels, and train/evaluate algorithms.
3. **Objective 3 work** will use these outputs—metrics, models, comparison data—to write the final recommendation.

Once the actual golden dataset arrives, drop it into `data/processed/golden_dataset_sample.json`, re-run the pipeline, and everything upgrades automatically.
