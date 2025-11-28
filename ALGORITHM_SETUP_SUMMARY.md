# Algorithm Development Setup Summary

## What We've Built

I've set up a complete algorithm development infrastructure for **Objective 2 (OKR 2)**. Even though you don't have the real golden dataset yet, you can start developing and testing algorithms right away using a sample dataset.

## Created Files

### Core Scripts

1. **`scripts/generate_sample_golden_dataset.py`**
   - Generates a sample ground truth dataset using rule-based heuristics
   - Creates train/validation/test splits (test set = 1,000 places for KR2)
   - When real golden dataset is ready, just swap the file!

2. **`scripts/baseline_heuristics.py`**
   - **MostRecentBaseline**: Always picks current (baseline for KR1)
   - **ConfidenceBaseline**: Picks higher confidence score
   - **CompletenessBaseline**: Picks more complete data
   - **HybridBaseline**: Combines multiple heuristics

3. **`scripts/extract_features.py`**
   - Extracts features for ML training
   - String similarity (Levenshtein, Jaro-Winkler)
   - Capitalization, punctuation, length features
   - Metadata features (confidence, sources)

4. **`scripts/train_models.py`**
   - Trains Logistic Regression, Random Forest, Gradient Boosting
   - Automatically selects best model
   - Saves models for inference

5. **`scripts/evaluate_models.py`**
   - Calculates F1-score, accuracy, precision, recall
   - Measures coverage (for KR3)
   - Compares multiple algorithms

6. **`scripts/run_algorithm_pipeline.py`**
   - Runs the complete pipeline end-to-end
   - One command to do everything

### Documentation

- **`docs/algorithm_development_guide.md`**: Complete guide on how to use everything

### Folder Structure

```
models/
  ├── rule_based/          # For saved rule-based models
  ├── ml_models/           # Trained ML models saved here
  └── feature_extractors/  # Feature extraction code

data/
  ├── processed/           # Generated datasets and features
  └── results/             # Evaluation results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Golden Dataset

```bash
python scripts/generate_sample_golden_dataset.py --split --attributes name
```

This creates:
- `data/processed/golden_dataset_sample.json` - Full dataset
- `data/processed/golden_dataset_train.json` - Training set
- `data/processed/golden_dataset_validation.json` - Validation set
- `data/processed/golden_dataset_test.json` - **Test set (1,000 places for KR2)**

### 3. Extract Features

```bash
python scripts/extract_features.py --attribute name
```

Creates: `data/processed/features_name.parquet`

### 4. Test Baseline (Most Recent)

```bash
python scripts/baseline_heuristics.py --baseline most_recent --attribute name
```

This is your **baseline for KR1** - you need to beat this by 15%+.

### 5. Train ML Models

```bash
python scripts/train_models.py --features data/processed/features_name.parquet
```

Trains 3 models and saves the best one to `models/ml_models/`

### 6. Or Run Everything at Once

```bash
python scripts/run_algorithm_pipeline.py --attribute name
```

## OKR 2 Targets

### KR1: Beat Baseline by 15%+
- **Baseline**: Most Recent heuristic F1-score
- **Target**: Your algorithm F1 ≥ baseline F1 × 1.15
- **How**: Compare your model's F1 to baseline F1

### KR2: Achieve >0.90 F1 on Test Set
- **Test Set**: 1,000 places (already split out)
- **Attribute**: Name (focus for now)
- **Target**: F1-score > 0.90
- **How**: Evaluate on `data/processed/golden_dataset_test.json`

### KR3: Process >99% Automatically
- **Coverage**: % of places that don't need manual review
- **Target**: Coverage > 99%
- **How**: Algorithm should make confident predictions (not "unclear") for >99% of cases

## When Real Golden Dataset is Ready

1. Replace `data/processed/golden_dataset_sample.json` with your real dataset
2. Make sure it has the same format:
   ```json
   [
     {
       "id": "record_id",
       "labels": {
         "name": "current|base|same|unclear"
       }
     }
   ]
   ```
3. Re-run feature extraction and training
4. Everything else stays the same!

## Next Steps

1. **Run the pipeline** to see baseline performance
2. **Improve features** - add more sophisticated string similarity, domain knowledge
3. **Try different models** - XGBoost, LightGBM, neural networks
4. **Hyperparameter tuning** - optimize model parameters
5. **Error analysis** - look at failure cases and improve
6. **Extend to other attributes** - phone, website, address, category

## Notes

- The sample golden dataset uses heuristics, so it's not perfect, but it's good enough to start algorithm development
- All scripts are designed to work with the sample dataset now, and easily swap in the real one later
- The baseline "Most Recent" is intentionally simple - it's meant to be beaten!
- Focus on the **name** attribute first (KR2 requirement), then extend to others

## Troubleshooting

**ModuleNotFoundError**: Install dependencies with `pip install -r requirements.txt`

**File not found**: Make sure you're in the project root directory

**Low scores**: This is expected with sample data! The real golden dataset will give better results.

---

You're all set! Start with `python scripts/generate_sample_golden_dataset.py --split --attributes name` and work your way through the pipeline.

