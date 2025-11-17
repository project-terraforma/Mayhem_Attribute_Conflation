# Algorithm Development Guide for Objective 2 (OKR 2)

This guide explains how to develop and evaluate attribute selection algorithms.

## Quick Start

### 1. Generate Sample Golden Dataset

Since we don't have the real golden dataset yet, we'll use a sample one generated from heuristics:

```bash
python scripts/generate_sample_golden_dataset.py --split --attributes name
```

This creates:
- `data/processed/golden_dataset_sample.json` - Full dataset with labels
- `data/processed/golden_dataset_train.json` - Training set
- `data/processed/golden_dataset_validation.json` - Validation set
- `data/processed/golden_dataset_test.json` - Test set (1,000 records)

### 2. Extract Features

Extract features for ML training:

```bash
python scripts/extract_features.py --attribute name
```

This creates:
- `data/processed/features_name.parquet` - Features and labels for training

### 3. Evaluate Baseline Heuristics

Test the baseline algorithms:

```bash
# Most Recent baseline (primary baseline for KR1)
python scripts/baseline_heuristics.py --baseline most_recent --attribute name

# Other baselines
python scripts/baseline_heuristics.py --baseline confidence --attribute name
python scripts/baseline_heuristics.py --baseline completeness --attribute name
python scripts/baseline_heuristics.py --baseline hybrid --attribute name
```

### 4. Train ML Models

Train machine learning models:

```bash
python scripts/train_models.py --features data/processed/features_name.parquet
```

This trains:
- Logistic Regression
- Random Forest
- Gradient Boosting

And saves the best model to `models/ml_models/`

### 5. Run Complete Pipeline

Run everything at once:

```bash
python scripts/run_algorithm_pipeline.py --attribute name
```

## Algorithm Components

### Baseline Heuristics

Located in `scripts/baseline_heuristics.py`:

1. **MostRecentBaseline**: Always selects current version (baseline for KR1)
2. **ConfidenceBaseline**: Selects version with higher confidence score
3. **CompletenessBaseline**: Selects version with more complete data
4. **HybridBaseline**: Combines multiple heuristics

### Feature Extraction

Located in `scripts/extract_features.py`:

Extracts features for name attribute:
- String similarity metrics (Levenshtein, Jaro-Winkler)
- Length features
- Capitalization quality
- Punctuation features
- Metadata features (confidence scores, source counts)

### ML Models

Located in `scripts/train_models.py`:

Trains three model types:
- Logistic Regression (with feature scaling)
- Random Forest (with feature importance)
- Gradient Boosting

### Evaluation

Located in `scripts/evaluate_models.py`:

Provides:
- F1-score, accuracy, precision, recall
- Coverage metrics (for KR3)
- Comparison between algorithms

## Key Results (OKR 2)

### KR1: Beat Baseline by 15%+
- Baseline: Most Recent heuristic F1-score
- Target: Your algorithm F1-score ≥ baseline F1 × 1.15

### KR2: Achieve >0.90 F1 on Test Set
- Evaluate on held-out test set (1,000 places)
- Focus on name attribute selection
- Target: F1-score > 0.90

### KR3: Process >99% Automatically
- Coverage: % of places that don't need manual review
- Target: Coverage > 99%

## File Structure

```
scripts/
  ├── generate_sample_golden_dataset.py  # Create sample ground truth
  ├── baseline_heuristics.py              # Rule-based algorithms
  ├── extract_features.py                # Feature engineering
  ├── train_models.py                     # ML model training
  ├── evaluate_models.py                  # Evaluation framework
  └── run_algorithm_pipeline.py           # Complete pipeline

models/
  ├── rule_based/                         # Saved rule-based models
  ├── ml_models/                          # Trained ML models
  └── feature_extractors/                 # Feature extraction code

data/
  ├── processed/
  │   ├── golden_dataset_sample.json      # Sample ground truth
  │   ├── golden_dataset_train.json       # Training set
  │   ├── golden_dataset_validation.json  # Validation set
  │   ├── golden_dataset_test.json        # Test set
  │   └── features_*.parquet              # Extracted features
  └── results/
      ├── baseline_metrics.json           # Baseline performance
      └── model_comparison.json           # All models compared
```

## When Real Golden Dataset is Ready

1. Replace `data/processed/golden_dataset_sample.json` with the real dataset
2. Ensure it follows the same format (JSON array with `id`, `labels` fields)
3. Re-run feature extraction: `python scripts/extract_features.py --attribute name`
4. Re-train models: `python scripts/train_models.py --features data/processed/features_name.parquet`
5. Re-evaluate everything

## Next Steps

1. **Improve Features**: Add more sophisticated features (e.g., semantic similarity, domain-specific rules)
2. **Try More Models**: Experiment with XGBoost, LightGBM, neural networks
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Error Analysis**: Analyze failure cases and improve
5. **Extend to Other Attributes**: Apply to phone, website, address, category

## Troubleshooting

**Import errors**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**File not found**: Make sure you're running scripts from the project root directory.

**Low F1-scores**: 
- Check feature quality
- Try different models
- Analyze error cases
- Consider ensemble methods

