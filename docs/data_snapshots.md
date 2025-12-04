# Data & Metrics Snapshots for Slides

Each subsection below is copy-ready for your deck. Screenshot the tables/blocks as-is (no graphs needed). I’ve noted the slide number where each snapshot fits best.

---

## Slide 5 — Ground Truth Datasets & Annotation

**Dataset Roles**  
| Dataset | Size | Source | Usage |
|---------|------|--------|-------|
| Synthetic Golden Dataset | 2,000 records | Yelp businesses with simulated noise (`data/synthetic_golden_dataset_2k.json`) | ML training + stress tests |
| Manual Golden Dataset | 200 records (183 usable after filtering) | AI + manual review on Overture pairs (`data/golden_dataset_200.json`) | Real-world validation & failure analysis |
| Overture Evaluation Set | 2,000 clusters | Pre-matched Overture sources (`data/project_b_samples_2k.parquet`) | End-to-end inference + scalability tests |

**LLM Annotation Study (Qwen vs Gemma)** — from `docs/AI_ANNOTATION_REPORT.md`  
| Metric | Value |
|--------|-------|
| Records Compared | 200 |
| Agreement Rate | 46% |
| Cohen’s Kappa | 0.227 (Fair) |
| Qwen Choices | 52% base, 29.5% current, 10% same, 8.5% unclear |
| Gemma Choices | 12% base, 77.5% current, 10% same, 0.5% unclear |
| Disagreements | 108 records → fed into manual review (`disagreements_qwen_vs_gemma.json`) |

**Annotation Flow**  
1. Generate synthetic conflicts (`scripts/generate_synthetic_dataset.py`).  
2. Run LLM annotator (`scripts/annotate_ai.py`) → structured JSON outputs.  
3. Manually reconcile disagreements via `review_disagreements.py`, storing final decisions in `data/manual_review_decisions.json`.

---

## Slide 8 — Performance: Synthetic vs Real Metrics

### Synthetic 2K (Yelp) — Name Attribute Test Set

_Source: `data/results/ml_model_test_results.json` & `baseline_heuristics.py` run on `data/processed/golden_dataset_test.json`_

| Selector | Accuracy | Precision | Recall | F1 | Notes |
|----------|----------|-----------|--------|----|-------|
| Logistic Regression | **0.9840** | 0.9842 | 0.9947 | **0.9894** | 1,000-record held-out test | 
| Most Recent Baseline | 0.7510 | — | — | ≈0.75 | Always picks current version |
| Coverage | 100% for both | | | | No “unclear” predictions |

### Real 200 (Manual Overture) — Per Attribute

_All pulled from `data/results/ml_evaluation_200_real_*.json` except Name final eval (`data/results/final_evaluation_report.txt`)._

| Attribute | Selector | Accuracy | Precision | Recall | F1 | Coverage |
|-----------|----------|----------|-----------|--------|----|----------|
| **Name** (183 usable) | Logistic Regression | 0.5683 | 0.6621 | 0.7619 | **0.7085** | 100% |
| Name Baseline (Most Recent)
| **Phone** | Logistic Regression | 0.7200 | 0.7351 | 0.9510 | **0.8293** | 100% |
| Phone Baseline (Most Recent)
| **Website** | Logistic Regression | 0.6200 | 0.7410 | 0.7203 | **0.7305** | 100% |
| **Address** | Logistic Regression | 0.3750 | 0.8750 | 0.1469 | 0.2515 | 100% |
| **Category** | Logistic Regression | 0.7150 | 0.7150 | 1.0000 | **0.8338** | 100% |

> **Note:** Rule-based approaches (Most Recent / Completeness / Hybrid) remain competitive on Name, Phone, and Website per `docs/OKRs.md` (~0.83 F1), highlighting domain shift between synthetic training and real-world labels.

---

## Slide 9 — Error Patterns & Production Constraints

### Failure Case Snippets (`data/results/failure_analysis.md`)
- **Name:** Prediction BASE vs Truth CURRENT — `08f446c25679a70e03572240a924ba2c` (“Chick-fil-A Grand Parkway North” vs “Chick-fil-A”).
- **Address:** BASE vs CURRENT — `08f3922211a9429003d34a89765a66c1` (street abbreviation vs fully qualified address).
- **Phone:** CURRENT vs BASE — `08f3cc5a1476a45403ed1a0c115b1102` (E.164 vs spaced digits).
- **Website:** CURRENT vs BASE — `08f44e38ed4518540380d8904cd5b34a` (HTTPS canonicalization).
- **Category:** CURRENT vs BASE — `08f26010a6db404403e339f85cbc6d43` (flat tag vs hierarchical path).

These highlight the main error taxonomy: formatting deltas, stale vs current data, missing subcomponents, canonical URLs, and taxonomy depth.

### Compute / Scalability Metrics (KR3)
_Logged via `scripts/run_inference.py` + `docs/OKRs.md`_

| Metric | Value |
|--------|-------|
| Inference speed (per record) | **~0.0066 ms** on `project_b_samples_2k.parquet` |
| Peak memory usage | **≈200 MB** |
| Hardware | Local CPU (no GPU required for inference) |
| Pipeline | End-to-end run via `scripts/run_algorithm_pipeline.py` |

These figures are well under the <100 ms/record target for 150–200M places/month.

---

## Slide Placement Summary

| Slide | Snapshot |
|-------|----------|
| 5 | Dataset table + LLM annotation metrics |
| 8 | Synthetic vs real performance tables |
| 9 | Failure patterns list + compute metrics table |

Feel free to lift any of the above blocks directly into your slides or screenshot them from this file. No plots required—everything is text/table-based.

---

## Slide 7 — Rule-Based vs ML vs Hybrid (Screenshot Options)

### A. Performance Matrix (Real 200 Manual Records)
_Source: `docs/OKRs.md` + `data/results/ml_evaluation_200_real_*.json`_

| Attribute | Best Approach | Best F1 | ML F1 | Rule-Based F1 (Most Recent / Completeness) | Hybrid F1 |
|-----------|---------------|---------|-------|---------------------------------------------|-----------|
| Category  | ML / Hybrid   | **0.8338** | 0.8338 | 0.8338 | 0.8094 |
| Address   | Hybrid / Rules | **0.8338** | 0.7921 | 0.8338 | 0.8338 |
| Phone     | Hybrid / Rules | **0.8554** | 0.6929 | 0.8554 | 0.8554 |
| Website   | Hybrid / Rules | **0.8323** | 0.4600 | 0.8323 | 0.8323 |
| Name      | Rules          | **0.8338** | 0.2209 | 0.8338 | 0.7667 |

**Key Callouts:**
- **Rules win** outright on Name and tie/win on Phone + Website.
- **Hybrid** matches rules on Address/Phone/Website (and keeps interpretability).
- **ML** only ties Rules on Category; struggles on real Name/Phone/Website due to domain shift.

### B. Approach Summary (use as bullets or mini-table)

| Approach | Where it Wins | Strengths | Weaknesses |
|----------|---------------|-----------|------------|
| Rule-Based (Most Recent / Completeness) | Name, Phone, Website | Deterministic, zero training, <1ms inference | Misses nuanced quality signals, brittle for noisy data |
| Machine Learning (Logistic Regression, etc.) | Category (ties), synthetic name (F1 0.989) | Captures subtle patterns, high test accuracy on synthetic data | Depends on label quality; real-world Name F1 drops to ~0.22 |
| Hybrid (weighted heuristics) | Address, Phone, Website (ties best) | Better balance of accuracy + interpretability; handles obvious cases via rules | Still inherits rule limitations; needs tuning per attribute |

### C. Speed & Coverage (same for all approaches)

| Metric | Value |
|--------|-------|
| Inference time per record | ~**0.0066 ms** (2,000-record run) |
| Coverage (200-record eval) | **100%** for ML, rules, hybrid |
| Platform | Pure Python (Sklearn + custom rules) |

Use any of the above tables/blocks as images for Slide 7.

### D. Concrete Hybrid Voting Example (Name Attribute)

_Extracted from `data/project_b_samples_2k.parquet` using `baseline_heuristics.py`_

```text
Record ID: 1407374885933937
Current name: {"primary":"Red Wing - Roswell, GA"}
Base name:    {"primary":"Red Wing"}

Individual rule outputs (name selector):
- MostRecentBaseline     → current
- ConfidenceBaseline     → same
- CompletenessBaseline   → current

Hybrid weights:
- recency_weight      = 0.3
- confidence_weight   = 0.5
- completeness_weight = 0.2

Hybrid score accumulation:
- From MostRecentBaseline (current):
    score_current += 0.3
- From ConfidenceBaseline (same):
    score_same    += 0.5
- From CompletenessBaseline (current):
    score_current += 0.2

Final scores:
- score_current = 0.3 + 0.2 = 0.5
- score_same    = 0.5
- score_base    = 0.0

Hybrid prediction: SAME (ties current in total weight, but confidence vote pushes it toward equivalence)
```

You can screenshot this block to illustrate how the hybrid ensemble combines disagreeing rules into a single decision.

### ML Feature Vector Example (Name Attribute)

_Extracted via `extract_features_for_record` on the first record of `project_b_samples_2k.parquet`_

```json
{
  "confidence_current": 0.9963,
  "confidence_base": 0.7700,
  "confidence_diff": 0.2263,
  "confidence_ratio": 1.2938,
  "sources_current_count": 24,
  "sources_base_count": 4,
  "sources_count_diff": 20,
  "name_exact_match": 1.0,
  "name_exact_match_lower": 1.0,
  "name_length_ratio": 1.0,
  "name_levenshtein_similarity": 1.0,
  "name_jaro_winkler_similarity": 1.0
  // ... ~15 more name-specific features (capitalization flags, punctuation, etc.)
}
```

Screenshot this block to show how each (current, base) pair becomes a numeric feature vector for the ML models.
