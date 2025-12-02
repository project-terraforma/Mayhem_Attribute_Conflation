# Rule-Based Baselines

This directory represents the Rule-Based selection approach.

## Approach
Unlike ML models, rule-based approaches do not require training artifacts. The logic is implemented directly in `scripts/baseline_heuristics.py`.

## Heuristics Implemented
1.  **Most Recent:** Selects the version from the most recently updated source (Current).
2.  **Confidence:** Selects the version with the higher confidence score from the data provider.
3.  **Completeness:** Selects the version with more complete data (e.g., valid JSON structure, more fields filled).

## Execution
Run via the main pipeline: `python scripts/run_algorithm_pipeline.py`
