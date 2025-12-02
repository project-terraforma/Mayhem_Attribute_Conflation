# Hybrid Approach

This directory represents the Hybrid selection approach.

## Approach
The Hybrid approach combines predictions from multiple baselines (and potentially ML models) using a weighted voting system.

## Logic
Implemented in `scripts/baseline_heuristics.py` (Class: `HybridBaseline`).

## Weights
*   Recency: 0.3
*   Confidence: 0.5
*   Completeness: 0.2

This weighted ensemble aims to balance the strengths of different heuristics.
