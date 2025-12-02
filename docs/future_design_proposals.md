# Future Design Proposals: Scalable Conflation

This document fulfills **Objective 3, KR3**. It outlines two critical architectural designs for scaling the conflation logic to the full Overture Maps ecosystem.

## Proposal 1: Handling Place Clusters (10-100 Records)

**Problem:** Our current model evaluates pairs (`Base` vs. `Current`). In production, clustering algorithms may group 10, 50, or 100 records for a single POI. Evaluating every pair ($N^2$) is computationally prohibitive.

### Proposed Architecture: "Tournament Selection"

**Concept:** Instead of evaluating all pairs, we treat conflation as a single-elimination tournament.

**Algorithm:**
1.  **Group:** Receive a cluster of $N$ records.
2.  **Sort:** Sort records by `timestamp` (descending) or `confidence` (descending) to seed the bracket.
3.  **Round 1:** Compare Record 1 vs. Record 2 using our **ML Model**. Winner advances.
4.  **Iterate:** Winner of (1 vs 2) plays Record 3.
5.  **Winner:** The final survivor is the "Golden Record."

**Benefits:**
*   **Efficiency:** Linear complexity $O(N)$. For 100 records, we make 99 comparisons, not 4,950.
*   **Consistency:** Order-dependence can be mitigated by seeding the "best" candidates (recent/high-confidence) first.

**Pseudocode:**
```python
def conflate_cluster(cluster_records):
    # Seed with the highest confidence record
    current_champion = sorted(cluster_records, key=lambda x: x.confidence, reverse=True)[0]
    
    for challenger in cluster_records[1:]:
        # Predict: Does Challenger beat Champion?
        winner = ml_model.predict(current_champion, challenger)
        if winner == 'challenger':
            current_champion = challenger
            
    return current_champion
```

---

## Proposal 2: Matcher Confidence Integration

**Problem:** The "matcher" (upstream system) provides a `confidence` score (0.0 - 1.0) indicating how likely two records represent the same place. Our current model treats all input pairs as valid matches.

### Proposed Architecture: "Confidence-Gated Logic"

**Concept:** Use the matcher's confidence score to dynamically switch between "Safe/Conservative" logic and "Aggressive/ML" logic.

**Thresholds:**
*   **High Confidence (> 0.9):** Treat as definitely same place. Use **ML Model** to pick best attributes. Focus on *cleaning*.
*   **Medium Confidence (0.6 - 0.9):** Ambiguous match. Use **Rule-Based "Safety" Mode**. Only merge attributes if they are strictly non-conflicting (e.g., fill nulls). Do *not* overwrite existing data.
*   **Low Confidence (< 0.6):** Treat as distinct places. **Do Not Conflate.**

**Implementation:**
1.  **Feature Engineering:** Add `matcher_confidence` as a primary feature to the ML model.
2.  **Hard Gate:** In `run_inference.py`, add a pre-check:
    ```python
    if row['matcher_confidence'] < 0.6:
        return "BASE" # Keep original, reject conflation
    ```

**Benefit:** drastically reduces the risk of "over-conflation" (merging two different businesses) which is a high-severity error in map data.
