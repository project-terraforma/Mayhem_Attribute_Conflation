# Detailed Project Progress: Mayhem Attribute Conflation

## Project Overview
This document provides a comprehensive overview of the Mayhem Attribute Conflation project's objectives, key results, and detailed progress.

## Objectives & Key Results (OKRs) - Revised

Here are the revised OKRs for the Mayhem project, along with a snapshot of our current progress against each.

### Objective 1: Establish a high-quality ground truth dataset for developing and evaluating attribute selection models

*   **KR1: Create a labeled "Golden Dataset" of 2,000 pre-matched Overture places validated against Yelp ground truth**
    *   **Status: ACHIEVED ✅**
    *   **Progress:** A 2,000-record **Synthetic Golden Dataset** has been successfully generated from Yelp data. This dataset serves as the primary training data, mimicking Overture-Yelp conflation scenarios. This approach was adopted after discovering the Overture 2k samples had minimal direct overlap with the Yelp Academic Dataset, making direct Yelp-based labeling infeasible for the full 2k Overture records.
    *   **Rationale Connection:** This method directly addresses the revised rationale for this KR, providing a meaningful scale for model development by leveraging the large Yelp dataset.

*   **KR2: Achieve ≥80% inter-annotator agreement on 200 records, documenting ≥10 distinct disagreement patterns**
    *   **Status: ACHIEVED ✅**
    *   **Progress:** An initial AI-to-AI agreement of 46% (Qwen vs. Gemma) on 200 records revealed task difficulty. Human review was then performed on 108 disagreements. The 200 records now represent a fully human-validated "diamond standard" ground truth. The `data/manual_review_decisions.json` captures all final decisions and notes.
    *   **Rationale Connection:** The manual review process, although challenging, provided invaluable insights into the nuances of attribute conflation, meeting the goal of understanding task difficulty and establishing a high-quality, albeit small, human-verified dataset.

*   **KR3: Document 5 key attributes with ≥15 edge cases and resolution strategies**
    *   **Status: PARTIALLY ACHIEVED 🟠**
    *   **Progress:** The 5 key attributes (name, phone, website, address, category) have been determined and `docs/attribute_guidelines.md` exists, outlining resolution strategies. The manual review of 108 disagreements provides a rich source for extracting edge cases.
    *   **Next Steps:** Systematically document 15 specific edge cases with their resolution strategies, drawing from the `data/manual_review_decisions.json` notes and attribute guidelines.

---

### Objective 2: Build and evaluate multiple attribute selection approaches to understand their comparative performance and trade-offs

*   **KR1: Develop 3 selection approaches (rule-based, ML, hybrid), each achieving ≥60% accuracy on held-out test set of 400 records**
    *   **Status: IN PROGRESS 🚧 (ML Achieved, Rule-Based Integrated, Hybrid Pending)**
    *   **Progress:**
        *   **ML Approach:** Fully developed and implemented for all 5 attributes (Name, Phone, Website, Address, Category). Models are trained on a 2,000-record synthetic dataset and show strong performance on a synthetic validation set (F1 ranging from ~0.78 for Name to ~1.00 for Website, Address, Category).
        *   **Real-world Evaluation (Name):** The ML model for the 'name' attribute achieved an F1-score of **0.7085 (70.85%)** on the 200 manually-labeled Overture records, surpassing the ≥60% target.
        *   **Rule-Based Approach:** `scripts/baseline_heuristics.py` has been integrated into the pipeline, allowing for the development and evaluation of rule-based selectors.
    *   **Next Steps:**
        1.  Evaluate the ML models for Phone, Website, Address, and Category on the 200 real Overture records.
        2.  Develop and evaluate the performance of at least three distinct rule-based heuristics using `scripts/baseline_heuristics.py` for each attribute on the 200 real records.
        3.  Develop a hybrid selection approach.

*   **KR2: Document ≥20 failure cases per approach, categorized into ≥5 distinct error types with analysis**
    *   **Status: NOT STARTED ⚪**
    *   **Next Steps:** After evaluating all approaches (ML, Rule-Based, Hybrid) on the 200 real records, perform a detailed error analysis for each. This involves comparing predictions against true labels to identify and categorize systematic failure patterns.

*   **KR3: Benchmark compute requirements showing inference time <100ms per record and memory <?GB for all approaches**
    *   **Status: IN PROGRESS 🚧**
    *   **Progress:** Compute logging for inference time and memory usage has been integrated into `scripts/train_models.py` (for training) and `scripts/run_inference.py` (for inference). This data is stored in training summaries and inference output JSONs.
    *   **Next Steps:**
        1.  Collect and consolidate these metrics for all approaches and attributes.
        2.  Present the average inference time per record and peak memory usage across a larger sample (e.g., the 2,000 Overture records for inference).
        3.  Compare against the target of <100ms per record.

---

### Objective 3: Deliver actionable insights and recommendations for Overture's conflation strategy

*   **Status: NOT STARTED ⚪**
*   **Next Steps:** This objective primarily involves synthesizing the results from Objectives 1 and 2 into a comprehensive report and presentation. Specific tasks include:
    *   Producing a technical report documenting the labeling process, pitfalls, and comparative analysis of approaches.
    *   Identifying top edge cases and proposing resolution strategies.
    *   Delivering design proposals for clusters and matcher confidence integration.
    *   Providing data-driven recommendations comparing all approaches across multiple dimensions.

---

## Project Structure (Detailed)

```
project/
├── .gitignore
├── LICENSE
├── README.md               <- High-level project overview
├── requirements.txt
├── data/
│   ├── agreement_sample_200.csv           <- Sample for IAA study
│   ├── annotations_gemma.json             <- Gemma's raw annotations
│   ├── annotations_jask.json              <- Jaskaran's raw annotations
│   ├── annotations_qwen.json              <- Qwen's raw annotations
│   ├── golden_dataset_200.json            <- 200 manually validated records (final ground truth)
│   ├── manual_review_decisions.json       <- Manual review decisions for disagreements
│   ├── project_b_samples_2k.parquet       <- Original 2k Overture samples
│   ├── synthetic_golden_dataset_2k.json   <- 2k Yelp-derived synthetic training data
│   ├── processed/                         <- Intermediate processed data (features)
│   │   └── features_name_synthetic.parquet
│   │   └── ... (features for other attributes)
│   └── results/                           <- Final pipeline outputs, reports
│       └── final_conflated_names_2k.json
│       └── final_golden_dataset_2k_consolidated.json
│       └── final_evaluation_report_name.txt
│       └── ... (reports, predictions for other attributes/baselines)
├── docs/                                  <- All project documentation
│   ├── README.md                          <- This detailed OKR and progress document
│   ├── AI_ANNOTATION_REPORT.md            <- Report on AI annotation experiments
│   ├── ALGORITHM_SETUP_SUMMARY.md         <- Summary of algorithm development setup
│   ├── QUICK_START_AI.md                  <- Quick start guide for AI tools
│   ├── OKRs.md                            <- Original OKRs
│   ├── ai_annotation_setup.md
│   ├── annotations_format.md
│   ├── attribute_guidelines.md
│   ├── data_exploration_report.md
│   ├── data_exploration_summary.json
│   └── var_updates.md                     <- Updates from Varnit on ML pipeline
├── notebooks/                             <- Jupyter notebooks for exploration, Colab
│   └── colab_pipeline_setup.ipynb         <- Colab setup and pipeline execution
├── scripts/                               <- All Python scripts for pipeline steps
│   ├── annotate.py                        <- Manual annotation CLI
│   ├── annotate_ai.py                     <- AI-powered annotation CLI
│   ├── calculate_agreement.py             <- Inter-annotator agreement calculation
│   ├── explore_dataset.py                 <- Initial data exploration
│   ├── generate_synthetic_dataset.py      <- Creates synthetic training data
│   ├── process_synthetic_data.py          <- Prepares synthetic data for features
│   ├── extract_features.py                <- Extracts features for ML models
│   ├── train_models.py                    <- Trains ML models
│   ├── baseline_heuristics.py             <- Implements rule-based baselines
│   ├── evaluate_models.py                 <- General evaluation framework
│   ├── evaluate_real_data.py              <- Evaluates ML models on real data
│   ├── run_inference.py                   <- Runs inference on Overture records
│   ├── run_algorithm_pipeline.py          <- Orchestrates the full pipeline
│   └── consolidate_results.py             <- Consolidates final inference results
├── models/                                <- Trained ML models and related artifacts
│   └── ml_models/
│       └── name/
│           └── best_model_*.joblib
│           └── training_summary.json
│       └── phone/
│       └── ... (for other attributes)
└── yelp/                                  <- Raw Yelp dataset files
    └── yelp_academic_dataset_business.json
```
