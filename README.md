# Mayhem: Places Attribute Conflation
**CRWN102 Project B** | Jaskaran Singh, Varnit Balivada

## Project Overview
This project addresses the challenge of place attribute conflation for Overture Maps. When multiple data sources (e.g., Meta, Microsoft, Foursquare) describe the same real-world place, key attributes like name, address, phone, and category often conflict. 

The **Mayhem** project implements an automated pipeline to select the most accurate and consistent attribute values, creating a high-quality, unified "Golden Record." We test two primary methods: **Rule-Based Heuristics** and **Machine Learning Models**, comparing their efficacy, scalability, and maintainability.

## Repository Structure

```markdown
project/
├── data/
│   ├── golden_dataset_200.json            # Manual ground truth (200 records, human-validated)
│   ├── synthetic_golden_dataset_2k.json   # Synthetic training data derived from Yelp
│   ├── project_b_samples_2k.parquet       # Original Overture input samples
│   ├── processed/                         # Extracted features for ML training
│   └── results/                           # Final metrics, reports, and conflated output files
│
├── docs/                                  # Project documentation & detailed reports
│   ├── OKRs.md                            # Detailed OKR tracking and progress
│   ├── attribute_guidelines.md            # Labeling rules and edge case definitions
│   ├── edge_cases.md                      # Documented edge cases and resolutions
│   └── ...
│
├── models/                                # Model artifacts and evaluation reports
│   ├── ml/                                # Trained ML models (.joblib) and training summaries
│   │   ├── name/, phone/, ...             # Subdirectories per attribute
│   ├── rule_based/                        # Evaluation results for heuristic baselines
│   │   ├── eval_most_recent/
│   │   ├── eval_confidence/
│   │   └── eval_completeness/
│   └── hybrid/                            # Evaluation results for hybrid ensemble approach
│
├── notebooks/                             # Jupyter notebooks
│   └── colab_pipeline_setup.ipynb         # Complete pipeline for Google Colab execution
│
├── scripts/                               # Core Python pipeline scripts
│   ├── run_algorithm_pipeline.py          # Main orchestrator (Data Gen -> Train -> Eval -> Inference)
│   ├── generate_synthetic_dataset.py      # Generates synthetic training data from Yelp
│   ├── extract_features.py                # Feature engineering (similarity, formats, etc.)
│   ├── train_models.py                    # ML training (Gradient Boosting, Random Forest, LogReg)
│   ├── baseline_heuristics.py             # Rule-based logic implementation
│   ├── evaluate_models.py                 # Evaluation metrics calculation
│   └── run_inference.py                   # Inference engine for Overture records
│
└── yelp/                                  # Raw Yelp dataset (tracked via Git LFS)
```

## Installation & Usage

### 1. Installation
Clone the repository and install dependencies. Note that the Yelp dataset is managed with Git LFS.

```bash
# Clone the repo
git clone https://github.com/project-terraforma/Mayhem_Attribute_Conflation.git
cd Mayhem_Attribute_Conflation

# Install dependencies
pip install -r requirements.txt

# Pull LFS data (Yelp dataset)
git lfs pull
```

### 2. Run the Full Pipeline (Local)
The master script orchestrates synthetic data generation, feature extraction, model training, evaluation, and final inference for all 5 attributes.

```bash
python scripts/run_algorithm_pipeline.py
```

**Options:**
*   `--attributes <list>`: Specific attributes to run (e.g., `name phone`). Default: all.
*   `--synthetic-limit <N>`: Number of synthetic records to generate (default 2000). Use `0` for all ~150k records.

### 3. Google Colab Workflow
For faster training on the full dataset, use the provided notebook:
*   Open `notebooks/colab_pipeline_setup.ipynb` in Google Colab.
*   Run all cells to execute the pipeline in the cloud.

### 4. Analyze Results
Generate a consolidated summary table of performance and compute metrics:

```bash
python scripts/analyze_results.py
```

## Results Summary

The project evaluated **Machine Learning** (Gradient Boosting/Logistic Regression) against **Rule-Based Baselines** (Most Recent, Confidence, Completeness) and a **Hybrid** approach.

**Performance Metrics (F1-Score on 200 Real-World Records):**

| Attribute   | Best Approach | F1-Score | ML F1 | Baseline F1 |
|:------------|:-------------:|---------:|------:|------------:|
| **Category**| **ML / Hybrid**| **0.8338** | 0.8338| 0.8338      |
| **Address** | **Rule-Based**| **0.8338** | 0.7921| 0.8338      |
| **Phone**   | **Rule-Based**| **0.8554** | 0.6929| 0.8554      |
| **Website** | **Rule-Based**| **0.8323** | 0.4600| 0.8323      |
| **Name**    | **Rule-Based**| **0.8338** | 0.2209| 0.8338      |

**Key Insights:**
*   **Rule-Based Wins:** For structured attributes (Address, Phone) and surprisingly Name/Website, simple heuristics (especially "Most Recent") proved highly effective and robust.
*   **ML Value:** ML demonstrated value in complex attributes like **Category**, where it matched the best baseline performance.
*   **Efficiency:** The pipeline is extremely efficient, with inference times averaging **~0.002 ms per record**, well below the 100ms target.

## Methodology

### Rule-Based Pipeline
Implemented in `scripts/baseline_heuristics.py`.
*   **Most Recent:** Selects based on source freshness.
*   **Confidence:** Uses the upstream provider's confidence score.
*   **Completeness:** select the value with the most information (e.g. fields in JSON).

### Machine Learning Pipeline
Implemented in `scripts/train_models.py` and `scripts/extract_features.py`.
*   **Training:** Trained on 2,000-10,000+ synthetic records generated from Yelp data to simulate "good" vs. "bad" attributes.
*   **Features:** String similarity (Levenshtein, Jaro-Winkler), formatting checks (HTTPS, international phone), and metadata features.
*   **Models:** Automatically selects between Logistic Regression, Random Forest, and Gradient Boosting based on validation F1.

---

## Acknowledgements
*   **Data:** Overture Maps Foundation, Yelp Academic Dataset.
*   **Project:** Created for CRWN102 at UCSC.

See [`LICENSE`](LICENSE) for terms.