# Mayhem Project: Places Attribute Conflation

## Overview
This project addresses the challenge of place attribute conflation for Overture Maps. When multiple data sources describe the same real-world place, attributes like name, address, phone, and category often conflict. The goal is to build an automated system to select the most accurate and consistent attribute values, thereby creating high-quality, unified place records.

This repository contains the proof-of-concept pipeline, including:
-   A **Golden Dataset** generation process, combining manual annotation with synthetic data.
-   **Machine Learning (ML)** models for attribute selection (Name, Phone, Website, Address, Category).
-   **Rule-based heuristics** for baseline comparison.
-   Tools for **evaluation, benchmarking**, and **workflow orchestration**.

## Key Achievements & Progress (Snapshot)
-   **Golden Dataset:** Created a 2,000-record synthetic training dataset and a 200-record real-world validation dataset.
-   **ML Pipeline:** Fully implemented training and inference for 5 core attributes (Name, Phone, Website, Address, Category).
-   **Baselines:** Integrated rule-based heuristics for comparative analysis.
-   **Metrics:** Pipeline is set up to collect performance (F1, Accuracy) and compute (time, memory) metrics.

## Getting Started
1.  **Clone the repository:** `git clone https://github.com/project-terraforma/Mayhem_Attribute_Conflation.git`
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Setup Yelp Data:** Ensure `yelp/yelp_academic_dataset_business.json` is present (it's tracked via Git LFS).
4.  **Run the pipeline:** Execute `python scripts/run_algorithm_pipeline.py` to generate models, predictions, and reports for all attributes.
5.  **Use Colab:** See `notebooks/colab_pipeline_setup.ipynb` for a cloud-based workflow.

## Detailed Project Information
For comprehensive details on Objectives & Key Results (OKRs), detailed progress updates, project structure, and workflow, please refer to:
[docs/README.md](docs/README.md)

See [`LICENSE`](LICENSE) for licensing and terms.
