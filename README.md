# Mayhem Attribute Conflation

**CRWN102 Project B: Places Attribute Conflation**  
Contributors: Jaskaran Singh, Varnit Balivada

## Project Overview

When a single real-world place (business, venue, etc.) is covered by multiple data sources, key attributes—such as name, address, phone, and category—often conflict or drift over time. Overture Maps requires an automated system that consistently selects the cleanest and most accurate value for every attribute, creating a unified, high-quality data record for each place.

This project builds a proof-of-concept pipeline that:
- Establishes a labeled ground-truth conflation dataset from pre-matched places (using sources like the Yelp academic dataset as proxy).
- Develops, compares, and benchmarks both machine learning (ML) and non-ML (heuristic, rule-based) selection algorithms.
- Analyzes the quality and reliability of attribute selection logic, guiding Overture's strategy for future production-scale data integration.

## Problem Statement

- **Challenge:** How do we reliably unify conflicting place attributes from multiple vendors/sources?
- **Goal:** Maximize data quality by selecting the best value for each attribute.
- **Key Questions:**
  - When sources disagree, which attribute do we pick—and how do we justify it?
  - Can advanced ML truly outperform rigorous, transparent heuristics?
  - How do we ensure both accuracy and scalability for vast datasets?

## Project Deliverables

- A labeled, ground-truth dataset for place attribute conflation.
- A ready-to-benchmark selection algorithm (ML and/or heuristic).
- A performance evaluation report for both technical and business stakeholders.
- Realistic recommendations for scaling to the production Overture dataset.

## Resources

- **Data:** Pre-matched places from Overture Maps; additional ground-truth proxy labeling from Yelp or similar open datasets.
- **Tools & Docs:** See below for scripts, guides, and intermediate artifacts.

---

## Objectives & Key Results (OKRs)

The full OKRs are available in [this Google Doc](https://docs.google.com/document/d/1OBkdFtXzfp6RGqbLMM2DDKwtH_EX1DsRVup6oDc-I-A/edit?usp=sharing).

**Objective 1: Establish a definitive ground truth for place attribute selection**  
*Foundation for reliable model development and benchmarking:*
- KR1: Create a high-quality labeled "Golden Dataset" with at least 5,000 pre-matched places, using a reliable ground-truth proxy for labeling.
- KR2: Achieve an inter-annotator agreement score of >95% on a sample of 200 records to ensure labeling consistency and quality.
- KR3: Define and document at least 5 key attributes for conflation (e.g., name, phone, website, address, category) with clear labeling guidelines.

**Objective 2: Develop a superior attribute selection algorithm**  
*Automate unified record creation with high accuracy:*
- KR1: Develop a selection algorithm that outperforms a “most recent” baseline by 15%+ on F1-score.
- KR2: Achieve >0.90 F1-score for “name” selection on a 1,000-place held-out test set.
- KR3: Reduce manual review by resolving attributes automatically for >99% of evaluation places.

**Objective 3: Final recommendations and analysis**  
*Deliver actionable insights and strategy:*
- KR1: Produce a comparative analysis report on ML vs. rule-based approaches (accuracy, speed, scalability).
- KR2: Identify and document the top 3 edge cases for future improvement.
- KR3: Provide a data-driven recommendation and cost-benefit analysis for scaling.
- KR4: Present findings to answer: “Can ML truly beat a well-designed heuristic in this context?”

*[Full OKRs and progress tracking](https://docs.google.com/document/d/1OBkdFtXzfp6RGqbLMM2DDKwtH_EX1DsRVup6oDc-I-A/edit?usp=sharing)*

---

## Project Workflow

1. **Data exploration and documentation:**  
   - Understand the pre-matched attribute dataset ([explore_dataset.py](scripts/explore_dataset.py); see [human summary](docs/data_exploration_report.md)).
2. **Guideline creation:**  
   - Develop and refine explicit, example-driven [attribute labeling guidelines](docs/attribute_guidelines.md).
3. **Annotation:**  
   - **Manual**: Use [annotate.py](scripts/annotate.py) to label a sample for IAA and the full dataset for model training.
   - **AI-Powered**: Use [annotate_ai.py](scripts/annotate_ai.py) with LM Studio for automatic annotation (see [setup guide](docs/ai_annotation_setup.md)).
   - Reference [annotations_format.md](docs/annotations_format.md) for format details.
4. **Agreement study:**  
   - Use [agreement_sample_200.csv](data/agreement_sample_200.csv) and [calculate_agreement.py](scripts/calculate_agreement.py) to ensure >95% IAA before expanding.
5. **Algorithm development:** (to be completed)  
   - Design, build, and evaluate attribute selectors (ML and rules).
6. **Final analysis and reporting:** (to be completed)  
   - Summarize findings, make technical recommendations, and document for future Overture data integration.

---

## Project Structure

### Data and Exploration
- [`data/project_b_samples_2k.parquet`](data/project_b_samples_2k.parquet): Source, pre-matched, multi-source records.
- [`scripts/explore_dataset.py`](scripts/explore_dataset.py): CLI for data profiling and sampling.
- [`docs/data_exploration_report.md`](docs/data_exploration_report.md): Human-focused summary of dataset stats and structure.
- [`docs/data_exploration_summary.json`](docs/data_exploration_summary.json): Machine-readable coverage/summary stats.

### Labeling & Agreement
- [`scripts/annotate.py`](scripts/annotate.py): Interactive labeling tool.
- [`scripts/annotate_ai.py`](scripts/annotate_ai.py): AI-powered automatic annotation using LM Studio (local LLM).
- [`docs/attribute_guidelines.md`](docs/attribute_guidelines.md): Attribute-specific labeling rules and edge cases.
- [`docs/annotations_format.md`](docs/annotations_format.md): Schema for annotation JSON output.
- [`docs/ai_annotation_setup.md`](docs/ai_annotation_setup.md): Setup guide for AI annotation with LM Studio.
- [`data/agreement_sample_200.csv`](data/agreement_sample_200.csv): Ready-to-label sample for agreement study.

### Agreement Analysis
- [`scripts/calculate_agreement.py`](scripts/calculate_agreement.py): Computes IAA, Cohen’s Kappa, and prints sample disagreements.

### Project Coordination
- [Full OKRs & progress (Google Doc)](https://docs.google.com/document/d/1OBkdFtXzfp6RGqbLMM2DDKwtH_EX1DsRVup6oDc-I-A/edit?usp=sharing): Project objectives, deliverables, and tracked progress.

---

## Getting Started

1. Clone the repo and install Python dependencies (`pandas`, `pyarrow`, etc.).
2. Explore the data and guidelines to understand project context.
3. Annotate a sample (or full dataset) using the CLI tools.
4. Run agreement and analysis scripts to prepare for modeling.

---

## Acknowledgements

- Data: Overture Maps and Yelp Academic Dataset (as proxy ground truth).
- Project and OKRs by Jaskaran Singh, Varnit Balivada.

See [`LICENSE`](LICENSE) for licensing and terms.