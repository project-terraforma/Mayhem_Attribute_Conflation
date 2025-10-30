# Annotations File Format

This document explains the JSON produced by `scripts/annotate.py`.

## Location
- Saved as `data/annotations_<annotator>.json`
  - Example: `data/annotations_jask.json`

## Structure
- A JSON array; each element corresponds to one annotated record.

```json
[
  {
    "record_index": 0,
    "id": "<place_id>",
    "choice": "c|b|s|u",
    "notes": "<optional free text>",
    "annotator": "<annotator name>"
  }
]
```

### Fields
- `record_index`: Zero-based index into the source DataFrame (`project_b_samples_2k.parquet`).
- `id`: Stable place identifier for that row (preferred for joining across annotators).
- `choice`:
  - `c` = CURRENT is better
  - `b` = BASE is better
  - `s` = SAME/equivalent
  - `u` = UNCLEAR/needs review
- `notes`: Optional rationale or context.
- `annotator`: The annotator’s name, provided via `--annotator`.

## Usage
- Two annotators independently generate JSONs using:
  - `python scripts/annotate.py --mode annotate --annotator <name> --load`
- Agreement is computed with:
  - `python scripts/calculate_agreement.py --ann1 data/annotations_A.json --ann2 data/annotations_B.json --output docs/agreement_analysis.json`

## Tips
- Use consistent `--annotator` names for clear provenance.
- Prefer annotating the pre-made sample: `data/agreement_sample_200.csv` for KR2.
- Add concise `notes` on tricky cases to improve guidelines and agreement.
