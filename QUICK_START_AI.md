# Quick Start: AI Annotation

## On Your Desktop (RTX 4070 Super)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start LM Studio
1. Open LM Studio
2. Download model: **Llama 3.1 8B Instruct** (Q4_K_M quantization)
3. Load model in Chat tab
4. Start Local Server (runs on `http://localhost:1234`)

### 3. Test Connection (5 records)
```bash
python scripts/annotate_ai.py --test
```

### 4. Run Full Annotation (200 records)
```bash
python scripts/annotate_ai.py --input data/agreement_sample_200.csv --output data/annotations_ai_agent.json
```

## Expected Time
- **200 records**: ~10-15 minutes
- **Progress saved every 25 records** (can resume if interrupted)

## Output
- File: `data/annotations_ai_agent.json`
- Format: Same as manual annotations (compatible with `calculate_agreement.py`)

## Full Documentation
See [`docs/ai_annotation_setup.md`](docs/ai_annotation_setup.md) for detailed instructions.

