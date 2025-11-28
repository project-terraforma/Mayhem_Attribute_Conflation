# AI Annotation Setup Guide

This guide explains how to set up and run the AI-powered annotation script using LM Studio on your desktop computer.

## Prerequisites

1. **Desktop Computer with RTX 4070 Super** (12GB VRAM)
2. **LM Studio** installed and running
3. **Python 3.8+** with pip

## Step 1: Install LM Studio

1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch LM Studio

## Step 2: Download and Load a Model

Recommended models for RTX 4070 Super (12GB VRAM):

### Primary Recommendation:
- **Llama 3.1 8B Instruct** (Q4_K_M quantization)
  - Download from: Hugging Face (search "Llama-3.1-8B-Instruct-GGUF")
  - Quantization: Q4_K_M (good balance of quality and speed)

### Alternatives:
- **Mistral 7B Instruct v0.3** (Q4_K_M)
- **Qwen 2.5 7B Instruct** (Q4_K_M)

### Steps in LM Studio:
1. Go to "Search" tab
2. Search for one of the recommended models
3. Download the Q4_K_M quantized version
4. Go to "Chat" tab
5. Select the downloaded model
6. Click "Start Server" (or ensure "Local Server" is enabled)
7. Note: Server runs on `http://localhost:1234` by default

## Step 3: Install Python Dependencies

On your desktop computer, navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

This installs:
- `openai` (for LM Studio API compatibility)
- `pandas` (data manipulation)
- `pyarrow` (parquet support)
- `tqdm` (progress bars)

## Step 4: Verify LM Studio is Running

Before running the annotation script, ensure:
1. LM Studio is open
2. A model is loaded in the Chat tab
3. The local server is running (check the "Local Server" section)

You can test the connection by running:

```bash
python scripts/annotate_ai.py --test
```

This will attempt to connect and process 5 test records.

## Step 5: Run Annotation

### Test Run (5 records):
```bash
python scripts/annotate_ai.py --test --input data/agreement_sample_200.csv --output data/annotations_ai_agent.json
```

### Full Run (200 records):
```bash
python scripts/annotate_ai.py --input data/agreement_sample_200.csv --output data/annotations_ai_agent.json
```

### Custom Options:
```bash
# Specify model name explicitly
python scripts/annotate_ai.py --model "llama-3.1-8b-instruct" --input data/agreement_sample_200.csv

# Adjust delay between API calls (default: 0.1 seconds)
python scripts/annotate_ai.py --delay 0.2 --input data/agreement_sample_200.csv

# Process specific range
python scripts/annotate_ai.py --start 0 --end 100 --input data/agreement_sample_200.csv
```

## Script Options

```
--input          Input CSV file (default: data/agreement_sample_200.csv)
--output         Output JSON file (default: data/annotations_ai_agent.json)
--start          Starting record index (default: 0)
--end            Ending record index (default: all)
--model          Model name (auto-detected if not specified)
--api-url        LM Studio API URL (default: http://localhost:1234/v1)
--delay          Delay between API calls in seconds (default: 0.1)
--save-interval  Save every N records (default: 25)
--annotator      Annotator name (default: ai_agent)
--test           Test mode: process only first 5 records
```

## Expected Performance

With RTX 4070 Super and Llama 3.1 8B (Q4_K_M):
- **Speed**: ~5-10 tokens/second
- **Time for 200 records**: ~10-15 minutes
- **Memory usage**: ~8-10GB VRAM

## Output Format

The script generates a JSON file with the same format as manual annotations:

```json
[
  {
    "record_index": 0,
    "id": "08f44f055a9a016e0390f050aa3c93c0",
    "choice": "c",
    "notes": "Current version has better phone format (+1 prefix) and more complete address",
    "annotator": "ai_agent",
    "model": "llama-3.1-8b-instruct"
  }
]
```

## Troubleshooting

### Connection Error
- **Problem**: "Could not connect to LM Studio"
- **Solution**: Ensure LM Studio is running with a model loaded and server started

### Out of Memory
- **Problem**: CUDA out of memory errors
- **Solution**: Use a smaller model or lower quantization (Q3_K_M instead of Q4_K_M)

### Slow Performance
- **Problem**: Very slow processing
- **Solution**: 
  - Check GPU utilization in Task Manager
  - Ensure model is using GPU (not CPU)
  - Try a smaller model or lower quantization

### Invalid JSON Responses
- **Problem**: Script marks many records as "unclear" due to JSON errors
- **Solution**: 
  - Try a different model (some models follow JSON format better)
  - Check LM Studio logs for errors
  - The script will retry up to 3 times automatically

## Next Steps

After annotation:
1. Validate output format: `python scripts/calculate_agreement.py --ann1 data/annotations_ai_agent.json --ann2 data/annotations_jask.json`
2. Compare AI annotations with manual annotations
3. Review edge cases and refine guidelines if needed

## Notes

- The script automatically saves progress every 25 records
- If interrupted, re-running will skip already-annotated records
- Temperature is set to 0.2 for consistency (lower = more deterministic)
- The script uses the attribute guidelines from `docs/attribute_guidelines.md`

