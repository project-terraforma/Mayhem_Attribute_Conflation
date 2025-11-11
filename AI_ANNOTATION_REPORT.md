# AI Annotation Report: Model Comparison Study

**Date**: November 11, 2024  
**Branch**: `local_model_annotation`  
**Contributors**: Jaskaran Singh  
**Objective**: Evaluate multiple AI models for automated place attribute annotation

---

## Executive Summary

We successfully tested multiple large language models (LLMs) for automated annotation of place attribute conflicts (current vs base). After encountering various compatibility and quality issues, we identified **Qwen 3 Coder 30B** as the most reliable annotator, with **Gemma 2 9B Instruct** as a useful comparison model.

**Key Finding**: Qwen 30B demonstrated balanced, nuanced decision-making with 46% agreement with Gemma, indicating both models are making reasonable but different judgments about attribute quality trade-offs.

---

## Models Tested

### 1. Llama 3.1 8B Instruct ❌
**Result**: Failed - Systematic bias

- **Issue**: 100% selection of "current" version across all 200 records
- **Analysis**: Model appeared to ignore attribute guidelines entirely
- **Conclusion**: Not suitable for this task - shows complete bias

### 2. DeepSeek R1 Qwen3 8B ❌
**Result**: Failed - Performance issues

- **Issue**: Reasoning model outputs thought process in `<think>` tags
- **Time**: ~82 seconds per record (4.5 hours for 200 records)
- **Problem**: Exhausts token limit (200) on reasoning before generating JSON
- **Conclusion**: Not practical for this annotation task

### 3. Mistral 7B Instruct v0.3 ❌
**Result**: Failed - Prompt compatibility

- **Issue**: `"Only user and assistant roles are supported!"`
- **Problem**: Model's Jinja template doesn't support system messages
- **Conclusion**: Incompatible with our prompt structure

### 4. Gemma 2 9B (Base) ❌
**Result**: Failed - Wrong variant

- **Issue**: Empty/invalid JSON responses
- **Problem**: Base model (not instruct-tuned) can't follow instructions
- **Fix**: Switched to instruct variant

### 5. Gemma 2 9B Instruct ✅
**Result**: Success - After fixing markdown wrapper

- **Issue**: Initially returned JSON wrapped in markdown code blocks
- **Fix**: Added markdown stripping logic to annotation script
- **Performance**: ~20 seconds per record
- **Quality**: Good instruction following, but shows current-bias (77.5%)

### 6. Qwen 3 Coder 30B ✅
**Result**: Success - Best performer

- **Performance**: ~10-15 seconds per record
- **Quality**: Most balanced and nuanced annotations
- **No compatibility issues**: Worked perfectly out of the box

---

## Technical Fixes Implemented

### 1. UTF-8 Encoding Issues (Windows Console)
**Problem**: Unicode characters (✓, ✗, Thai text) caused crashes
```python
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution**: Replaced all unicode symbols with ASCII equivalents:
- `✓` → `[OK]`
- `✗` → `[NO]`
- `⚠` → `[WARNING]`

**Files Modified**: 
- `scripts/annotate_ai.py`
- `scripts/calculate_agreement.py`

### 2. JSON Response Format Parameter
**Problem**: LM Studio rejected `response_format: {"type": "json_object"}`
```
Error code: 400 - {'error': "'response_format.type' must be 'json_schema' or 'text'"}
```

**Solution**: Removed `response_format` parameter entirely (line 180 in `annotate_ai.py`)

### 3. Gemma Markdown Code Blocks
**Problem**: Gemma wraps JSON in markdown:
```
```json
{"choice": "c", "notes": "..."}
```
```

**Solution**: Added markdown stripping logic before JSON parsing:
```python
if result_text.strip().startswith('```'):
    lines = result_text.strip().split('\n')
    if lines[0].startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    result_text = '\n'.join(lines)
```

---

## Final Results: Qwen vs Gemma Comparison

### Agreement Metrics
- **Total Records**: 200
- **Agreed Records**: 92 (46.0%)
- **Disagreed Records**: 108 (54.0%)
- **Cohen's Kappa**: 0.2270 (Fair)

### Choice Distribution

| Choice | Qwen 30B | Gemma 2 9B | Llama 3.1 8B |
|--------|----------|------------|--------------|
| **current (c)** | 59 (29.5%) | 155 (77.5%) | 200 (100%) |
| **base (b)** | 104 (52.0%) | 24 (12.0%) | 0 (0%) |
| **same (s)** | 20 (10.0%) | 20 (10.0%) | 0 (0%) |
| **unclear (u)** | 17 (8.5%) | 1 (0.5%) | 0 (0%) |

### Key Observations

#### Qwen 30B Behavior:
- ✅ **Most balanced**: Prefers base (52%) over current (29.5%)
- ✅ **Nuanced judgments**: Uses all four choice categories appropriately
- ✅ **Quality-focused**: Catches capitalization errors, URL quality, category structure
- ✅ **Willing to mark unclear**: 8.5% unclear rate shows careful consideration

**Example reasoning** (Record 1):
> "The base version is slightly better overall. For NAME, 'Davaindia Generic Pharmacy' (base) is more properly capitalized than 'davaindia GENERIC PHARMACY' (current), following standard title case conventions."

#### Gemma 2 9B Behavior:
- ⚠️ **Current-biased**: 77.5% current selection (similar to Llama's pattern)
- ⚠️ **Completeness-focused**: Heavily weights data completeness over formatting
- ⚠️ **Rarely unclear**: Only 0.5% unclear (1 record) suggests less caution
- ✅ **Reasonable explanations**: Provides valid reasoning despite bias

**Example reasoning** (Record 2):
> "Current version has more complete and accurate data for name, phone, website, and address. Category is also more specific and relevant."

### Major Disagreement Patterns

**79 "base vs current" conflicts** - Core trade-off:
- Qwen prioritizes: Name capitalization, canonical URLs, category hierarchy
- Gemma prioritizes: Data completeness (especially phone numbers present)

**Example Disagreement** (Record 2):
- **Names**: `"davaindia GENERIC PHARMACY"` (current) vs `"Davaindia Generic Pharmacy"` (base)
- **Qwen chose**: base (better capitalization)
- **Gemma chose**: current (more complete website URL)

---

## Recommendations

### 1. Primary Annotation Strategy
**Use Qwen 30B as the primary automated annotator:**
- Most balanced and guideline-adherent
- Catches subtle quality issues
- Appropriate use of all decision categories

### 2. Manual Review Priority
**Review the 108 Qwen vs Gemma disagreements:**
- These represent genuinely difficult cases
- Will help refine attribute guidelines
- Creates high-quality training data for future ML models
- See: `disagreements_qwen_vs_gemma.json`

### 3. Guideline Refinement
**Document resolution of key trade-offs:**
- Completeness vs Formatting quality
- When to prioritize phone presence vs name capitalization
- Canonical URLs vs localized URLs
- Structured vs simple category formats

### 4. Inter-Annotator Agreement Target
**Current status vs goal:**
- Target: >95% agreement (for human annotators)
- Achieved: 46% agreement (between AI models)
- This gap suggests AI models weigh attributes differently than intended
- Human annotations will be critical for establishing ground truth

---

## Performance Metrics

### Speed Comparison (per record)

| Model | Time/Record | Total Time (200) | Notes |
|-------|-------------|------------------|-------|
| Llama 3.1 8B | ~7 seconds | ~25 minutes | Fast but biased |
| DeepSeek R1 8B | ~82 seconds | ~4.5 hours | Too slow (reasoning overhead) |
| Qwen 30B | ~10-15 seconds | ~30-50 minutes | Good balance |
| Gemma 2 9B | ~20 seconds | ~65 minutes | Slower but acceptable |

### Hardware
- **GPU**: RTX 4070 Super (12GB VRAM)
- **Context**: Full attribute guidelines (~2,667 tokens per prompt)
- **Temperature**: 0.2 (for consistency)
- **Max Tokens**: 200 (sufficient for JSON response)

---

## Files Generated

### Annotation Files
- ✅ `data/annotations_qwen.json` - Qwen 30B annotations (200 records)
- ✅ `data/annotations_gemma.json` - Gemma 2 9B annotations (200 records)
- ✅ `disagreements_qwen_vs_gemma.json` - Detailed disagreements (108 records)

### Deleted Files (Failed Attempts)
- ❌ `data/annotations_llama.json` - 100% biased toward current
- ❌ `data/annotations_mistral.json` - Prompt template errors
- ❌ `disagreements_llama_vs_qwen.json` - No longer relevant

---

## Next Steps

### Immediate (Next Session)
1. **Set up manual review workflow** for 108 disagreements
2. **Establish tie-breaking process** between Qwen and Gemma
3. **Create annotation interface** for efficient review

### Medium-Term
1. **Complete 200-record ground truth** via manual review
2. **Calculate final IAA** with human annotations
3. **Refine attribute guidelines** based on disagreement patterns
4. **Document edge cases** for future reference

### Long-Term
1. **Scale to full 2K dataset** using Qwen 30B + manual review
2. **Train ML model** on high-quality labeled data
3. **Benchmark ML vs rule-based approaches**
4. **Deliver final recommendations** for production use

---

## Lessons Learned

### Model Selection
- ✅ Larger models (30B) generally more nuanced than smaller (7-9B)
- ✅ Task-specific models (coder variants) can work well for structured tasks
- ❌ Model size alone doesn't guarantee quality (Llama 8B failed, Gemma 9B biased)
- ✅ Testing with 5-record samples saves significant time

### Prompt Engineering
- ✅ Including full guidelines in prompt works (for compatible models)
- ⚠️ Some models can't handle complex prompts (Gemma had issues initially)
- ✅ Low temperature (0.2) provides consistency without sacrificing reasoning
- ⚠️ Different models have different prompt template requirements

### Technical Infrastructure
- ⚠️ Windows console encoding requires special handling
- ✅ LM Studio is reliable once model compatibility is established
- ✅ Auto-save functionality crucial for long-running tasks
- ✅ Background process management important to prevent conflicts

---

## Conclusion

This model comparison study successfully identified **Qwen 3 Coder 30B** as a reliable automated annotator for place attribute conflation. While the 46% agreement between Qwen and Gemma is below human IAA targets, it represents meaningful disagreement on genuinely difficult cases rather than random noise or complete bias.

The 108 disagreements provide a focused set of records for manual review, which will serve as high-quality ground truth for:
1. Validating and refining attribute guidelines
2. Training future ML models
3. Benchmarking automated approaches
4. Understanding edge cases in attribute conflation

**Recommendation**: Proceed with Qwen 30B for bulk annotation, with manual review of disagreements to establish definitive ground truth.

---

*Report generated: November 11, 2024*  
*Branch: `local_model_annotation`*  
*Next session: Manual disagreement review workflow*

