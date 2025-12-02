
import json
import pandas as pd
from pathlib import Path
import random

RESULTS_DIR = Path('data/results')
GOLDEN_PATH = 'data/golden_dataset_200.json'
OUTPUT_FILE = 'data/results/failure_analysis.md'

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_record_by_id(golden_data, record_id):
    for r in golden_data:
        if r['id'] == record_id:
            return r
    return None

def generate_report():
    print("Generating Failure Analysis Report...")
    
    golden_data = load_json(GOLDEN_PATH)
    golden_lookup = {r['id']: r for r in golden_data}
    
    report_lines = []
    report_lines.append("# Failure Analysis Report (OKR 2, KR 2)")
    report_lines.append("This document lists failure cases where the model/algorithm prediction disagreed with the ground truth.\n")
    
    # Attributes to analyze
    attributes = ['name', 'address', 'phone', 'website', 'category']
    
    for attr in attributes:
        report_lines.append(f"## Attribute: {attr.upper()}")
        
        # 1. ML Failures
        ml_file = RESULTS_DIR / f'ml_predictions_200_real_{attr}.json'
        if ml_file.exists():
            preds = load_json(ml_file)
            failures = []
            
            for p in preds:
                rid = p['id']
                pred_source = p['selected_source'] # 'current' or 'base'
                
                # Ground Truth
                # Note: We need to use the same logic as evaluate_models.py (Smart Derivation or flat label)
                # For simplicity here, we'll look at the flat label which is what the 200 set provided
                r_gt = golden_lookup.get(rid)
                if not r_gt: continue
                
                label_map = {'b': 'base', 'c': 'current', 's': 'same', 'u': 'unclear'}
                gt_label = label_map.get(r_gt.get('label', 'unclear'), 'unclear')
                
                if gt_label in ['same', 'unclear']: continue # Skip ambiguous
                
                if pred_source != gt_label:
                    failures.append({
                        'id': rid,
                        'pred': pred_source,
                        'truth': gt_label,
                        'val_c': p['candidates']['current'],
                        'val_b': p['candidates']['base']
                    })
            
            # Sample 5 failures
            sample = random.sample(failures, min(5, len(failures)))
            report_lines.append(f"\n### ML Model Failures (Count: {len(failures)})")
            if not sample:
                report_lines.append("No failures found (or no valid comparison).")
            for i, f in enumerate(sample, 1):
                report_lines.append(f"**{i}. Record {f['id']}**")
                report_lines.append(f"- Prediction: {f['pred'].upper()} | Truth: {f['truth'].upper()}")
                report_lines.append(f"- Current: `{f['val_c']}`")
                report_lines.append(f"- Base:    `{f['val_b']}`")
                report_lines.append("")

        # 2. Baseline Failures (Most Recent)
        base_file = RESULTS_DIR / f'predictions_baseline_most_recent_200_real_{attr}.json'
        if base_file.exists():
            preds_dict = load_json(base_file)
            failures = []
            
            for rid, pred_source in preds_dict.items():
                r_gt = golden_lookup.get(rid)
                if not r_gt: continue
                
                label_map = {'b': 'base', 'c': 'current', 's': 'same', 'u': 'unclear'}
                gt_label = label_map.get(r_gt.get('label', 'unclear'), 'unclear')
                
                if gt_label in ['same', 'unclear']: continue
                
                if pred_source != gt_label:
                    # Need to fetch values from golden data since baseline output is just id->pred
                    try:
                        # Data key mapping
                        key_map = {'name': 'names', 'phone': 'phones', 'website': 'websites', 'address': 'addresses', 'category': 'categories'}
                        key = key_map.get(attr, attr+'s')
                        val_c = r_gt['data']['current'].get(key)
                        val_b = r_gt['data']['base'].get(f'base_{key}')
                    except:
                        val_c = "N/A"
                        val_b = "N/A"

                    failures.append({
                        'id': rid,
                        'pred': pred_source,
                        'truth': gt_label,
                        'val_c': val_c,
                        'val_b': val_b
                    })
            
            sample = random.sample(failures, min(5, len(failures)))
            report_lines.append(f"\n### Baseline (Most Recent) Failures (Count: {len(failures)})")
            if not sample:
                report_lines.append("No failures found.")
            for i, f in enumerate(sample, 1):
                report_lines.append(f"**{i}. Record {f['id']}**")
                report_lines.append(f"- Prediction: {f['pred'].upper()} | Truth: {f['truth'].upper()}")
                report_lines.append(f"- Current: `{f['val_c']}`")
                report_lines.append(f"- Base:    `{f['val_b']}`")
                report_lines.append("")
                
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Failure analysis report generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
