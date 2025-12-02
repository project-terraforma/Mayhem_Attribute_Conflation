import json
import pandas as pd
from pathlib import Path
import glob

RESULTS_DIR = Path('data/results')

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_results():
    print("="*80)
    print("MAYHEM PROJECT: FINAL RESULTS ANALYSIS")
    print("="*80)
    
    # 1. Performance Metrics (Accuracy, F1, Precision, Recall)
    eval_files = list(RESULTS_DIR.glob('*_evaluation_*.json'))
    
    performance_data = []
    
    for f in eval_files:
        data = load_json(f)
        
        name_parts = f.stem.split('_')
        attribute = name_parts[-1]
        
        if 'ml_evaluation' in f.name:
            algo = "ML Model"
        elif 'baseline' in f.name:
            algo_parts = name_parts[4:-1]
            algo = "Baseline: " + " ".join(algo_parts).title()
        else:
            algo = "Unknown"
            
        metrics = data.get('metrics', {})
        if not metrics:
            continue
            
        row = {
            'Attribute': attribute.capitalize(),
            'Algorithm': algo,
            'F1-Score': metrics.get('f1', 0.0),
            'Accuracy': metrics.get('accuracy', 0.0),
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'Coverage': data.get('coverage', 0.0)
        }
        performance_data.append(row)
        
    df_perf = pd.DataFrame(performance_data)
    if not df_perf.empty:
        df_perf = df_perf.sort_values(['Attribute', 'F1-Score'], ascending=[True, False])
        print("\n--- PERFORMANCE METRICS (200 Real Records) ---")
        print(df_perf.to_markdown(index=False, floatfmt=".4f"))
        
        print("\n--- OKR 2.1 CHECK (ML vs. Most Recent Baseline) ---")
        for attr in df_perf['Attribute'].unique():
            subset = df_perf[df_perf['Attribute'] == attr]
            ml_row = subset[subset['Algorithm'].str.contains("ML Model")]
            base_row = subset[subset['Algorithm'].str.contains("Most Recent")]
            
            if not ml_row.empty and not base_row.empty:
                ml_f1 = ml_row.iloc[0]['F1-Score']
                base_f1 = base_row.iloc[0]['F1-Score']
                
                if base_f1 > 0:
                    improvement = (ml_f1 - base_f1) / base_f1 * 100
                    status = "PASSED ✅" if improvement >= 15 else "FAILED ❌"
                    print(f"{attr}: ML ({ml_f1:.4f}) vs Baseline ({base_f1:.4f}) -> {improvement:+.2f}% Improvement  {status}")
                else:
                    print(f"{attr}: Baseline F1 is 0.0. ML F1: {ml_f1:.4f}. Infinite Improvement ✅")

    # 2. Compute Metrics (Time, Memory)
    print("\n--- COMPUTE METRICS (Training) ---")
    model_dirs = list(Path('models/ml').glob('*'))
    compute_data = []
    
    for d in model_dirs:
        if not d.is_dir(): continue
        summary_file = d / 'training_summary.json'
        if summary_file.exists():
            summary = load_json(summary_file)
            best_model = summary['best_model']
            stats = summary['all_models'][best_model]
            
            row = {
                'Attribute': d.name.capitalize(),
                'Best Model': best_model,
                'Train Time (s)': stats.get('train_duration_seconds', 0.0),
                'Peak Memory (MB)': stats.get('peak_memory_mb', 0.0)
            }
            compute_data.append(row)
            
    df_compute = pd.DataFrame(compute_data)
    if not df_compute.empty:
        print(df_compute.to_markdown(index=False, floatfmt=".4f"))
        
    # 3. Inference Metrics
    print("\n--- COMPUTE METRICS (Inference on 2k Records) ---")
    inf_data = []
    result_files = list(RESULTS_DIR.glob('final_conflated_*_2k.json'))
    
    for f in result_files:
        attr = f.stem.split('_')[2]
        data = load_json(f)
        if data and isinstance(data, list) and len(data) > 0:
            meta = data[0].get('inference_meta', {})
            if meta:
                duration_total = meta.get('duration_seconds', 0.0)
                # Assuming batch duration, calculate per record
                duration_per_record_ms = (duration_total / 2000) * 1000
                
                row = {
                    'Attribute': attr.capitalize(),
                    'Total Inference Time (s)': duration_total,
                    'Time Per Record (ms)': duration_per_record_ms,
                    'Peak Memory (MB)': meta.get('peak_memory_mb', 0.0)
                }
                inf_data.append(row)
    
    df_inf = pd.DataFrame(inf_data)
    if not df_inf.empty:
        print(df_inf.to_markdown(index=False, floatfmt=".4f"))
        
        print("\n--- OKR 2.3 CHECK (< 100ms Inference Time) ---")
        avg_time = df_inf['Time Per Record (ms)'].mean()
        status = "PASSED ✅" if avg_time < 100 else "FAILED ❌"
        print(f"Average Time Per Record: {avg_time:.4f} ms  {status}")

if __name__ == "__main__":
    analyze_results()