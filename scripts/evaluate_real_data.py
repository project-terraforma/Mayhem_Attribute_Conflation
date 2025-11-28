
import pandas as pd
import json
import joblib
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import time
import psutil
import os
from scripts.extract_features import extract_features_batch

REAL_GOLDEN_PATH = 'data/golden_dataset_200.json'

def evaluate_on_real_data():
    parser = argparse.ArgumentParser(description='Evaluate ML model on real manual ground truth.')
    parser.add_argument('--attribute', default='name', choices=['name', 'phone', 'website', 'address', 'category'])
    parser.add_argument('--model', default=None, help='Path to the model joblib file.')
    args = parser.parse_args()

    # Determine model path dynamically
    model_dir = Path(f'models/ml_models/{args.attribute}')
    summary_path = model_dir / 'training_summary.json'
    
    if args.model:
        model_path = Path(args.model)
    elif summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        best_model_name = summary['best_model']
        model_path = model_dir / f"best_model_{best_model_name}.joblib"
    else:
        print(f"Error: No model or training summary found for attribute '{args.attribute}'. Please train the model first or specify --model.")
        return

    output_report_path = Path(f'data/results/final_evaluation_report_{args.attribute}.txt')

    print("="*80)
    print(f"FINAL EVALUATION: ML Model ({args.attribute.upper()}) vs. Real Manual Ground Truth (200 Records)")
    print("="*80)

    # Measure initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_memory_mb:.2f} MB")

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found.")
        return
        
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    print(f"Model loaded: {model_data['model_type']}")

    # 2. Load Real Golden Dataset (200 records)
    print(f"Loading validation data from {REAL_GOLDEN_PATH}...")
    with open(REAL_GOLDEN_PATH, 'r') as f:
        records = json.load(f)
    
    # Convert to DataFrame for feature extraction
    rows = []
    y_true = []
    
    valid_labels = ['c', 'b', 'current', 'base']
    
    for r in records:
        label = r['label']
        # For simplicity, we assume 'same' and 'unclear' are not expected as ground truth for this binary evaluation.
        # This will filter out records with 's' or 'u' labels.
        if label not in valid_labels:
            continue
            
        # Map label to binary: 'c'/'current' -> 1, 'b'/'base' -> 0
        y_val = 1 if label in ['c', 'current'] else 0
        y_true.append(y_val)
        
        # Populate row with all necessary data for feature extraction
        row_data = {
            'id': r['id'],
            'names': r['data']['current']['names'],
            'base_names': r['data']['base']['names'],
            'phones': r['data']['current']['phones'],
            'base_phones': r['data']['base']['phones'],
            'websites': r['data']['current']['websites'],
            'base_websites': r['data']['base']['websites'],
            'addresses': r['data']['current']['addresses'],
            'base_addresses': r['data']['base']['addresses'],
            'categories': r['data']['current']['categories'],
            'base_categories': r['data']['base']['categories'],
            'confidence': r['data']['current']['confidence'],
            'base_confidence': r['data']['base']['confidence'],
            'sources': '', # Not used in features, but good to have a placeholder
            'base_sources': '' # Not used in features, but good to have a placeholder
        }
        rows.append(row_data)
        
    df_eval = pd.DataFrame(rows)
    print(f"Evaluated on {len(df_eval)} records (excluding 'same'/'unclear' labels in ground truth).")

    # 3. Extract Features
    print(f"Extracting features for '{args.attribute}' attribute from real data...")
    X_eval = extract_features_batch(df_eval, attribute=args.attribute)
    
    # Ensure columns align with training (add missing cols with 0)
    for col in feature_cols:
        if col not in X_eval.columns:
            X_eval[col] = 0.0
    X_eval = X_eval[feature_cols].fillna(0.0) # Final fill for any NaN from real data
    
    # 4. Predict
    print("Running inference...")
    start_time = time.time()
    y_pred = model.predict(X_eval)
    end_time = time.time()
    inference_duration_seconds = end_time - start_time
    
    # Measure peak memory usage
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # 5. Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    report = f"""
    FINAL RESULTS (Real-World Validation - {args.attribute.upper()})
    -------------------------------------
    Accuracy:  {acc:.4f}
    F1-Score:  {f1:.4f}
    Precision: {prec:.4f}
    Recall:    {rec:.4f}
    
    Confusion Matrix:
    [[TN (Base Correct)  FP (Base Wrong)]
     [FN (Curr Wrong)    TP (Curr Correct)]]
    {cm}
    
    Total Records: {len(y_true)}
    
    Compute Metrics:
      Inference Duration: {inference_duration_seconds:.4f} seconds
      Initial Memory: {initial_memory_mb:.2f} MB
      Peak Memory: {peak_memory_mb:.2f} MB
    """
    
    print(report)
    
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {output_report_path}")

if __name__ == "__main__":
    evaluate_on_real_data()
