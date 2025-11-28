
import pandas as pd
import json
import joblib
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from scripts.extract_features import extract_features_batch

# Config
MODEL_PATH = 'models/ml_models/best_model_gradient_boosting.joblib'
REAL_GOLDEN_PATH = 'data/golden_dataset_200.json'
OUTPUT_REPORT = 'data/results/final_evaluation_report.txt'

def evaluate_on_real_data():
    print("="*80)
    print("FINAL EVALUATION: ML Model vs. Real Manual Ground Truth (200 Records)")
    print("="*80)

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model_data = joblib.load(MODEL_PATH)
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
        if label not in valid_labels:
            continue # Skip unclear/same for binary classification test
            
        # Map label to binary (Current=1, Base=0)
        y_val = 1 if label in ['c', 'current'] else 0
        y_true.append(y_val)
        
        row = {
            'id': r['id'],
            'names': r['data']['current']['names'],
            'base_names': r['data']['base']['names'],
            'confidence': r['data']['current']['confidence'],
            'base_confidence': r['data']['base']['confidence'],
            'sources': '',
            'base_sources': ''
        }
        rows.append(row)
        
    df_eval = pd.DataFrame(rows)
    print(f"Evaluated on {len(df_eval)} records (excluding 'same'/'unclear').")

    # 3. Extract Features
    print("Extracting features from real data...")
    X_eval = extract_features_batch(df_eval, attribute='name')
    
    # Ensure columns match training
    X_eval = X_eval[feature_cols].fillna(0.0)
    
    # 4. Predict
    print("Running inference...")
    y_pred = model.predict(X_eval)
    
    # 5. Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    report = f"""
    FINAL RESULTS (Real-World Validation)
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
    """
    
    print(report)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    evaluate_on_real_data()
