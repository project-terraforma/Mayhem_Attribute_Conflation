"""
Evaluate Trained ML Model on Test Set

This script loads a trained model and evaluates it on the held-out test set.
"""

import pandas as pd
import json
import joblib
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_features import extract_features_batch, load_golden_labels


def evaluate_model_on_test_set(
    model_file: str = 'models/ml_models/best_model_logistic_regression.joblib',
    scaler_file: str = 'models/ml_models/scaler_logistic_regression.joblib',
    test_data_file: str = 'data/project_b_samples_2k.parquet',
    test_labels_file: str = 'data/processed/golden_dataset_test.json',
    attribute: str = 'name'
):
    """
    Evaluate trained model on test set.
    
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("EVALUATING ML MODEL ON TEST SET")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from {model_file}...")
    model_data = joblib.load(model_file)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # Load scaler if exists
    scaler = None
    if Path(scaler_file).exists():
        print(f"Loading scaler from {scaler_file}...")
        scaler = joblib.load(scaler_file)
    
    # Load test data
    print(f"\nLoading test data from {test_data_file}...")
    df = pd.read_parquet(test_data_file)
    
    # Load test labels
    print(f"Loading test labels from {test_labels_file}...")
    test_labels_dict = load_golden_labels(test_labels_file, attribute)
    
    # Get test set record IDs
    with open(test_labels_file, 'r') as f:
        test_data = json.load(f)
    test_ids = {record['id'] for record in test_data}
    
    # Filter dataframe to test set
    df_test = df[df['id'].isin(test_ids)].copy()
    print(f"Test set size: {len(df_test)} records")
    
    # Extract features
    print(f"\nExtracting features for '{attribute}' attribute...")
    features_df = extract_features_batch(df_test, attribute)
    
    # Prepare features (same columns as training)
    X_test = features_df[feature_cols].fillna(0.0)
    
    # Scale if scaler exists
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    print("Making predictions...")
    predictions_binary = model.predict(X_test_scaled)
    
    # Convert binary predictions to labels
    predictions = ['current' if p == 1 else 'base' for p in predictions_binary]
    
    # Get true labels
    labels = [test_labels_dict.get(rid, 'unclear') for rid in features_df['id']]
    
    # Convert to binary for metrics
    labels_binary = [1 if l in ['current', 'same'] else 0 for l in labels]
    
    # Calculate metrics
    accuracy = accuracy_score(labels_binary, predictions_binary)
    precision = precision_score(labels_binary, predictions_binary, zero_division=0)
    recall = recall_score(labels_binary, predictions_binary, zero_division=0)
    f1 = f1_score(labels_binary, predictions_binary, zero_division=0)
    
    # Count coverage (non-unclear predictions)
    unclear_count = sum(1 for p in predictions if p == 'unclear')
    coverage = 1.0 - (unclear_count / len(predictions)) if len(predictions) > 0 else 0.0
    
    results = {
        'model_type': model_data.get('model_type', 'unknown'),
        'test_set_size': len(df_test),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_samples': len(predictions_binary)
        },
        'coverage': float(coverage),
        'n_unclear': unclear_count,
        'kr2_target_met': f1 > 0.90,
        'kr3_target_met': coverage > 0.99
    }
    
    # Print results
    print("\n" + "="*80)
    print("TEST SET EVALUATION RESULTS")
    print("="*80)
    print(f"\nModel Type: {results['model_type']}")
    print(f"Test Set Size: {results['test_set_size']}")
    print(f"\nMetrics:")
    print(f"  F1-Score:    {results['metrics']['f1']:.4f} ({results['metrics']['f1']*100:.2f}%)")
    print(f"  Accuracy:    {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    print(f"  Precision:   {results['metrics']['precision']:.4f} ({results['metrics']['precision']*100:.2f}%)")
    print(f"  Recall:      {results['metrics']['recall']:.4f} ({results['metrics']['recall']*100:.2f}%)")
    print(f"\nCoverage:")
    print(f"  Coverage:    {results['coverage']:.4f} ({results['coverage']*100:.2f}%)")
    print(f"  Unclear:     {results['n_unclear']}")
    print(f"\nKR Targets:")
    print(f"  KR2 (F1 > 0.90): {'[YES]' if results['kr2_target_met'] else '[NO]'}")
    print(f"  KR3 (Coverage > 99%): {'[YES]' if results['kr3_target_met'] else '[NO]'}")
    
    return results


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ML model on test set')
    parser.add_argument('--model', default='models/ml_models/best_model_logistic_regression.joblib',
                       help='Trained model file')
    parser.add_argument('--scaler', default='models/ml_models/scaler_logistic_regression.joblib',
                       help='Feature scaler file')
    parser.add_argument('--test-data', default='data/project_b_samples_2k.parquet',
                       help='Test data file')
    parser.add_argument('--test-labels', default='data/processed/golden_dataset_test.json',
                       help='Test labels file')
    parser.add_argument('--attribute', default='name',
                       help='Attribute being evaluated')
    parser.add_argument('--output', default=None,
                       help='Output file for results JSON')
    
    args = parser.parse_args()
    
    results = evaluate_model_on_test_set(
        model_file=args.model,
        scaler_file=args.scaler,
        test_data_file=args.test_data,
        test_labels_file=args.test_labels,
        attribute=args.attribute
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

