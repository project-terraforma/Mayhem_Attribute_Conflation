"""
Evaluation Framework for Attribute Selection Algorithms

This module provides evaluation metrics and comparison tools for different
algorithms (baselines and ML models).
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def load_golden_labels(golden_file: str, attribute: str = 'name') -> Dict[str, str]:
    """Load golden labels from JSON file."""
    with open(golden_file, 'r') as f:
        golden_data = json.load(f)
    
    labels = {}
    for record in golden_data:
        record_id = record['id']
        
        # Get raw values to check for equality
        # Map attribute name to data keys
        attr_key_map = {
            'name': ('names', 'base_names'),
            'phone': ('phones', 'base_phones'),
            'website': ('websites', 'base_websites'),
            'address': ('addresses', 'base_addresses'),
            'category': ('categories', 'base_categories')
        }
        
        curr_key, base_key = attr_key_map.get(attribute, (None, None))
        
        # Fallback label from manual review
        manual_label = record.get('label', 'unclear')
        label_map = {'b': 'base', 'c': 'current', 's': 'same', 'u': 'unclear'}
        final_label = label_map.get(manual_label, manual_label)

        # Smart derivation: If values are identical, label is 'same'
        if curr_key and 'data' in record:
            try:
                val_c = str(record['data']['current'].get(curr_key, '')).strip()
                val_b = str(record['data']['base'].get(base_key, '')).strip()
                
                # Normalize nulls/NaNs
                if val_c in ['None', 'nan', 'NaN', '[]', '{}'] and val_b in ['None', 'nan', 'NaN', '[]', '{}']:
                    final_label = 'same'
                elif val_c == val_b:
                    final_label = 'same'
            except:
                pass
        
        labels[record_id] = final_label
    
    return labels


def convert_predictions_to_binary(predictions: List[str], labels: List[str]) -> tuple:
    """
    Convert predictions and labels to binary format for evaluation.
    
    Maps: 'current' -> 1, 'base' -> 0, 'same' -> 1 (treat as current), 'unclear' -> excluded
    """
    binary_preds = []
    binary_labels = []
    
    for pred, label in zip(predictions, labels):
        # Skip unclear cases
        if pred == 'unclear' or label == 'unclear':
            continue
        
        # Convert to binary
        pred_binary = 1 if pred in ['current', 'same'] else 0
        label_binary = 1 if label in ['current', 'same'] else 0
        
        binary_preds.append(pred_binary)
        binary_labels.append(label_binary)
    
    return np.array(binary_preds), np.array(binary_labels)


def calculate_metrics(predictions: List[str], labels: List[str], 
                     metric_type: str = 'binary') -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of predictions ('current', 'base', 'same', 'unclear')
        labels: List of true labels
        metric_type: 'binary' (current vs base) or 'multiclass' (all classes)
    
    Returns:
        Dictionary of metrics
    """
    if metric_type == 'binary':
        # Convert to binary
        binary_preds, binary_labels = convert_predictions_to_binary(predictions, labels)
        
        if len(binary_preds) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'n_samples': 0
            }
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision = precision_score(binary_labels, binary_preds, zero_division=0)
        recall = recall_score(binary_labels, binary_preds, zero_division=0)
        f1 = f1_score(binary_labels, binary_preds, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_samples': len(binary_preds)
        }
    
    else:
        # Multiclass evaluation
        # Filter out unclear
        filtered_preds = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if pred != 'unclear' and label != 'unclear':
                filtered_preds.append(pred)
                filtered_labels.append(label)
        
        if len(filtered_preds) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'n_samples': 0
            }
        
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        
        # For multiclass, calculate macro-averaged metrics
        try:
            precision = precision_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
            recall = recall_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
            f1 = f1_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_samples': len(filtered_preds)
        }


def evaluate_algorithm(
    predictions: Dict[str, str],
    golden_labels: Dict[str, str],
    algorithm_name: str = "Algorithm"
) -> Dict[str, Any]:
    """
    Evaluate an algorithm's predictions against golden labels.
    
    Args:
        predictions: Dictionary mapping record ID to prediction
        golden_labels: Dictionary mapping record ID to true label
        algorithm_name: Name of the algorithm
    
    Returns:
        Dictionary with evaluation results
    """
    # Align predictions with labels
    aligned_preds = []
    aligned_labels = []
    
    for record_id in golden_labels.keys():
        if record_id in predictions:
            aligned_preds.append(predictions[record_id])
            aligned_labels.append(golden_labels[record_id])
    
    if len(aligned_preds) == 0:
        return {
            'algorithm': algorithm_name,
            'error': 'No overlapping predictions and labels'
        }
    
    # Calculate metrics
    metrics = calculate_metrics(aligned_preds, aligned_labels, metric_type='binary')
    
    # Calculate coverage (percentage of records with non-unclear predictions)
    unclear_count = sum(1 for p in aligned_preds if p == 'unclear')
    coverage = 1.0 - (unclear_count / len(aligned_preds)) if len(aligned_preds) > 0 else 0.0
    
    results = {
        'algorithm': algorithm_name,
        'metrics': metrics,
        'coverage': float(coverage),
        'n_total': len(aligned_preds),
        'n_unclear': unclear_count
    }
    
    return results


def compare_algorithms(
    algorithm_results: List[Dict[str, Any]],
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple algorithms and create a comparison table.
    
    Args:
        algorithm_results: List of evaluation result dictionaries
        output_file: Optional path to save comparison table
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for result in algorithm_results:
        if 'error' in result:
            continue
        
        row = {
            'Algorithm': result['algorithm'],
            'F1-Score': result['metrics']['f1'],
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'Coverage': result['coverage'],
            'N Samples': result['metrics']['n_samples']
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        # Sort by F1-score descending
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_file, index=False)
        print(f"Comparison table saved to {output_file}")
    
    return comparison_df


def print_evaluation_report(results: Dict[str, Any]):
    """Print a formatted evaluation report."""
    if 'error' in results:
        print(f"\nError: {results['error']}")
        return
    
    print("\n" + "="*80)
    print(f"EVALUATION REPORT: {results['algorithm']}")
    print("="*80)
    
    metrics = results['metrics']
    print(f"\nMetrics:")
    print(f"  F1-Score:    {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    
    print(f"\nCoverage:")
    print(f"  Total Records: {results['n_total']}")
    print(f"  Unclear:       {results['n_unclear']}")
    print(f"  Coverage:      {results['coverage']:.4f} ({results['coverage']*100:.2f}%)")
    
    # Check KR targets
    print(f"\nKR Targets:")
    print(f"  KR1 (Beat baseline by 15%+): {'[TBD]'}")
    print(f"  KR2 (F1 > 0.90): {'[YES]' if metrics['f1'] > 0.90 else '[NO]'}")
    print(f"  KR3 (Coverage > 99%): {'[YES]' if results['coverage'] > 0.99 else '[NO]'}")


def calculate_baseline_improvement(
    baseline_f1: float,
    model_f1: float
) -> Dict[str, float]:
    """
    Calculate improvement over baseline for KR1.
    
    Returns:
        Dictionary with improvement metrics
    """
    if baseline_f1 == 0:
        improvement_pct = float('inf') if model_f1 > 0 else 0.0
    else:
        improvement_pct = ((model_f1 - baseline_f1) / baseline_f1) * 100
    
    improvement_absolute = model_f1 - baseline_f1
    
    return {
        'baseline_f1': baseline_f1,
        'model_f1': model_f1,
        'improvement_absolute': improvement_absolute,
        'improvement_percentage': improvement_pct,
        'kr1_target_met': improvement_pct >= 15.0
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate attribute selection algorithms')
    parser.add_argument('--predictions', required=True,
                       help='JSON file with predictions (dict: record_id -> prediction)')
    parser.add_argument('--golden', required=True,
                       help='Golden labels file')
    parser.add_argument('--attribute', default='name',
                       help='Attribute being evaluated')
    parser.add_argument('--algorithm-name', default='Algorithm',
                       help='Name of the algorithm')
    parser.add_argument('--output', default=None,
                       help='Output file for results JSON')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, 'r') as f:
        raw_predictions = json.load(f)
        
    # Handle list format (from run_inference.py) vs dict format (from baseline_heuristics.py)
    if isinstance(raw_predictions, list):
        predictions = {}
        for item in raw_predictions:
            if 'id' in item and 'selected_source' in item:
                predictions[item['id']] = item['selected_source']
    else:
        predictions = raw_predictions
    
    # Load golden labels
    print(f"Loading golden labels from {args.golden}...")
    golden_labels = load_golden_labels(args.golden, args.attribute)
    
    # Evaluate
    results = evaluate_algorithm(
        predictions=predictions,
        golden_labels=golden_labels,
        algorithm_name=args.algorithm_name
    )
    
    # Print report
    print_evaluation_report(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

