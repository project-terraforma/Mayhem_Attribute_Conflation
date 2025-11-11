"""
Inter-Annotator Agreement Analysis

This script calculates agreement metrics between two annotators
for the attribute selection task.
"""

import pandas as pd
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from pathlib import Path


def load_annotations(*filenames):
    """Load annotation files from multiple annotators."""
    annotations = {}
    
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotator_name = Path(filename).stem.replace('annotations_', '')
            annotations[annotator_name] = data
    
    return annotations


def extract_choices(annotations_dict):
    """Extract annotation choices by record ID."""
    choices_dict = {}
    
    for annotator, ann_list in annotations_dict.items():
        for ann in ann_list:
            rec_id = ann['id']
            if rec_id not in choices_dict:
                choices_dict[rec_id] = {}
            
            # Map choice code to full label
            choice_map = {
                'c': 'current',
                'b': 'base',
                's': 'same',
                'u': 'unclear'
            }
            
            choices_dict[rec_id][annotator] = choice_map.get(ann['choice'], ann['choice'])
    
    return choices_dict


def calculate_agreement(choices_dict):
    """Calculate various agreement metrics."""
    agreement_results = {
        'total_records': len(choices_dict),
        'agreed_records': 0,
        'disagreed_records': 0,
        'agreement_rate': 0.0,
        'conflict_breakdown': {}
    }
    
    for rec_id, choices in choices_dict.items():
        if len(choices) == 2:
            annotators = list(choices.keys())
            choice_a = choices[annotators[0]]
            choice_b = choices[annotators[1]]
            
            if choice_a == choice_b:
                agreement_results['agreed_records'] += 1
            else:
                agreement_results['disagreed_records'] += 1
                
                # Track what types of conflicts
                conflict_key = f"{choice_a} vs {choice_b}"
                agreement_results['conflict_breakdown'][conflict_key] = \
                    agreement_results['conflict_breakdown'].get(conflict_key, 0) + 1
    
    agreement_results['agreement_rate'] = (
        agreement_results['agreed_records'] / 
        agreement_results['total_records'] * 100
    )
    
    return agreement_results


def calculate_cohen_kappa(choices_dict):
    """Calculate Cohen's Kappa for inter-annotator agreement."""
    # Convert choices to numeric format for Kappa calculation
    label_to_num = {
        'current': 0,
        'base': 1,
        'same': 2,
        'unclear': 3
    }
    
    annotator_choices = {'annotator_1': [], 'annotator_2': []}
    
    for rec_id, choices in choices_dict.items():
        if len(choices) == 2:
            annotators = list(choices.keys())
            choice_a = choices[annotators[0]]
            choice_b = choices[annotators[1]]
            
            annotator_choices['annotator_1'].append(label_to_num.get(choice_a, -1))
            annotator_choices['annotator_2'].append(label_to_num.get(choice_b, -1))
    
    if len(annotator_choices['annotator_1']) == 0:
        return None
    
    kappa = cohen_kappa_score(
        annotator_choices['annotator_1'],
        annotator_choices['annotator_2']
    )
    
    return kappa


def print_agreement_report(agreement_results, kappa=None):
    """Print a formatted agreement report."""
    print("\n" + "="*80)
    print("INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Records Annotated: {agreement_results['total_records']}")
    print(f"Agreed Records: {agreement_results['agreed_records']}")
    print(f"Disagreed Records: {agreement_results['disagreed_records']}")
    print(f"\nOverall Agreement Rate: {agreement_results['agreement_rate']:.2f}%")
    
    if kappa is not None:
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        # Interpret Kappa
        if kappa < 0:
            interpretation = "Poor (negative)"
        elif kappa < 0.20:
            interpretation = "Slight"
        elif kappa < 0.40:
            interpretation = "Fair"
        elif kappa < 0.60:
            interpretation = "Moderate"
        elif kappa < 0.80:
            interpretation = "Substantial"
        else:
            interpretation = "Almost Perfect"
        
        print(f"Interpretation: {interpretation}")
    
    print(f"\nTarget Agreement: >95%")
    target_met = agreement_results['agreement_rate'] >= 95
    print(f"Target Met: {'[YES]' if target_met else '[NO]'}")
    
    if agreement_results['conflict_breakdown']:
        print(f"\nConflict Breakdown:")
        for conflict, count in agreement_results['conflict_breakdown'].items():
            print(f"  {conflict}: {count}")


def analyze_disagreements(choices_dict, df):
    """Analyze specific cases where annotators disagreed."""
    disagreements = []
    
    for rec_id, choices in choices_dict.items():
        if len(choices) == 2:
            annotators = list(choices.keys())
            choice_a = choices[annotators[0]]
            choice_b = choices[annotators[1]]
            
            if choice_a != choice_b:
                # Get the original record
                rec = df[df['id'] == rec_id].iloc[0]
                
                disagreements.append({
                    'id': rec_id,
                    'annotator_a': annotators[0],
                    'choice_a': choice_a,
                    'annotator_b': annotators[1],
                    'choice_b': choice_b,
                    'record': rec
                })
    
    return disagreements


def print_sample_disagreements(disagreements, n=5):
    """Print sample disagreement cases for review."""
    if len(disagreements) == 0:
        return
    
    print("\n" + "="*80)
    print(f"SAMPLE DISAGREEMENTS (showing {min(n, len(disagreements))} of {len(disagreements)})")
    print("="*80)
    
    for i, d in enumerate(disagreements[:n]):
        print(f"\n--- Disagreement {i+1} ---")
        print(f"ID: {d['id']}")
        print(f"{d['annotator_a']} chose: {d['choice_a']}")
        print(f"{d['annotator_b']} chose: {d['choice_b']}")
        print(f"\nNames:")
        print(f"  Current: {d['record']['names']}")
        print(f"  Base: {d['record']['base_names']}")
        print(f"\nPhones:")
        print(f"  Current: {d['record']['phones']}")
        print(f"  Base: {d['record']['base_phones']}")


def main():
    """Main agreement analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate inter-annotator agreement')
    parser.add_argument('--ann1', required=True, help='Annotation file 1')
    parser.add_argument('--ann2', required=True, help='Annotation file 2')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    # Load data for reference
    df = pd.read_parquet('data/project_b_samples_2k.parquet')
    
    # Load annotations
    annotations = load_annotations(args.ann1, args.ann2)
    
    # Extract choices
    choices_dict = extract_choices(annotations)
    
    # Calculate agreement
    agreement_results = calculate_agreement(choices_dict)
    kappa = calculate_cohen_kappa(choices_dict)
    
    # Print report
    print_agreement_report(agreement_results, kappa)
    
    # Analyze disagreements
    disagreements = analyze_disagreements(choices_dict, df)
    print_sample_disagreements(disagreements)
    
    # Save report
    if args.output:
        report = {
            'agreement_results': agreement_results,
            'cohen_kappa': float(kappa) if kappa is not None else None,
            'disagreement_count': len(disagreements)
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Report saved to: {args.output}")
    
    return agreement_results, kappa


if __name__ == "__main__":
    main()

