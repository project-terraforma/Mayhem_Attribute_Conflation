"""
Main Pipeline Script for Objective 2 (OKR 2)

This script runs the complete pipeline:
1. Generate sample golden dataset (if needed)
2. Extract features
3. Train baseline heuristics
4. Train ML models
5. Evaluate and compare all approaches

Usage:
    python scripts/run_algorithm_pipeline.py --attribute name
"""

import pandas as pd
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def run_step(step_name: str, command: list):
    """Run a pipeline step and handle errors."""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Step failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    """Run the complete algorithm pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete algorithm pipeline for OKR 2')
    parser.add_argument('--attribute', default='name',
                       choices=['name', 'phone', 'website', 'address', 'category'],
                       help='Attribute to work on (default: name)')
    parser.add_argument('--skip-golden', action='store_true',
                       help='Skip golden dataset generation (use existing)')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature extraction (use existing)')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='Skip baseline evaluation')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML model training')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OBJECTIVE 2 (OKR 2) ALGORITHM PIPELINE")
    print("="*80)
    print(f"Attribute: {args.attribute}")
    print(f"Working directory: {Path.cwd()}")
    
    # Step 1: Generate sample golden dataset
    if not args.skip_golden:
        golden_file = 'data/processed/golden_dataset_sample.json'
        if not Path(golden_file).exists():
            success = run_step(
                "Generate Sample Golden Dataset",
                [sys.executable, 'scripts/generate_sample_golden_dataset.py',
                 '--split', '--attributes', args.attribute]
            )
            if not success:
                print("Warning: Golden dataset generation failed, but continuing...")
        else:
            print(f"\nGolden dataset already exists: {golden_file}")
    else:
        print("\nSkipping golden dataset generation")
    
    # Step 2: Extract features
    if not args.skip_features:
        features_file = f'data/processed/features_{args.attribute}.parquet'
        if not Path(features_file).exists():
            success = run_step(
                "Extract Features",
                [sys.executable, 'scripts/extract_features.py',
                 '--attribute', args.attribute,
                 '--output', features_file]
            )
            if not success:
                print("ERROR: Feature extraction failed. Cannot continue.")
                return
        else:
            print(f"\nFeatures already exist: {features_file}")
    else:
        print("\nSkipping feature extraction")
    
    # Step 3: Evaluate baselines
    if not args.skip_baselines:
        print("\n" + "="*80)
        print("EVALUATING BASELINE HEURISTICS")
        print("="*80)
        
        baselines = ['most_recent', 'confidence', 'completeness', 'hybrid']
        baseline_results = []
        
        for baseline_name in baselines:
            print(f"\nEvaluating {baseline_name} baseline...")
            # This would require running the baseline and saving predictions
            # For now, we'll note that this needs to be done
            print(f"  [TODO: Run baseline evaluation for {baseline_name}]")
        
        print("\nNote: To evaluate baselines, run:")
        print(f"  python scripts/baseline_heuristics.py --baseline most_recent --attribute {args.attribute}")
    else:
        print("\nSkipping baseline evaluation")
    
    # Step 4: Train ML models
    if not args.skip_ml:
        features_file = f'data/processed/features_{args.attribute}.parquet'
        if Path(features_file).exists():
            success = run_step(
                "Train ML Models",
                [sys.executable, 'scripts/train_models.py',
                 '--features', features_file]
            )
            if not success:
                print("Warning: ML training failed, but continuing...")
        else:
            print(f"ERROR: Features file not found: {features_file}")
            print("Run feature extraction first.")
    else:
        print("\nSkipping ML model training")
    
    # Step 5: Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Evaluate baselines: python scripts/baseline_heuristics.py --baseline most_recent")
    print("2. Compare all models: [TODO: create comparison script]")
    print("3. Evaluate on test set: [TODO: create test evaluation script]")
    print("\nFor detailed evaluation, use:")
    print(f"  python scripts/evaluate_models.py --predictions <predictions.json> --golden data/processed/golden_dataset_test.json")


if __name__ == "__main__":
    main()

