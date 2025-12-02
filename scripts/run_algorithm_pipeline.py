"""
Main Pipeline Script for Objective 2 (OKR 2)

This script runs the complete pipeline:
1. Generate synthetic golden dataset
2. Extract features (for training and evaluation)
3. Train baseline heuristics
4. Train ML models
5. Evaluate and compare all approaches
6. Run inference on the 2000 Overture records

Usage:
    python scripts/run_algorithm_pipeline.py
"""

import pandas as pd
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

ALL_ATTRIBUTES = ['name', 'phone', 'website', 'address', 'category']
REAL_GOLDEN_PATH = 'data/golden_dataset_200.json' # Your manually reviewed 200 records

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


def run_pipeline_for_attribute(attribute: str, args):
    """Run the complete algorithm pipeline for a single attribute."""
    
    print("\n" + "#"*80)
    print(f"PROCESSING ATTRIBUTE: {attribute.upper()}")
    print("#"*80)

    # Step 1: Generate synthetic golden dataset (if needed)
    # This step generates the synthetic data ONCE, then process_synthetic_data uses it
    if attribute == ALL_ATTRIBUTES[0] and not args.skip_golden: # Only generate once for the first attribute
        cmd = [sys.executable, 'scripts/generate_synthetic_dataset.py']
        if args.synthetic_limit:
            cmd.extend(['--limit', str(args.synthetic_limit)])
            
        success = run_step(
            "Generate Synthetic Golden Dataset",
            cmd
        )
        if not success:
            print("ERROR: Synthetic dataset generation failed. Cannot continue.")
            return False
    elif args.skip_golden:
        print("\nSkipping synthetic golden dataset generation.")
    elif attribute != ALL_ATTRIBUTES[0]:
        print("\nSynthetic golden dataset already generated (skip for subsequent attributes).")
    
    # Step 2: Process synthetic data (extract features for training)
    if not args.skip_features:
        synthetic_features_file = f'data/processed/features_{attribute}_synthetic.parquet'
        # Features for an attribute are always regenerated if not skipped
        success = run_step(
            f"Process Synthetic Data (Features for {attribute})",
            [sys.executable, '-m', 'scripts.process_synthetic_data',
             '--attribute', attribute]
        )
        if not success:
            print(f"ERROR: Feature extraction for {attribute} failed. Cannot continue.")
            return False
    else:
        print(f"\nSkipping feature extraction for {attribute}.")

    # Step 3: Train ML models
    if not args.skip_ml:
        synthetic_features_file = f'data/processed/features_{attribute}_synthetic.parquet'
        output_model_dir = f'models/ml/{attribute}'
        
        success = run_step(
            f"Train ML Models ({attribute})",
            [sys.executable, 'scripts/train_models.py',
             '--features', synthetic_features_file,
             '--output-dir', output_model_dir]
        )
        if not success:
            print(f"Warning: ML training for {attribute} failed, but continuing...")
    else:
        print(f"\nSkipping ML model training for {attribute}.")
    
    # Step 4: Evaluate ML Model on Real Golden Dataset (200 records)
    if not args.skip_ml_eval:
        ml_predictions_200_file = f'data/results/ml_predictions_200_real_{attribute}.json'
        
        output_model_dir = Path(f'models/ml/{attribute}')
        summary_path = output_model_dir / 'training_summary.json'
        
        # Check if model training output directory and summary exist
        if not (output_model_dir.exists() and summary_path.exists()):
            print(f"Warning: No trained ML model found for {attribute} at {output_model_dir}. Skipping ML evaluation on 200 real records.")
            return True # Allow pipeline to continue
        
        # Load best model name from summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        best_model_name = summary['best_model']
        model_path = output_model_dir / f"best_model_{best_model_name}.joblib"
        
        # Check if the actual best model file exists
        if not model_path.exists():
            print(f"Warning: Best model file not found at {model_path}. Skipping ML evaluation on 200 real records.")
            return True # Allow pipeline to continue

        # First, run inference on the 200 real records using the dynamically selected best model
        success = run_step(
            f"Run ML Inference on 200 Real Records ({attribute})",
            [sys.executable, '-m', 'scripts.run_inference',
             '--attribute', attribute,
             '--data', REAL_GOLDEN_PATH, # The 200 real records are in golden_dataset_200.json
             '--model', str(model_path),
             '--output', ml_predictions_200_file
            ]
        )
        if not success:
            print(f"Warning: ML inference on 200 real records for {attribute} failed.")
        else:
            # Then evaluate using evaluate_models.py
            success = run_step(
                f"Evaluate ML Model on 200 Real Records ({attribute})",
                [sys.executable, 'scripts/evaluate_models.py',
                 '--predictions', ml_predictions_200_file,
                 '--golden', REAL_GOLDEN_PATH,
                 '--attribute', attribute,
                 '--algorithm-name', f'ML Model ({attribute})',
                 '--output', f'data/results/ml_evaluation_200_real_{attribute}.json'
                ]
            )
            if not success:
                print(f"Warning: ML evaluation for {attribute} failed.")
    else:
        print(f"\nSkipping ML model evaluation on 200 real records for {attribute}.")


    # Step 5: Evaluate Baselines
    # These baselines also need to predict on the 200 real records
    if not args.skip_baselines:
        print("\n" + "="*80)
        print(f"EVALUATING BASELINE HEURISTICS ({attribute.upper()})")
        print("="*80)
        
        baselines = ['most_recent', 'confidence', 'completeness', 'hybrid'] # 'hybrid' is often a combination of these
        
        for baseline_name in baselines:
            baseline_predictions_file = f'data/results/predictions_baseline_{baseline_name}_200_real_{attribute}.json'
            
            # Run baseline prediction on the 200 real records
            success = run_step(
                f"Run Baseline '{baseline_name}' Predictions ({attribute})",
                [sys.executable, 'scripts/baseline_heuristics.py',
                 '--baseline', baseline_name,
                 '--attribute', attribute,
                 '--data', REAL_GOLDEN_PATH,
                 '--output', baseline_predictions_file
                ]
            )
            if not success:
                print(f"Warning: Baseline '{baseline_name}' prediction for {attribute} failed.")
            else:
                # Evaluate baseline predictions
                success = run_step(
                    f"Evaluate Baseline '{baseline_name}' ({attribute})",
                    [sys.executable, 'scripts/evaluate_models.py',
                     '--predictions', baseline_predictions_file,
                     '--golden', REAL_GOLDEN_PATH,
                     '--attribute', attribute,
                     '--algorithm-name', f'Baseline {baseline_name.replace("_", " ").title()} ({attribute})',
                     '--output', f'data/results/baseline_evaluation_200_real_{baseline_name}_{attribute}.json'
                    ]
                )
                if not success:
                    print(f"Warning: Baseline '{baseline_name}' evaluation for {attribute} failed.")
    else:
        print(f"\nSkipping baseline evaluation for {attribute}.")
    
    # Step 6: Run final inference on 2000 Overture records (for ML model)
    if not args.skip_inference_2k:
        output_model_dir = Path(f'models/ml/{attribute}')
        summary_path = output_model_dir / 'training_summary.json'
        
        # Check if model training output directory and summary exist
        if not (output_model_dir.exists() and summary_path.exists()):
            print(f"Warning: No trained ML model found for {attribute} at {output_model_dir}. Skipping final ML inference.")
            return True # Allow pipeline to continue
        
        # Load best model name from summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        best_model_name = summary['best_model']
        model_path = output_model_dir / f"best_model_{best_model_name}.joblib"
        
        # Check if the actual best model file exists
        if not model_path.exists():
            print(f"Warning: Best model file not found at {model_path}. Skipping final ML inference.")
            return True # Allow pipeline to continue
            
        success = run_step(
            f"Run Final ML Inference on 2000 Overture Records ({attribute})",
            [sys.executable, '-m', 'scripts.run_inference',
             '--attribute', attribute,
             '--data', 'data/project_b_samples_2k.parquet',
             '--model', str(model_path),
             '--output', f'data/results/final_conflated_{attribute}_2k.json'
            ]
        )
        if not success:
            print(f"Warning: Final ML inference for {attribute} on 2000 records failed.")
    else:
        print(f"\nSkipping final ML inference for {attribute} on 2000 records.")

    return True # Indicate successful run for attribute


def main():
    """Run the complete algorithm pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete algorithm pipeline for OKR 2')
    parser.add_argument('--attributes', nargs='*', default=ALL_ATTRIBUTES,
                       help='Attributes to work on (default: all)')
    parser.add_argument('--skip-golden', action='store_true',
                       help='Skip synthetic golden dataset generation (use existing)')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature extraction (use existing)')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML model training')
    parser.add_argument('--skip-ml-eval', action='store_true',
                       help='Skip ML model evaluation on 200 real records')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='Skip baseline evaluation')
    parser.add_argument('--skip-inference-2k', action='store_true',
                       help='Skip final inference on 2000 Overture records')
    parser.add_argument('--skip-consolidation', action='store_true',
                       help='Skip final consolidation of 2k inference results')
    parser.add_argument('--synthetic-limit', type=int, default=2000,
                       help='Number of synthetic records to generate (0 for all)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OBJECTIVE 2 (OKR 2) ALGORITHM PIPELINE")
    print("="*80)
    print(f"Attributes to process: {', '.join(args.attributes)}")
    print(f"Working directory: {Path.cwd()}")
    
    # Run pipeline for each specified attribute
    for attr in args.attributes:
        if not run_pipeline_for_attribute(attr, args):
            print(f"Pipeline failed for attribute {attr}. Aborting.")
            return

    # Final step: Consolidate 2k inference results
    if not args.skip_consolidation:
        success = run_step(
            "Consolidate All 2k Inference Results",
            [sys.executable, 'scripts/consolidate_results.py']
        )
        if not success:
            print("Warning: Final consolidation failed.")
    else:
        print("\nSkipping final consolidation.")

    # Final Summary
    print("\n" + "="*80)
    print("COMPLETE PIPELINE RUN FINISHED")
    print("="*80)
    print("\nNext steps: Analyze evaluation reports in data/results/ and compare performance metrics.")


if __name__ == "__main__":
    main()