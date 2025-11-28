"""
Train ML Models for Attribute Selection

This script trains various ML models on extracted features to predict
which attribute version (current vs base) is better.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import time
import psutil
import os

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, classification_report
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    raise


def load_training_data(features_file: str) -> tuple:
    """
    Load training data with features and labels.
    
    Returns:
        (X, y) where X is features DataFrame and y is labels Series
    """
    print(f"Loading training data from {features_file}...")
    df = pd.read_parquet(features_file)
    
    # Separate features and labels
    label_col = 'label'
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")
    
    # Get feature columns (exclude metadata columns)
    exclude_cols = ['record_index', 'id', label_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0.0)
    y = df[label_col]
    
    # Convert labels to binary: 'current'/'same'/'c' -> 1, 'base'/'b' -> 0
    y_binary = y.apply(lambda x: 1 if x in ['current', 'same', 'c'] else 0)
    
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y_binary, feature_cols


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    """Train a Logistic Regression model."""
    print("\nTraining Logistic Regression...")
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    train_duration_seconds = end_time - start_time
    
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Val F1:   {val_f1:.4f}")
    print(f"  Val Acc:  {val_acc:.4f}")
    print(f"  Train Duration: {train_duration_seconds:.2f}s")
    print(f"  Peak Memory: {peak_memory_mb:.2f}MB")
    
    return {
        'model': model,
        'scaler': scaler,
        'model_type': 'logistic_regression',
        'train_f1': train_f1,
        'val_f1': val_f1,
        'val_acc': val_acc,
        'train_duration_seconds': train_duration_seconds,
        'initial_memory_mb': initial_memory_mb,
        'peak_memory_mb': peak_memory_mb
    }


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       n_estimators: int = 100, max_depth: Optional[int] = None) -> Dict[str, Any]:
    """Train a Random Forest model."""
    print("\nTraining Random Forest...")
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_duration_seconds = end_time - start_time
    
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Val F1:   {val_f1:.4f}")
    print(f"  Val Acc:  {val_acc:.4f}")
    print(f"  Train Duration: {train_duration_seconds:.2f}s")
    print(f"  Peak Memory: {peak_memory_mb:.2f}MB")
    
    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top 5 features: {[f[0] for f in top_features[:5]]}")
    
    return {
        'model': model,
        'scaler': None,
        'model_type': 'random_forest',
        'train_f1': train_f1,
        'val_f1': val_f1,
        'val_acc': val_acc,
        'train_duration_seconds': train_duration_seconds,
        'initial_memory_mb': initial_memory_mb,
        'peak_memory_mb': peak_memory_mb,
        'feature_importance': feature_importance
    }


def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_estimators: int = 100, learning_rate: float = 0.1) -> Dict[str, Any]:
    """Train a Gradient Boosting model."""
    print("\nTraining Gradient Boosting...")
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_duration_seconds = end_time - start_time
    
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Val F1:   {val_f1:.4f}")
    print(f"  Val Acc:  {val_acc:.4f}")
    print(f"  Train Duration: {train_duration_seconds:.2f}s")
    print(f"  Peak Memory: {peak_memory_mb:.2f}MB")
    
    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top 5 features: {[f[0] for f in top_features[:5]]}")
    
    return {
        'model': model,
        'scaler': None,
        'model_type': 'gradient_boosting',
        'train_f1': train_f1,
        'val_f1': val_f1,
        'val_acc': val_acc,
        'train_duration_seconds': train_duration_seconds,
        'initial_memory_mb': initial_memory_mb,
        'peak_memory_mb': peak_memory_mb,
        'feature_importance': feature_importance
    }


def train_all_models(
    features_file: str,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = 'models/ml_models'
) -> Dict[str, Any]:
    """
    Train all ML models and return results.
    
    Returns:
        Dictionary with trained models and results
    """
    # Load data
    X, y, feature_cols = load_training_data(features_file)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set:   {len(X_val)} samples")
    
    # Train all models
    results = {}
    
    # Logistic Regression
    try:
        lr_result = train_logistic_regression(X_train, y_train, X_val, y_val)
        results['logistic_regression'] = lr_result
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
    
    # Random Forest
    try:
        rf_result = train_random_forest(X_train, y_train, X_val, y_val)
        results['random_forest'] = rf_result
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    # Gradient Boosting
    try:
        gb_result = train_gradient_boosting(X_train, y_train, X_val, y_val)
        results['gradient_boosting'] = gb_result
    except Exception as e:
        print(f"Error training Gradient Boosting: {e}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['val_f1'])
        best_model = results[best_model_name]
        print(f"\n{'='*80}")
        print(f"Best Model: {best_model_name} (F1: {best_model['val_f1']:.4f})")
        print(f"{'='*80}")
        
        # Save best model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / f"best_model_{best_model_name}.joblib"
        scaler_file = output_path / f"scaler_{best_model_name}.joblib" if best_model['scaler'] else None
        
        # Save model
        joblib.dump({
            'model': best_model['model'],
            'model_type': best_model['model_type'],
            'feature_cols': feature_cols,
            'val_f1': best_model['val_f1'],
            'val_acc': best_model['val_acc'],
            'train_duration_seconds': best_model.get('train_duration_seconds'),
            'initial_memory_mb': best_model.get('initial_memory_mb'),
            'peak_memory_mb': best_model.get('peak_memory_mb')
        }, model_file)
        print(f"Saved best model to {model_file}")
        
        # Save scaler if exists
        if scaler_file and best_model['scaler']:
            joblib.dump(best_model['scaler'], scaler_file)
            print(f"Saved scaler to {scaler_file}")
        
        # Save results summary
        summary = {
            'best_model': best_model_name,
            'best_val_f1': float(best_model['val_f1']),
            'best_val_acc': float(best_model['val_acc']),
            'all_models': {
                name: {
                    'val_f1': float(r['val_f1']),
                    'val_acc': float(r['val_acc']),
                    'model_type': r['model_type'],
                    'train_duration_seconds': r.get('train_duration_seconds'),
                    'initial_memory_mb': r.get('initial_memory_mb'),
                    'peak_memory_mb': r.get('peak_memory_mb')
                }
                for name, r in results.items()
            }
        }
        
        summary_file = output_path / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training summary to {summary_file}")
    
    return results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models for attribute selection')
    parser.add_argument('--features', required=True,
                       help='Features file (parquet)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Validation set size (default: 0.2)')
    parser.add_argument('--output-dir', default='models/ml_models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    train_all_models(
        features_file=args.features,
        test_size=args.test_size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


