import pandas as pd
import json
import joblib
import argparse
from pathlib import Path
import time
import psutil
import os
from scripts.extract_features import extract_features_batch

def get_attribute_value(row, attribute, source='current'):
    """Helper to extract the raw value for a given attribute."""
    col_map = {
        'name': ('names', 'base_names', 'primary'),
        'phone': ('phones', 'base_phones', 0),
        'website': ('websites', 'base_websites', 0),
        'address': ('addresses', 'base_addresses', 0),
        'category': ('categories', 'base_categories', 'primary')
    }
    
    if attribute not in col_map:
        return "N/A"
        
    curr_col, base_col, key = col_map[attribute]
    col = curr_col if source == 'current' else base_col
    val_str = row.get(col, '')
    
    try:
        parsed = json.loads(val_str)
        if isinstance(parsed, dict):
            return str(parsed.get(key, ''))
        elif isinstance(parsed, list) and len(parsed) > 0:
            # Special case for address which returns a dict in list
            if attribute == 'address':
                return str(parsed[0])
            return str(parsed[0])
    except:
        pass
    return str(val_str)

def run_inference():
    parser = argparse.ArgumentParser(description='Run inference on Overture data')
    parser.add_argument('--attribute', default='name', choices=['name', 'phone', 'website', 'address', 'category'])
    parser.add_argument('--data', default='data/project_b_samples_2k.parquet')
    parser.add_argument('--model', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    if args.model is None:
        # Try to find best model automatically
        model_dir = Path('models/ml_models')
        candidates = list(model_dir.glob(f'best_model_*.joblib'))
        if candidates:
            # Prefer gradient boosting, then random forest
            gb = list(model_dir.glob('best_model_gradient_boosting.joblib'))
            rf = list(model_dir.glob('best_model_random_forest.joblib'))
            args.model = str(gb[0]) if gb else (str(rf[0]) if rf else str(candidates[0]))
        else:
            print("Error: No model found. Please specify --model.")
            return

    if args.output is None:
        args.output = f'data/results/final_conflated_{args.attribute}_2k.json'

    print("="*80)
    print(f"RUNNING INFERENCE ON 2,000 OVERTURE RECORDS ({args.attribute.upper()} ATTRIBUTE)")
    print("="*80)

    # Measure initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_memory_mb:.2f} MB")

    # 1. Load Data
    print(f"Loading Overture data from {args.data}...")
    
    data_path = Path(args.data)
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Transform JSON data to a DataFrame structure similar to parquet
        processed_records = []
        for record in json_data:
            row_data = {
                'id': record['id'],
                'names': record['data']['current']['names'] if 'current' in record['data'] and 'names' in record['data']['current'] else None,
                'base_names': record['data']['base']['names'] if 'base' in record['data'] and 'names' in record['data']['base'] else None,
                'phones': record['data']['current']['phones'] if 'current' in record['data'] and 'phones' in record['data']['current'] else None,
                'base_phones': record['data']['base']['phones'] if 'base' in record['data'] and 'phones' in record['data']['base'] else None,
                'websites': record['data']['current']['websites'] if 'current' in record['data'] and 'websites' in record['data']['current'] else None,
                'base_websites': record['data']['base']['websites'] if 'base' in record['data'] and 'websites' in record['data']['base'] else None,
                'addresses': record['data']['current']['addresses'] if 'current' in record['data'] and 'addresses' in record['data']['current'] else None,
                'base_addresses': record['data']['base']['addresses'] if 'base' in record['data'] and 'addresses' in record['data']['base'] else None,
                'categories': record['data']['current']['categories'] if 'current' in record['data'] and 'categories' in record['data']['current'] else None,
                'base_categories': record['data']['base']['categories'] if 'base' in record['data'] and 'categories' in record['data']['base'] else None,
                'confidence': record['data']['current'].get('confidence', 0.0) if 'current' in record['data'] else 0.0,
                'base_confidence': record['data']['base'].get('confidence', 0.0) if 'base' in record['data'] else 0.0
            }
            processed_records.append(row_data)
        df = pd.DataFrame(processed_records)
    else:
        print(f"Error: Unsupported file type for {args.data}. Only .parquet and .json are supported.")
        return
        
    print(f"Loaded {len(df)} records.")

    # 2. Load Model
    print(f"Loading model from {args.model}...")
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found.")
        return
        
    model_data = joblib.load(args.model)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    print(f"Model loaded: {model_data['model_type']}")

    # 3. Extract Features
    print(f"Extracting features for '{args.attribute}' attribute...")
    features_df = extract_features_batch(df, attribute=args.attribute)
    
    # Ensure columns align with training
    # Add missing cols with 0
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
            
    X = features_df[feature_cols].fillna(0.0)
    
    # 4. Predict
    print("Predicting best attributes...")
    start_time = time.time()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    end_time = time.time()
    inference_duration_seconds = end_time - start_time
    
    # Measure peak memory usage
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # 5. Construct Results
    results = []
    stats = {"current": 0, "base": 0}
    
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        row = df.iloc[idx]
        
        val_c = get_attribute_value(row, args.attribute, 'current')
        val_b = get_attribute_value(row, args.attribute, 'base')
            
        choice = "current" if pred == 1 else "base"
        stats[choice] += 1
        
        chosen_value = val_c if choice == "current" else val_b
        confidence = max(prob)
        
        results.append({
            "id": row['id'],
            "record_index": idx,
            "attribute": args.attribute,
            "selected_source": choice,
            "model_confidence": float(confidence),
            "conflated_value": chosen_value,
            "candidates": {
                "current": val_c,
                "base": val_b
            },
            "inference_meta": {
                "duration_seconds": inference_duration_seconds,
                "initial_memory_mb": initial_memory_mb,
                "peak_memory_mb": peak_memory_mb
            }
        })
        
    # 6. Save Results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Total Records: {len(results)}")
    print(f"Decisions: Current={stats['current']}, Base={stats['base']}")
    print(f"Results saved to: {args.output}")
    print(f"Inference Duration: {inference_duration_seconds:.4f} seconds")
    print(f"Initial Memory: {initial_memory_mb:.2f} MB")
    print(f"Peak Memory: {peak_memory_mb:.2f} MB")
    
    # Show examples
    print("\nSample Decisions:")
    for r in results[:5]:
        print(f"[{r['selected_source'].upper()}] {str(r['candidates']['current'])[:30]}... vs {str(r['candidates']['base'])[:30]}... -> {str(r['conflated_value'])[:30]}...")

if __name__ == "__main__":
    run_inference()