
import json
import pandas as pd
from scripts.extract_features import extract_features_batch

INPUT_FILE = 'data/synthetic_golden_dataset_2k.json'
OUTPUT_FEATURES = 'data/processed/features_name_synthetic.parquet'

def process_synthetic():
    print(f"Loading synthetic dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        records = json.load(f)
        
    # Convert to DataFrame format expected by feature extractor
    df_rows = []
    labels = {}
    
    for r in records:
        row = {
            'id': r['id'],
            'names': r['data']['current']['names'],
            'base_names': r['data']['base']['names'],
            # Add other fields if feature extractor needs them (it currently focuses on names)
            'confidence': 1.0, # Synthetic 'current' is perfect
            'base_confidence': 0.5, # Synthetic 'base' is degraded
            'sources': '',
            'base_sources': ''
        }
        df_rows.append(row)
        labels[r['id']] = r['label']
        
    df = pd.DataFrame(df_rows)
    print(f"Converted {len(df)} records to DataFrame.")
    
    # Extract Features
    print("Extracting features...")
    features_df = extract_features_batch(df, attribute='name')
    
    # Add Labels
    print("Applying labels...")
    features_df['label'] = features_df['id'].map(labels)
    
    # Save
    features_df.to_parquet(OUTPUT_FEATURES, index=False)
    print(f"✅ Saved features with labels to {OUTPUT_FEATURES}")

if __name__ == "__main__":
    process_synthetic()
