"""
Generate Sample Golden Dataset for Algorithm Development

This script creates a synthetic ground truth dataset based on heuristics.
When the real golden dataset is ready, we can swap it in easily.

The sample dataset uses rule-based logic to simulate human annotation decisions:
- Higher confidence scores → more likely to be correct
- Better formatting → preferred
- More complete data → preferred
- Recency (current vs base) → slight preference for current
"""

#sample golden dataset created with heuristics

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def parse_json_field(value):
    """Safely parse JSON field (may be string or already dict)."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return None
    return value


def extract_name_primary(names_field):
    """Extract primary name from names field."""
    parsed = parse_json_field(names_field)
    if parsed and isinstance(parsed, dict):
        return parsed.get('primary', '').strip()
    return ''


def compare_names(current_names, base_names):
    """
    Compare name attributes and decide which is better.
    Returns: 'current', 'base', or 'same'
    """
    current_primary = extract_name_primary(current_names)
    base_primary = extract_name_primary(base_names)
    
    # If identical, return 'same'
    if current_primary.lower() == base_primary.lower():
        return 'same'
    
    # Heuristics for better name:
    # 1. More complete/formal (longer, proper capitalization)
    # 2. Better formatting (has apostrophes, proper punctuation)
    # 3. Not all caps or all lowercase
    
    current_score = 0
    base_score = 0
    
    # Length (prefer longer, more complete names)
    if len(current_primary) > len(base_primary):
        current_score += 1
    elif len(base_primary) > len(current_primary):
        base_score += 1
    
    # Capitalization quality (prefer proper case)
    if current_primary.istitle() or (current_primary[0].isupper() if current_primary else False):
        current_score += 1
    if base_primary.istitle() or (base_primary[0].isupper() if base_primary else False):
        base_score += 1
    
    # Punctuation (prefer names with proper punctuation)
    if "'" in current_primary or "-" in current_primary:
        current_score += 0.5
    if "'" in base_primary or "-" in base_primary:
        base_score += 0.5
    
    # Avoid all caps or all lowercase
    if not current_primary.isupper() and not current_primary.islower():
        current_score += 0.5
    if not base_primary.isupper() and not base_primary.islower():
        base_score += 0.5
    
    # If scores are equal, slight preference for current (recency)
    if abs(current_score - base_score) < 0.1:
        return 'current'
    
    return 'current' if current_score > base_score else 'base'


def generate_sample_labels(df: pd.DataFrame, attribute: str = 'name') -> pd.DataFrame:
    """
    Generate sample ground truth labels for a given attribute.
    
    Args:
        df: DataFrame with current and base attribute columns
        attribute: Attribute name ('name', 'phone', 'website', 'address', 'category')
    
    Returns:
        DataFrame with added 'label_{attribute}' column
    """
    labels = []
    
    if attribute == 'name':
        for idx, row in df.iterrows():
            current_val = row['names']
            base_val = row['base_names']
            
            # If both missing, mark as unclear
            if pd.isna(current_val) and pd.isna(base_val):
                labels.append('unclear')
                continue
            
            # If only one exists, prefer that one
            if pd.isna(current_val):
                labels.append('base')
                continue
            if pd.isna(base_val):
                labels.append('current')
                continue
            
            # Compare and decide
            decision = compare_names(current_val, base_val)
            labels.append(decision)
    
    elif attribute == 'phone':
        for idx, row in df.iterrows():
            current_val = row['phones']
            base_val = row['base_phones']
            
            if pd.isna(current_val) and pd.isna(base_val):
                labels.append('unclear')
                continue
            if pd.isna(current_val):
                labels.append('base')
                continue
            if pd.isna(base_val):
                labels.append('current')
                continue
            
            # Prefer international format, more complete
            current_str = str(current_val) if not isinstance(current_val, list) else str(current_val[0]) if current_val else ''
            base_str = str(base_val) if not isinstance(base_val, list) else str(base_val[0]) if base_val else ''
            
            # Prefer E.164 format (+country code)
            if current_str.startswith('+') and not base_str.startswith('+'):
                labels.append('current')
            elif base_str.startswith('+') and not current_str.startswith('+'):
                labels.append('base')
            # Prefer longer (more complete) number
            elif len(current_str.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) > \
                 len(base_str.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')):
                labels.append('current')
            elif len(base_str.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) > \
                 len(current_str.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')):
                labels.append('base')
            else:
                # Default to current (recency)
                labels.append('current')
    
    elif attribute == 'website':
        for idx, row in df.iterrows():
            current_val = row['websites']
            base_val = row['base_websites']
            
            if pd.isna(current_val) and pd.isna(base_val):
                labels.append('unclear')
                continue
            if pd.isna(current_val):
                labels.append('base')
                continue
            if pd.isna(base_val):
                labels.append('current')
                continue
            
            # Prefer HTTPS over HTTP
            current_str = str(current_val) if not isinstance(current_val, list) else str(current_val[0]) if current_val else ''
            base_str = str(base_val) if not isinstance(base_val, list) else str(base_val[0]) if base_val else ''
            
            if 'https://' in current_str.lower() and 'http://' in base_str.lower():
                labels.append('current')
            elif 'https://' in base_str.lower() and 'http://' in current_str.lower():
                labels.append('base')
            else:
                # Default to current
                labels.append('current')
    
    else:
        # For other attributes, use confidence-based heuristic
        for idx, row in df.iterrows():
            current_val = row.get(attribute + 's', None) if attribute != 'address' else row.get('addresses', None)
            base_val = row.get('base_' + attribute + 's', None) if attribute != 'address' else row.get('base_addresses', None)
            
            if pd.isna(current_val) and pd.isna(base_val):
                labels.append('unclear')
                continue
            if pd.isna(current_val):
                labels.append('base')
                continue
            if pd.isna(base_val):
                labels.append('current')
                continue
            
            # Use confidence scores
            current_conf = row.get('confidence', 0.5)
            base_conf = row.get('base_confidence', 0.5)
            
            if current_conf > base_conf + 0.05:
                labels.append('current')
            elif base_conf > current_conf + 0.05:
                labels.append('base')
            else:
                labels.append('current')  # Default to current (recency)
    
    return labels


def create_golden_dataset(
    input_file: str = 'data/project_b_samples_2k.parquet',
    output_file: str = 'data/processed/golden_dataset_sample.json',
    attributes: list = ['name', 'phone', 'website', 'address', 'category']
):
    """
    Create a sample golden dataset with labels for specified attributes.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output JSON file
        attributes: List of attributes to label
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} records")
    
    # Generate labels for each attribute
    golden_data = []
    
    for idx, row in df.iterrows():
        record_labels = {
            'id': row['id'],
            'base_id': row['base_id'],
            'record_index': int(idx),
            'labels': {}
        }
        
        for attr in attributes:
            labels = generate_sample_labels(df.iloc[[idx]], attr)
            if labels:
                record_labels['labels'][attr] = labels[0]
        
        golden_data.append(record_labels)
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(golden_data, f, indent=2)
    
    print(f"\nSample golden dataset created: {output_file}")
    print(f"Total records: {len(golden_data)}")
    print(f"Attributes labeled: {', '.join(attributes)}")
    
    # Print label distribution
    print("\nLabel distribution:")
    for attr in attributes:
        attr_labels = [r['labels'].get(attr, 'unclear') for r in golden_data]
        from collections import Counter
        dist = Counter(attr_labels)
        print(f"  {attr}: {dict(dist)}")
    
    return golden_data


def create_train_test_split(
    golden_file: str = 'data/processed/golden_dataset_sample.json',
    test_size: int = 1000,
    random_state: int = 42
):
    """
    Split golden dataset into train/validation/test sets.
    
    Args:
        golden_file: Path to golden dataset JSON
        test_size: Number of records for test set
        random_state: Random seed
    """
    with open(golden_file, 'r') as f:
        golden_data = json.load(f)
    
    # Shuffle
    np.random.seed(random_state)
    indices = np.random.permutation(len(golden_data))
    
    # Split: test (1000), then 80/20 for train/val from remainder
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]
    
    val_size = int(len(remaining_indices) * 0.2)
    val_indices = remaining_indices[:val_size]
    train_indices = remaining_indices[val_size:]
    
    splits = {
        'train': [golden_data[i] for i in train_indices],
        'validation': [golden_data[i] for i in val_indices],
        'test': [golden_data[i] for i in test_indices]
    }
    
    # Save splits
    for split_name, split_data in splits.items():
        output_file = f'data/processed/golden_dataset_{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Created {split_name} set: {len(split_data)} records -> {output_file}")
    
    return splits


def main():
    """Main function to generate sample golden dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample golden dataset')
    parser.add_argument('--input', default='data/project_b_samples_2k.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', default='data/processed/golden_dataset_sample.json',
                       help='Output JSON file')
    parser.add_argument('--attributes', nargs='+', 
                       default=['name', 'phone', 'website', 'address', 'category'],
                       help='Attributes to label')
    parser.add_argument('--split', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--test-size', type=int, default=1000,
                       help='Test set size')
    
    args = parser.parse_args()
    
    # Generate golden dataset
    golden_data = create_golden_dataset(
        input_file=args.input,
        output_file=args.output,
        attributes=args.attributes
    )
    
    # Create splits if requested
    if args.split:
        create_train_test_split(
            golden_file=args.output,
            test_size=args.test_size
        )


if __name__ == "__main__":
    main()

