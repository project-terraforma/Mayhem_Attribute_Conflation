"""
Data Exploration Script for project_b_samples_2k.parquet

This script analyzes the structure, coverage, and quality of the attribute conflation dataset.
"""

import pandas as pd
import json
from pathlib import Path


def load_data(filepath='data/project_b_samples_2k.parquet'):
    """Load the parquet dataset."""
    return pd.read_parquet(filepath)


def analyze_structure(df):
    """Analyze basic structure of the dataset."""
    print("=" * 80)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")
    
    print(f"\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:25} {str(dtype)}")
    
    print(f"\nNull value counts:")
    null_counts = df.isnull().sum()
    for col in df.columns:
        count = null_counts[col]
        pct = (count / len(df)) * 100
        if count > 0:
            print(f"  {col:25} {count:4} ({pct:5.1f}%)")


def analyze_attributes(df):
    """Analyze attribute coverage and format."""
    print("\n" + "=" * 80)
    print("ATTRIBUTE COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Key attributes to analyze
    attributes = {
        'names': 'names',
        'phones': 'phones',
        'websites': 'websites',
        'addresses': 'addresses',
        'categories': 'categories',
        'emails': 'emails',
        'socials': 'socials',
        'brand': 'brand'
    }
    
    for attr_name, col in attributes.items():
        print(f"\n--- {attr_name.upper()} ---")
        current_col = col
        base_col = f'base_{col}'
        
        # Count non-null
        current_non_null = df[current_col].notna().sum()
        base_non_null = df[base_col].notna().sum()
        
        print(f"  Current version: {current_non_null}/{len(df)} ({current_non_null/len(df)*100:.1f}%)")
        print(f"  Base version:    {base_non_null}/{len(df)} ({base_non_null/len(df)*100:.1f}%)")
        
        # Check differences
        if attr_name in ['names', 'phones', 'websites', 'categories']:
            # These are JSON-like strings
            try:
                diff_count = 0
                for idx in df.index:
                    current_val = df.loc[idx, current_col]
                    base_val = df.loc[idx, base_col]
                    
                    if pd.notna(current_val) and pd.notna(base_val):
                        if current_val.strip() != base_val.strip():
                            diff_count += 1
                
                print(f"  Different values: {diff_count}/{len(df)} ({diff_count/len(df)*100:.1f}%)")
            except:
                pass


def show_sample_records(df, n=3):
    """Display sample records for understanding data format."""
    print("\n" + "=" * 80)
    print(f"SAMPLE RECORDS (showing {n})")
    print("=" * 80)
    
    for i in range(min(n, len(df))):
        print(f"\n{'='*80}")
        print(f"RECORD {i+1}")
        print(f"{'='*80}")
        
        row = df.iloc[i]
        
        print(f"\nID: {row['id']}")
        print(f"Base ID: {row['base_id']}")
        print(f"Confidence: Current={row['confidence']:.3f}, Base={row['base_confidence']:.3f}")
        
        # Parse and display key attributes
        attrs = [
            ('names', 'Names'),
            ('phones', 'Phones'),
            ('websites', 'Websites'),
            ('categories', 'Categories'),
            ('addresses', 'Addresses'),
        ]
        
        for attr, label in attrs:
            print(f"\n{label}:")
            print(f"  Current: {row[attr]}")
            print(f"  Base:    {row[f'base_{attr}']}")
        
        # Sources
        print(f"\nSources:")
        print(f"  Current: {row['sources'][:100]}...")
        print(f"  Base:    {row['base_sources'][:100]}...")


def analyze_json_structure(df):
    """Analyze the structure of JSON-encoded fields."""
    print("\n" + "=" * 80)
    print("JSON FIELD STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Check if fields are JSON strings or actual JSON
    json_fields = ['names', 'categories', 'addresses', 'base_names', 'base_categories', 'base_addresses']
    
    for field in json_fields:
        if field in df.columns:
            print(f"\n--- {field} ---")
            sample_value = df[field].dropna().iloc[0] if not df[field].dropna().empty else None
            
            if sample_value:
                print(f"  Sample value: {sample_value}")
                try:
                    parsed = json.loads(sample_value) if isinstance(sample_value, str) else sample_value
                    print(f"  Type: {type(parsed)}")
                    if isinstance(parsed, dict):
                        print(f"  Keys: {list(parsed.keys())}")
                        for key in list(parsed.keys())[:2]:
                            print(f"    - {key}: {parsed[key]}")
                except:
                    print(f"  Could not parse as JSON")


def main():
    """Run all analyses."""
    print("\nLoading data...")
    df = load_data()
    
    analyze_structure(df)
    analyze_attributes(df)
    analyze_json_structure(df)
    show_sample_records(df, n=3)
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    
    # Save summary statistics
    summary = {
        'total_records': int(len(df)),
        'total_columns': int(len(df.columns)),
        'attribute_coverage': {}
    }
    
    for col in df.columns:
        summary['attribute_coverage'][col] = {
            'non_null_count': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum()),
            'null_percentage': float(df[col].isna().sum() / len(df) * 100)
        }
    
    output_file = Path('docs') / 'data_exploration_summary.json'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()

