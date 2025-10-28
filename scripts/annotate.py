"""
Annotation Interface for Attribute Selection

This script provides a simple interface for comparing and labeling attribute values
from the current vs. base versions of place data.
"""

import pandas as pd
import json
import os
from pathlib import Path


class AttributeAnnotator:
    """Annotation interface for comparing attribute versions."""
    
    def __init__(self, data_file='data/project_b_samples_2k.parquet'):
        """Initialize with data file."""
        self.df = pd.read_parquet(data_file)
        self.annotations = []
        self.annotator_name = ""
        self.current_index = 0
        
        # Attribute fields to annotate
        self.attributes = [
            ('names', 'Name'),
            ('phones', 'Phone'),
            ('websites', 'Website'),
            ('addresses', 'Address'),
            ('categories', 'Category'),
            ('socials', 'Social Media'),
            ('brand', 'Brand')
        ]
    
    def print_separator(self):
        """Print visual separator."""
        print("\n" + "="*80 + "\n")
    
    def display_record(self, index):
        """Display current and base versions of attributes for a record."""
        row = self.df.iloc[index]
        
        self.print_separator()
        print(f"Record {index + 1} / {len(self.df)}")
        print(f"ID: {row['id']}")
        print(f"Base ID: {row['base_id']}")
        print(f"Confidence: Current={row['confidence']:.3f}, Base={row['base_confidence']:.3f}")
        self.print_separator()
        
        for attr_key, attr_label in self.attributes:
            print(f"\n--- {attr_label} ---")
            current_val = row[attr_key]
            base_val = row[f'base_{attr_key}']
            
            print(f"\nCurrent:")
            if pd.isna(current_val):
                print("  [MISSING]")
            else:
                print(f"  {current_val}")
            
            print(f"\nBase:")
            if pd.isna(base_val):
                print("  [MISSING]")
            else:
                print(f"  {base_val}")
    
    def get_annotation(self, index):
        """Get annotation for a single record."""
        self.display_record(index)
        
        print("\nOptions:")
        print("  c - Current version is better")
        print("  b - Base version is better")
        print("  s - Same/equivalent")
        print("  u - Unclear/need review")
        print("  q - Quit and save")
        print("  n - Skip to next")
        
        while True:
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'q':
                return 'quit'
            elif choice == 'n':
                return 'next'
            elif choice in ['c', 'b', 's', 'u']:
                notes = input("Optional notes: ").strip()
                
                annotation = {
                    'record_index': index,
                    'id': self.df.iloc[index]['id'],
                    'choice': choice,
                    'notes': notes,
                    'annotator': self.annotator_name
                }
                
                return annotation
            else:
                print("Invalid choice. Please try again.")
    
    def save_annotations(self, filename=None):
        """Save annotations to file."""
        if filename is None:
            filename = f'data/annotations_{self.annotator_name}.json'
        
        # Create directory if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"\nAnnotations saved to {filename}")
    
    def load_annotations(self, filename):
        """Load existing annotations."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.annotations = json.load(f)
            
            # Track which records already annotated
            annotated_indices = set(a['record_index'] for a in self.annotations)
            print(f"\nLoaded {len(self.annotations)} existing annotations")
            return annotated_indices
        return set()
    
    def annotate_records(self, start_index=0, annotator_name=None, load_existing=False):
        """Annotate records starting from a given index."""
        self.annotator_name = annotator_name or input("Annotator name: ")
        
        # Load existing annotations if requested
        annotated_indices = set()
        if load_existing:
            filename = f'data/annotations_{self.annotator_name}.json'
            annotated_indices = self.load_annotations(filename)
        
        self.current_index = start_index
        
        print(f"\nStarting annotation from record {start_index + 1}")
        print(f"Annotating as: {self.annotator_name}")
        
        while self.current_index < len(self.df):
            # Skip if already annotated
            if self.current_index in annotated_indices:
                print(f"\nSkipping record {self.current_index + 1} (already annotated)")
                self.current_index += 1
                continue
            
            result = self.get_annotation(self.current_index)
            
            if result == 'quit':
                self.save_annotations()
                print("\nAnnotation session ended. Progress saved.")
                break
            elif result == 'next':
                self.current_index += 1
            elif isinstance(result, dict):
                self.annotations.append(result)
                self.current_index += 1
                print(f"\nAnnotated. Total: {len(self.annotations)}")
        else:
            print(f"\nCompleted all records!")
            self.save_annotations()


def sample_records_for_agreement(df, n=200, seed=42):
    """Sample random records for inter-annotator agreement study."""
    return df.sample(n=n, random_state=seed)


def prepare_agreement_samples(output_file='data/agreement_sample_200.csv', n=200):
    """Prepare a CSV file with 200 sampled records for agreement study."""
    df = pd.read_parquet('data/project_b_samples_2k.parquet')
    
    # Sample random records
    sample_df = sample_records_for_agreement(df, n=n)
    
    # Add annotation columns
    for attr_key, attr_label in [
        ('names', 'name'),
        ('phones', 'phone'),
        ('websites', 'website'),
        ('addresses', 'address'),
        ('categories', 'category')
    ]:
        sample_df[f'{attr_label}_annotation_1'] = ''
        sample_df[f'{attr_label}_annotation_2'] = ''
        sample_df[f'{attr_label}_notes'] = ''
    
    # Save to CSV
    sample_df.to_csv(output_file, index=False)
    print(f"Created agreement sample with {n} records: {output_file}")
    
    return sample_df


def main():
    """Main annotation interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Attribute annotation interface')
    parser.add_argument('--mode', choices=['annotate', 'prepare'], default='annotate',
                       help='Mode: annotate records or prepare agreement samples')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting record index')
    parser.add_argument('--annotator', type=str,
                       help='Annotator name')
    parser.add_argument('--load', action='store_true',
                       help='Load existing annotations')
    
    args = parser.parse_args()
    
    if args.mode == 'annotate':
        annotator = AttributeAnnotator()
        annotator.annotate_records(
            start_index=args.start,
            annotator_name=args.annotator,
            load_existing=args.load
        )
    elif args.mode == 'prepare':
        prepare_agreement_samples()


if __name__ == "__main__":
    main()

