import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, Literal
from pathlib import Path
from collections import Counter # Moved import to top for clarity


def parse_json_field(value):
    """Safely parse JSON field."""
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


class MostRecentBaseline:
    """
    Baseline 1: Most Recent Heuristic
    Always selects the current version (assumes newer = better).
    This is the primary baseline for KR1 comparison.
    """
    
    def __init__(self):
        self.name = "Most Recent"
    
    def predict(self, row: pd.Series, attribute: str = 'name') -> Literal['current', 'base', 'same', 'unclear']:
        """
        Predict which version is better.
        
        Returns:
            'current': Current version is better
            'base': Base version is better
            'same': Both are equivalent
            'unclear': Cannot determine
        """
        if attribute == 'name':
            current_val = row['names']
            base_val = row['base_names']
            
            if pd.isna(current_val) and pd.isna(base_val):
                return 'unclear'
            if pd.isna(current_val):
                return 'base'
            if pd.isna(base_val):
                return 'current'
            
            # Check if same
            current_primary = extract_name_primary(current_val)
            base_primary = extract_name_primary(base_val)
            
            if current_primary and base_primary and current_primary.lower() == base_primary.lower():
                return 'same'
            
            # Always prefer current (most recent)
            return 'current'
        
        # For other attributes, similar logic
        attr_col_curr = attribute + 's' if attribute != 'address' else 'addresses'
        attr_col_base = 'base_' + attr_col_curr
        
        current_val = row.get(attr_col_curr)
        base_val = row.get(attr_col_base)
        
        if pd.isna(current_val) and pd.isna(base_val):
            return 'unclear'
        if pd.isna(current_val):
            return 'base'
        if pd.isna(base_val):
            return 'current'
        
        # Check if same (simple string comparison)
        if str(current_val).strip() == str(base_val).strip():
            return 'same'
        
        return 'current'
    
    def predict_batch(self, df: pd.DataFrame, attribute: str = 'name') -> list:
        """Predict for a batch of records."""
        return [self.predict(row, attribute) for _, row in df.iterrows()]


class ConfidenceBaseline:
    """
    Baseline 2: Confidence-Based Heuristic
    Selects the version with higher confidence score.
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: Minimum difference in confidence to prefer one over the other
        """
        self.name = "Confidence-Based"
        self.threshold = threshold
    
    def predict(self, row: pd.Series, attribute: str = 'name') -> Literal['current', 'base', 'same', 'unclear']:
        """Predict based on confidence scores."""
        attr_col_curr = attribute + 's' if attribute != 'address' else 'addresses'
        attr_col_base = 'base_' + attr_col_curr
        
        current_data_val = row.get(attr_col_curr)
        base_data_val = row.get(attr_col_base)
        
        if pd.isna(current_data_val) and pd.isna(base_data_val):
            return 'unclear'
        if pd.isna(current_data_val):
            return 'base'
        if pd.isna(base_data_val):
            return 'current'
        
        # Check if same
        if attribute == 'name':
            current_primary = extract_name_primary(current_data_val)
            base_primary = extract_name_primary(base_data_val)
            if current_primary and base_primary and current_primary.lower() == base_primary.lower():
                return 'same'
        else:
            if str(current_data_val).strip() == str(base_data_val).strip():
                return 'same'
        
        # Compare confidence
        current_conf = row.get('confidence', 0.5)
        base_conf = row.get('base_confidence', 0.5)
        
        diff = current_conf - base_conf
        
        if abs(diff) < self.threshold:
            return 'same'
        
        return 'current' if diff > 0 else 'base'
    
    def predict_batch(self, df: pd.DataFrame, attribute: str = 'name') -> list:
        """Predict for a batch of records."""
        return [self.predict(row, attribute) for _, row in df.iterrows()]


class CompletenessBaseline:
    """
    Baseline 3: Completeness-Based Heuristic
    Selects the version with more complete data.
    """
    
    def __init__(self):
        self.name = "Completeness-Based"
    
    def _calculate_completeness(self, value):
        """Calculate a completeness score for a value."""
        if pd.isna(value):
            return 0.0
        
        score = 1.0 # Base score for existence
        
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    if 'primary' in parsed and parsed['primary']:
                        score += 0.5 # Has primary
                    if 'alternate' in parsed and parsed['alternate'] and len(parsed['alternate']) > 0:
                        score += 0.2 # Has alternates
                    if len(parsed) > 1: # More complex structure
                        score += 0.1
                elif isinstance(parsed, list):
                    score += len(parsed) * 0.1 # More items in list is more complete
                else: # Simple string
                    if len(value.strip()) > 0:
                        score += 0.1
            except json.JSONDecodeError: # Not JSON
                if len(value.strip()) > 0:
                    score += 0.1
        
        return score
    
    def predict(self, row: pd.Series, attribute: str = 'name') -> Literal['current', 'base', 'same', 'unclear']:
        """Predict based on completeness."""
        attr_col_curr = attribute + 's' if attribute != 'address' else 'addresses'
        attr_col_base = 'base_' + attr_col_curr
        
        current_val = row.get(attr_col_curr)
        base_val = row.get(attr_col_base)
        
        if pd.isna(current_val) and pd.isna(base_val):
            return 'unclear'
        if pd.isna(current_val):
            return 'base'
        if pd.isna(base_val):
            return 'current'
        
        # Check if same
        if attribute == 'name':
            current_primary = extract_name_primary(current_val)
            base_primary = extract_name_primary(base_val)
            if current_primary and base_primary and current_primary.lower() == base_primary.lower():
                return 'same'
        else:
            if str(current_val).strip() == str(base_val).strip():
                return 'same'
        
        # Compare completeness
        current_complete = self._calculate_completeness(current_val)
        base_complete = self._calculate_completeness(base_val)
        
        if abs(current_complete - base_complete) < 0.01: # Small epsilon for float comparison
            # Tie-break: if completeness is almost same, default to 'current' (Most Recent)
            return 'current'
        
        return 'current' if current_complete > base_complete else 'base'
    
    def predict_batch(self, df: pd.DataFrame, attribute: str = 'name') -> list:
        """Predict for a batch of records."""
        return [self.predict(row, attribute) for _, row in df.iterrows()]


class HybridBaseline:
    """
    Baseline 4: Hybrid Heuristic
    Combines multiple heuristics with weights.
    """
    
    def __init__(self, recency_weight: float = 0.3, confidence_weight: float = 0.5, completeness_weight: float = 0.2):
        """
        Args:
            recency_weight: Weight for recency (prefer current)
            confidence_weight: Weight for confidence scores
            completeness_weight: Weight for data completeness
        """
        self.name = "Hybrid"
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight
        self.completeness_weight = completeness_weight
        
        self.most_recent = MostRecentBaseline()
        self.confidence = ConfidenceBaseline()
        self.completeness = CompletenessBaseline()
    
    def predict(self, row: pd.Series, attribute: str = 'name') -> Literal['current', 'base', 'same', 'unclear']:
        """Predict using weighted combination of heuristics."""
        # Get predictions from each baseline
        recency_pred = self.most_recent.predict(row, attribute)
        confidence_pred = self.confidence.predict(row, attribute)
        completeness_pred = self.completeness.predict(row, attribute)
        
        # Handle unclear cases
        if recency_pred == 'unclear' and confidence_pred == 'unclear' and completeness_pred == 'unclear':
            return 'unclear'
        
        # Score each option
        scores = {'current': 0.0, 'base': 0.0, 'same': 0.0}
        
        # Recency: current gets weight
        if recency_pred == 'current':
            scores['current'] += self.recency_weight
        elif recency_pred == 'base':
            scores['base'] += self.recency_weight
        elif recency_pred == 'same':
            # If recency says same, it doesn't push current/base, so no score here
            pass
        
        # Confidence
        if confidence_pred == 'current':
            scores['current'] += self.confidence_weight
        elif confidence_pred == 'base':
            scores['base'] += self.confidence_weight
        elif confidence_pred == 'same':
            # Split confidence weight if both seem equal based on confidence
            scores['current'] += self.confidence_weight / 2
            scores['base'] += self.confidence_weight / 2
        
        # Completeness
        if completeness_pred == 'current':
            scores['current'] += self.completeness_weight
        elif completeness_pred == 'base':
            scores['base'] += self.completeness_weight
        elif completeness_pred == 'same':
             # Split completeness weight
            scores['current'] += self.completeness_weight / 2
            scores['base'] += self.completeness_weight / 2
        
        # Return highest scoring option
        if abs(scores['current'] - scores['base']) < 0.01: # Epsilon for ties
             # If scores are very close, try to keep 'same' if any baseline suggested it
            if recency_pred == 'same' or confidence_pred == 'same' or completeness_pred == 'same':
                return 'same'
            return 'current' # Default tie-break to current
        
        return 'current' if scores['current'] > scores['base'] else 'base'
    
    def predict_batch(self, df: pd.DataFrame, attribute: str = 'name') -> list:
        """Predict for a batch of records."""
        return [self.predict(row, attribute) for _, row in df.iterrows()]


def load_data_for_baselines(data_file: str) -> pd.DataFrame:
    """Load input data for baselines, handling both parquet and JSON."""
    data_path = Path(data_file)
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
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
        raise ValueError(f"Unsupported file type for {data_file}. Only .parquet and .json are supported.")
    
    return df


def main():
    """Test baseline heuristics."""
    import argparse
    from collections import Counter
    
    parser = argparse.ArgumentParser(description='Test baseline heuristics')
    parser.add_argument('--data', required=True,
                       help='Input data file (parquet or json)')
    parser.add_argument('--attribute', default='name',
                       choices=['name', 'phone', 'website', 'address', 'category'],
                       help='Attribute to evaluate')
    parser.add_argument('--baseline', default='most_recent',
                       choices=['most_recent', 'confidence', 'completeness', 'hybrid'],
                       help='Baseline algorithm to use')
    parser.add_argument('--output', help='Output JSON file for predictions (dict: record_id -> prediction)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data_for_baselines(args.data)
    
    # Initialize baseline
    if args.baseline == 'most_recent':
        baseline = MostRecentBaseline()
    elif args.baseline == 'confidence':
        baseline = ConfidenceBaseline()
    elif args.baseline == 'completeness':
        baseline = CompletenessBaseline()
    else:
        baseline = HybridBaseline()
    
    print(f"\nUsing baseline: {baseline.name} for attribute: {args.attribute}")
    
    # Make predictions
    predictions_list = baseline.predict_batch(df, args.attribute)
    
    # Format predictions for output
    output_predictions = {}
    for idx, row in df.iterrows():
        output_predictions[row['id']] = predictions_list[idx]
    
    # Save predictions
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {args.output}")
    else:
        print("\nPredictions made but not saved (no --output specified).")
    
    print(f"\nPrediction distribution: {dict(Counter(predictions_list))}")


if __name__ == "__main__":
    main()