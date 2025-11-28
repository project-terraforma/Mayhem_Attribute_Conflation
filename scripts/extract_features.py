"""
Feature Extraction for Attribute Selection

This module extracts features from current vs base attribute pairs
to train ML models for attribute selection.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path


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


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity (simplified version)."""
    if s1 == s2:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    # Simplified Jaro similarity
    match_window = max(len(s1), len(s2)) // 2 - 1
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    matches = 0
    transpositions = 0
    
    for i in range(len(s1)):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len(s2))
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches) / 3.0
    
    # Winkler modification
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + (0.1 * prefix * (1 - jaro))


def calculate_string_similarity(s1: str, s2: str) -> Dict[str, float]:
    """Calculate various string similarity metrics."""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    
    # Exact match
    exact_match = 1.0 if s1 == s2 else 0.0
    exact_match_lower = 1.0 if s1_lower == s2_lower else 0.0
    
    # Length ratio
    max_len = max(len(s1), len(s2), 1)
    min_len = min(len(s1), len(s2))
    length_ratio = min_len / max_len
    
    # Levenshtein distance
    lev_dist = levenshtein_distance(s1, s2)
    lev_similarity = 1.0 - (lev_dist / max_len) if max_len > 0 else 0.0
    
    # Jaro-Winkler
    jw_similarity = jaro_winkler_similarity(s1, s2)
    
    return {
        'exact_match': exact_match,
        'exact_match_lower': exact_match_lower,
        'length_ratio': length_ratio,
        'levenshtein_similarity': lev_similarity,
        'jaro_winkler_similarity': jw_similarity
    }


def extract_name_features(current_names: Any, base_names: Any) -> Dict[str, float]:
    """Extract features for name attribute comparison."""
    current_primary = extract_name_primary(current_names)
    base_primary = extract_name_primary(base_names)
    
    features = {}
    
    # Basic string features
    if current_primary and base_primary:
        sim_features = calculate_string_similarity(current_primary, base_primary)
        features.update({f'name_{k}': v for k, v in sim_features.items()})
        
        # Length features
        features['name_current_length'] = len(current_primary)
        features['name_base_length'] = len(base_primary)
        features['name_length_diff'] = abs(len(current_primary) - len(base_primary))
        features['name_length_ratio'] = min(len(current_primary), len(base_primary)) / max(len(current_primary), len(base_primary), 1)
        
        # Capitalization features
        features['name_current_has_proper_case'] = 1.0 if current_primary.istitle() or (current_primary[0].isupper() if current_primary else False) else 0.0
        features['name_base_has_proper_case'] = 1.0 if base_primary.istitle() or (base_primary[0].isupper() if base_primary else False) else 0.0
        features['name_current_all_caps'] = 1.0 if current_primary.isupper() else 0.0
        features['name_base_all_caps'] = 1.0 if base_primary.isupper() else 0.0
        features['name_current_all_lower'] = 1.0 if current_primary.islower() else 0.0
        features['name_base_all_lower'] = 1.0 if base_primary.islower() else 0.0
        
        # Punctuation features
        features['name_current_has_apostrophe'] = 1.0 if "'" in current_primary else 0.0
        features['name_base_has_apostrophe'] = 1.0 if "'" in base_primary else 0.0
        features['name_current_has_hyphen'] = 1.0 if "-" in current_primary else 0.0
        features['name_base_has_hyphen'] = 1.0 if "-" in base_primary else 0.0
        features['name_current_has_punctuation'] = 1.0 if any(c in current_primary for c in ".,;:!?") else 0.0
        features['name_base_has_punctuation'] = 1.0 if any(c in base_primary for c in ".,;:!?") else 0.0
    else:
        # Missing values
        features['name_current_missing'] = 1.0 if not current_primary else 0.0
        features['name_base_missing'] = 1.0 if not base_primary else 0.0
        # Fill with zeros for similarity features
        for k in ['exact_match', 'exact_match_lower', 'length_ratio', 'levenshtein_similarity', 'jaro_winkler_similarity']:
            features[f'name_{k}'] = 0.0
        for k in ['current_length', 'base_length', 'length_diff', 'length_ratio', 
                  'current_has_proper_case', 'base_has_proper_case',
                  'current_all_caps', 'base_all_caps', 'current_all_lower', 'base_all_lower',
                  'current_has_apostrophe', 'base_has_apostrophe',
                  'current_has_hyphen', 'base_has_hyphen',
                  'current_has_punctuation', 'base_has_punctuation']:
            features[f'name_{k}'] = 0.0
    
    return features


def extract_phone_primary(phone_field):
    """Extract primary phone from phone field."""
    parsed = parse_json_field(phone_field)
    if parsed and isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[0]) if parsed[0] else ''
    return ''

def extract_phone_features(current_phones: Any, base_phones: Any) -> Dict[str, float]:
    """Extract features for phone attribute comparison."""
    current = extract_phone_primary(current_phones)
    base = extract_phone_primary(base_phones)
    
    features = {}
    
    # Presence features
    features['phone_current_exists'] = 1.0 if current and current != 'nan' else 0.0
    features['phone_base_exists'] = 1.0 if base and base != 'nan' else 0.0
    
    if features['phone_current_exists'] and features['phone_base_exists']:
        # Length features
        features['phone_current_len'] = len(current)
        features['phone_base_len'] = len(base)
        features['phone_len_diff'] = abs(len(current) - len(base))
        
        # Format features
        features['phone_current_has_plus'] = 1.0 if '+' in current else 0.0
        features['phone_base_has_plus'] = 1.0 if '+' in base else 0.0
        features['phone_current_has_paren'] = 1.0 if '(' in current else 0.0
        features['phone_base_has_paren'] = 1.0 if '(' in base else 0.0
        
        # Digit only comparison
        curr_digits = ''.join(filter(str.isdigit, current))
        base_digits = ''.join(filter(str.isdigit, base))
        
        features['phone_digits_match'] = 1.0 if curr_digits == base_digits else 0.0
        features['phone_digits_len_diff'] = abs(len(curr_digits) - len(base_digits))
        
        # String similarity
        sim_features = calculate_string_similarity(current, base)
        features.update({f'phone_{k}': v for k, v in sim_features.items()})
        
    else:
        # Fill zeros
        for k in ['current_len', 'base_len', 'len_diff', 'current_has_plus', 'base_has_plus', 
                  'current_has_paren', 'base_has_paren', 'digits_match', 'digits_len_diff']:
            features[f'phone_{k}'] = 0.0
        for k in ['exact_match', 'exact_match_lower', 'length_ratio', 'levenshtein_similarity', 'jaro_winkler_similarity']:
            features[f'phone_{k}'] = 0.0
            
    return features

def extract_website_primary(website_field):
    """Extract primary website from website field."""
    parsed = parse_json_field(website_field)
    if parsed and isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[0]) if parsed[0] else ''
    return ''

def extract_website_features(current_websites: Any, base_websites: Any) -> Dict[str, float]:
    """Extract features for website attribute comparison."""
    current = extract_website_primary(current_websites)
    base = extract_website_primary(base_websites)
    
    features = {}
    
    # Presence
    features['web_current_exists'] = 1.0 if current and current != 'nan' else 0.0
    features['web_base_exists'] = 1.0 if base and base != 'nan' else 0.0
    
    if features['web_current_exists'] and features['web_base_exists']:
        # Protocol
        features['web_current_https'] = 1.0 if current.startswith('https') else 0.0
        features['web_base_https'] = 1.0 if base.startswith('https') else 0.0
        
        # Subdomain
        features['web_current_www'] = 1.0 if 'www.' in current else 0.0
        features['web_base_www'] = 1.0 if 'www.' in base else 0.0
        
        # Length
        features['web_len_diff'] = abs(len(current) - len(base))
        features['web_len_ratio'] = min(len(current), len(base)) / max(len(current), len(base), 1)
        
        # Similarity
        sim_features = calculate_string_similarity(current, base)
        features.update({f'web_{k}': v for k, v in sim_features.items()})
    else:
        for k in ['current_https', 'base_https', 'current_www', 'base_www', 'len_diff', 'len_ratio']:
            features[f'web_{k}'] = 0.0
        for k in ['exact_match', 'exact_match_lower', 'length_ratio', 'levenshtein_similarity', 'jaro_winkler_similarity']:
            features[f'web_{k}'] = 0.0
            
    return features

def extract_address_primary(address_field):
    """Extract primary address object."""
    parsed = parse_json_field(address_field)
    if parsed and isinstance(parsed, list) and len(parsed) > 0:
        return parsed[0] if parsed[0] else {}
    return {}

def extract_address_features(current_addr: Any, base_addr: Any) -> Dict[str, float]:
    """Extract features for address attribute comparison."""
    curr_obj = extract_address_primary(current_addr)
    base_obj = extract_address_primary(base_addr)
    
    features = {}
    
    # Component checks
    components = ['freeform', 'locality', 'region', 'postcode', 'country']
    
    curr_present_count = sum(1 for c in components if curr_obj.get(c))
    base_present_count = sum(1 for c in components if base_obj.get(c))
    
    features['addr_current_components'] = float(curr_present_count)
    features['addr_base_components'] = float(base_present_count)
    features['addr_components_diff'] = features['addr_current_components'] - features['addr_base_components']
    
    # Specific field presence
    features['addr_current_has_postcode'] = 1.0 if curr_obj.get('postcode') else 0.0
    features['addr_base_has_postcode'] = 1.0 if base_obj.get('postcode') else 0.0
    
    features['addr_current_has_region'] = 1.0 if curr_obj.get('region') else 0.0
    features['addr_base_has_region'] = 1.0 if base_obj.get('region') else 0.0
    
    # Freeform text similarity
    curr_free = str(curr_obj.get('freeform', '')).strip()
    base_free = str(base_obj.get('freeform', '')).strip()
    
    if curr_free and base_free:
        sim_features = calculate_string_similarity(curr_free, base_free)
        features.update({f'addr_{k}': v for k, v in sim_features.items()})
    else:
        for k in ['exact_match', 'exact_match_lower', 'length_ratio', 'levenshtein_similarity', 'jaro_winkler_similarity']:
            features[f'addr_{k}'] = 0.0
            
    return features

def extract_category_primary(category_field):
    """Extract primary category string."""
    parsed = parse_json_field(category_field)
    if parsed and isinstance(parsed, dict):
        val = parsed.get('primary', '')
        return str(val).strip() if val is not None else ''
    return ''

def extract_category_features(current_cat: Any, base_cat: Any) -> Dict[str, float]:
    """Extract features for category attribute comparison."""
    curr_prim = extract_category_primary(current_cat)
    base_prim = extract_category_primary(base_cat)
    
    features = {}
    
    # Presence
    features['cat_current_exists'] = 1.0 if curr_prim else 0.0
    features['cat_base_exists'] = 1.0 if base_prim else 0.0
    
    # Similarity
    if curr_prim and base_prim:
        sim_features = calculate_string_similarity(curr_prim, base_prim)
        features.update({f'cat_{k}': v for k, v in sim_features.items()})
        
        # Specificity proxy (length or ' > ' counts for hierarchy)
        features['cat_current_depth'] = float(curr_prim.count('>'))
        features['cat_base_depth'] = float(base_prim.count('>'))
        features['cat_depth_diff'] = features['cat_current_depth'] - features['cat_base_depth']
    else:
        features['cat_current_depth'] = 0.0
        features['cat_base_depth'] = 0.0
        features['cat_depth_diff'] = 0.0
        for k in ['exact_match', 'exact_match_lower', 'length_ratio', 'levenshtein_similarity', 'jaro_winkler_similarity']:
            features[f'cat_{k}'] = 0.0
            
    return features


def extract_metadata_features(row: pd.Series) -> Dict[str, float]:
    """Extract metadata features (confidence, sources, etc.)."""
    features = {}
    
    # Confidence scores
    features['confidence_current'] = float(row.get('confidence', 0.0))
    features['confidence_base'] = float(row.get('base_confidence', 0.0))
    features['confidence_diff'] = features['confidence_current'] - features['confidence_base']
    features['confidence_ratio'] = features['confidence_current'] / max(features['confidence_base'], 0.001)
    
    # Source information (if available)
    current_sources = row.get('sources', '')
    base_sources = row.get('base_sources', '')
    features['sources_current_count'] = len(str(current_sources).split(',')) if current_sources else 0.0
    features['sources_base_count'] = len(str(base_sources).split(',')) if base_sources else 0.0
    features['sources_count_diff'] = features['sources_current_count'] - features['sources_base_count']
    
    return features


def extract_features_for_record(row: pd.Series, attribute: str = 'name') -> Dict[str, float]:
    """
    Extract all features for a single record and attribute.
    
    Args:
        row: DataFrame row
        attribute: Attribute name ('name', 'phone', 'website', etc.)
    
    Returns:
        Dictionary of feature names to values
    """
    features = {}
    
    # Metadata features (same for all attributes)
    metadata_features = extract_metadata_features(row)
    features.update(metadata_features)
    
    # Attribute-specific features
    if attribute == 'name':
        name_features = extract_name_features(row['names'], row['base_names'])
        features.update(name_features)
    elif attribute == 'phone':
        phone_features = extract_phone_features(row['phones'], row['base_phones'])
        features.update(phone_features)
    elif attribute == 'website':
        website_features = extract_website_features(row['websites'], row['base_websites'])
        features.update(website_features)
    elif attribute == 'address':
        address_features = extract_address_features(row['addresses'], row['base_addresses'])
        features.update(address_features)
    elif attribute == 'category':
        category_features = extract_category_features(row['categories'], row['base_categories'])
        features.update(category_features)
    
    return features


def extract_features_batch(df: pd.DataFrame, attribute: str = 'name') -> pd.DataFrame:
    """
    Extract features for a batch of records.
    
    Returns:
        DataFrame with one row per record, columns are features
    """
    feature_rows = []
    
    for idx, row in df.iterrows():
        features = extract_features_for_record(row, attribute)
        features['record_index'] = idx
        features['id'] = row['id']
        feature_rows.append(features)
    
    features_df = pd.DataFrame(feature_rows)
    
    # Fill NaN values with 0
    features_df = features_df.fillna(0.0)
    
    return features_df


def load_golden_labels(golden_file: str, attribute: str = 'name') -> pd.Series:
    """
    Load golden labels and return as Series indexed by record ID.
    
    Returns:
        Series with record IDs as index and labels as values
    """
    with open(golden_file, 'r') as f:
        golden_data = json.load(f)
    
    labels = {}
    for record in golden_data:
        record_id = record['id']
        label = record.get('labels', {}).get(attribute, 'unclear')
        labels[record_id] = label
    
    return pd.Series(labels)


def prepare_training_data(
    data_file: str = 'data/project_b_samples_2k.parquet',
    golden_file: str = 'data/processed/golden_dataset_sample.json',
    attribute: str = 'name',
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare training data with features and labels.
    
    Returns:
        DataFrame with features and labels
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} records")
    
    print(f"Extracting features for '{attribute}' attribute...")
    features_df = extract_features_batch(df, attribute)
    
    print(f"Loading golden labels from {golden_file}...")
    labels_series = load_golden_labels(golden_file, attribute)
    
    # Merge features with labels
    features_df['label'] = features_df['id'].map(labels_series).fillna('unclear')
    
    # Filter out unclear labels for training
    trainable_df = features_df[features_df['label'] != 'unclear'].copy()
    
    print(f"Prepared {len(trainable_df)} records with labels (excluding unclear)")
    print(f"Label distribution: {trainable_df['label'].value_counts().to_dict()}")
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trainable_df.to_parquet(output_file, index=False)
        print(f"Saved training data to {output_file}")
    
    return trainable_df


def main():
    """Main function for feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features for ML training')
    parser.add_argument('--data', default='data/project_b_samples_2k.parquet',
                       help='Input data file')
    parser.add_argument('--golden', default='data/processed/golden_dataset_sample.json',
                       help='Golden labels file')
    parser.add_argument('--attribute', default='name',
                       choices=['name', 'phone', 'website', 'address', 'category'],
                       help='Attribute to extract features for')
    parser.add_argument('--output', default=None,
                       help='Output file for training data (optional)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'data/processed/features_{args.attribute}.parquet'
    
    prepare_training_data(
        data_file=args.data,
        golden_file=args.golden,
        attribute=args.attribute,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

