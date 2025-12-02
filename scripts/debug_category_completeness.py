import json
import pandas as pd
from pathlib import Path
from scripts.baseline_heuristics import CompletenessBaseline, parse_json_field

def debug_category_completeness_all_records():
    golden_path = 'data/golden_dataset_200.json'
    
    print(f"Debugging completeness score for Category attribute for all 200 records...")
    
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    
    completeness_baseline = CompletenessBaseline()
    
    score_differences = []
    identical_scores_count = 0
    
    for record_index, record in enumerate(golden_data):
        current_categories_raw = record['data']['current'].get('categories')
        base_categories_raw = record['data']['base'].get('categories')
        
        current_score = completeness_baseline._calculate_completeness(current_categories_raw, attribute='category')
        base_score = completeness_baseline._calculate_completeness(base_categories_raw, attribute='category')
        
        score_differences.append(abs(current_score - base_score))
        
        if current_score == base_score:
            identical_scores_count += 1
            # print(f"Record {record_index}: Scores identical ({current_score:.2f}) -> Current: {current_categories_raw}, Base: {base_categories_raw}")
        # else:
            # print(f"Record {record_index}: Scores different (C={current_score:.2f}, B={base_score:.2f}) -> Current: {current_categories_raw}, Base: {base_categories_raw}")

    print(f"\n--- Summary for all 200 Records (Category) ---")
    print(f"Number of records with identical completeness scores: {identical_scores_count} / {len(golden_data)}")
    
    if identical_scores_count == len(golden_data):
        print("All records have identical completeness scores. This explains 'unclear' predictions.")
    else:
        print("Not all records have identical completeness scores.")
        print(f"Min score difference: {min(score_differences):.2f}")
        print(f"Max score difference: {max(score_differences):.2f}")
        print(f"Average score difference: {sum(score_differences) / len(score_differences):.2f}")
        
    # Final check: let's run a prediction on one record where scores are different to see what it predicts
    print("\n--- Test prediction for a record with different scores ---")
    
    found_diff_score_record = False
    for record_index, record in enumerate(golden_data):
        current_categories_raw = record['data']['current'].get('categories')
        base_categories_raw = record['data']['base'].get('categories')
        
        current_score = completeness_baseline._calculate_completeness(current_categories_raw, attribute='category')
        base_score = completeness_baseline._calculate_completeness(base_categories_raw, attribute='category')
        
        if current_score != base_score:
            print(f"Record {record_index} (ID: {record['id']}): C={current_score:.2f}, B={base_score:.2f}")
            
            # Construct a pseudo-DataFrame row for testing the predict method
            pseudo_row_data = {}
            for key, val in record['data']['current'].items():
                pseudo_row_data[key] = val
            for key, val in record['data']['base'].items():
                pseudo_row_data[f'base_{key}'] = val
            
            # Also add direct confidence values if they are not nested in 'data'
            pseudo_row_data['confidence'] = record['data']['current'].get('confidence', 0.0)
            pseudo_row_data['base_confidence'] = record['data']['base'].get('confidence', 0.0)

            # Convert to pd.Series, which is what the predict method expects
            pseudo_row = pd.Series(pseudo_row_data)

            predicted_choice = completeness_baseline.predict(pseudo_row, attribute='category')
            print(f"Predicted by CompletenessBaseline: {predicted_choice}")
            found_diff_score_record = True
            break
            
    if not found_diff_score_record:
        print("No record found with different completeness scores to test prediction.")

if __name__ == "__main__":
    debug_category_completeness_all_records()