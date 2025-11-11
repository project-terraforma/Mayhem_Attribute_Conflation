"""
AI-Powered Annotation Script for Attribute Selection

This script uses LM Studio's local LLM API to automatically annotate records
by comparing current vs base versions of attributes according to the guidelines.
"""

import pandas as pd
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm


class AIAnnotator:
    """AI-powered annotation interface using LM Studio's local LLM API."""
    
    def __init__(self, data_file: str = 'data/agreement_sample_200.csv',
                 api_base_url: str = 'http://localhost:1234/v1',
                 model: Optional[str] = None,
                 annotator_name: str = 'ai_agent'):
        """
        Initialize AI annotator.
        
        Args:
            data_file: Path to CSV data file
            api_base_url: LM Studio API base URL
            model: Model name (if None, will auto-detect from LM Studio)
            annotator_name: Name for annotator in output
        """
        # Load data
        if data_file.endswith('.csv'):
            self.df = pd.read_csv(data_file)
        else:
            self.df = pd.read_parquet(data_file)
        
        self.annotations = []
        self.annotator_name = annotator_name
        self.api_base_url = api_base_url
        self.model = model
        
        # Load attribute guidelines
        guidelines_path = Path('docs/attribute_guidelines.md')
        if guidelines_path.exists():
            with open(guidelines_path, 'r', encoding='utf-8') as f:
                self.guidelines = f.read()
        else:
            self.guidelines = "Follow standard data quality principles."
        
        # Initialize OpenAI client (compatible with LM Studio)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=api_base_url,
                api_key="lm-studio"  # LM Studio doesn't require real API key
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        # Test connection and get model name
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to LM Studio and get available model."""
        try:
            models = self.client.models.list()
            if models.data:
                available_model = models.data[0].id
                if self.model is None:
                    self.model = available_model
                    print(f"✓ Connected to LM Studio. Using model: {self.model}")
                else:
                    print(f"✓ Connected to LM Studio. Model: {self.model}")
            else:
                print("⚠ Warning: No models found in LM Studio. Make sure a model is loaded.")
                if self.model is None:
                    self.model = "unknown"
        except Exception as e:
            print(f"⚠ Warning: Could not connect to LM Studio: {e}")
            print("Make sure LM Studio is running with a model loaded on port 1234")
            if self.model is None:
                self.model = "unknown"
    
    def format_record_for_llm(self, index: int) -> str:
        """Format a record for LLM comparison."""
        row = self.df.iloc[index]
        
        # Helper to format values
        def format_value(val):
            if pd.isna(val):
                return "[MISSING]"
            if isinstance(val, str):
                # Try to parse JSON strings for better formatting
                try:
                    parsed = json.loads(val)
                    return json.dumps(parsed, indent=2, ensure_ascii=False)
                except:
                    return val
            return str(val)
        
        record_text = f"""Record ID: {row['id']}
Base ID: {row['base_id']}
Confidence: Current={row['confidence']:.3f}, Base={row['base_confidence']:.3f}

ATTRIBUTES TO COMPARE:

1. NAME:
   Current: {format_value(row['names'])}
   Base:    {format_value(row['base_names'])}

2. PHONE:
   Current: {format_value(row['phones'])}
   Base:    {format_value(row['base_phones'])}

3. WEBSITE:
   Current: {format_value(row['websites'])}
   Base:    {format_value(row['base_websites'])}

4. ADDRESS:
   Current: {format_value(row['addresses'])}
   Base:    {format_value(row['base_addresses'])}

5. CATEGORY:
   Current: {format_value(row['categories'])}
   Base:    {format_value(row['base_categories'])}
"""
        return record_text
    
    def create_annotation_prompt(self, record_text: str) -> str:
        """Create prompt for LLM annotation."""
        prompt = f"""You are an expert data quality annotator. Your task is to compare two versions of place attributes (CURRENT vs BASE) and determine which is better according to the guidelines.

{self.guidelines}

For the following record, analyze ALL attributes (name, phone, website, address, category) and determine the overall best version. Consider:
- Which version has more complete and accurate data?
- Which version follows better formatting standards?
- Which version is more reliable overall?

{record_text}

Based on the attribute guidelines, make a decision:
- If CURRENT version is better overall → respond with "c"
- If BASE version is better overall → respond with "b"  
- If they are SAME/equivalent → respond with "s"
- If UNCLEAR/needs review → respond with "u"

Respond in JSON format:
{{
  "choice": "c|b|s|u",
  "notes": "Brief explanation of your decision, focusing on which attributes influenced the choice"
}}

Only respond with the JSON object, nothing else."""
        return prompt
    
    def annotate_record(self, index: int, max_retries: int = 3) -> Dict[str, Any]:
        """Annotate a single record using AI."""
        record_text = self.format_record_for_llm(index)
        prompt = self.create_annotation_prompt(record_text)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise data quality annotator. Always respond with valid JSON only, no additional text."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,  # Low temperature for consistency
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                
                # Parse JSON response
                result = json.loads(result_text)
                choice = result.get('choice', 'u').lower().strip()
                notes = result.get('notes', '')
                
                # Validate choice
                if choice not in ['c', 'b', 's', 'u']:
                    choice = 'u'
                    notes = f"Invalid choice returned: {result.get('choice')}. {notes}"
                
                annotation = {
                    'record_index': index,
                    'id': str(self.df.iloc[index]['id']),
                    'choice': choice,
                    'notes': notes,
                    'annotator': self.annotator_name,
                    'model': self.model
                }
                
                return annotation
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                # Fallback on JSON error
                annotation = {
                    'record_index': index,
                    'id': str(self.df.iloc[index]['id']),
                    'choice': 'u',
                    'notes': f'JSON parsing error: {str(e)}',
                    'annotator': self.annotator_name,
                    'model': self.model
                }
                return annotation
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                # Fallback on error
                annotation = {
                    'record_index': index,
                    'id': str(self.df.iloc[index]['id']),
                    'choice': 'u',
                    'notes': f'Error: {str(e)}',
                    'annotator': self.annotator_name,
                    'model': self.model
                }
                return annotation
        
        # Final fallback
        return {
            'record_index': index,
            'id': str(self.df.iloc[index]['id']),
            'choice': 'u',
            'notes': 'Max retries exceeded',
            'annotator': self.annotator_name,
            'model': self.model
        }
    
    def load_existing_annotations(self, filename: str) -> set:
        """Load existing annotations to avoid re-processing."""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            annotated_indices = {a['record_index'] for a in existing if 'record_index' in a}
            self.annotations = existing
            print(f"Loaded {len(existing)} existing annotations")
            return annotated_indices
        return set()
    
    def annotate_all(self, start_index: int = 0, end_index: Optional[int] = None,
                     output_file: Optional[str] = None,
                     delay: float = 0.1, save_interval: int = 25):
        """
        Annotate all records (or a range).
        
        Args:
            start_index: Starting record index
            end_index: Ending record index (None = all)
            output_file: Output file path
            delay: Delay between API calls (seconds)
            save_interval: Save every N records
        """
        if output_file is None:
            output_file = f'data/annotations_{self.annotator_name}.json'
        
        # Load existing annotations
        annotated_indices = self.load_existing_annotations(output_file)
        
        end_index = end_index or len(self.df)
        total = end_index - start_index
        
        print(f"\nStarting AI annotation: {total} records")
        print(f"Model: {self.model}")
        print(f"Annotator: {self.annotator_name}")
        print(f"Output: {output_file}\n")
        
        processed = 0
        skipped = 0
        
        for i in tqdm(range(start_index, end_index), desc="Annotating"):
            # Skip if already annotated
            if i in annotated_indices:
                skipped += 1
                continue
            
            annotation = self.annotate_record(i)
            self.annotations.append(annotation)
            processed += 1
            
            # Rate limiting
            if delay > 0:
                time.sleep(delay)
            
            # Save periodically
            if len(self.annotations) % save_interval == 0:
                self.save_annotations(output_file)
        
        # Final save
        self.save_annotations(output_file)
        print(f"\n✓ Completed!")
        print(f"  Processed: {processed} new records")
        print(f"  Skipped: {skipped} already annotated")
        print(f"  Total annotations: {len(self.annotations)}")
    
    def save_annotations(self, filename: str):
        """Save annotations to file."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)


def main():
    """Main annotation interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-powered attribute annotation using LM Studio')
    parser.add_argument('--input', type=str, default='data/agreement_sample_200.csv',
                       help='Input CSV file with records to annotate')
    parser.add_argument('--output', type=str, default='data/annotations_ai_agent.json',
                       help='Output JSON file for annotations')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting record index')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending record index (None = all)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (auto-detected if not specified)')
    parser.add_argument('--api-url', type=str, default='http://localhost:1234/v1',
                       help='LM Studio API base URL')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between API calls (seconds)')
    parser.add_argument('--save-interval', type=int, default=25,
                       help='Save every N records')
    parser.add_argument('--annotator', type=str, default='ai_agent',
                       help='Annotator name')
    parser.add_argument('--test', action='store_true',
                       help='Test with first 5 records only')
    
    args = parser.parse_args()
    
    annotator = AIAnnotator(
        data_file=args.input,
        api_base_url=args.api_url,
        model=args.model,
        annotator_name=args.annotator
    )
    
    # Test mode: only process 5 records
    if args.test:
        print("TEST MODE: Processing first 5 records only")
        annotator.annotate_all(
            start_index=0,
            end_index=5,
            output_file=args.output,
            delay=args.delay,
            save_interval=5
        )
    else:
        annotator.annotate_all(
            start_index=args.start,
            end_index=args.end,
            output_file=args.output,
            delay=args.delay,
            save_interval=args.save_interval
        )


if __name__ == "__main__":
    main()

