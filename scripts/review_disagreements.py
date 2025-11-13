"""
Interactive Disagreement Review Tool

This script provides an interactive CLI for reviewing disagreements between
Qwen and Gemma annotations, allowing manual establishment of ground truth.
"""

import pandas as pd
import json
import os
from pathlib import Path


class DisagreementReviewer:
    """Interactive tool for reviewing model disagreements."""
    
    def __init__(self, disagreements_file='disagreements_qwen_vs_gemma.json',
                 data_file='data/agreement_sample_200.csv',
                 review_file='data/manual_review_decisions.json'):
        """Initialize the reviewer."""
        # Load disagreements
        with open(disagreements_file, 'r', encoding='utf-8') as f:
            self.disagreements = json.load(f)
        
        # Load original data
        if data_file.endswith('.csv'):
            self.df = pd.read_csv(data_file)
        else:
            self.df = pd.read_parquet(data_file)
        
        self.review_file = review_file
        self.decisions = self.load_existing_decisions()
        
        print(f"Loaded {len(self.disagreements)} disagreements")
        print(f"Already reviewed: {len(self.decisions)}")
        print(f"Remaining: {len(self.disagreements) - len(self.decisions)}")
    
    def load_existing_decisions(self):
        """Load existing review decisions if available."""
        if os.path.exists(self.review_file):
            with open(self.review_file, 'r', encoding='utf-8') as f:
                decisions = json.load(f)
            return {d['record_index']: d for d in decisions}
        return {}
    
    def save_decisions(self):
        """Save current decisions to file."""
        decisions_list = list(self.decisions.values())
        with open(self.review_file, 'w', encoding='utf-8') as f:
            json.dump(decisions_list, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Progress saved ({len(decisions_list)} reviews)")
    
    def format_value(self, val):
        """Format a value for display."""
        if pd.isna(val):
            return "[MISSING]"
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except:
                return val
        return str(val)
    
    def display_record(self, record_index):
        """Display a single record with disagreement info."""
        # Get disagreement info
        disagreement = next((d for d in self.disagreements if d['record_index'] == record_index), None)
        if not disagreement:
            return False
        
        # Get original data
        row = self.df.iloc[record_index]
        
        # Clear screen (works on both Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 100)
        print(f"DISAGREEMENT REVIEW - Record {record_index + 1}/{len(self.disagreements)}")
        print(f"Progress: {len(self.decisions)}/{len(self.disagreements)} reviewed")
        print("=" * 100)
        
        print(f"\nRecord ID: {row['id']}")
        print(f"Base ID: {row['base_id']}")
        print(f"Confidence: Current={row['confidence']:.3f}, Base={row['base_confidence']:.3f}\n")
        
        print("-" * 100)
        print("ATTRIBUTES COMPARISON:")
        print("-" * 100)
        
        # Display each attribute
        attributes = [
            ('NAME', 'names', 'base_names'),
            ('PHONE', 'phones', 'base_phones'),
            ('WEBSITE', 'websites', 'base_websites'),
            ('ADDRESS', 'addresses', 'base_addresses'),
            ('CATEGORY', 'categories', 'base_categories')
        ]
        
        for attr_name, curr_col, base_col in attributes:
            print(f"\n{attr_name}:")
            print(f"  Current: {self.format_value(row[curr_col])}")
            print(f"  Base:    {self.format_value(row[base_col])}")
        
        print("\n" + "=" * 100)
        print("MODEL DECISIONS:")
        print("=" * 100)
        
        choice_labels = {
            'c': 'CURRENT',
            'b': 'BASE',
            's': 'SAME',
            'u': 'UNCLEAR'
        }
        
        print(f"\n[QWEN 30B] Chose: {choice_labels.get(disagreement['qwen_choice'], disagreement['qwen_choice']).upper()}")
        print(f"Reasoning: {disagreement['qwen_notes']}")
        
        print(f"\n[GEMMA 2 9B] Chose: {choice_labels.get(disagreement['gemma_choice'], disagreement['gemma_choice']).upper()}")
        print(f"Reasoning: {disagreement['gemma_notes']}")
        
        print("\n" + "=" * 100)
        
        return disagreement
    
    def get_user_decision(self, disagreement):
        """Get user's decision on the disagreement."""
        print("\nYour Decision:")
        print("  [q] Qwen is correct")
        print("  [g] Gemma is correct")
        print("  [c] Choose CURRENT")
        print("  [b] Choose BASE")
        print("  [s] Mark as SAME")
        print("  [u] Mark as UNCLEAR")
        print("  [n] Skip (review later)")
        print("  [x] Save and exit")
        
        while True:
            choice = input("\nEnter your choice: ").lower().strip()
            
            if choice == 'x':
                return 'exit'
            elif choice == 'n':
                return 'skip'
            elif choice == 'q':
                return disagreement['qwen_choice']
            elif choice == 'g':
                return disagreement['gemma_choice']
            elif choice in ['c', 'b', 's', 'u']:
                return choice
            else:
                print("Invalid choice. Please try again.")
    
    def review_all(self, start_from=0):
        """Interactive review of all disagreements."""
        print("\n" + "=" * 100)
        print("DISAGREEMENT REVIEW SESSION")
        print("=" * 100)
        print("\nInstructions:")
        print("- Review each record and decide which version is better")
        print("- You can agree with Qwen (q), Gemma (g), or make your own choice")
        print("- Progress is saved automatically")
        print("- Press 'x' to exit and save at any time")
        print("\nLet's begin!\n")
        input("Press Enter to start...")
        
        unreviewed = [d for d in self.disagreements if d['record_index'] not in self.decisions]
        
        if not unreviewed:
            print("\n[COMPLETE] All disagreements have been reviewed!")
            return
        
        for i, disagreement in enumerate(unreviewed, 1):
            record_index = disagreement['record_index']
            
            # Display record
            self.display_record(record_index)
            
            # Get decision
            decision = self.get_user_decision(disagreement)
            
            if decision == 'exit':
                print("\n[EXIT] Saving progress and exiting...")
                self.save_decisions()
                break
            elif decision == 'skip':
                print("\n[SKIPPED] Moving to next record...")
                continue
            
            # Record decision
            notes = input("\nOptional notes (press Enter to skip): ").strip()
            
            self.decisions[record_index] = {
                'record_index': record_index,
                'id': disagreement['id'],
                'final_choice': decision,
                'qwen_choice': disagreement['qwen_choice'],
                'gemma_choice': disagreement['gemma_choice'],
                'notes': notes if notes else '',
                'reviewer': 'manual'
            }
            
            # Save every 5 decisions
            if len(self.decisions) % 5 == 0:
                self.save_decisions()
        
        # Final save
        self.save_decisions()
        
        print("\n" + "=" * 100)
        print("REVIEW SESSION COMPLETE")
        print("=" * 100)
        print(f"Total reviewed: {len(self.decisions)}/{len(self.disagreements)}")
        print(f"Remaining: {len(self.disagreements) - len(self.decisions)}")
    
    def generate_summary(self):
        """Generate summary statistics of review decisions."""
        if not self.decisions:
            print("No decisions to summarize yet.")
            return
        
        from collections import Counter
        
        final_choices = [d['final_choice'] for d in self.decisions.values()]
        choice_counts = Counter(final_choices)
        
        # Agreement analysis
        agreed_with_qwen = sum(1 for d in self.decisions.values() if d['final_choice'] == d['qwen_choice'])
        agreed_with_gemma = sum(1 for d in self.decisions.values() if d['final_choice'] == d['gemma_choice'])
        
        print("\n" + "=" * 100)
        print("REVIEW SUMMARY")
        print("=" * 100)
        print(f"\nTotal decisions made: {len(self.decisions)}/{len(self.disagreements)}")
        
        print("\nFinal choice distribution:")
        for choice, count in sorted(choice_counts.items()):
            print(f"  {choice}: {count} ({count/len(self.decisions)*100:.1f}%)")
        
        print(f"\nAgreement with models:")
        print(f"  Agreed with Qwen: {agreed_with_qwen} ({agreed_with_qwen/len(self.decisions)*100:.1f}%)")
        print(f"  Agreed with Gemma: {agreed_with_gemma} ({agreed_with_gemma/len(self.decisions)*100:.1f}%)")
        
        print("\n" + "=" * 100)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive disagreement review tool')
    parser.add_argument('--disagreements', type=str, default='disagreements_qwen_vs_gemma.json',
                       help='Disagreements file')
    parser.add_argument('--data', type=str, default='data/agreement_sample_200.csv',
                       help='Original data file')
    parser.add_argument('--output', type=str, default='data/manual_review_decisions.json',
                       help='Output file for review decisions')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of existing reviews and exit')
    
    args = parser.parse_args()
    
    reviewer = DisagreementReviewer(
        disagreements_file=args.disagreements,
        data_file=args.data,
        review_file=args.output
    )
    
    if args.summary:
        reviewer.generate_summary()
    else:
        reviewer.review_all()
        reviewer.generate_summary()


if __name__ == "__main__":
    main()

