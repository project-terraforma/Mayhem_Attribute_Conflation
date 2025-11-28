
import json
import random
import pandas as pd
import re

YELP_PATH = 'yelp/yelp_academic_dataset_business.json'
OUTPUT_PATH = 'data/synthetic_golden_dataset_2k.json'

def perturb_name(name):
    """Simulate name data quality issues."""
    choice = random.random()
    if choice < 0.3:
        return name.lower() # Lowercase
    elif choice < 0.5:
        return name.upper() # Uppercase
    elif choice < 0.7:
        return name.replace("'", "") # Remove punctuation
    else:
        return name # No change

def perturb_phone(phone):
    """Simulate phone formatting issues."""
    if not phone: return None
    # Yelp usually has clean data. Let's make it messy.
    digits = re.sub(r'\D', '', phone)
    choice = random.random()
    if choice < 0.4:
        return digits # Plain digits
    elif choice < 0.7:
        return f"({digits[-10:-7]}) {digits[-7:-4]}-{digits[-4:]}" # US Format
    else:
        return phone

def generate_fake_phone():
    return f"+1{random.randint(200,999)}{random.randint(200,999)}{random.randint(1000,9999)}"

def generate_fake_website(name, business_id):
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
    return f"https://www.{clean_name}.com"

def create_synthetic_dataset(limit=2000):
    print(f"Generating {limit} synthetic records from Yelp...")
    
    records = []
    with open(YELP_PATH, 'r', encoding='utf-8') as f:
        yelp_data = [json.loads(line) for line in f]
    
    # Sample random records
    sampled_yelp = random.sample(yelp_data, min(limit, len(yelp_data)))
    
    for i, y in enumerate(sampled_yelp):
        # Generate fake data if missing (Yelp public dataset often lacks phone/web)
        phone_val = y.get('phone')
        if not phone_val:
            phone_val = generate_fake_phone()
            
        website_val = y.get('website')
        if not website_val:
            website_val = generate_fake_website(y['name'], y['business_id'])

        # Create "Current" (The Good Version)
        # We assume Yelp data is generally "Ground Truth" quality
        current = {
            "names": json.dumps({"primary": y['name']}),
            "phones": json.dumps([phone_val]),
            "websites": json.dumps([website_val]),
            "addresses": json.dumps([{
                "freeform": y['address'],
                "locality": y['city'],
                "region": y['state'],
                "postcode": y['postal_code'],
                "country": "US"
            }]),
            "categories": json.dumps({"primary": y['categories']})
        }
        
        # Create "Base" (The Stale/Noisy Version)
        base = {
            "names": json.dumps({"primary": perturb_name(y['name'])}),
            "phones": json.dumps([perturb_phone(phone_val)]),
            "websites": json.dumps([website_val.replace('https://', 'http://')]) if random.random() < 0.5 else "[]", # Degrade website
            "addresses": json.dumps([{
                "freeform": y['address'],
                "locality": y['city'],
                "region": y['state'],
                # Simulate missing postcode in base
                "country": "US"
            }]),
            "categories": "{}" # Simulate missing category
        }
        
        # Randomly swap to balance labels
        if random.random() < 0.5:
            # Case: Base is better (The Good Version)
            final_current = base
            final_base = current
            label = "b"
        else:
            # Case: Current is better (The Good Version)
            final_current = current
            final_base = base
            label = "c"

        record = {
            "id": f"synthetic_yelp_{y['business_id']}",
            "data": {
                "current": final_current,
                "base": final_base
            },
            "label": label,
            "method": "synthetic_yelp_proxy"
        }
        records.append(record)
        
    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(records, f, indent=2)
        
    print(f"✅ Generated {len(records)} synthetic records to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_synthetic_dataset()
