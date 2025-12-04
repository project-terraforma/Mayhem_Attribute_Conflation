# Dataset Samples for Slide 5 - Screenshot Ready

---

## 1. Manual Golden Dataset (200 Records)

**File**: `data/golden_dataset_200.json`  
**Source**: Overture pre-matched places, fully human-validated  
**Usage**: Real-world validation "diamond standard"

### Sample Record #1

```json
{
  "id": "08f3da18ccad52ad03cc06b87820910f",
  "record_index": 1,
  "label": "c",
  "method": "manual_review (manual)",
  "data": {
    "current": {
      "names": "{\"primary\":\"davaindia GENERIC PHARMACY\"}",
      "phones": "[\"09818559767\"]",
      "websites": "[\"http://sector15rohini.davaindia.com/\"]",
      "addresses": "[{\"freeform\":\"Shop No F21/66, Rohini, Sector 15\",\"locality\":\"New Delhi\",\"postcode\":\"110089\",\"region\":\"Delhi\",\"country\":\"IN\"}]",
      "categories": "{\"primary\":\"pharmacy\",\"alternate\":[\"retail\"]}",
      "confidence": 0.77
    },
    "base": {
      "names": "{\"primary\":\"Davaindia Generic Pharmacy\"}",
      "phones": "[\"09818559767\"]",
      "websites": "[\"https://pharmacy-nearme.davaindia.com/davaindia-generic-pharmacy-pharmacy-rohini-sector-15-new-delhi-173545/Home?utm_source=locator&utm_medium=bing\"]",
      "addresses": "[{\"freeform\":\"Shop No F21/66, Rohini, Sector 15\",\"locality\":\"New Delhi\",\"region\":\"Delhi\",\"country\":\"IN\",\"postcode\":\"110089\"}]",
      "categories": "{\"primary\":\"pharmacy\",\"alternate\":[\"retail\"]}",
      "confidence": 0.77
    }
  }
}
```

**Key Fields**:
- `label: "c"` = **Current** version is better (human decision)
- `method: "manual_review (manual)"` = Fully human-validated
- Contains all 5 attributes: names, phones, websites, addresses, categories

### Sample Record #2

```json
{
  "id": "08f64a59990a63a0030784450be43ba4",
  "record_index": 0,
  "label": "b",
  "method": "manual_review (manual)",
  "data": {
    "current": {
      "names": "{\"primary\":\"ร้านตัดผมพี่ริน ห้วยตะแคง\"}",
      "phones": NaN,
      "websites": NaN,
      "addresses": "[{\"country\":\"TH\"}]",
      "categories": "{\"primary\":\"beauty_salon\",\"alternate\":[\"barber\",\"thai_restaurant\"]}",
      "confidence": 0.2315484804630969
    },
    "base": {
      "names": "{\"primary\":\"ร้านตัดผมพี่ริน ห้วยตะแคง\",\"common\":{},\"rules\":[]}",
      "phones": "[null]",
      "websites": "[null]",
      "addresses": "[{\"country\":\"TH\"}]",
      "categories": "{\"primary\":\"Business and Professional Services > Health and Beauty Service > Hair Salon\",\"alternate\":[]}",
      "confidence": 1.0
    }
  }
}
```

**Key Fields**:
- `label: "b"` = **Base** version is better (more structured category)
- Shows real-world edge cases: missing data, international characters, category hierarchy differences

---

## 2. Synthetic Golden Dataset (2,000 Records from Yelp)

**File**: `data/synthetic_golden_dataset_2k.json`  
**Source**: Yelp Academic Dataset with simulated noise  
**Usage**: Primary training dataset for ML models

### Sample Record #1

```json
{
  "id": "synthetic_yelp_c4Y_RZKBXsXENA9y7JIBaQ",
  "data": {
    "current": {
      "names": "{\"primary\": \"kool tortas\"}",
      "phones": "[\"(599) 376-7457\"]",
      "websites": "[\"http://kooltortas.com\"]",
      "addresses": "[{\"freeform\": \"4547 S 6th Ave\", \"locality\": \"Tucson\", \"region\": \"AZ\", \"country\": \"US\"}]",
      "categories": "{\"primary\": \"Mexican\", \"alternate\": []}"
    },
    "base": {
      "names": "{\"primary\": \"Kool Tortas\"}",
      "phones": "[\"+15993767457\"]",
      "websites": "[\"https://www.kooltortas.com\"]",
      "addresses": "[{\"freeform\": \"4547 S 6th Ave\", \"locality\": \"Tucson\", \"region\": \"AZ\", \"postcode\": \"85714\", \"country\": \"US\"}]",
      "categories": "{\"primary\": \"Mexican, Restaurants\"}"
    }
  },
  "label": "b",
  "method": "synthetic_yelp_proxy"
}
```

**Key Differences** (simulated noise):
- **Name**: `"kool tortas"` (lowercase) vs `"Kool Tortas"` (proper case)
- **Phone**: `"(599) 376-7457"` vs `"+15993767457"` (formatting difference)
- **Website**: `"http://kooltortas.com"` vs `"https://www.kooltortas.com"` (HTTP vs HTTPS, www)
- **Address**: Missing postcode in current vs complete in base
- **Category**: Simple `"Mexican"` vs detailed `"Mexican, Restaurants"`
- `label: "b"` = Base is better (more complete + proper formatting)

### Sample Record #2

```json
{
  "id": "synthetic_yelp_pyrLuxTNqCSOoweWZIw-yA",
  "data": {
    "current": {
      "names": "{\"primary\": \"HAMMERS HEALTHY HOUNDS\"}",
      "phones": "[\"+14277606045\"]",
      "websites": "[\"http://hammershealthyhounds.com\"]",
      "addresses": "[{\"freeform\": \"4820 Vista Blvd, Ste 106\", \"locality\": \"Sparks\", \"region\": \"NV\", \"country\": \"US\"}]",
      "categories": "{\"primary\": \"Pet Stores\"}"
    },
    "base": {
      "names": "{\"primary\": \"Hammers Healthy Hounds\"}",
      "phones": "[\"+14277606045\"]",
      "websites": "[\"https://www.hammershealthyhounds.com\"]",
      "addresses": "[{\"freeform\": \"4820 Vista Blvd, Ste 106\", \"locality\": \"Sparks\", \"region\": \"NV\", \"postcode\": \"89436\", \"country\": \"US\"}]",
      "categories": "{\"primary\": \"Pet Stores, Shopping, Pets, Fashion\"}"
    }
  },
  "label": "b",
  "method": "synthetic_yelp_proxy"
}
```

**Key Differences**:
- **Name**: `"HAMMERS HEALTHY HOUNDS"` (ALL CAPS) vs `"Hammers Healthy Hounds"` (proper case)
- **Website**: HTTP vs HTTPS + www
- **Address**: Missing postcode in current
- **Category**: Single category vs multiple categories
- `label: "b"` = Base is better (proper formatting + completeness)

---

## Dataset Comparison Summary

| Feature | Manual 200 Dataset | Synthetic 2K Dataset |
|---------|-------------------|----------------------|
| **Source** | Overture pre-matched places | Yelp Academic Dataset |
| **Labeling** | Human-validated (AI + manual review) | Synthetic (simulated noise) |
| **Record Count** | 200 records | 2,000 records |
| **Use Case** | Final validation + failure analysis | ML training + experimentation |
| **Label Values** | `"c"` (current), `"b"` (base), `"s"` (same), `"u"` (unclear) | `"c"`, `"b"`, `"s"` |
| **Method Field** | `"manual_review (manual)"` or `"ai_agreement"` | `"synthetic_yelp_proxy"` |
| **Data Quality** | Real-world conflicts, edge cases | Controlled noise patterns |

---

## How to Use These Screenshots

1. **Screenshot Option 1**: Show one complete record from each dataset side-by-side
   - Left: Manual 200 sample (shows human decision + real Overture data)
   - Right: Synthetic 2K sample (shows simulated noise + Yelp structure)

2. **Screenshot Option 2**: Show the comparison table at the bottom
   - Clean summary of dataset differences and roles

3. **Screenshot Option 3**: Show just the JSON structure
   - Focus on the `label` field and `data.current` vs `data.base` structure

**Recommended**: Use Option 1 (side-by-side records) for maximum visual impact on Slide 5.

