# Attribute Labeling Guidelines for Golden Dataset

## Overview

This document defines the key attributes for place conflation and provides clear labeling guidelines for creating a high-quality golden dataset. Each attribute requires annotators to select the "best" value when two versions (current vs. base) conflict.

## Core Attributes (Required for KR3)

### 1. Name

**Definition**: Primary business/place name as it should appear in the final conflation record.

**Format**: JSON object with `primary` field
- Example: `{"primary":"Goin' Postal Jacksonville"}`

**Labeling Criteria**:
- **Prefer the more complete/formal name**
- **Prefer the official business name** (avoid shortened versions)
- Consider: spelling accuracy, capitalization consistency
- If names are identical, mark as "SAME"

**Edge Cases**:
- Minor spelling variations → Prefer the more commonly used spelling
- Abbreviations (e.g., "Ltd" vs "Limited") → Prefer the formal/legal name
- Punctuation differences → Usually equivalent, but prefer the more standard format
- Missing punctuation or special characters → Prefer the version with proper punctuation

**Examples**:
- `"McDonald's"` vs `"Mcdonalds"` → Prefer "McDonald's"
- `"Starbucks Coffee"` vs `"Starbucks"` → Prefer the more complete name "Starbucks Coffee"
- `"Joe's Pizza"` vs `"Joes Pizza"` → Prefer "Joe's Pizza"

---

### 2. Phone

**Definition**: Primary contact phone number(s) for the business.

**Format**: JSON array of phone strings
- Example: `["+19049989600"]` or `["9049989600"]`

**Labeling Criteria**:
- **Prefer international format** (with country code and + prefix) when available
- Prefer the complete/standardized format
- If formats differ, prefer E.164 format (e.g., `+19049989600`)
- If phone numbers are truly different (not just format differences), mark both if both are valid

**Edge Cases**:
- Format differences (`+1-904-998-9600` vs `9049989600` vs `+19049989600`) → Prefer standardized format `+19049989600`
- Extra numbers vs missing → Prefer the more complete number
- Text in number (e.g., extensions) → Prefer the pure numeric format
- Area code differences → Likely different businesses, review carefully

**Examples**:
- `["+19049989600"]` vs `["9049989600"]` → Prefer `+19049989600`
- `["1-904-998-9600"]` vs `["+19049989600"]` → Prefer `+19049989600` (E.164 format)

---

### 3. Website

**Definition**: Official website URL for the business.

**Format**: JSON array of website URLs
- Example: `["http://www.goinpostaljacksonville.com/"]` or `["https://www.goinpostaljacksonville.com/"]`

**Labeling Criteria**:
- **Prefer HTTPS over HTTP** when both available
- Prefer the canonical URL (with or without www consistently)
- Prefer full URL over shortened URLs (e.g., `https://example.com` vs `http://bit.ly/example`)
- Prefer the more stable/long-term URL

**Edge Cases**:
- HTTP vs HTTPS → Always prefer HTTPS
- With/without trailing slash → Usually equivalent, but prefer canonical form
- With/without www → Usually equivalent, but verify which is the canonical domain
- Shortened URLs (bit.ly, etc.) → Prefer the full URL if available
- Social media links → Only if no official website exists, treat as secondary

**Examples**:
- `["http://www.goinpostaljacksonville.com/"]` vs `["https://www.goinpostaljacksonville.com/"]` → Prefer HTTPS
- `["http://bit.ly/NelspruitMazda"]` vs `["https://nelspruitmazda.co.za/"]` → Prefer the full URL

---

### 4. Address

**Definition**: Physical location information for the place.

**Format**: JSON array of address objects with fields: freeform, locality, region, country, postcode
- Example: `[{"freeform":"7643 Gate Pkwy","locality":"Jacksonville","postcode":"32256-2892","region":"FL","country":"US"}]`

**Labeling Criteria**:
- **Prefer the most complete address** with all components (street, city, state, zip, country)
- Prefer standardized formatting (especially for postal codes)
- Prefer the address with fewer inconsistencies across fields
- Consider: completeness, accuracy of individual fields

**Edge Cases**:
- Postal code format differences (`32256` vs `32256-2892`) → Usually equivalent, but prefer 9-digit format if available
- City name variations → Prefer the official locality name
- Incomplete addresses (missing region) → Prefer the complete version
- Street address format differences → Usually equivalent unless significant (Suite/Unit numbers)
- Freeform inconsistencies → Verify against structured fields

**Examples**:
- Address with Suite: `"7643 Gate Pkwy Ste 104"` vs without Suite `"7643 Gate Pkwy"` → Prefer the version with Suite number if it's a multi-tenant building
- Postal code: `"32256"` vs `"32256-2892"` → Usually both acceptable, prefer the more specific if verifying actual +4 extension

---

### 5. Category

**Definition**: Business type/classification describing what the place does.

**Format**: JSON object with `primary` category and optional `alternate` array
- Example: `{"primary":"shipping_center","alternate":["freight_and_cargo_service","post_office"]}`

**Labeling Criteria**:
- **Prefer the more specific/accurate categorization**
- If one version has more complete categorization (has alternate categories), prefer that
- Consider: which classification better describes the business
- Prefer the version that matches the business name or describes the primary activity

**Edge Cases**:
- More vs fewer categories → Prefer the version with more categories if they're all relevant
- Specific vs general → Prefer specific (e.g., "transmission_repair" vs "automotive")
- Semantic differences → Consider which better matches the actual business
- Missing primary category → Prefer the version with primary category

**Examples**:
- `{"primary":"shipping_center"}` vs `{"primary":"vehicle_shipping"}` → Prefer the more specific "vehicle_shipping" if accurate
- `{"primary":"car_dealer","alternate":["automotive","automotive_dealer"]}` vs `{"primary":"used_car_dealer","alternate":["automotive"]}` → Prefer version with more categories if all are accurate
- `{"primary":"automotive_repair"}` (no alternate) vs `{"primary":"transmission_repair","alternate":["automotive"]}` → Prefer the version that provides more specific context

---

## Optional Attributes (Bonuses)

### 6. Social Media Profiles

**Definition**: Social media URLs associated with the business.

**Format**: JSON array of social media URLs
- Example: `["https://www.facebook.com/463273470392736"]`

**Labeling Criteria**:
- Prefer complete/stable URLs
- Prefer official business pages over personal profiles
- Consider recency and authenticity
- If only one version has social media data, prefer that version

**Note**: This is less critical than name, phone, website, address, or category.

---

### 7. Brand

**Definition**: Brand information if the place is part of a chain or franchise.

**Format**: JSON object (varies)
- Example: `{"names":{}}` or more complex brand structures

**Labeling Criteria**:
- Prefer the more complete brand information
- If brand info exists in one version, prefer that
- Consider relevance to the specific location (local vs corporate brand)

**Note**: Not all businesses have brand information.

---

## General Labeling Principles

### Selection Rules (in priority order):

1. **Completeness**: Prefer the version with more complete data
2. **Accuracy**: Choose the version that appears more accurate based on cross-validation
3. **Formatting**: Prefer standardized, well-formatted data
4. **Canonical Forms**: Prefer canonical URLs, standard phone formats, etc.
5. **Recency**: If all else equal, prefer more recent data (but this is lower priority)

### When to Mark as "SAME":
- Values are functionally identical despite minor formatting differences
- Both versions are equally valid and equivalent

### When to Mark as "NEITHER" or "REVIEW REQUIRED":
- Both versions appear incorrect
- Significant data quality issues in both versions
- Unable to determine which is "better" without additional research

### Annotation Format:

For each record, annotators will:
1. Compare each attribute (current vs base)
2. Select "CURRENT", "BASE", "SAME", or "UNCLEAR"
3. Add optional notes for difficult cases

---

## Quality Assurance

- **Consistency**: Annotators should make similar decisions for similar cases
- **Documentation**: Document reasoning for unusual or edge cases
- **Inter-annotator agreement**: Target >95% agreement on sample of 200 records
- **Iterative refinement**: Update guidelines based on disagreements during annotation

---

## Revision History

- Initial version: Created based on exploratory analysis of project_b_samples_2k.parquet
- Expected updates based on inter-annotator agreement study findings

