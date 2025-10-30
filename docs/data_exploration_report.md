# Data Exploration Report

This report summarizes findings from the exploratory analysis of `data/project_b_samples_2k.parquet` using `scripts/explore_dataset.py`.

---

## 1. Dataset Structure

- **Records:** 2,000
- **Columns:** 22

**Columns:**

| # | Name             | # | Name             |
|---|------------------|---|------------------|
| 1 | id               | 12| addresses        |
| 2 | base_id          | 13| base_sources     |
| 3 | sources          | 14| base_names       |
| 4 | names            | 15| base_categories  |
| 5 | categories       | 16| base_confidence  |
| 6 | confidence       | 17| base_websites    |
| 7 | websites         | 18| base_socials     |
| 8 | socials          | 19| base_emails      |
| 9 | emails           | 20| base_phones      |
|10 | phones           | 21| base_brand       |
|11 | brand            | 22| base_addresses   |

**Data types** (string/object unless float shown): `id`, `base_id`, `sources`, `names`, `categories`, `websites`, `socials`, `emails`, `phones`, `brand`, `addresses`, + respective `base_*` fields. Numeric: `confidence`, `base_confidence` (float).

**Null value counts** (non-required fields):
- categories: 16 (0.8%)
- websites: 288 (14.4%)
- socials: 608 (30.4%)
- emails: 2000 (100%)
- phones: 109 (5.5%)
- brand: 1296 (64.8%)
- base_categories: 12 (0.6%)
- base_websites: 3 (0.1%)
- base_socials: 983 (49.1%)
- base_emails: 1000 (50%)
- base_phones: 4 (0.2%)
- base_brand: 46 (2.3%)

---

## 2. Attribute Coverage and Difference Rates

Attribute pairs (current vs. base version for each attribute):

| Attribute   | Coverage (Current) | Coverage (Base) | Different Values |
|-------------|--------------------|-----------------|-----------------|
| names       | 100.0%             | 100.0%          | 68.8%           |
| phones      | 94.5%              | 99.8%           | 76.8%           |
| websites    | 85.6%              | 99.9%           | 64.6%           |
| addresses   | 100.0%             | 100.0%          | —               |
| categories  | 99.2%              | 99.4%           | 80.4%           |
| emails      | 0%                 | 50%             | —               |
| socials     | 69.6%              | 50.8%           | —               |
| brand       | 35.2%              | 97.7%           | —               |

Notes:
- “Different Values”: % of rows where current and base values are not the same (string compare; does not mean semantically different in all cases!).
- “—”: No diff calc for lists w/ missing data or low coverage.

---

## 3. JSON Field Structure

Some fields store structured JSON as strings, typically with these formats:

- **names/base_names**: `{ "primary": <business_name> }`
- **categories/base_categories**: `{ "primary": <category>, "alternate": [ ... ] }`
- **addresses/base_addresses**: List of `{ freeform, locality, region, country, postcode }` objects

**Example values:**
```
names:         {"primary":"Goin' Postal Jacksonville"}
categories:   {"primary":"shipping_center","alternate":["freight_and_cargo_service","post_office"]}
addresses:    [{"freeform":"7643 Gate Pkwy","locality":"Jacksonville","postcode":"32256-2892","region":"FL","country":"US"}]

base_names:       {"primary":"Goin' Postal Jacksonville"}
base_categories:  {"primary":"vehicle_shipping","alternate":[...]}  (see example)
base_addresses:   [{"freeform":"7643 Gate Pkwy Ste 104","locality":"Jacksonville",...}]
```
---

## 4. Sample Records

### RECORD 1
- **ID**: 08f44f055a9a016e0390f050aa3c93c0
- **Base ID**: 1688849865669487
- **Confidence**: Current=0.996, Base=0.770
- **Names**: {"primary":"Goin' Postal Jacksonville"} (same for current/base)
- **Phones**: Current: ["+19049989600"], Base: ["9049989600"]
- **Websites**: ["http://www.goinpostaljacksonville.com/"] vs ["https://www.goinpostaljacksonville.com/"]
- **Categories**:
    - Current: {"primary":"shipping_center","alternate":["freight_and_cargo_service","post_office"]}
    - Base:    {"primary":"vehicle_shipping","alternate":["courier_and_delivery_services","mailbox_center","shipping_center","post_office","automotive"]}
- **Addresses**:
    - Current: [{"freeform":"7643 Gate Pkwy","locality":"Jacksonville","postcode":"32256-2892","region":"FL","country":"US"}]
    - Base:    [{"freeform":"7643 Gate Pkwy Ste 104","locality":"Jacksonville","region":"FL","country":"US","postcode":"32256"}]

---

### RECORD 2
- **ID**: 08f29a456e42e5830324637954145c50
- **Base ID**: 1125899907111860
- **Confidence**: Current=0.996, Base=0.770
- **Names**: {"primary":"Valley Transmission"} (identical)
- **Phones**: Current: ["+16194474353"], Base: ["6194474353"]
- **Websites**: ["http://valleytransca.com/"] (both current and base)
- **Categories**:
    - Current: {"primary":"automotive_repair","alternate":["automotive","professional_services"]}
    - Base:    {"primary":"transmission_repair","alternate":["automotive"]}
- **Addresses**:
    - Current: [{"freeform":"1158 N 2nd St","locality":"El Cajon","postcode":"92021-5023","region":"CA","country":"US"}]
    - Base:    [{"freeform":"1158 N 2nd St","locality":"El Cajon","region":"CA","country":"US","postcode":"92021"}]

---

### RECORD 3
- **ID**: 08fbcd0030da5323031bcafa8c2fa0dc
- **Base ID**: 844424934845986
- **Confidence**: Current=0.998, Base=0.770
- **Names**: {"primary":"Mazda Nelspruit"} (identical)
- **Phones**: Current: ["+27137573800"], Base: ["0137573800"]
- **Websites**: ["http://bit.ly/NelspruitMazda"] (current) vs ["https://nelspruitmazda.co.za/"] (base)
- **Categories**:
    - Current: {"primary":"car_dealer","alternate":["automotive","automotive_dealer"]}
    - Base:    {"primary":"used_car_dealer","alternate":["automotive"]}
- **Addresses**:
    - Current: [{"freeform":"Mazda Nelspruit, 49 Emnotweni Avenue","locality":"Riverside Park","postcode":"1201","region":"MP","country":"ZA"}]
    - Base:    [{"freeform":"49 Emnotweni Avenue","locality":"Mbombela","region":"","country":"ZA","postcode":"1201"}]

---

**Summary**: The dataset consists of paired records with substantial disagreement rates in key fields—especially phone (77%), category (80%), and name (69%). It is well-suited as a test ground for attribute selection/conflation logic, with robust primary and base attribute coverage, but missing data for some optional fields.
