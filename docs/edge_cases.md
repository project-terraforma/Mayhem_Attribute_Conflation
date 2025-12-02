# Edge Cases & Resolution Strategies

This document fulfills **Objective 1, KR3** and **Objective 3, KR2**. It documents specific edge cases encountered during the manual review of 200 Overture records and provides the resolution strategies applied.

## 1. Attribute: Name

### Case 1.1: Capitalization Inconsistency
*   **Scenario:** `STARBUCKS COFFEE` vs `Starbucks Coffee`.
*   **Resolution:** Prefer **Title Case**. It is more standard for map display.
*   **Decision:** `Base` (if Title Case) > `Current` (if ALL CAPS).

### Case 1.2: Official Legal Name vs. Common Name
*   **Scenario:** `Goin' Postal Jacksonville` vs `Goin' Postal`.
*   **Resolution:** Prefer **Specific/Full Name**. It provides more context, especially for chains with multiple locations in a city.
*   **Decision:** `Current` (Longer) > `Base` (Short).

### Case 1.3: Affixes and Suffixes
*   **Scenario:** `Valley Transmission` vs `Valley Transmission Inc.`
*   **Resolution:** Prefer **Natural Name** (without legal suffixes like Inc/LLC) unless strict legal entity matching is required.
*   **Decision:** `Base` (No Inc) > `Current` (With Inc).

---

## 2. Attribute: Address

### Case 2.1: Missing Region/State
*   **Scenario:** `{"locality": "Paris", "country": "FR"}` vs `{"locality": "Paris", "region": "Ile-de-France", "country": "FR"}`.
*   **Resolution:** Prefer **Completeness**. Region/State is vital for geocoding and disambiguation.
*   **Decision:** Version with `region` wins.

### Case 2.2: Unit/Suite Numbers
*   **Scenario:** `123 Main St` vs `123 Main St Ste 100`.
*   **Resolution:** Prefer **Specificity**. The unit number is critical for navigation in multi-tenant buildings.
*   **Decision:** Version with `Ste/Unit` wins.

### Case 2.3: Postcode Formatting
*   **Scenario:** `90210` vs `90210-1234`.
*   **Resolution:** Prefer **Standard 5-digit** (US) for general display, but **9-digit** for precision if backend supports it.
*   **Decision:** `Same` (functionally equivalent for most users), or prefer 5-digit for cleaner display.

---

## 3. Attribute: Website

### Case 3.1: Protocol Mismatch
*   **Scenario:** `http://example.com` vs `https://example.com`.
*   **Resolution:** **Always Prefer HTTPS**. Security best practice.
*   **Decision:** `https` version wins.

### Case 3.2: Canonical vs. Specific Page
*   **Scenario:** `www.chain.com` vs `www.chain.com/locations/store-123`.
*   **Resolution:** Prefer **Specific Location Page**. It provides relevant hours/phone for that specific POI.
*   **Decision:** Long URL wins.

### Case 3.3: Dead Links vs. No Link
*   **Scenario:** `http://deadlink.com` (404) vs `[Null]`.
*   **Resolution:** Prefer **Null**. A broken link is a worse user experience than no link.
*   **Decision:** `Base` (Null) if `Current` is verified broken (requires live check).

---

## 4. Attribute: Phone

### Case 4.1: Formatting
*   **Scenario:** `(555) 123-4567` vs `+15551234567`.
*   **Resolution:** Prefer **E.164 International Format** (`+1...`) for backend storage and programmatic use.
*   **Decision:** `+1` version wins.

### Case 4.2: Local vs. Toll-Free
*   **Scenario:** `555-1234` (Local) vs `800-555-5555` (Corporate).
*   **Resolution:** Prefer **Local Number**. Connects directly to the venue.
*   **Decision:** Local wins.

---

## 5. Attribute: Category

### Case 5.1: Hierarchy vs. Flat
*   **Scenario:** `Restaurant` vs `Food > Restaurants > Italian`.
*   **Resolution:** Prefer **Hierarchical/Granular**. Provides more filtering capability.
*   **Decision:** Hierarchical wins.

### Case 5.2: Primary vs. Alternate Lists
*   **Scenario:** `{"primary": "Bar"}` vs `{"primary": "Bar", "alternate": ["Pub", "Nightclub"]}`.
*   **Resolution:** Prefer **Rich Metadata**. Alternates help search recall.
*   **Decision:** Version with alternates wins.

### Case 5.3: Specificity Conflict
*   **Scenario:** `Retail` vs `Shoe Store`.
*   **Resolution:** Prefer **Specific Leaf Node**.
*   **Decision:** `Shoe Store` wins.
