# Failure Analysis Report (OKR 2, KR 2)
This document lists failure cases where the model/algorithm prediction disagreed with the ground truth.

## Attribute: NAME

### ML Model Failures (Count: 124)
**1. Record 08f3da18ccad52ad03cc06b87820910f**
- Prediction: BASE | Truth: CURRENT
- Current: `davaindia GENERIC PHARMACY`
- Base:    `Davaindia Generic Pharmacy`

**2. Record 08f08ed358889889037e291e52eb62a5**
- Prediction: BASE | Truth: CURRENT
- Current: `Full Steam Tromso`
- Base:    `Full Steam`

**3. Record 08f2b9b36479862e0351baadfa912c6b**
- Prediction: BASE | Truth: CURRENT
- Current: `Franco's Pizza`
- Base:    `Franco's Pizza`

**4. Record 08f446c25679a70e03572240a924ba2c**
- Prediction: BASE | Truth: CURRENT
- Current: `Chick-fil-A Grand Parkway North`
- Base:    `Chick-fil-A`

**5. Record 281474979747202**
- Prediction: BASE | Truth: CURRENT
- Current: `Interstate All Battery Center`
- Base:    `Interstate All Battery Center`


### Baseline (Most Recent) Failures (Count: 116)
**1. Record 08f48932e69a12dd03d5671400e23b78**
- Prediction: SAME | Truth: CURRENT
- Current: `{"primary":"AT&T Store"}`
- Base:    `None`

**2. Record 08f446b259454ad503533f12dc4d0f9c**
- Prediction: SAME | Truth: CURRENT
- Current: `{"primary":"Love's RV Hookup"}`
- Base:    `None`

**3. Record 08f194ac2641827303a8c9d8d0f63cfb**
- Prediction: SAME | Truth: CURRENT
- Current: `{"primary":"Paddy Power"}`
- Base:    `None`

**4. Record 562949958305102**
- Prediction: SAME | Truth: CURRENT
- Current: `{"primary":"Oxford Glen Memory Care at Sachse"}`
- Base:    `None`

**5. Record 08f41695326f2812035673b147fa47f6**
- Prediction: CURRENT | Truth: BASE
- Current: `{"primary":"สนามบอลราชภัฏสกลนคร"}`
- Base:    `None`

## Attribute: ADDRESS

### ML Model Failures (Count: 108)
**1. Record 08f2986b8089b2cb03220db3aa72c816**
- Prediction: BASE | Truth: CURRENT
- Current: `{'freeform': '3663 S Las Vegas Blvd', 'locality': 'Las Vegas', 'postcode': '89109', 'region': 'NV', 'country': 'US'}`
- Base:    `{'freeform': '3663 Las Vegas Blvd South', 'locality': 'Las Vegas', 'region': 'NV', 'country': 'US', 'postcode': '89109'}`

**2. Record 08f3da18ccad52ad03cc06b87820910f**
- Prediction: BASE | Truth: CURRENT
- Current: `{'freeform': 'Shop No F21/66, Rohini, Sector 15', 'locality': 'New Delhi', 'postcode': '110089', 'region': 'Delhi', 'country': 'IN'}`
- Base:    `{'freeform': 'Shop No F21/66, Rohini, Sector 15', 'locality': 'New Delhi', 'region': 'Delhi', 'country': 'IN', 'postcode': '110089'}`

**3. Record 08f275922226a40b03fa3c4693070159**
- Prediction: BASE | Truth: CURRENT
- Current: `{'freeform': '516 W Wise Rd', 'locality': 'Schaumburg', 'postcode': '60193-3815', 'region': 'IL', 'country': 'US'}`
- Base:    `{'freeform': '516 W Wise Rd', 'locality': 'Schaumburg', 'region': 'IL', 'country': 'US', 'postcode': '60193'}`

**4. Record 08f3922211a9429003d34a89765a66c1**
- Prediction: BASE | Truth: CURRENT
- Current: `{'freeform': 'Avenida do Futebol Clube do Porto 779', 'locality': 'Marco de Canaveses', 'postcode': '4630-203', 'country': 'PT'}`
- Base:    `{'freeform': 'Avenida Futebol Clube do Porto 779', 'locality': 'Marco de Canaveses', 'region': 'Porto', 'country': 'PT', 'postcode': '4630-203'}`

**5. Record 08f1f9b089cf330d03b9a3ab6ad274ba**
- Prediction: CURRENT | Truth: BASE
- Current: `{'freeform': 'Via Edmondo De Amicis, 2', 'locality': 'Genova', 'postcode': '16122', 'region': '42', 'country': 'IT'}`
- Base:    `{'freeform': 'Via Edmondo de Amicis 2', 'locality': 'Genova', 'region': 'Genova', 'country': 'IT'}`


### Baseline (Most Recent) Failures (Count: 68)
**1. Record 562949958305102**
- Prediction: SAME | Truth: CURRENT
- Current: `[{"freeform":"4546 Bunker Hill Rd","locality":"Sachse","region":"TX","country":"US","postcode":"75048"}]`
- Base:    `None`

**2. Record 08f64a59990a63a0030784450be43ba4**
- Prediction: SAME | Truth: BASE
- Current: `[{"country":"TH"}]`
- Base:    `None`

**3. Record 08f6d231146cbb8203a6b43c5463e884**
- Prediction: CURRENT | Truth: BASE
- Current: `[{"freeform":"Calle Central Norte 12","locality":"Tuxtla Gutiérrez","postcode":"29000","country":"MX"}]`
- Base:    `None`

**4. Record 08f195e2017138a60344368b12283837**
- Prediction: CURRENT | Truth: BASE
- Current: `[{"freeform":"Builth Wells Quarry","postcode":"LD2 3UB","country":"GB"}]`
- Base:    `None`

**5. Record 1970324838691605**
- Prediction: SAME | Truth: BASE
- Current: `[{"freeform":"1550 Ingram Blvd","locality":"West Memphis","region":"AR","country":"US","postcode":"72301"}]`
- Base:    `None`

## Attribute: PHONE

### ML Model Failures (Count: 56)
**1. Record 562949958207347**
- Prediction: CURRENT | Truth: BASE
- Current: `8133302175`
- Base:    `8133302175`

**2. Record 08f3cc5a1476a45403ed1a0c115b1102**
- Prediction: CURRENT | Truth: BASE
- Current: `+9118604195555`
- Base:    `1860 419 5555`

**3. Record 08f64b018d54d71403e6f180aca04fd1**
- Prediction: CURRENT | Truth: BASE
- Current: `nan`
- Base:    `None`

**4. Record 1970324838691605**
- Prediction: CURRENT | Truth: BASE
- Current: `8004676182`
- Base:    `8004676182`

**5. Record 08f28308106929aa0322048f73b8a6a1**
- Prediction: CURRENT | Truth: BASE
- Current: `5104633888`
- Base:    `5106479364`


### Baseline (Most Recent) Failures (Count: 54)
**1. Record 1125899908396217**
- Prediction: SAME | Truth: CURRENT
- Current: `["01634671651"]`
- Base:    `None`

**2. Record 08f1f13d06cdb4dd03c2287ba2cf48a4**
- Prediction: CURRENT | Truth: BASE
- Current: `["+49523192570"]`
- Base:    `None`

**3. Record 08f446b259454ad503533f12dc4d0f9c**
- Prediction: SAME | Truth: CURRENT
- Current: `["9036184002"]`
- Base:    `None`

**4. Record 08f424927109622803649838ac7eb6a2**
- Prediction: CURRENT | Truth: BASE
- Current: `[""]`
- Base:    `None`

**5. Record 08f2a92a8c54d533034329dc2c045529**
- Prediction: CURRENT | Truth: BASE
- Current: `["+17658553633"]`
- Base:    `None`

## Attribute: WEBSITE

### ML Model Failures (Count: 75)
**1. Record 08f1e82252c1c0ae03c90f2f5d5feef9**
- Prediction: CURRENT | Truth: BASE
- Current: `nan`
- Base:    `None`

**2. Record 08f44e38ed4518540380d8904cd5b34a**
- Prediction: BASE | Truth: CURRENT
- Current: `http://earlycountysheriff.com/`
- Base:    `http://www.earlycountynews.com/news/2015-07-01/Front_Page/County_hires_Spencer_Mueller.html`

**3. Record 08fa96e0d31aaa0003dff57d6d7652d6**
- Prediction: CURRENT | Truth: BASE
- Current: `nan`
- Base:    `None`

**4. Record 08f41695326f2812035673b147fa47f6**
- Prediction: CURRENT | Truth: BASE
- Current: `nan`
- Base:    `None`

**5. Record 08f441ab99d1e40903520592b478eaee**
- Prediction: BASE | Truth: CURRENT
- Current: `http://delicaegourmet.com/`
- Base:    `https://www.delicaegourmet.com/`


### Baseline (Most Recent) Failures (Count: 77)
**1. Record 08f2aa859dba3af003731e8ef72ef347**
- Prediction: SAME | Truth: CURRENT
- Current: `["https://international-bakery-inc-md.hub.biz"]`
- Base:    `None`

**2. Record 562949955127994**
- Prediction: CURRENT | Truth: BASE
- Current: `[""]`
- Base:    `None`

**3. Record 08f1f13d06cdb4dd03c2287ba2cf48a4**
- Prediction: CURRENT | Truth: BASE
- Current: `["https://www.boeckmann-mode.de/lingerie.html"]`
- Base:    `None`

**4. Record 1407374883710721**
- Prediction: SAME | Truth: CURRENT
- Current: `["http://www.redwingrichmond.com/"]`
- Base:    `None`

**5. Record 08f2aab52140802c0395a9524fcac41d**
- Prediction: SAME | Truth: CURRENT
- Current: `["https://www.libertytax.com/12077"]`
- Base:    `None`

## Attribute: CATEGORY

### ML Model Failures (Count: 57)
**1. Record 08f26010a6db404403e339f85cbc6d43**
- Prediction: CURRENT | Truth: BASE
- Current: `automotive_repair`
- Base:    `Business and Professional Services > Automotive Service > Automotive Repair Shop`

**2. Record 1970324838691605**
- Prediction: CURRENT | Truth: BASE
- Current: `hotel`
- Base:    `hotel`

**3. Record 08f4440d8b89c95803ef5bd3cc80eac4**
- Prediction: CURRENT | Truth: BASE
- Current: `mexican_restaurant`
- Base:    `Dining and Drinking > Restaurant > Mexican Restaurant > Burrito Restaurant`

**4. Record 562949955293136**
- Prediction: CURRENT | Truth: BASE
- Current: `tire_shop`
- Base:    `tire_shop`

**5. Record 08f3961561d3094c038780c81f579575**
- Prediction: CURRENT | Truth: BASE
- Current: `shopping`
- Base:    `Retail > Food and Beverage Retail > Butcher`


### Baseline (Most Recent) Failures (Count: 68)
**1. Record 08f19424ccc1e0820363bbececd4900d**
- Prediction: SAME | Truth: BASE
- Current: `{"primary":"food_delivery_service","alternate":["courier_and_delivery_services","food"]}`
- Base:    `None`

**2. Record 08f1e82252c1c0ae03c90f2f5d5feef9**
- Prediction: CURRENT | Truth: BASE
- Current: `{"primary":"bar","alternate":["arts_and_entertainment","professional_services"]}`
- Base:    `None`

**3. Record 08f3cc5a1476a45403ed1a0c115b1102**
- Prediction: CURRENT | Truth: BASE
- Current: `{"primary":"atms","alternate":["financial_service","bank_credit_union"]}`
- Base:    `None`

**4. Record 1125899912289624**
- Prediction: SAME | Truth: CURRENT
- Current: `{"primary":"roofing","alternate":["siding","ceiling_and_roofing_repair_and_service"]}`
- Base:    `None`

**5. Record 08f41695326f2812035673b147fa47f6**
- Prediction: CURRENT | Truth: BASE
- Current: `{"primary":"topic_concert_venue","alternate":["soccer_field"]}`
- Base:    `None`
