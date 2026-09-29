# Corpus cleaning report

- Source: `training_data.txt`
- The original source was preserved.
- Retained records were copied without technical rewriting.

## Cleaning policy

Keep one deterministic representative for each normalized template family and technical topic; preserve preamble records.

The technical topic is included in the deduplication key. Therefore,
different questions about the same aircraft system are not removed merely
because they share a subject. Only records with the same normalized
template family and subject compete for one representative.

## Statistics before and after

| Measure | Before | After |
|---|---:|---:|
| Records | 22,829 | 11,824 |
| UTF-8 bytes | 14,266,938 | 8,787,562 |
| Template families | 9,594 | 9,594 |

## Duplicate and repetition counts

- Exact duplicate groups: **0**
- Exact duplicate records beyond the first: **0**
- Strict near-duplicate pairs: **119** (190 records; 5-word shingle Jaccard >= 0.85)
- Template-equivalent groups: **2,376**
- Template-equivalent records removed: **11,005**
- Repeated answer groups: **2,126**
- Records participating in repeated answers: **7,072**
- Repeated `Training variation` structures: **26**
- Records participating in repeated variation structures: **2,224**

## Boilerplate and low-information analysis

| Pattern | Occurrences |
|---|---:|
| generic training context | 20,052 |
| simulator caveat | 424 |
| configuration caveat | 3,073 |
| learning-angle label | 4,346 |
| training-variation label | 2,224 |
| scenario scaffolding | 3,073 |

Low-information candidates (flagged, not independently deleted): **21**.
They are only removed when they are also template-equivalent duplicates;
unique records remain available for review.

## Removal summary

- Records removed: **11,005**
- UTF-8 bytes removed: **5,479,376**
- Removal reason: template-equivalent duplicate within the same technical topic.

## Template-family counts

The complete before/after family counts are in the JSON report.
The largest families before cleaning are:

- `family-39b73131889b18bc`: 254
- `family-065965ee03d69e99`: 250
- `family-d942e6db95d9f4ba`: 246
- `family-632e3214bc518dbf`: 237
- `family-7340c8f1aa72fed4`: 237
- `family-b57cdf895aa7b4d9`: 236
- `family-83dc89f9f629c6b7`: 232
- `family-e6621a06f32f553f`: 231
- `family-008cb22f386c6f4f`: 221
- `family-32920f13fc6c7fce`: 221
- `family-ea177ba9af50c6d5`: 201
- `family-c09aa9234cacfa51`: 193
- `family-9a7f98472300ff2e`: 191
- `family-a9d83026953f4f98`: 189
- `family-22dbe3f1c45de1a7`: 187
- `family-26de65303f50f7e8`: 184
- `family-3bb4f0aaf84904a9`: 178
- `family-301a5bf0921c3c83`: 170
- `family-a4cb3bae2febd412`: 157
- `family-f695dc4b672a4f00`: 155
- `family-a13d3c3b2b3c1f0a`: 126
- `family-b7fa2d8c22dbc495`: 126
- `family-8e5e7451f654a95c`: 125
- `family-9004860406f7411d`: 125
- `family-a807a9e78fce458d`: 125

## Outputs

- `training_data_cleaned.txt` — cleaned corpus; blank-line-delimited records.
- `training_data_cleaned.families.json` — record-to-family assignments.
- `corpus_cleaning_report.json` — machine-readable full report.
- `corpus_cleaning_report.md` — this human-readable report.
