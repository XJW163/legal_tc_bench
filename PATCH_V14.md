# v1.4 — Robust occurrence-level placeholder control

## Why v1.3 failed

v1.3 reused one identical placeholder for every occurrence of the same term. In long legal paragraphs the model had to count strings such as `【术语0001】` 14 or 16 times. Models frequently merged one repeated mention, omitted one token, or duplicated one token. The whole paragraph was then discarded after retries.

## Changes

1. **Occurrence-unique placeholders**
   - One occurrence: `【术语0002】`
   - Repeated occurrences: `【术语0002_01】`, `【术语0002_02】`, ...
   - All occurrence tokens restore to the same canonical document-level term.

2. **Inventory-aware validation**
   - Separately reports missing, duplicated, and unknown placeholders.
   - Uses a mismatch score to retain the best attempt.

3. **Minimum-edit placeholder repair**
   - The first request translates the protected source.
   - Later attempts repair the best malformed Chinese translation instead of blindly retranslating from scratch.
   - The repair prompt receives the protected source, faulty translation, required token ledger, and exact mismatch diagnostics.

4. **Audited glossary fallback**
   - If hard-placeholder repair still fails, the paragraph is not dropped.
   - It is translated using the same definition-grounded canonical glossary.
   - The output is explicitly marked `constraint_strategy_used=glossary_fallback` and is not counted as hard-placeholder compliance.

5. **New audit fields**
   - `placeholder_strategy_version`
   - `constraint_strategy_used`
   - `placeholder_repair_attempts`
   - `placeholder_fallback_used`
   - `placeholder_diagnostics`
   - `placeholder_failure`
   - manifest statistics for repaired and fallback paragraphs.

## Methodological note

Hard placeholder conservation is an experimental strict-consistency mode. Chinese may legitimately omit repeated noun phrases. The fallback rate should therefore be reported alongside terminology consistency and translation quality rather than hidden.
