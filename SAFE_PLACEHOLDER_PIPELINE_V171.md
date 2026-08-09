# LegalTC v1.7.1 — Calibrated Safe Placeholder Pipeline

Version: 1.7.1
Placeholder strategy: `occurrence-unique-retry-quality-gate-v4`

## Why this patch exists

The v1.7 pilot showed that all 73 restored-stage rejections were reported as
`DUPLICATED_CANONICAL_RENDERING`. The old check scanned the whole paragraph, so a
legal term appearing normally in two different source occurrences was mistaken for
an accidental adjacent duplicate.

The same pilot also showed repeated wrapper rejections in acronym/alias definitions,
repeated identical rejection loops, three max-token truncations, and a semantic
verifier that occasionally demanded retention of English protected terms despite the
Chinese canonical-term policy.

## Changes

### 1. Span-local restored checks

Placeholder restoration now records the exact target span inserted for every token.
Post-restoration checks inspect only the local window around that span. Legitimate
repetition in another sentence or source occurrence no longer triggers a duplicate
error.

The following remain hard errors when local to one restored span:

- immediately adjacent duplicate canonical rendering;
- nested book-title marks around a restored legal-title term;
- risky rendering immediately beside the canonical rendering;
- unresolved placeholders.

### 2. Definition-alias normalization

Validated `DEFINIENDUM` patterns such as:

```text
员工股票购买计划（“【术语0011】”）
首席执行官（“【术语0007】”）
《联邦存款保险法》（“【术语0019】”）
```

are normalized to one canonical Chinese expression. The removed alias occurrence is
recorded as `contextually_satisfied`, rather than forcing a redundant output such as
`员工股票购买计划（“员工股票购买计划”）`.

A same-term pattern such as:

```text
【术语0010_01】董事会（即“【术语0010_02】”）
```

is also collapsed to one canonical `董事会`, with both source occurrences explicitly
audited as contextual omissions.

New row fields:

- `placeholder_contextual_omission_tokens`
- `placeholder_contextual_omission_count`

New manifest statistics:

- `placeholder_contextual_omission_paragraphs`
- `placeholder_contextual_omission_occurrences`

### 3. Early deterministic fallback

The same deterministic rejection no longer consumes all translation retries by
default. After it repeats twice, the paragraph goes directly to glossary fallback.

CLI:

```bash
--placeholder-repeat-rejection-limit 2
```

Use `0` to disable this optimization.

### 4. Max-token escalation

A `finish_reason=length` response now raises a dedicated truncation error. The next
translation attempt automatically increases the output budget, up to 8192 tokens.

New statistic:

- `max_tokens_escalations`

### 5. Semantic verifier policy correction

The verifier prompt now states explicitly that:

- the output must be Simplified Chinese;
- listed Chinese canonical renderings are mandatory;
- protected terms must not be required to remain in English;
- a cross-term canonical collision should still be reported when it destroys a legal
  distinction.

Verifier request/JSON parsing can retry without regenerating the translation:

```bash
--semantic-verifier-retries 2
```

### 6. Term-memory collision warnings

Each document output now contains `term_memory_collision_warnings`. For example,
`Cause` and `Good Reason` sharing one Chinese canonical rendering is reported as:

```text
CROSS_TERM_CANONICAL_COLLISION
```

This is an audit warning. v1.7.1 does not silently rewrite the term memory.

### 7. More complete rejection statistics

New manifest statistics include:

- `placeholder_request_or_processing_rejections`
- `placeholder_early_fallback_paragraphs`
- `placeholder_contextual_omission_paragraphs`
- `placeholder_contextual_omission_occurrences`
- `max_tokens_escalations`

## Recommended pilot command

Occurrence validation has not changed, so the v1.6.1 validation directory can be
reused.

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-placeholder-v171-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode placeholder \
  --occurrence-validation data/occurrences_validated_v161_pilot \
  --placeholder-repair-policy retry_then_fallback \
  --placeholder-semantic-verifier retry \
  --placeholder-repeat-rejection-limit 2 \
  --semantic-verifier-retries 2 \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

Use a new experiment name. The placeholder strategy signature changed.
