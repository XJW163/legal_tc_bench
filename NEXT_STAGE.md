# Next stage: occurrence-level rendering extraction

1. Create review rows from annotated datasets and translations:

```bash
legal-tc make-reviews data/annotated data/translations data/reviews
```

2. Check `data/reviews/review_manifest.json`. `missing_translation_count` should be 0.

3. Extract exact Chinese renderings with resumable vLLM calls:

```bash
legal-tc llm-extract-renderings \
  data/reviews \
  data/renderings \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 10 \
  --max-retries 5 \
  --timeout 300
```

Rerun the same command after interruption. Existing completed results and the
checkpoint are reused automatically.

4. Inspect `data/renderings/renderings.audit.json`:

- `status` should be `complete`
- `remaining_count` should be `0`
- manually review low-confidence, `AMBIGUOUS`, and `TRANSLATION_ERROR` rows

5. Compute formal metrics for a rendered review file:

```bash
legal-tc metrics data/renderings/<file>.review.json data/metrics/<file>.metrics.json
```
