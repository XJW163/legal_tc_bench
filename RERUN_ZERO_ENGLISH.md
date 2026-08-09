# Zero-English translation mode with soft Latin checking

This version keeps the `zero-english-v1` prompt, but changes Latin-letter validation from a hard failure to a quality warning.

Behavior:

- The prompt still asks the model to produce fully Chinese output.
- If the result contains `A-Z` or `a-z`, the result is saved immediately.
- The translator does not retry merely because Latin letters remain.
- Each paragraph records `latin_residue_count`, `latin_residue_samples`, `latin_residue_warning`, and `latin_residue_policy`.
- Empty output, request failure, timeout, connection failure, and `max_tokens` truncation remain hard errors.
- Structural validation is `valid` when every source paragraph has a non-empty aligned translation, even if Latin residues exist.
- The validation report uses `quality_status: warning` when Latin residues are present.

Continue an existing experiment without deleting its checkpoint:

```bash
legal-tc batch-translate \
  data/annotated \
  data/translations \
  --experiment gpt-oss-20b-zero-english-v1 \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8000/v1 \
  --prompt-profile zero-english-v1 \
  --temperature 0 \
  --context-paragraphs 0 \
  --max-retries 8 \
  --timeout 300 \
  --continue-on-error
```

Do not use `--overwrite`. Existing completed paragraphs are restored, and paragraphs that previously failed only because of Latin residues will now be saved.

Validate after translation:

```bash
legal-tc validate-translations \
  data/annotated \
  data/translations/gpt-oss-20b-zero-english-v1
```

Example successful structural validation with a quality warning:

```json
{
  "status": "valid",
  "quality_status": "warning",
  "missing_paragraphs": 0,
  "empty_translations": 0,
  "source_mismatches": 0,
  "paragraphs_with_latin_letters": 12,
  "latin_letter_count": 38
}
```
