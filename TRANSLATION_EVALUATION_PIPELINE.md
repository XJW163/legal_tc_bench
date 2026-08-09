# Low-Cost Translation Quality Evaluation (v1.8.1)

`legal-tc llm-evaluate-translation` performs a blind paragraph-level comparison
between a baseline translation and a controlled translation.

## v1.8.1 cost controls

The command now defaults to a targeted evaluation scope:

- evaluate every paragraph containing at least one benchmark term;
- evaluate a deterministic 10% sample of non-term-bearing paragraphs;
- use `response_format={"type":"json_object"}` when supported;
- request at most 600 output tokens initially;
- make at most two attempts by default;
- when an answer is truncated, double the token limit once rather than repeating
  the same doomed request five times;
- shorten the rationale to at most 60 Chinese characters;
- record candidate and selected paragraph counts in every document and summary.

The sampling decision is stable for the same document ID, paragraph ID, rate,
and seed. Baseline, glossary, and placeholder experiments therefore use the same
non-term sample when the same `--sample-seed` is supplied.

## Evaluation scopes

```text
all
  Evaluate every aligned paragraph.

term-bearing
  Evaluate only paragraphs containing benchmark-term occurrences.

term-bearing-plus-sample
  Evaluate all term-bearing paragraphs plus a deterministic sample of other
  paragraphs. This is the low-cost default.
```

Relevant options:

```bash
--evaluation-scope term-bearing-plus-sample
--non-term-sample-rate 0.10
--sample-seed 2026
--max-tokens 600
--max-retries 2
--max-token-escalations 1
```

JSON mode is enabled by default. Use `--disable-json-mode` only for a server that
does not support OpenAI-compatible JSON output. The runner automatically retries
without JSON mode when the server explicitly rejects `response_format`.

## DeepSeek low-cost development evaluation

Set the API key once:

```bash
export DEEPSEEK_API_KEY="your-key"
```

Evaluate baseline versus glossary:

```bash
PYTHONUNBUFFERED=1 nohup legal-tc llm-evaluate-translation \
  data/annotated \
  data/translations/gpt-oss-20b-zero-english-v2 \
  data/translations_controlled/gpt-oss-20b-glossary-final \
  data/translation_evaluations \
  --experiment gpt-oss-20b-glossary-deepseek-v4-flash-lowcost \
  --term-memory data/term_memory_frozen_v1 \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --controlled-label glossary \
  --position-control deterministic \
  --evaluation-scope term-bearing-plus-sample \
  --non-term-sample-rate 0.10 \
  --sample-seed 2026 \
  --extra-body-json '{"thinking":{"type":"disabled"}}' \
  --max-tokens 600 \
  --max-retries 2 \
  --max-token-escalations 1 \
  --workers 16 \
  --timeout 300 \
  > data/logs/gpt-oss-20b-glossary-deepseek-v4-flash-lowcost.log 2>&1 &
```

Placeholder uses the same fixed sample:

```bash
PYTHONUNBUFFERED=1 nohup legal-tc llm-evaluate-translation \
  data/annotated \
  data/translations/gpt-oss-20b-zero-english-v2 \
  data/translations_controlled/gpt-oss-20b-placeholder-final \
  data/translation_evaluations \
  --experiment gpt-oss-20b-placeholder-deepseek-v4-flash-lowcost \
  --term-memory data/term_memory_frozen_v1 \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --controlled-label placeholder \
  --position-control deterministic \
  --evaluation-scope term-bearing-plus-sample \
  --non-term-sample-rate 0.10 \
  --sample-seed 2026 \
  --extra-body-json '{"thinking":{"type":"disabled"}}' \
  --max-tokens 600 \
  --max-retries 2 \
  --max-token-escalations 1 \
  --workers 16 \
  --timeout 300 \
  > data/logs/gpt-oss-20b-placeholder-deepseek-v4-flash-lowcost.log 2>&1 &
```

View progress written by `tqdm` through `nohup`:

```bash
tail -f data/logs/gpt-oss-20b-glossary-deepseek-v4-flash-lowcost.log | tr '\r' '\n'
```

## Pilot

```bash
--max-documents 1 --max-paragraphs-per-document 10 --workers 2
```

The paragraph cap is applied after scope selection.

## Full evaluation when required

For an explicit all-paragraph experiment:

```bash
--evaluation-scope all
```

This should not be used for routine prompt development because it is much more
expensive.

## Output

```text
data/translation_evaluations/<experiment>/
├── <document_id>.translation-evaluation.json
├── translation_evaluation_manifest.json
├── translation_evaluation_summary.json
└── translation_evaluation_summary.csv
```

The summary contains:

- candidate, term-bearing, non-term-bearing, and selected paragraph counts;
- baseline, controlled, and tie win counts and rates;
- mean semantic fidelity, legal accuracy, terminology compliance, fluency, and
  overall quality scores;
- controlled-minus-baseline deltas;
- critical-error rates;
- separate term-bearing and sampled non-term-bearing results;
- exact successful and failed-attempt token usage.

Rerun the same command without `--overwrite` to resume. Completed judgments are
restored. Failed paragraphs are retried, and one failure never terminates the
full experiment.
