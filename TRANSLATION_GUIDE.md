# Multi-model resumable translation

## Recommended benchmark setup

For a fair comparison across models, keep these identical:

- input: `data/annotated`
- prompt profile: `baseline-v1`
- temperature: `0`
- context paragraphs: `0`
- source paragraph boundaries
- source chunk limit

Only change the model identifier and, when necessary, the endpoint.

## Check the served model ID

```bash
legal-tc check-models --base-url http://10.12.143.51:8000/v1
```

## Run one model

```bash
legal-tc batch-translate \
  data/annotated \
  data/translations \
  --experiment gpt-oss-120b-baseline \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --prompt-profile baseline-v1 \
  --temperature 0 \
  --max-retries 5 \
  --timeout 300
```

The output directory is:

```text
data/translations/gpt-oss-120b-baseline/
```

## Resume after failure

Run the exact same command again. Each successful paragraph is already saved
inside its document translation JSON. The source text hash and experiment
configuration are verified before reuse.

Do not add `--overwrite` when resuming.

## Run another model

Use a different experiment name:

```bash
legal-tc batch-translate \
  data/annotated \
  data/translations \
  --experiment qwen3-32b-baseline \
  --model Qwen3-32B \
  --base-url http://OTHER_HOST:8000/v1 \
  --prompt-profile baseline-v1 \
  --temperature 0
```

If the same vLLM server loads only one model at a time, finish or pause one
experiment, switch the served model, and run the second command. The two output
directories are independent.

## Optional endpoint-specific request fields

Extra fields can be passed to `chat.completions` without changing code:

```bash
--extra-body-json '{"your_endpoint_option": true}'
```

The exact JSON is stored in experiment metadata and therefore forms part of the
resume signature.

## Validate output

```bash
legal-tc validate-translations \
  data/annotated \
  data/translations/gpt-oss-120b-baseline
```

Only proceed when the report contains no missing documents, missing paragraphs,
empty translations, or source mismatches.

## Run an experiment matrix

Edit `translation_matrix.example.json`, enable only models that are currently
reachable, then run:

```bash
legal-tc translate-matrix translation_matrix.example.json
```
