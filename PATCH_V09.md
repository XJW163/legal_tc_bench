# v0.9 judge-consistency reliability patch

This patch addresses failures where a local vLLM judge returns an empty final answer or
valid-but-incomplete JSON after a long run.

## Changes

- Empty `message.content` is detected explicitly and reported with `finish_reason`.
- Non-standard `reasoning_content` is used as a last-resort JSON source.
- Embedded JSON is recovered with `JSONDecoder.raw_decode` rather than brittle first/last-brace slicing.
- Minimum judge generation budget increased from 1,536 to 4,096 tokens.
- Malformed multi-group outputs still fall back to per-group and then per-variant requests.
- A persistently malformed/empty *single-variant* answer is recorded as `UNRESOLVED`
  (`fallback_status=MODEL_OUTPUT_UNUSABLE`) instead of aborting the complete experiment.
- Real transport/server failures still stop the run, preserving checkpoints for a later retry;
  they are not silently converted into `UNRESOLVED` labels.

## Resume an interrupted run

Do not delete `data/judgments` and do not use `--overwrite`.

```bash
pip install -e .

legal-tc llm-judge-consistency \
  data/aggregated \
  data/judgments \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --batch-size 2 \
  --confidence-threshold 0.8 \
  --max-retries 5 \
  --timeout 300
```

Previously saved groups are loaded from the existing judgment JSON and checkpoint files.
Changing `--batch-size` does not force completed groups to be judged again.
