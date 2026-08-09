# vLLM gpt-oss-120b fix

The LLM term annotator now uses `/v1/chat/completions` instead of the OpenAI
Responses API. This is required for the vLLM service at:

- Base URL: `http://10.12.143.51:8000/v1`
- Model ID: `gpt-oss-120b`
- API key: not required (`EMPTY` is used internally)

Reinstall after extracting:

```bash
pip install -e .
```

Run:

```bash
legal-tc llm-annotate-terms \
  data/dataset \
  data/output/term_review.csv \
  --batch-size 5
```

The model and base URL now have the correct defaults, so they do not need to
be specified.
