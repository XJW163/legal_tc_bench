# v1.8.1 Low-Cost Evaluation Patch

- Added `all`, `term-bearing`, and `term-bearing-plus-sample` evaluation scopes.
- Added deterministic non-term sampling with configurable rate and seed.
- Enabled OpenAI-compatible JSON mode by default, with automatic unsupported-mode fallback.
- Reduced default output budget from 1200 to 600 tokens and default attempts from 5 to 2.
- Replaced repeated truncation retries with at most one token-budget escalation.
- Shortened judge rationale and compacted term-scope evidence.
- Added selection counts to document outputs, manifests, and summaries.
- Treats zero matched input documents as an error instead of a successful empty run.
