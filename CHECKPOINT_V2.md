# Checkpoint v2 fix

The previous implementation only recorded completed batch IDs. It did not
persist the annotations themselves, and the CSV/audit files were written only
after the full job ended. Therefore it could not safely resume.

Version 2 now:

- uses `httpx.Client(trust_env=False)` for the internal vLLM endpoint;
- displays a tqdm batch progress bar;
- saves complete per-batch annotations and audit data;
- writes the partial CSV and audit JSON after every successful batch;
- uses atomic temporary-file replacement;
- restores matching batches and starts the progress bar at the restored count;
- ignores legacy batch-only checkpoints that cannot reconstruct annotations.

Checkpoint file naming:

    OUTPUT.csv.checkpoint.json
