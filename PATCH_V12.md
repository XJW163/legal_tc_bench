# v1.2 parallel-write safety patch

This patch fixes `FileNotFoundError` races such as:

- `repair.checkpoint.json.tmp -> repair.checkpoint.json`
- `<document>.review.json.tmp -> <document>.review.json`

## Cause

Two processes were writing the same output/checkpoint and both used the same fixed
`.tmp` filename. One process renamed the temporary file while the other still
expected it to exist.

## Changes

1. Every atomic write now uses a process-unique temporary file in the same directory.
2. `llm-repair-renderings` takes an exclusive lock for its output directory.
3. Different experiment output directories can still run in parallel.
4. A duplicate worker for the same experiment fails immediately with a clear lock error.
5. Existing outputs/checkpoints remain compatible and can be resumed.

## Safe parallelism

Run at most one process per experiment/output directory. Before restarting, inspect:

```bash
pgrep -af 'legal-tc llm-repair-renderings'
```

Then rerun the experiment that failed. Completed batches are restored.

## Recovery after the observed crash

1. Stop duplicate workers targeting the same experiment/output.
2. Install v1.2.
3. Keep `data/renderings_clean`; do not delete checkpoints.
4. Rerun only failed experiments or rerun the parallel launcher. Existing completed
   batches are restored.
