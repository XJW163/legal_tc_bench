# LegalTC-Bench Builder

A reproducible pipeline for building a **legal terminology consistency** benchmark from public SEC EDGAR material-contract exhibits.


## v1.8.1 low-cost blind translation-quality evaluation

`legal-tc llm-evaluate-translation` now evaluates all term-bearing paragraphs plus a deterministic sample of non-term paragraphs by default. It adds JSON response mode, compact prompts, bounded token escalation, two-attempt defaults, exact selection/token accounting, and an explicit all-paragraph override. See `TRANSLATION_EVALUATION_PIPELINE.md`.

The controlled-translation path retains the safe `occurrence-unique-retry-quality-gate-v6` placeholder strategy and the fault-tolerant term-memory builder.

## What it does

1. Downloads quarterly EDGAR master indexes.
2. Samples filings and discovers `EX-10` / agreement-like exhibits.
3. Downloads contract HTML/TXT with caching and SEC-compliant throttling.
4. Converts each contract into paragraph-structured text.
5. Extracts explicitly defined legal terms such as `"Purchaser" means ...`.
6. Links every occurrence of each term to paragraph and character offsets.
7. Optionally translates documents through an OpenAI-compatible Responses API.
8. Produces a conservative occurrence-level review file instead of pretending automatic word alignment is ground truth.
9. Computes DTR, TVR, entropy, and normalized entropy after renderings are annotated.

## Installation

```bash
cd legal_tc_bench
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## 1. Download contracts from SEC EDGAR

The SEC requires a declared user agent containing an identity and contact email. Keep the request rate at or below 10 requests/second; this project defaults to 5.

```bash
export SEC_USER_AGENT="Jiwei Xu xujiwei@gmail.com"
legal-tc download-sec \
  --year 2023 \
  --quarters 1,2,3,4 \
  --max-contracts 100 \
  --output data/raw
```

Windows PowerShell:

```powershell
$env:SEC_USER_AGENT="Jiwei Xu xujiwei@gmail.com"
legal-tc download-sec --year 2023 --max-contracts 100 --output data/raw
```

## 2. Build the source benchmark

```bash
legal-tc build --raw data/raw --output data/dataset --min-occurrences 3
```

Each retained document contains:

```json
{
  "document_id": "...",
  "paragraphs": [{"paragraph_id": 0, "text": "..."}],
  "terms": [{
    "term_id": "T0001",
    "source_term": "Purchaser",
    "definition_paragraph_ids": [4],
    "occurrences": [{
      "paragraph_id": 4,
      "char_start": 1,
      "char_end": 10,
      "surface": "Purchaser",
      "context": "..."
    }]
  }]
}
```

## 3. Batch translation with resume and multi-model experiments

The translation stage uses the OpenAI-compatible **Chat Completions API** and
defaults to the self-hosted vLLM endpoint:

```text
http://10.12.143.51:8000/v1
```

No key is required. The client sends `api_key="EMPTY"` unless
`OPENAI_API_KEY` is explicitly set.

First verify the model ID advertised by vLLM:

```bash
legal-tc check-models \
  --base-url http://10.12.143.51:8000/v1
```

Translate all annotated contracts with one model:

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

Output is isolated by experiment:

```text
data/translations/
└── gpt-oss-120b-baseline/
    ├── <document_id>.translation.json
    └── translation_manifest.json
```

Every successful paragraph is written atomically. After a network, SSH, Python,
or vLLM interruption, rerun the **same command**. Completed paragraphs are
validated against their source hash and skipped. Do not use `--overwrite` when
resuming.

The default `baseline-v1` prompt does not explicitly tell the model to enforce
terminology consistency. This is the recommended profile for comparing the
models' natural document-level consistency. A second profile is available for
an intervention experiment:

```bash
--prompt-profile consistency-aware-v1 --context-paragraphs 2
```

Keep the prompt profile, context setting, temperature, and paragraph boundaries
identical when comparing models.

### Run several models from one experiment matrix

Copy and edit `translation_matrix.example.json`:

```json
{
  "input": "data/annotated",
  "output": "data/translations",
  "defaults": {
    "temperature": 0,
    "prompt_profile": "baseline-v1",
    "context_paragraphs": 0,
    "max_retries": 5,
    "timeout_seconds": 300
  },
  "experiments": [
    {
      "name": "gpt-oss-120b-baseline",
      "model": "gpt-oss-120b",
      "base_url": "http://10.12.143.51:8000/v1"
    },
    {
      "name": "qwen3-32b-baseline",
      "model": "Qwen3-32B",
      "base_url": "http://ANOTHER_HOST:8000/v1"
    }
  ]
}
```

Run:

```bash
legal-tc translate-matrix translation_matrix.json
```

If one vLLM host serves only one loaded model at a time, run each experiment
separately after switching the served model. The experiment directories are
independent, so each model resumes from its own checkpointed output.

Validate an experiment before occurrence-level alignment:

```bash
legal-tc validate-translations \
  data/annotated \
  data/translations/gpt-oss-120b-baseline
```

A valid experiment must have zero missing documents, zero missing paragraphs,
zero empty translations, and zero source mismatches.

A single document can still be translated with:

```bash
legal-tc translate \
  data/annotated/DOCUMENT.json \
  data/translations/test.translation.json \
  --model gpt-oss-120b
```

## 4. Produce the terminology review file

```bash
legal-tc make-review \
  data/dataset/DOCUMENT.json \
  data/translations/DOCUMENT.MODEL.json \
  data/reviews/DOCUMENT.MODEL.json
```

Fill the `rendering` field with the Chinese realization of the source term. Use `label` for a three-way annotation such as:

- `CONSISTENT`
- `ACCEPTABLE_VARIANT`
- `CRITICAL_INCONSISTENCY`

This human/LLM-reviewed stage is deliberate: free-form legal translation makes automatic word alignment too unreliable to serve as benchmark truth.

## 5. Compute metrics

```bash
legal-tc metrics data/reviews/DOCUMENT.MODEL.json data/metrics/DOCUMENT.MODEL.json
```

Metrics:

- **DTR**: dominant translation count / reviewed occurrences.
- **TVR**: `(variant_count - 1) / (occurrence_count - 1)`; 0 means one rendering, 1 means every occurrence differs.
- **Entropy**: Shannon entropy of the rendering distribution.
- **Normalized entropy**: entropy divided by `log2(variant_count)`.

These are descriptive consistency metrics. They do **not** determine whether variants are legally acceptable; that requires the annotation label and term criticality.

## Using Stanford MCC instead

Stanford's bulk MCC contract archive is over 21 GB. Download it from the MCC site, extract it locally, then point `legal-tc build --raw` at the extracted directory. The parser recursively accepts HTML and text files, so the source-building stage is unchanged.

## Recommended benchmark protocol

- Split by **contract**, never by paragraph.
- Keep source contracts and model outputs versioned separately.
- Record model identifier, date, prompt, decoding settings, and chunking policy.
- Double-annotate at least the test set; adjudicate disagreements.
- Report extraction recall separately from consistency scores.
- Weight critical defined terms only after publishing a transparent criticality rubric.

## Limitations

- The regex extractor focuses on explicit defined terms. It will miss implicit legal terminology.
- EDGAR descriptions and exhibit typing are noisy; downloaded candidates still require filtering.
- PDF/OCR is intentionally excluded because SEC exhibits are predominantly HTML/TXT and OCR would introduce uncontrolled noise.
- The pipeline does not claim automatic source-target term alignment is ground truth.

## SEC 原始文件为什么看起来像 HTML 源码？

SEC 有时返回带 `<DOCUMENT><TEXT>...</TEXT></DOCUMENT>` 外层封装的申报附件。这不是空文件，合同正文位于 `<TEXT>` 内。新版解析器会自动剥离 SEC 外层封装，并在下载时额外生成可直接阅读的 `*.htm.txt` 文件。

也可以单独清洗已有文件：

```bash
legal-tc parse "Pasted code.html" contract_clean.txt
```

随后建库：

```bash
legal-tc build --raw data/raw --output data/dataset --min-occurrences 2
```

新版同时支持合同中常见的括号定义形式，如 `This Agreement (“Agreement”)` 和 `February 28, 2023 (“Effective Date”)`，不再只识别 `"Term" means ...`。

## 术语筛选与人工标注

先把自动候选导出为 Excel/WPS 可直接打开的 UTF-8 CSV：

```bash
legal-tc export-term-review data/dataset data/term_review.csv
```

人工重点填写最后四列：

- `term_type`：`PARTY_NAME`、`DEFINED_LEGAL_TERM`、`RIGHT_OR_OBLIGATION`、`LEGAL_CONCEPT`、`COMMERCIAL_TERM`、`DOCUMENT_REFERENCE`、`GENERAL_TERM`
- `criticality`：`CRITICAL`、`IMPORTANT`、`AUXILIARY`、`EXCLUDED`
- `include_in_benchmark`：`true` 或 `false`
- `reviewer_notes`：可选说明

`suggested_*` 列只是预标建议，不是 ground truth。审核完成后回写：

```bash
legal-tc apply-term-review \
  data/dataset \
  data/term_review.csv \
  data/annotated
```

若只是快速 pilot，可接受所有未修改的启发式建议：

```bash
legal-tc apply-term-review \
  data/dataset \
  data/term_review.csv \
  data/annotated \
  --accept-suggestions
```

输出 JSON 保留原始 `terms`，每个术语新增 `annotation`；同时生成仅包含纳入项的 `benchmark_terms`。目录模式还会生成 `annotation_manifest.json`。

## 使用大模型筛选和标注术语

大模型可以直接填写 `term_type`、`criticality` 和 `include_in_benchmark`，并为每条标注保存置信度与理由：

```bash
export OPENAI_API_KEY="..."
legal-tc llm-annotate-terms \
  data/dataset \
  data/term_review.llm.csv \
  --model YOUR_MODEL \
  --batch-size 20 \
  --confidence-threshold 0.8
```

兼容接口可设置：

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
legal-tc llm-annotate-terms data/dataset data/term_review.llm.csv --model local-model
```

输出包括：

- `term_review.llm.csv`：已填写完成的审核表，可直接回写数据集。
- `term_review.llm.csv.audit.json`：模型、置信度、理由、每批原始输出和验证结果。
- 低于 `--confidence-threshold` 的记录会在 `reviewer_notes` 中标记为 `LOW_CONFIDENCE`，便于只抽查不确定项。

回写：

```bash
legal-tc apply-term-review data/dataset data/term_review.llm.csv data/annotated
```

正式论文建议至少人工复核全部低置信度项，并随机抽查 10% 的高置信度项。模型标注应视为银标（silver labels），而不是未经验证的人工金标。

### LLM 标注进度与断点续传

`llm-annotate-terms` 会显示按批次计算的进度条。每个批次成功后立即原子保存三个文件：

- `term_review.llm.csv.checkpoint.json`：该批完整标注、置信度、理由和原始响应，是恢复时的权威数据；
- `term_review.llm.csv`：当前已经完成的部分 CSV，即使任务中断也可查看；
- `term_review.llm.csv.audit.json`：当前审计记录，`status` 为 `in_progress`、`failed` 或 `complete`。

再次执行完全相同的命令时，程序会校验文档 ID、术语 ID 和术语文本，恢复匹配的批次，并从未完成批次继续。若输入数据或 `--batch-size` 改变，签名不匹配的旧批次不会被错误复用。

旧版生成的 `term_review.llm.checkpoint.json` 只保存批次编号、没有保存标注内容，不能安全恢复；新版不会把这种记录当作已完成结果。


## Occurrence-level rendering extraction

After `apply-term-review`, pair the annotated source documents with completed
translations:

```bash
legal-tc make-reviews \
  data/annotated \
  data/translations/gpt-oss-120b-baseline \
  data/reviews/gpt-oss-120b-baseline
```

The command matches files by the `document_id` stored inside JSON rather than
by filename. It writes one `*.review.json` file per document and a
`review_manifest.json` summary. Only `benchmark_terms` are included.

Extract exact Chinese renderings with the self-hosted vLLM endpoint:

```bash
legal-tc llm-extract-renderings \
  data/reviews \
  data/renderings \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 10 \
  --confidence-threshold 0.8 \
  --max-retries 5 \
  --timeout 300
```

No API key is required. The client uses `EMPTY` unless `OPENAI_API_KEY` is set.

The extractor writes output and a checkpoint after every successful batch:

```text
data/renderings/
├── <document>.review.json
├── renderings.audit.json
└── renderings.checkpoint.json
```

Rerun exactly the same command after a network, SSH, vLLM, or Python failure.
Completed occurrences are read from the existing output and skipped. Existing
batch checkpoint data are also reused when the batch signature still matches.

Each occurrence receives:

```json
{
  "rendering": "生效日期",
  "extraction_status": "FOUND",
  "extraction_confidence": 0.97,
  "extraction_rationale": "Exact target span corresponding to Effective Date.",
  "extraction_model": "gpt-oss-120b"
}
```

Allowed statuses are `FOUND`, `OMITTED`, `IMPLICIT`, `AMBIGUOUS`, and
`TRANSLATION_ERROR`. For `FOUND` and `TRANSLATION_ERROR`, the code validates
that `rendering` is an exact substring of the translated paragraph.

Use `--overwrite` only when all completed extraction results must be regenerated.

### Formal metrics for one rendered review file

```bash
legal-tc metrics \
  data/renderings/<document>.review.json \
  data/metrics/<document>.metrics.json
```

The current metrics command reports DTR, TVR, entropy and normalized entropy.
Semantic variant judgment and legal-risk scoring remain a later stage.

## v0.4 strict Chinese legal translation

Use `--prompt-profile strict-zh-v1` to force Chinese translations of contractual roles, defined terms, legal concepts and document names. See `RERUN_STRICT_ZH.md`.

## Zero-English translation profile

Use `--prompt-profile zero-english-v1` to strongly request a fully Chinese translation. Latin residues are recorded in each paragraph as quality warnings, but they do not trigger retries, do not discard the model output, and do not make structural validation fail. This preserves the model's actual output for later comparison. See `RERUN_ZERO_ENGLISH.md`.

## v1.0 rendering cleanup stage

After occurrence renderings have been extracted, run the validation and targeted
minimal-span repair pipeline before final aggregation and judgment. See
[`RENDERING_CLEANUP_PIPELINE.md`](RENDERING_CLEANUP_PIPELINE.md).

New commands:

```text
validate-renderings
llm-repair-renderings
reuse-judgments
```

## Repair robustness (v1.1.0)

A malformed or empty LLM response for one occurrence no longer terminates the
entire `llm-repair-renderings` run. The original rendering is retained with:

```json
{
  "validation_status": "REPAIR_FAILED_RETAINED",
  "include_in_consistency": false,
  "needs_rendering_repair": true,
  "repair_fallback_status": "MODEL_OUTPUT_UNUSABLE"
}
```

Resume the normal run by repeating the same command. After it completes, retry
only failed retained rows with the stronger model or the same endpoint:

```bash
legal-tc llm-repair-renderings \
  data/renderings_validated \
  data/renderings_clean \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --batch-size 1 \
  --max-retries 5 \
  --timeout 300 \
  --retry-failed
```

## Active terminology consistency control (v1.3)

The project now contains an intervention pipeline in addition to evaluation:

```bash
legal-tc build-term-memory data/annotated data/term_memory \
  --aggregated data/aggregated_clean \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1

legal-tc controlled-translate data/annotated data/term_memory data/translations_controlled \
  --experiment qwen3-32b-placeholder-v1 \
  --model Qwen3-32B \
  --base-url http://10.12.143.51:8000/v1 \
  --constraint-mode placeholder
```

See `CONTROLLED_TRANSLATION_PIPELINE.md` for the recommended pilot and ablations.

## Definition-use disambiguation (v1.5)

Before hard placeholder injection, validate whether each string match is actually a use of the contract-defined term:

```bash
legal-tc validate-term-occurrences \
  data/annotated \
  data/occurrences_validated

legal-tc controlled-translate \
  data/annotated \
  data/term_memory \
  data/translations_controlled \
  --experiment gpt-oss-20b-placeholder-v15 \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode placeholder \
  --occurrence-validation data/occurrences_validated
```

Occurrences are classified as `HARD`, `SOFT`, or `NONE`. This prevents ordinary lowercase uses such as `cause` and components of other legal names such as `United States Code` from being replaced by defined-term placeholders. See [`DEFINITION_USE_DISAMBIGUATION_V15.md`](DEFINITION_USE_DISAMBIGUATION_V15.md).

## Context-sensitive compound disambiguation (v1.6)

The v1.6 validator separates external modifiers from components of other named expressions. Possessives, determiners, safe acronym modifiers, and entity-relation phrases can remain `HARD`; job-title, statute-name, entity-full-name, and named-plan components become `NONE`; unresolved compounds remain `SOFT`.

See `COMPOUND_DISAMBIGUATION_V16.md` for the decision table, migration steps, and pilot commands.

## 6. Blind baseline-versus-controlled quality evaluation

After baseline and controlled experiments are complete, run a blind LLM judge:

```bash
export DEEPSEEK_API_KEY="your-key"

legal-tc llm-evaluate-translation \
  data/annotated \
  data/translations/gpt-oss-20b-zero-english-v1 \
  data/translations_controlled/gpt-oss-20b-placeholder-final \
  data/translation_evaluations \
  --experiment gpt-oss-20b-placeholder-deepseek \
  --term-memory data/term_memory_frozen_v1 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key-env DEEPSEEK_API_KEY \
  --controlled-label placeholder \
  --position-control deterministic \
  --workers 16
```

See `TRANSLATION_EVALUATION_PIPELINE.md` for glossary, pilot, bidirectional,
resume, and final OpenAI-judge commands.
