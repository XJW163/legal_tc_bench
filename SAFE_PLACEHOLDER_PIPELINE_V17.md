# LegalTC v1.7.0：安全占位符重试、质量门控与选择性语义验证

## 目标

v1.6.1 的占位符库存检查只能证明每个 token 出现一次，不能证明 token 位于正确语义位置。人工审查发现，旧的 LLM 局部修复可能产生：

- 为补齐 token 而新增原文不存在的句子；
- `《国内税收法典【术语0006】》`，恢复后形成双重法典名称；
- `雇员【术语0001】`，恢复后形成“雇员执行官”；
- token 数量正确但附着在错误谓词或句子位置。

v1.7.0 将默认流程改为：

```text
首次从受保护源文翻译
  → token 库存检查
  → 占位符上下文污染检查
  → 恢复规范译法
  → 恢复后异常检查
  → 对重试成功候选进行选择性语义验证
  → 通过则保存；否则从源文重新生成
  → 达到最大次数后 glossary fallback
```

默认不再编辑已有中文译文来补 token。

## 新增参数

```text
--placeholder-repair-policy retry_then_fallback | llm_repair
--placeholder-semantic-verifier off | retry | all
--disable-placeholder-quality-gate
```

默认值：

```text
placeholder_repair_policy = retry_then_fallback
placeholder_semantic_verifier = retry
placeholder_quality_gate = enabled
```

`llm_repair` 仅保留为消融实验，不建议用于正式结果。

## 确定性质量门控

恢复前检测：

- `PLACEHOLDER_WRAPPER_CONTAMINATION`
- `REDUNDANT_TITLE_MARK_WRAPPER`
- 缺失、重复、未知 placeholder

恢复后检测：

- `NESTED_TITLE_MARKS`
- `DUPLICATED_CANONICAL_RENDERING`
- `RISKY_AND_CANONICAL_ADJACENCY`
- 未恢复 placeholder
- `POSSIBLE_PERIOD_SUFFIX_REDUNDANCY`（仅警告）

## 选择性语义 verifier

默认 `retry`：只验证第 2 次及之后成功通过确定性门控的候选，避免为所有首次成功段落增加一次模型调用。

Verifier 检查：

- unsupported addition
- meaning omission
- unnatural term attachment
- broken legal logic
- protected-term semantics

Verifier 拒绝后不会继续编辑该中文译文，而是重新从受保护英文生成；达到重试上限后进入 glossary fallback。

## 新状态与统计字段

段落策略：

```text
placeholder_exact
placeholder_retry_exact
placeholder_repaired        # 仅 legacy llm_repair 模式
conditional_glossary
uncontrolled
glossary_fallback
```

新增统计：

```text
generation_retries
glossary_compliance_retries
placeholder_regeneration_attempts
placeholder_legacy_repair_attempts
placeholder_quality_gate_rejections
placeholder_inventory_rejections
placeholder_context_rejections
placeholder_restored_rejections
placeholder_semantic_verifier_calls
placeholder_semantic_rejections
placeholder_retry_success_paragraphs
```

旧字段 `placeholder_retries` 保留为兼容别名，但正式分析应使用新字段。

## 运行

v1.6.1 的 occurrence validation 可以继续使用，因为本次未改变 occurrence validator：

```bash
legal-tc controlled-translate \
  data/annotated \
  data/term_memory_pilot \
  data/translations_controlled \
  --experiment gpt-oss-20b-placeholder-v17-pilot \
  --model gpt-oss-20b \
  --base-url http://10.12.143.51:8001/v1 \
  --constraint-mode placeholder \
  --occurrence-validation data/occurrences_validated_v161_pilot \
  --placeholder-repair-policy retry_then_fallback \
  --placeholder-semantic-verifier retry \
  --context-paragraphs 2 \
  --workers 2 \
  --max-retries 5 \
  --timeout 600 \
  --max-documents 5
```

纯确定性版本：

```bash
--placeholder-semantic-verifier off
```

对所有首次成功结果也调用 verifier：

```bash
--placeholder-semantic-verifier all
```

旧修复方式消融：

```bash
--placeholder-repair-policy llm_repair \
--placeholder-semantic-verifier retry
```

## 推荐评价方式

正式报告时区分：

```text
placeholder inventory satisfaction
placeholder deterministic-gate pass
placeholder semantic-verifier pass
retry success
fallback success
```

不要再把 token 数量满足直接称为语义层面的 hard-control success。
