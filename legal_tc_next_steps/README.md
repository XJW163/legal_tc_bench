# LegalTC 后续实验脚本包

本脚本包对应当前 `gpt-oss-20b` 三组翻译：

- Baseline：`data/translations/gpt-oss-20b-zero-english-v2`
- Glossary：`data/translations_controlled/gpt-oss-20b-glossary-final`
- Placeholder：`data/translations_controlled/gpt-oss-20b-placeholder-final`

以及已经完成的两组全量 DeepSeek Judge 结果。

## 0. 安装与配置

在项目根目录执行：

```bash
cd ~/Downloads/legal_tc_bench
unzip legal_tc_next_steps.zip
cp legal_tc_next_steps/analysis_config.example.json analysis_config.json
```

检查并修改：

```bash
open analysis_config.json
```

尤其确认以下目录真实存在：

```text
data/annotated
data/term_memory_frozen_v1
data/translations/gpt-oss-20b-zero-english-v2
data/translations_controlled/gpt-oss-20b-glossary-final
data/translations_controlled/gpt-oss-20b-placeholder-final
```

涉及 DeepSeek API 的步骤前设置密钥：

```bash
read -s DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
echo
```

所有付费命令都显式写明 `--evaluation-scope`。不要添加 `--overwrite`。

---

## 1. 审计当前全量结果与失败段落

不调用 API：

```bash
python legal_tc_next_steps/00_audit_current_evaluations.py \
  --config analysis_config.json
```

输出：

```text
data/analysis/gpt-oss-20b/00_current_evaluation_audit.json
data/analysis/gpt-oss-20b/00_failed_paragraphs.csv
data/analysis/gpt-oss-20b/00_content_mismatches.csv
```

预期：

```text
Glossary successful: 19033
Placeholder successful: 19034
Common successful: 19033
Content mismatches: 0
```

---

## 2. 计算确定性术语指标

### 2.1 精确字符串指标（不调用 API）

```bash
python legal_tc_next_steps/01_deterministic_term_metrics.py \
  --config analysis_config.json
```

输出：

```text
01_deterministic_term_metrics.csv
01_deterministic_term_occurrences.csv
```

包含：

- canonical presence micro
- canonical + allowed variants presence micro
- risky variant presence
- strict term rate
- strict document rate
- term-bearing paragraph compliance

这些是**字符串存在性指标**，不是语义对齐指标。

### 2.2 可选：运行项目已有的 DTR/SLCR/LIR 语义术语流水线

这一步会调用 `config/final_experiment.json` 中配置的本地 Judge，通常是本地 `gpt-oss-120b`：

```bash
bash legal_tc_next_steps/01b_run_semantic_term_pipeline.sh \
  analysis_config.json
```

输出位于：

```text
data/analysis/gpt-oss-20b/semantic_term_metrics/
```

其中包含 DTR、SLCR、LIR、Omission rate、WLCS 等结果。

---

## 3. Glossary 与 Placeholder 直接盲测

### 3.1 构造固定 700 段样本

默认包含 500 个含术语段落和 200 个非术语段落：

```bash
python legal_tc_next_steps/02_prepare_direct_ab_sample.py \
  --config analysis_config.json
```

输出：

```text
data/analysis/gpt-oss-20b/direct_ab_sample/
```

### 3.2 启动直接 A/B Judge

```bash
bash legal_tc_next_steps/03_run_direct_ab.sh \
  analysis_config.json
```

该命令明确使用：

```text
--baseline-label glossary
--controlled-label placeholder
--position-control bidirectional
--evaluation-scope all
```

这里的 `all` 仅指预先构造的 700 段样本，不是原始 19,041 段全量数据。

查看日志：

```bash
EXP=gpt-oss-20b-glossary-vs-placeholder-direct-700
tail -f "data/logs/$EXP.log" | tr '\r' '\n'
```

### 3.3 分析直接 A/B 结果

```bash
python legal_tc_next_steps/04_analyze_direct_ab.py \
  --config analysis_config.json
```

输出：

```text
04_direct_ab_summary.json
04_direct_ab_significance.csv
```

其中 effect 定义为：

```text
Glossary score − Placeholder score
```

正值表示 Glossary 更好。

---

## 4. 两名人工评审验证

### 4.1 生成盲审表

直接 A/B 完成后执行：

```bash
python legal_tc_next_steps/05_prepare_human_review.py \
  --config analysis_config.json \
  --sample-size 200 \
  --term-fraction 1.0
```

输出：

```text
data/analysis/gpt-oss-20b/human_review/reviewer_1.csv
data/analysis/gpt-oss-20b/human_review/reviewer_2.csv
data/analysis/gpt-oss-20b/human_review/BLIND_KEY_DO_NOT_SHARE.csv
```

将前两个文件分别交给两名评审者，不要把 key 文件发给评审者。

填写要求：

```text
preferred: A / B / TIE
term_correct_A/B: 1–5
legal_accuracy_A/B: 1–5
fluency_A/B: 1–5
critical_error_A/B: 0 或 1
```

### 4.2 统计人工一致性与 Human–LLM agreement

```bash
python legal_tc_next_steps/06_analyze_human_review.py \
  data/analysis/gpt-oss-20b/human_review/reviewer_1.csv \
  data/analysis/gpt-oss-20b/human_review/reviewer_2.csv \
  --config analysis_config.json
```

输出：

```text
06_human_review_summary.json
06_human_review_items.csv
```

包括：

- raw reviewer agreement
- Cohen's κ
- 人工共识 Glossary/Placeholder 胜率
- Human consensus 与 LLM Judge 的一致率
- 三项人工评分的 Glossary−Placeholder 均值

---

## 5. 其余模型固定低成本评测

### 5.1 编辑模型目录矩阵

```bash
cp legal_tc_next_steps/model_matrix.example.json model_matrix.json
open model_matrix.json
```

逐项核对真实目录，并将暂时不跑的模型设置为：

```json
"enabled": false
```

### 5.2 生成命令

```bash
python legal_tc_next_steps/07_generate_lowcost_commands.py \
  --config analysis_config.json \
  --matrix model_matrix.json \
  --non-term-sample-rate 0.10 \
  --sample-seed 2026
```

先检查生成的脚本：

```bash
cat data/analysis/gpt-oss-20b/07_run_remaining_models_lowcost.sh
```

确认后运行：

```bash
bash data/analysis/gpt-oss-20b/07_run_remaining_models_lowcost.sh
```

每个命令都显式使用：

```text
--evaluation-scope term-bearing-plus-sample
--non-term-sample-rate 0.10
--sample-seed 2026
```

### 5.3 从当前 gpt-oss-20B 全量结果离线提取相同低成本子集

不调用 API：

```bash
python legal_tc_next_steps/07_extract_lowcost_subset_from_full.py \
  --config analysis_config.json \
  --non-term-sample-rate 0.10 \
  --sample-seed 2026
```

这样主表中所有模型使用同一抽样规则，而 gpt-oss-20B 不需要重跑。

---

## 6. 闭环组件消融

当前消融脚本针对 Placeholder 闭环中的以下组件：

```text
full
no-semantic-verifier
no-retry
no-deterministic-quality-gate
legacy-llm-repair
```

### 6.1 选择代表性文档

默认选 20 个文档，并覆盖低、中、高术语复杂度：

```bash
python legal_tc_next_steps/08_prepare_ablation_docs.py \
  --config analysis_config.json \
  --documents 20
```

### 6.2 运行翻译消融

确认本地 `gpt-oss-20b` 服务可用后：

```bash
bash legal_tc_next_steps/08_run_ablations.sh \
  analysis_config.json \
  config/final_experiment.json \
  gpt-oss-20b
```

消融翻译单独写入：

```text
data/analysis/gpt-oss-20b/ablations/translations_controlled/
```

不会覆盖正式翻译结果。

### 6.3 生成并运行消融 Judge 命令

```bash
python legal_tc_next_steps/08_generate_ablation_judge_commands.py \
  --config analysis_config.json \
  --model-key gpt-oss-20b

cat data/analysis/gpt-oss-20b/ablations/run_ablation_judges.sh
bash data/analysis/gpt-oss-20b/ablations/run_ablation_judges.sh
```

消融 Judge 只评含术语段落：

```text
--evaluation-scope term-bearing
```

---

## 7. 错误类型与案例分析

不调用 API：

```bash
python legal_tc_next_steps/09_export_error_cases.py \
  --config analysis_config.json \
  --per-category 50
```

输出：

```text
data/analysis/gpt-oss-20b/error_analysis/selected_qualitative_cases.csv
```

自动导出：

- Glossary 整体优势案例
- Placeholder 优势案例
- Glossary 术语合规优势案例
- Placeholder 流畅性受损案例
- Placeholder critical error 更多的案例

从每类人工挑选 3–5 个放入论文案例表。

---

## 8. 效率与成本表

在 `analysis_config.json` 中填写真实费用：

```json
"costs_cny": {
  "glossary_judge_actual": 26.14,
  "placeholder_judge_actual": 你的实际费用
}
```

然后执行：

```bash
python legal_tc_next_steps/10_efficiency_cost_table.py \
  --config analysis_config.json
```

输出：

```text
10_efficiency_cost_table.csv
10_efficiency_cost_table.json
```

包含：

- 翻译耗时
- Judge token
- 每段 token
- 实际或估算费用
- generation retries
- verifier calls
- fallback 数量
- unresolved 数量

---

## 9. 生成论文表格

前面结果完成到哪一步，本脚本就纳入到哪一步：

```bash
python legal_tc_next_steps/11_make_paper_tables.py \
  --config analysis_config.json
```

输出：

```text
data/analysis/gpt-oss-20b/11_main_llm_judge_results.csv
data/analysis/gpt-oss-20b/11_glossary_vs_placeholder_significance.csv
data/analysis/gpt-oss-20b/11_paper_results.md
```

---

## 推荐执行顺序

```bash
python legal_tc_next_steps/00_audit_current_evaluations.py --config analysis_config.json
python legal_tc_next_steps/01_deterministic_term_metrics.py --config analysis_config.json
python legal_tc_next_steps/02_prepare_direct_ab_sample.py --config analysis_config.json
bash legal_tc_next_steps/03_run_direct_ab.sh analysis_config.json
python legal_tc_next_steps/04_analyze_direct_ab.py --config analysis_config.json
python legal_tc_next_steps/05_prepare_human_review.py --config analysis_config.json
python legal_tc_next_steps/09_export_error_cases.py --config analysis_config.json
python legal_tc_next_steps/10_efficiency_cost_table.py --config analysis_config.json
python legal_tc_next_steps/11_make_paper_tables.py --config analysis_config.json
```

人工评审、其余模型和消融可以并行推进。
