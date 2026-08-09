# 使用严格中文法律翻译 Prompt 重新运行

新 Prompt 配置名：`strict-zh-v1`

它明确要求除真实专名、标准缩写、网址、邮箱、法规编号、条款号和 `[***]` 外，所有英文合同角色、定义术语、法律概念和文件名称必须翻译为中文。

## 不覆盖旧实验

旧结果目录可以保留，例如：

```text
data/translations/gpt-oss-120b-baseline/
```

新结果使用不同实验名：

```text
data/translations/gpt-oss-120b-strict-zh-v1/
```

## 运行 gpt-oss-120b

```bash
legal-tc batch-translate \
  data/annotated \
  data/translations \
  --experiment gpt-oss-120b-strict-zh-v1 \
  --model gpt-oss-120b \
  --base-url http://10.12.143.51:8000/v1 \
  --prompt-profile strict-zh-v1 \
  --temperature 0 \
  --context-paragraphs 0 \
  --max-retries 5 \
  --timeout 300
```

网络中断后直接重新执行同一命令，已完成段落会自动跳过。

## 运行 Qwen3

Qwen3 关闭 thinking：

```bash
legal-tc batch-translate \
  data/annotated \
  data/translations \
  --experiment qwen3-8b-strict-zh-v1 \
  --model Qwen3-8B \
  --base-url http://10.12.143.51:8000/v1 \
  --prompt-profile strict-zh-v1 \
  --temperature 0 \
  --context-paragraphs 0 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --max-retries 5 \
  --timeout 300
```

## 多模型配置

编辑 `translation_matrix.strict-zh-v1.json`，每次只把当前 vLLM 已加载模型的 `enabled` 改为 `true`，其他保持 `false`：

```bash
legal-tc translate-matrix translation_matrix.strict-zh-v1.json
```

不要为不同 Prompt 复用旧实验名，否则程序会检测到实验签名变化并拒绝混用断点。
