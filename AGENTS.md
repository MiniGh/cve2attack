# CVE → ATT&CK Stage-1 项目结构与命令行说明

本文档介绍项目的目标、代码组织、数据流、实验配置、文件格式和命令行用法。它同时面向项目使用者和需要理解代码的 Agent。

## 1. 项目目标与范围

本项目实现 CVE → MITRE ATT&CK Technique 映射流程的第一阶段：根据 CVE 信息生成一个按相关性排序的 ATT&CK Technique 候选集。候选集会交给映射流程的后续阶段继续判断或筛选。

项目中需要区分两种第一阶段方法：

- `main` 分支保存分层映射方法。
- `new_method` 分支保存基于文本改写与向量检索的候选生成方法。

两者是第一阶段的两种独立实现，不是前后相接的两个阶段。本文档描述的是重构后的 `new_method` 方法。

当前选定方案为 V3a：

1. 读取 CVE 描述和 CWE 信息。
2. 使用 `sec-i1` 将漏洞描述改写为 ATT&CK 风格的攻击者动作描述。
3. 使用 `basel/ATTACK-BERT` 分别编码 CVE 查询文本和 ATT&CK Technique 文本。
4. 通过归一化向量点积计算余弦相似度。
5. 返回相似度最高的顶层 Technique 候选。

项目还保留了其他实验方案，用于比较原始描述、procedure examples 和结构化知识链等因素的影响。

## 2. 整体数据流

```text
实验 YAML
   │
   ▼
选择 CVE 集合 ──────────────── benchmark CVE / 全部 Enterprise CVE
   │
   ▼
构造查询文本 ───────────────── 原始 CVE 描述 / LLM rewrite cache
   │
   ▼
构造 Technique 文档 ────────── 名称 + 描述 [+ procedure examples]
   │
   ▼
ATTACK-BERT 编码与相似度排序
   │
   ▼
候选 Technique 列表
   │
   ├── 可选：CWE → CAPEC → ATT&CK 结构化链融合
   │
   ▼
规范化候选 JSONL
   │
   ├── 在指定 benchmark 上计算指标
   │
   ▼
runs/<run_id>/
```

完整流程由 `src/cve2attack/pipeline.py` 组织。各处理步骤分别实现在 `data`、`rewrite`、`retrieval`、`fusion` 和 `evaluation` 子包中。

## 3. 项目目录

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── experiments/
│   ├── v1_raw_attackbert.yaml
│   ├── v2_raw_procedures.yaml
│   ├── v3a_llm_rewrite.yaml
│   ├── v3b_llm_rewrite_procedures.yaml
│   └── v4_rewrite_chain.yaml
├── src/cve2attack/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── schemas.py
│   ├── data/
│   │   └── loaders.py
│   ├── domain/
│   │   └── classifier.py
│   ├── rewrite/
│   │   ├── ollama.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── generator.py
│   │   └── technique_kb.py
│   ├── fusion/
│   │   └── structured_chain.py
│   └── evaluation/
│       ├── metrics.py
│       └── report.py
├── tests/
├── data/
│   ├── benchmarks/
│   ├── knowledge/
│   ├── raw/
│   └── derived/
├── docs/
├── runs/
├── comparisons/
└── archive/
```

根目录文件的作用：

- `AGENTS.md`：项目结构和命令行参考，也就是本文档。
- `README.md`：较简短的项目介绍和快速运行示例。
- `pyproject.toml`：Python 包信息、依赖、可执行命令入口和测试配置。
- `experiments/`：可复现的实验定义。这里只保存方法和参数，不保存实验结果。
- `src/cve2attack/`：当前生效的项目代码。
- `tests/`：不依赖完整数据运行的快速单元测试。
- `docs/experiment_history.md`：各实验方案的背景、结果和取舍记录。
- `runs/`：每次候选生成的独立输出目录，不进入 Git。
- `comparisons/`：多个 run 的统一评估结果，不进入 Git。
- `archive/`：重构前代码、旧 TF-IDF 实现和本地历史材料；不属于当前运行路径。

## 4. Python 代码结构

### 4.1 命令入口：`cli.py` 与 `__main__.py`

`src/cve2attack/cli.py` 定义全部命令行子命令和参数，并把命令分发给对应模块。

它提供以下命令：

- `validate`：读取并校验实验 YAML。
- `inspect`：在不加载嵌入模型的情况下检查输入与数据覆盖。
- `classify-domain`：重新生成 CVE 的 Enterprise、ICS、Mobile 分类。
- `rewrite`：调用兼容 Ollama 的服务生成 CVE 改写缓存。
- `run`：执行完整候选生成和评估流程。
- `compare`：在同一 benchmark 上重新比较一个或多个 run。

`src/cve2attack/__main__.py` 使项目可以通过 `python -m cve2attack` 启动。安装项目后，也可以使用 `cve2attack-stage1` 命令，两种入口的功能相同。

### 4.2 配置：`config.py`

`src/cve2attack/config.py` 负责：

- 确定项目根目录 `PROJECT_ROOT`。
- 定义实验配置的默认值 `DEFAULTS`。
- 读取 YAML 并递归合并默认配置。
- 校验 input、query 和 fusion 的 strategy 名称。
- 将配置中的相对路径解析为项目根目录下的路径。

当前支持：

- `input.mode`：`benchmark`、`full_enterprise`。
- `query.strategy`：`raw_description`、`rewrite_cache`。
- `fusion.strategy`：`none`、`structured_chain`。

### 4.3 流程编排：`pipeline.py`

`src/cve2attack/pipeline.py` 是一次实验运行的总入口。

主要函数：

- `select_input_ids`：根据配置选择需要处理的 CVE ID。
- `build_queries`：读取原始 CVE 描述或 rewrite cache，生成 `CVE ID → 查询文本`。
- `make_run_id`：用时间和实验名生成默认运行 ID。
- `run_experiment`：执行 Technique 文档构造、嵌入、检索、可选融合、结果写入和评估。

运行开始时会先创建 `manifest.json`，状态为 `running`。成功后状态变为 `complete`；发生异常时状态变为 `failed`，并在 manifest 中记录错误类型和消息。

### 4.4 数据读取：`data/loaders.py`

`src/cve2attack/data/loaders.py` 集中处理项目数据格式：

- `iter_jsonl`：逐行读取 JSONL，并在格式错误时报告文件和行号。
- `benchmark_truth`：读取 benchmark 的 CVE → Technique 标注；默认将子技术上卷到父 Technique。
- `candidate_records`：读取新格式或历史格式的候选结果。
- `write_candidate_records`：按 CVE 年份写出候选 JSONL。
- `enterprise_cve_ids`：从 domain mapping 中选择 Enterprise CVE。
- `CVERepository`：按年份延迟加载原始 CVE 文件，并提供描述和 CWE 查询。

`CVERepository` 对每个年份文件最多读取一次，避免反复解析体积较大的原始数据。

### 4.5 数据对象与兼容格式：`schemas.py`

`src/cve2attack/schemas.py` 定义统一候选格式：

- `TechniqueCandidate`：一个 Technique 候选及其分数、来源和元数据。
- `CandidateRecord`：一个 CVE 的完整候选列表。
- `parent_technique_id`：规范化 Technique ID，并把 `Txxxx.xxx` 上卷到 `Txxxx`。
- `parse_candidate`：解析规范格式和两种历史候选格式。

新运行只写统一 schema，但读取器仍兼容历史结果中的 Technique 字符串列表和带分数字典列表。

### 4.6 Domain 分类：`domain/classifier.py`

`src/cve2attack/domain/classifier.py` 根据 CVE 描述、来源标识和 CPE 中的关键词，将 CVE 分类为：

- `ICS`
- `Mobile`
- `Enterprise`

优先级为 ICS、Mobile、Enterprise。没有命中 ICS 或 Mobile 关键词的记录默认归入 Enterprise。分类结果按年份写入 `data/derived/domain_mapping/`。

### 4.7 LLM 改写：`rewrite/`

`src/cve2attack/rewrite/pipeline.py` 负责：

- 从 `data/knowledge/cwe.xml` 读取 CWE 名称和描述。
- 将 CVE 描述与 CWE 信息组合成提示词。
- 要求模型输出 3–5 句 ATT&CK 风格的攻击者动作描述。
- 并行处理 CVE，并定期将结果安全写入 JSON cache。
- 跳过已有缓存，统计成功、失败、缺少描述等数量。

`src/cve2attack/rewrite/ollama.py` 是一个小型 HTTP 客户端。它向实验配置中的 generate endpoint 发送非流式请求，支持超时、重试和指数等待。

### 4.8 Technique 语料：`retrieval/technique_kb.py`

`src/cve2attack/retrieval/technique_kb.py` 从 `enterprise-attack.json` 构建检索语料。

它会：

- 读取 STIX `attack-pattern` 对象。
- 排除子技术、已撤销和已弃用的 Technique。
- 提取 MITRE external ID、名称、描述、tactic 和 STIX ID。
- 根据配置决定是否把 relationship 中的 procedure examples 加入 Technique 文本。
- 对 procedure 文本去重，并按 `procedure_char_limit` 截断。

最终每个 Technique 被表示为一个 `TechniqueDocument`。

### 4.9 嵌入与候选检索：`retrieval/embedder.py`、`generator.py`

`src/cve2attack/retrieval/embedder.py` 提供：

- 通用 `Embedder` 接口。
- 基于 `sentence-transformers` 的 `SentenceTransformerEmbedder`。
- `l2_normalize` 向量归一化函数。

模型在创建 `SentenceTransformerEmbedder` 时才加载，因此 schema、评估和配置命令不需要加载 Torch 或模型。

`src/cve2attack/retrieval/generator.py` 负责：

- 根据模型、ATT&CK 数据文件和 Technique 文本设置计算 embedding cache key。
- 复用或生成 Technique embedding cache。
- 分批编码 CVE 查询文本。
- 通过归一化向量点积计算余弦相似度。
- 使用 Top-K 排序生成 `CandidateRecord`。

候选元数据会保存 Technique 名称和 tactics，候选来源标记为 `embedding`。

### 4.10 结构化链融合：`fusion/structured_chain.py`

该模块实现 V4 使用的历史 CWE → CAPEC → ATT&CK 融合方法。

它会：

- 读取 CWE 的 abstraction，只接受 `Base` 和 `Variant`。
- 读取每个 CVE 的 CWE、CAPEC 和 Technique 链。
- 根据 CWE/CAPEC 的 fan-out 计算结构化链贡献分数。
- 使用 `alpha` 把链分数加到已有嵌入候选。
- 当链的 Technique 数量不超过 `fanout_threshold` 时，允许加入原候选中没有的 Technique。
- 最终重新排序并截取 `top_k`。

融合后的候选来源可以同时包含 `embedding` 和 `structured_chain`，元数据中包含 `chain_score`。

### 4.11 评估与报告：`evaluation/`

`src/cve2attack/evaluation/metrics.py` 在 benchmark 的完整 CVE 集合上计算：

- 预测覆盖率。
- Hit@10、Hit@20。
- Recall@10、Recall@20。

没有生成预测的 benchmark CVE 仍保留在分母中，并按未命中处理。

`src/cve2attack/evaluation/report.py` 将单次运行或多次运行对比写成 Markdown 表格。原始数值同时保存在 `metrics.json`。

## 5. 测试结构

```text
tests/
├── test_schemas.py
├── test_retrieval.py
├── test_technique_kb.py
└── test_metrics.py
```

- `test_schemas.py`：规范候选输出和历史格式兼容。
- `test_retrieval.py`：使用假 Embedder 验证候选排序和结构。
- `test_technique_kb.py`：父 Technique 筛选与 procedure 文本开关。
- `test_metrics.py`：固定 benchmark cohort、缺失预测和覆盖率语义。

## 6. 数据目录与文件格式

### 6.1 `data/benchmarks/`

这里保存用于评估的 CVE → Technique 人工或论文标注数据。

```text
data/benchmarks/
├── data_result/
│   ├── dataset.yaml
│   └── CVE-<year>.jsonl
└── cve2attack_result/
    ├── dataset.yaml
    └── CVE-<year>.jsonl
```

`data_result` 与 `cve2attack_result` 来自不同论文，标注范围和策略不同。它们是两个独立 benchmark，程序不会自动合并。

每行 benchmark 数据至少包含：

```json
{
  "cve_id": "CVE-2022-0014",
  "techniques": ["T1574"]
}
```

Technique 也可以是带 ID 字段的对象。评估读取时，子技术默认上卷到顶层 Technique。

### 6.2 `data/raw/`

`data/raw/cve/CVE-<year>.json` 保存原始 CVE 记录。每个年份文件是以 CVE ID 为键的 JSON 对象。流水线主要使用其中的：

- `description`
- `cwes`
- `cpes`
- `sourceIdentifier`

`data/raw/downloads/` 保存 CWE、CAPEC 等来源数据的原始压缩包。

### 6.3 `data/knowledge/`

- `enterprise-attack.json`：MITRE ATT&CK Enterprise STIX bundle。
- `cwe.xml`：CWE catalog，用于重写上下文和结构化链过滤。
- `capec.xml`：CAPEC catalog，作为历史知识源保留。

### 6.4 `data/derived/`

- `domain_mapping/`：按年份保存 CVE domain 分类。
- `rewrite_cache/`：LLM 生成的 CVE 查询改写，格式为 `CVE ID → 文本` 的 JSON 对象。
- `embedding_cache/`：Technique embedding 的 `.npz` 缓存，可重建且不进入 Git。
- `structured_chain/`：V4 使用的历史 CWE-CAPEC-ATT&CK 链数据。

rewrite cache 示例：

```json
{
  "CVE-2022-0014": "An attacker exploits ..."
}
```

实验配置中的 `{benchmark}` 会替换为当前输入 benchmark 名称。例如：

```text
data/derived/rewrite_cache/cve2attack_result_sec_i1.json
data/derived/rewrite_cache/data_result_sec_i1.json
```

## 7. 实验配置

一个实验 YAML 描述“采用什么方法”，而不是“一次运行的结果”。同一实验可以用不同 benchmark 或不同 run ID 多次执行。

### 7.1 当前实验

| 配置 | 查询文本 | Technique 文本 | 融合 |
| --- | --- | --- | --- |
| `v1_raw_attackbert.yaml` | 原始 CVE 描述 | 名称 + 描述 | 无 |
| `v2_raw_procedures.yaml` | 原始 CVE 描述 | 名称 + 描述 + procedures | 无 |
| `v3a_llm_rewrite.yaml` | sec-i1 改写 | 名称 + 描述 | 无 |
| `v3b_llm_rewrite_procedures.yaml` | sec-i1 改写 | 名称 + 描述 + procedures | 无 |
| `v4_rewrite_chain.yaml` | sec-i1 改写 | 名称 + 描述 + procedures | structured chain |

### 7.2 配置字段

```yaml
name: v3a_llm_rewrite
description: Human-readable experiment description.

input:
  mode: benchmark
  benchmark: cve2attack_result

query:
  strategy: rewrite_cache
  cache: data/derived/rewrite_cache/{benchmark}_sec_i1.json
  llm:
    base_url: http://host:11434/api/generate
    model: sec-i1
    timeout_seconds: 120
    max_retries: 3

technique_document:
  include_procedures: false
  procedure_char_limit: 1500

retrieval:
  model: basel/ATTACK-BERT
  top_k: 20
  batch_size: 32

fusion:
  strategy: none

evaluation:
  benchmarks: [input]
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `name` | 实验唯一名称，也是默认 run ID 的一部分。 |
| `description` | 实验目的或组成的可读说明。 |
| `input.mode` | `benchmark` 表示只处理某套 benchmark 中的 CVE；`full_enterprise` 表示处理 domain mapping 中全部 Enterprise CVE。 |
| `input.benchmark` | 输入 benchmark 目录名。命令行 `--benchmark` 可以临时覆盖。 |
| `query.strategy` | 使用 `raw_description` 或 `rewrite_cache` 作为检索查询。 |
| `query.cache` | rewrite cache 路径；支持 `{benchmark}` 占位符。 |
| `query.llm.*` | `rewrite` 命令使用的服务地址、模型、单次请求超时和最大尝试次数。 |
| `technique_document.include_procedures` | 是否把 ATT&CK procedure examples 加入 Technique 文本。 |
| `procedure_char_limit` | 每个 Technique 最多保留多少个 procedure 字符；小于等于 0 表示不截断。 |
| `retrieval.model` | sentence-transformers 模型名或本地模型路径。 |
| `retrieval.top_k` | 每个 CVE 最终保留的候选数量。 |
| `retrieval.batch_size` | CVE 和 Technique 文本编码批大小。 |
| `fusion.strategy` | `none` 或 `structured_chain`。 |
| `evaluation.benchmarks` | 要评估的 benchmark；`input` 表示使用当前 input benchmark。 |

V4 还使用：

| 字段 | 含义 |
| --- | --- |
| `fusion.chain_file` | CVE-CWE-CAPEC-Technique 链文件。 |
| `fusion.cwe_xml` | CWE catalog 路径。 |
| `fusion.alpha` | 结构化链贡献加入检索分数时的权重。 |
| `fusion.fanout_threshold` | 链候选数不超过此值时，允许向检索列表加入新 Technique。 |

## 8. 候选输出与运行目录

新候选记录使用 schema 1.0：

```json
{
  "schema_version": "1.0",
  "cve_id": "CVE-2022-0014",
  "domain": "Enterprise",
  "candidates": [
    {
      "technique_id": "T1574",
      "score": 0.6575,
      "sources": ["embedding"],
      "metadata": {
        "name": "Hijack Execution Flow",
        "tactics": ["persistence", "privilege-escalation"]
      }
    }
  ]
}
```

字段说明：

- `schema_version`：候选文件格式版本。
- `cve_id`：当前 CVE。
- `domain`：当前流程中的 ATT&CK domain，现为 Enterprise。
- `candidates`：按分数从高到低排列的候选。
- `technique_id`：顶层 ATT&CK Technique ID。
- `score`：检索相似度或融合后的分数。
- `sources`：分数来源，例如 `embedding`、`structured_chain`。
- `metadata`：Technique 名称、tactics、chain score 等附加信息。

每次 `run` 创建独立目录：

```text
runs/<run_id>/
├── manifest.json
├── candidates/
│   ├── CVE-2008.jsonl
│   └── ...
├── metrics.json
└── report.md
```

`manifest.json` 记录：

- schema version、run ID、实验名和创建时间。
- 当前 Git commit。
- 原始配置文件与合并默认值后的完整配置。
- `running`、`complete` 或 `failed` 状态。
- 查询覆盖情况、Technique 数量和候选记录数量。
- 使用的 embedding cache 路径。
- 失败时的异常类型和错误消息。

`metrics.json` 保存各 benchmark 的原始指标；`report.md` 保存便于阅读的百分比表格。

## 9. 指标含义

| 指标 | 含义 |
| --- | --- |
| `benchmark_cves` | benchmark 中的 CVE 总数，也是所有指标的固定分母。 |
| `predicted_cves` | 实际存在候选记录的 benchmark CVE 数量。 |
| `coverage` | `predicted_cves / benchmark_cves`。 |
| `hit_rate_at_10` | 前 10 个候选中至少命中一个真实 Technique 的 CVE 比例。 |
| `hit_rate_at_20` | 前 20 个候选中至少命中一个真实 Technique 的 CVE 比例。 |
| `recall_at_10` | 对每个 CVE 计算 Top-10 命中真实 Technique 的比例，再在完整 benchmark 上求平均。 |
| `recall_at_20` | 对每个 CVE 计算 Top-20 命中真实 Technique 的比例，再在完整 benchmark 上求平均。 |

没有候选记录的 CVE 按未命中处理，不会从分母中删除。`data_result` 和 `cve2attack_result` 始终分别报告。

## 10. 安装与命令入口

项目要求 Python 3.10 或更高版本。核心依赖为：

- NumPy
- PyYAML
- sentence-transformers

在项目根目录安装为 editable package：

```bash
.venv/bin/pip install -e .
```

以下两种调用方式等价：

```bash
.venv/bin/python -m cve2attack <command> [arguments]
.venv/bin/cve2attack-stage1 <command> [arguments]
```

查看总帮助：

```bash
.venv/bin/python -m cve2attack --help
```

## 11. 命令行参考

### 11.1 `validate`：校验实验配置

```bash
.venv/bin/python -m cve2attack validate <experiment>
```

参数：

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `experiment` | 是 | 实验 YAML 路径，例如 `experiments/v3a_llm_rewrite.yaml`。 |

该命令读取 YAML、合并默认值并校验当前支持的 input/query/fusion strategy。它不检查所有数据文件是否存在，也不加载模型。

示例：

```bash
.venv/bin/python -m cve2attack validate experiments/v3a_llm_rewrite.yaml
```

成功时输出：

```text
Valid experiment: v3a_llm_rewrite
```

### 11.2 `inspect`：检查输入覆盖

```bash
.venv/bin/python -m cve2attack inspect <experiment> \
  [--max-cves N] \
  [--benchmark NAME]
```

参数：

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `experiment` | 是 | 实验 YAML 路径。 |
| `--max-cves N` | 否 | 只检查排序后最前面的 N 个 CVE；不指定时检查完整输入集合。 |
| `--benchmark NAME` | 否 | 临时把输入切换到 `data/benchmarks/NAME/`，不修改 YAML。 |

该命令会检查 CVE 选择、原始描述或 rewrite 覆盖，并读取 ATT&CK 知识库统计 Technique 数量。它不会加载 embedding 模型，也不会创建 run。

输出字段：

| 字段 | 含义 |
| --- | --- |
| `selected_cves` | 输入集合选择出的 CVE 数量。 |
| `query_cves` | 实际找到查询文本的 CVE 数量。 |
| `missing_description` | raw description 策略下缺少描述的数量。 |
| `missing_rewrite` | rewrite cache 策略下缺少改写的数量。 |
| `technique_count` | 检索语料中的顶层、有效 Technique 数量。 |

示例：

```bash
.venv/bin/python -m cve2attack inspect \
  experiments/v3a_llm_rewrite.yaml \
  --benchmark cve2attack_result
```

### 11.3 `classify-domain`：重新生成 domain mapping

```bash
.venv/bin/python -m cve2attack classify-domain
```

该命令没有额外参数。它读取 `data/raw/cve/CVE-<year>.json`，并覆盖写入：

```text
data/derived/domain_mapping/CVE-<year>.jsonl
```

完成后输出总数以及 Enterprise、ICS、Mobile 的数量。该命令会重建整个年份范围的 domain mapping。

### 11.4 `rewrite`：生成 LLM 改写缓存

```bash
.venv/bin/python -m cve2attack rewrite <experiment> \
  [--workers N] \
  [--max-cves N] \
  [--no-cache] \
  [--benchmark NAME]
```

参数：

| 参数 | 必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `experiment` | 是 | — | 必须使用 `query.strategy: rewrite_cache` 的实验 YAML。 |
| `--workers N` | 否 | `4` | 同时发送 LLM 请求的线程数；最小有效并发为 1。 |
| `--max-cves N` | 否 | 全部 | 只处理排序后最前面的 N 个输入 CVE。 |
| `--no-cache` | 否 | false | 忽略已有 cache，从空映射重新生成，并最终覆盖该 cache 文件。 |
| `--benchmark NAME` | 否 | YAML 中的值 | 临时选择 benchmark，并影响 `{benchmark}` cache 路径。 |

默认情况下，已有且非空的改写会被保留并跳过。程序每完成 20 个请求或完成全部任务时写一次 cache，写入采用临时文件替换，降低中途损坏风险。

`--no-cache` 与 `--max-cves` 同时使用时，最终文件只包含本次选择范围内成功生成的内容，因此不要把这种小样本命令用于需要保留的完整 cache。

输出统计：

- `requested`：本次选择的 CVE 数量。
- `already_cached`：已有有效改写的数量。
- `success`：本次成功生成数量。
- `failed`：请求失败或返回空文本的数量。
- `missing_description`：缺少原始描述、无法构造提示词的数量。
- `cache_size`：写入后 cache 的总条目数。

小规模示例：

```bash
.venv/bin/python -m cve2attack rewrite \
  experiments/v3a_llm_rewrite.yaml \
  --workers 2 \
  --max-cves 20
```

### 11.5 `run`：执行候选生成实验

```bash
.venv/bin/python -m cve2attack run <experiment> \
  [--run-id ID] \
  [--max-cves N] \
  [--benchmark NAME]
```

参数：

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `experiment` | 是 | 实验 YAML 路径。 |
| `--run-id ID` | 否 | 指定 `runs/ID/`；不指定时使用 `时间_实验名`。目标目录已存在时命令会停止，不覆盖旧 run。 |
| `--max-cves N` | 否 | 只运行排序后最前面的 N 个 CVE，适合冒烟测试。 |
| `--benchmark NAME` | 否 | 临时覆盖 input benchmark，同时影响 rewrite cache 的 `{benchmark}` 路径和 `evaluation: [input]`。 |

该命令会加载 embedding 模型。模型在本机没有缓存时，sentence-transformers 可能尝试下载模型。

V3a/V3b/V4 的 `run` 只读取现有 rewrite cache，不会自动调用 LLM 补齐缺失项；缺少 rewrite 的 CVE 会记录在 input coverage 中并跳过候选生成。

完整 V3a 示例：

```bash
.venv/bin/python -m cve2attack run \
  experiments/v3a_llm_rewrite.yaml \
  --benchmark cve2attack_result
```

两条 CVE 冒烟测试：

```bash
.venv/bin/python -m cve2attack run \
  experiments/v1_raw_attackbert.yaml \
  --max-cves 2 \
  --run-id smoke_v1
```

在另一套论文数据上执行相同方法：

```bash
.venv/bin/python -m cve2attack run \
  experiments/v1_raw_attackbert.yaml \
  --benchmark data_result
```

### 11.6 `compare`：统一比较多个 run

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark NAME \
  [--comparison-id ID] \
  <run> [<run> ...]
```

参数：

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `runs` | 是 | 一个或多个 run 目录，可使用相对项目根目录或绝对路径。目录中可以有 `candidates/` 子目录，也可以直接包含年度候选 JSONL。 |
| `--benchmark NAME` | 是 | 用于所有 run 的同一固定 benchmark，例如 `cve2attack_result`。 |
| `--comparison-id ID` | 否 | 指定 `comparisons/ID/`；默认使用 `时间_benchmark`。目录已存在时不会覆盖。 |

输出目录：

```text
comparisons/<comparison_id>/
├── metrics.json
├── cohort.json
└── report.md
```

- `metrics.json`：所有 run 的指标。
- `cohort.json`：本次比较使用的固定 CVE ID 集合。
- `report.md`：覆盖率与 Recall 的对比表格。

示例：

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark cve2attack_result \
  --comparison-id v1_vs_v3a \
  runs/20260716_v1_raw_attackbert \
  runs/20260716_v3a_llm_rewrite
```

## 12. 常用使用流程

### 12.1 检查并运行已有 V3a cache

```bash
.venv/bin/python -m cve2attack validate experiments/v3a_llm_rewrite.yaml

.venv/bin/python -m cve2attack inspect experiments/v3a_llm_rewrite.yaml

.venv/bin/python -m cve2attack run experiments/v3a_llm_rewrite.yaml
```

### 12.2 先补充 rewrite，再运行

```bash
.venv/bin/python -m cve2attack rewrite \
  experiments/v3a_llm_rewrite.yaml \
  --workers 4

.venv/bin/python -m cve2attack inspect experiments/v3a_llm_rewrite.yaml

.venv/bin/python -m cve2attack run experiments/v3a_llm_rewrite.yaml
```

### 12.3 运行测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

如果安装了可选测试依赖，也可以运行：

```bash
.venv/bin/pytest
```

## 13. 运行时注意事项

- `runs/`、`comparisons/` 和 `data/derived/embedding_cache/` 被 Git 忽略。
- `rewrite` 会访问实验 YAML 中配置的外部 LLM 服务。
- `run` 可能加载或下载体积较大的 sentence-transformers 模型。
- `--max-cves` 取排序后前 N 个 CVE，不是随机采样。
- `run` 和 `compare` 不覆盖已存在的目标目录。
- `classify-domain` 会重新写入所有年度 domain mapping。
- `--benchmark` 只覆盖本次命令使用的输入配置，不会修改实验 YAML。
- `archive/` 中的代码不参与当前 Python 包导入和运行。
- 历史混合实验结果保存在 `runs/legacy_import_20260628/`，仅用于追溯。
