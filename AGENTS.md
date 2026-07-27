# CVE → ATT&CK 映射项目结构与命令行说明

本文档介绍项目的目标、代码组织、数据流、实验配置、文件格式和命令行用法。它同时面向项目使用者和需要理解代码的 Agent。项目现已在同一仓库中加入第二阶段攻击图上下文提取；第一阶段候选接入、图上下文重排序和最终评价仍在后续实现。

处理第二阶段任务前，必须先阅读根目录 `STAGE2_PLAN.md`。该文档定义毕设闭环范围、工作包顺序、验收标准、标签泄漏防线和 worktree 规则；本文档继续作为整个项目的代码结构与命令行参考。

## 1. 项目目标与范围

本项目实现 CVE → MITRE ATT&CK Technique 的两阶段映射流程：第一阶段根据 CVE 信息生成按相关性排序的 Technique 候选集；第二阶段从 MulVAL 攻击图提取该 CVE 的局部条件和上游路径证据，后续据此重排第一阶段候选。

项目中需要区分两种第一阶段方法：

- `main` 分支保存分层映射方法。
- `new_method` 分支保存基于文本改写与向量检索的候选生成方法。

两者是第一阶段的两种独立实现，不是前后相接的两个阶段。本文档描述的是重构后的 `new_method` 方法。

当前选定方案为 V3a：

1. 读取 CVE 描述和 CWE 信息。
2. 使用 `sec-i1-cve-rewrite:v1` 将漏洞描述改写为 ATT&CK 风格的攻击者动作描述。
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
├── STAGE2_PLAN.md
├── pyproject.toml
├── experiments/
│   ├── diagnostics/
│   │   ├── v1_raw_attackbert_fullranking.yaml
│   │   ├── v2_raw_procedures_fullranking.yaml
│   │   ├── v3a_llm_rewrite_fullranking.yaml
│   │   └── v3b_llm_rewrite_procedures_fullranking.yaml
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
│   │   ├── rrf.py
│   │   └── structured_chain.py
│   ├── evaluation/
│   │   ├── diagnostics.py
│   │   ├── metrics.py
│   │   ├── ranking.py
│   │   ├── triage.py
│   │   └── report.py
│   └── stage2/
│       ├── candidate_joiner.py
│       ├── evaluation.py
│       ├── graph_parser.py
│       ├── path_expander.py
│       ├── context_extractor.py
│       ├── reranker.py
│       └── pipeline.py
├── tests/
│   └── fixtures/mulval/AttackGraph.xml
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
- `STAGE2_PLAN.md`：第二阶段总行动指南；记录必做工作包、验收标准、评价边界和下一任务。
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
- `import-kev`：从固定的 CTID KEV CSV 快照生成 `all`、`exploitation` 和 `nonoverlap` 三个公开 benchmark 视图。
- `import-triage`：从公开复现实验包的冻结 split 与标签生成两个 60-CVE TRIAGE 测试视图。
- `compare-triage`：在论文原始测试 split 上，将项目 run 与公开的 TRIAGE、SMET 预测统一复评并输出逐 CVE 分歧。
- `diagnose-triage`：在同一 60-CVE/143-parent-label 口径下，对完整排名 run、SMET 和 TRIAGE 做候选互补性诊断。
- `fuse-rrf`：读取多个已完成 run，仅使用候选名次执行无训练 Reciprocal Rank Fusion，并写出受控 Top-K 标准 run。
- `extract-graph-context`：读取 MulVAL `AttackGraph.xml`，输出带版本号的局部上下文与完整上游证据分支。
- `run-stage2`：读取已有第一阶段 run 和一张攻击图，连接 `CandidateRecord`、执行 topology-only 确定性重排序并输出前后评价。

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

当前 V3a、V3b 和 V4 共用同一个版本化改写条件：

- Ollama 服务运行在 `172.23.216.73:11434`；项目代码运行在 `172.23.216.47`，只通过 HTTP 调用模型，不在代码主机保存模型权重。
- 原始标签 `sec-i1:latest` 是从本地 GGUF 导入的 Llama 3 8B Q4 模型，其模板只有 `{{ .Prompt }}`，不会应用 system prompt，部分输入会直接返回空文本。
- 修复后的标签是 `sec-i1-cve-rewrite:v1`，它复用原始权重，并增加 Llama 3 system/user/assistant 模板、`num_ctx=8192`、`num_predict=512`、`temperature=0` 和固定 seed。
- 模型标签和缓存文件名都是实验条件的一部分。旧标签生成的 cache 不能与 `v1` 模板生成的 cache 混用。

模型定义保存在 `models/ollama/sec-i1-cve-rewrite-v1.Modelfile`，仓库不保存权重。如果 Ollama 标签丢失，在能读取项目目录且安装了 Ollama CLI 的机器上执行：

```bash
OLLAMA_HOST=http://172.23.216.73:11434 \
  ollama create sec-i1-cve-rewrite:v1 \
  -f models/ollama/sec-i1-cve-rewrite-v1.Modelfile
```

`OLLAMA_HOST` 让 CLI 修改 73 上的服务，而不是本机 Ollama。执行命令时，`FROM sec-i1:latest` 必须能在该远程服务中解析；若从项目目录以外执行，应将 `-f` 改为 Modelfile 的绝对路径。

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

`src/cve2attack/fusion/rrf.py` 实现多个已完成 run 的 Reciprocal Rank Fusion。它不比较原始相似度，而对每个来源名次累加 `weight / (rank_constant + rank)`。`source_depth` 控制每路参与融合的内部排名深度，`top_k` 控制最终交给下游阶段的候选预算。输出仍使用统一 `CandidateRecord`，并在每个候选的 metadata 中记录来源名次和逐来源贡献；run manifest 会保存全部参数、输入路径、输入 experiment、Git commit 和 resolved config。

### 4.11 评估与报告：`evaluation/`

`src/cve2attack/evaluation/metrics.py` 在 benchmark 的完整 CVE 集合上计算：

- 预测覆盖率。
- Hit@10、Hit@20。
- Recall@10、Recall@20。

没有生成预测的 benchmark CVE 仍保留在分母中，并按未命中处理。

`src/cve2attack/evaluation/report.py` 将单次运行或多次运行对比写成 Markdown 表格。原始数值同时保存在 `metrics.json`。

`src/cve2attack/evaluation/ranking.py` 提供 TRIAGE 兼容的排名评测。它明确区分两种不能混用的 Recall：

- `macro_recall_at_k`：先对每个 CVE 计算真实 Technique 覆盖率，再对固定 CVE cohort 求平均；这是项目原有 Recall 的语义。
- `micro_recall_at_k`：将所有 CVE 的真实标签汇总后计算命中比例；这是 TRIAGE 论文表格中的 Recall@K。

`src/cve2attack/evaluation/triage.py` 读取公开 reference predictions，核验其内嵌真值与冻结 benchmark 一致，并生成 MAP、Hit@K、两种 Recall、按 mapping type 的诊断和逐 CVE Top-10 分歧。

`src/cve2attack/evaluation/diagnostics.py` 用于第一阶段候选源诊断。它计算 Recall@1/3/5/10/20/30/50、正确标签排名分布、两两候选交集、独有正确命中、并集 oracle，以及按 mapping type、CVE 年份、训练标签频率、CWE 数量和描述长度分组的 Micro Recall。TRIAGE 公开预测只到 Top-20，因此其 Recall@30/@50 会写为 `N/A`，不会把截断误报为零。

### 4.12 第二阶段攻击图上下文：`stage2/`

`src/cve2attack/stage2/` 将外部 MulVAL 生成的 `AttackGraph.xml` 转换成带版本号的上下文 JSON：

- `graph_parser.py`：解析 XML，并显式把 MulVAL 原始边反转为“条件 → 规则 → 结果”。
- `path_expander.py`：展开 OR 状态的全部 producer rules；不再任意选择第一条路径。
- `context_extractor.py`：为每个 `vulExists` 分别生成直接的 `local_context` 和上游 `graph_context`。
- `pipeline.py`：组织文件读写、原子输出和终端进度。
- `candidate_joiner.py`：按规范化 CVE ID 连接第一阶段候选与图上下文，并报告缺失、未解析和重复输入。
- `reranker.py`：实现不使用目标语义字段和标签的 `topology-rule-priority-v1` 基线。
- `evaluation.py`：在候选集合不变的前提下比较 Top-1/3/5、MRR 和正确标签名次。

攻击图生成器仍是外部组件，本仓库不依赖 `ldh_attackgraph` 父目录。原 MulVAL 固定回归输入复制在 `tests/fixtures/mulval/AttackGraph.xml`；最小闭环场景位于 `tests/fixtures/stage2/`。单独提取上下文时 `candidates` 为空，`run-stage2` 会用现有 `CandidateRecord` 填充并执行确定性重排序。详细数据契约见 `docs/stage2_graph_context.md`。

## 5. 测试结构

```text
tests/
├── test_diagnostics.py
├── test_rrf.py
├── test_stage2_context.py
├── test_stage2_closed_loop.py
├── test_schemas.py
├── test_retrieval.py
├── test_technique_kb.py
└── test_metrics.py
```

- `test_schemas.py`：规范候选输出和历史格式兼容。
- `test_retrieval.py`：使用假 Embedder 验证候选排序和结构。
- `test_technique_kb.py`：父 Technique 筛选与 procedure 文本开关。
- `test_metrics.py`：固定 benchmark cohort、缺失预测和覆盖率语义。
- `test_diagnostics.py`：任意 cutoff、公开排名截断、正确标签 rank bin 和并集候选预算语义。
- `test_rrf.py`：RRF 共识排序、内部来源深度、确定性 tie break 和非法权重校验。
- `test_stage2_context.py`：MulVAL XML 解析、边方向、局部上下文、全部分支保留和文件输出契约。
- `test_stage2_closed_loop.py`：候选接入、缺失/重复检查、三类 topology-only 规则、候选集合不变和端到端输出。

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

TRIAGE 接入后还包含两个固定测试视图：

- `triage_2025_test_all`：论文公开 test split 的 60 个 CVE，使用 exploitation、primary impact、secondary impact 的并集。
- `triage_2025_test_no_secondary`：同一批 60 个 CVE，但排除 secondary impact，复现论文的排除次要影响实验。

两者都来自同一份 296-CVE CTID KEV 标签，不是新的独立人工标注集。它们的价值在于冻结了 TRIAGE 的 236/60 划分，并可与作者公开预测做完全同 cohort 的比较。

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
data/derived/rewrite_cache/cve2attack_result_sec_i1_llama3_chat_v1.json
data/derived/rewrite_cache/data_result_sec_i1_llama3_chat_v1.json
```

## 7. 实验配置

一个实验 YAML 描述“采用什么方法”，而不是“一次运行的结果”。同一实验可以用不同 benchmark 或不同 run ID 多次执行。

### 7.1 当前实验

| 配置 | 查询文本 | Technique 文本 | 融合 |
| --- | --- | --- | --- |
| `v1_raw_attackbert.yaml` | 原始 CVE 描述 | 名称 + 描述 | 无 |
| `v2_raw_procedures.yaml` | 原始 CVE 描述 | 名称 + 描述 + procedures | 无 |
| `v3a_llm_rewrite.yaml` | sec-i1 Llama 3 模板 v1 改写 | 名称 + 描述 | 无 |
| `v3b_llm_rewrite_procedures.yaml` | sec-i1 Llama 3 模板 v1 改写 | 名称 + 描述 + procedures | 无 |
| `v4_rewrite_chain.yaml` | sec-i1 Llama 3 模板 v1 改写 | 名称 + 描述 + procedures | structured chain |

`experiments/diagnostics/` 中的四份 `*_fullranking.yaml` 不是新方法，而是 V1/V2/V3a/V3b 的诊断运行条件：输入固定为 `triage_2025_test_all`，`retrieval.top_k=202`，保存 ATT&CK 15.1 全部父 Technique 的排序。这样才能区分“正确标签在 21–50 位”和“正确标签在所有实用 Top-50 候选之外”。V3 诊断配置直接复用已经完成的 296-CVE rewrite cache，执行 `run` 时不会调用 Ollama。

### 7.2 配置字段

```yaml
name: v3a_llm_rewrite
description: Human-readable experiment description.

input:
  mode: benchmark
  benchmark: cve2attack_result

query:
  strategy: rewrite_cache
  cache: data/derived/rewrite_cache/{benchmark}_sec_i1_llama3_chat_v1.json
  llm:
    base_url: http://172.23.216.73:11434/api/generate
    model: sec-i1-cve-rewrite:v1
    timeout_seconds: 300
    max_retries: 3

technique_document:
  include_procedures: false
  procedure_char_limit: 1500

retrieval:
  model: basel/ATTACK-BERT
  top_k: 20
  batch_size: 32
  local_files_only: true

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
| `retrieval.local_files_only` | 默认 `true`，只从本机 Hugging Face cache 加载模型，避免实验因网络阻塞。首次有意下载模型时才设为 `false`。 |
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
- NetworkX
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

运行时会输出已有缓存数、剩余请求数、成功/失败数、checkpoint、耗时和预计剩余时间。`failed` 同时包括请求异常和模型返回空文本；失败详情会按 CVE 输出到终端。

V3a、V3b 和 V4 当前都使用 `sec-i1-cve-rewrite:v1`。不要把旧的 `*_sec_i1.json` 复制或重命名为新的 `*_sec_i1_llama3_chat_v1.json`，否则会混合两种提示模板生成的数据。

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

该命令会加载 embedding 模型。默认只使用本机缓存；模型缺失时会立即说明错误。只有在 YAML 中显式设置 `retrieval.local_files_only: false` 后，才允许 sentence-transformers 下载模型。

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

### 11.7 `import-kev`：生成固定 CTID KEV benchmark

```bash
.venv/bin/python -m cve2attack import-kev \
  [--source PATH] \
  [--benchmark-root PATH] \
  [--cve2attack-benchmark PATH]
```

默认原始输入为
`data/raw/kev/kev-02.13.2025_attack-15.1-enterprise.csv`，即 CTID 在 Zenodo
公开的 `02.13.2025` KEV 快照，ATT&CK Enterprise 版本为 `15.1`。命令会验证其
框架版本与 ATT&CK 版本，并创建下列互不混合的 benchmark：

- `ctid_kev_2025_02_13_all`：利用技术、主要影响和次要影响的并集；这是第一阶段的 KEV 主结果。
- `ctid_kev_2025_02_13_exploitation`：只保留 `exploitation_technique`，用于检查显式漏洞利用动作。
- `ctid_kev_2025_02_13_nonoverlap`：从 `all` 中去除 `cve2attack_result` 已包含的 CVE，作为严格外部结果。

每个目录包含年度 JSONL 和 `dataset.yaml`。`techniques` 是现有顶层候选生成器
使用的父 Technique 标签；`techniques_raw` 保存源数据中的精确 Technique 或子
Technique；`labels_by_mapping_type` 和 `label_metadata` 保存 CTID 的语义角色与证据。
目标目录已存在时命令会失败，避免无意覆盖冻结基准。

每个 KEV `dataset.yaml` 还指定
`data/knowledge/enterprise-attack-15.1.json`。`inspect` 与 `run` 会自动选择
这份冻结语料并生成独立 embedding cache；CVE2ATT&CK 等其他 benchmark 仍使用
当前的 `data/knowledge/enterprise-attack.json`。只有明确进行 ATT&CK 版本迁移
实验时，才应在 YAML 的 `technique_document.attack_bundle` 中覆盖该选择。

KEV 评测的模型输入始终是 `data/raw/cve/` 中的 CVE 描述；不要把 KEV 的
`comments` 或 `references` 用作模型输入。

### 11.8 `import-triage`：生成 TRIAGE 固定测试视图

```bash
.venv/bin/python -m cve2attack import-triage \
  [--source-dir PATH] \
  [--benchmark-root PATH]
```

默认输入目录为 `data/raw/triage/triage_2025/`。该目录只保存从公开
`TRIAGE.zip` 中选出的 split、806 条标签和四套 reference predictions，不保存
773.7 MB 的完整压缩包。`source.yaml` 记录压缩包 MD5、原始路径、文件 SHA-256、
预期数据量和论文报告指标。

命令会验证 train/test 数量、不重叠、split 与标签一致以及两个主测试视图均有标签。目标 benchmark 已存在时不会覆盖。

### 11.9 `compare-triage`：与公开预测统一复评

```bash
.venv/bin/python -m cve2attack compare-triage \
  [--comparison-id ID] \
  <run> [<run> ...]
```

该命令不加载模型，也不重新生成候选。它读取已有 run，在精确的 60-CVE test
split 上与公开 SMET、TRIAGE 预测比较，输出到 `comparisons/<ID>/`：

- `metrics.json`：all/no-secondary 两个视图及三个 mapping type 的完整原始指标；
- `report.md`：MAP、Hit@K、宏平均 Recall 和论文式微平均 Recall 表格；
- `disagreements_<run>.jsonl`：逐 CVE 的真实标签、双方 Top-10、各自命中和分歧类别。

reference predictions 中保存的真值如果与生成后的 benchmark 不一致，或复算结果无法重现 `source.yaml` 记录的论文数值，命令会立即失败。

### 11.10 `diagnose-triage`：候选互补性与失败类型诊断

```bash
.venv/bin/python -m cve2attack diagnose-triage \
  [--comparison-id ID] \
  [--source-dir PATH] \
  <full-ranking-run> [<full-ranking-run> ...]
```

参数：

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `full-ranking-run` | 是 | 一个或多个已完成的完整排名 run。标准诊断应传入 V1、V2、V3a、V3b 四个 `top_k=202` run。 |
| `--comparison-id ID` | 否 | 指定 `comparisons/ID/`；默认使用时间戳。目标目录已经存在时不会覆盖。 |
| `--source-dir PATH` | 否 | TRIAGE split、标签和公开参考预测目录；默认 `data/raw/triage/triage_2025`。 |

命令固定使用 `triage_2025_test_all` 的 60 个 CVE 和 143 个父 Technique 标签。它不会加载嵌入模型、生成候选或调用 Ollama，只读取现有 run 和公开预测。运行期间会依次输出来源覆盖率、存储排名范围、曲线计算、互补性计算和报告写入进度。

输出目录：

```text
comparisons/<comparison_id>/
├── diagnostics.json
├── label_ranks.jsonl
├── practical_failure_labels.jsonl
└── report.md
```

- `diagnostics.json`：完整 Recall 曲线、来源可观测深度、rank 分布、两两交集、独有命中、并集 oracle 和全部分组结果。
- `label_ranks.jsonl`：每个真实父标签一行，包含各来源排名、mapping type、年份、训练标签频率、CWE 和描述长度特征。
- `practical_failure_labels.jsonl`：按项目四路的最佳排名将标签分为 Top-20、21–50、50 以后或未排名，便于逐例检查。
- `report.md`：核心曲线、主要失败判断和分组表格。

并集 oracle 将每个来源自己的 Top-K 取并集，实际候选数通常大于 K；报告会同时给出平均/最大并集大小，因此不能把它当作受控 Recall@K。TRIAGE 公开历史最多到 20 位，它的 Recall@30/@50 必须保持 `N/A`。

标准运行方式：

```bash
.venv/bin/python -m cve2attack run \
  experiments/diagnostics/v1_raw_attackbert_fullranking.yaml \
  --run-id triage_diag_v1_fullranking

# 对 v2、v3a、v3b 的 fullranking 配置分别运行后：
.venv/bin/python -m cve2attack diagnose-triage \
  --comparison-id triage_candidate_complementarity \
  runs/triage_diag_v1_fullranking \
  runs/triage_diag_v2_fullranking \
  runs/triage_diag_v3a_fullranking \
  runs/triage_diag_v3b_fullranking
```

### 11.11 `fuse-rrf`：无训练的候选排名融合

```bash
.venv/bin/python -m cve2attack fuse-rrf \
  --run-id ID \
  --benchmark NAME \
  [--top-k N] \
  [--source-depth N] \
  [--rank-constant FLOAT] \
  [--weights W1 W2 ...] \
  <run> <run> [<run> ...]
```

参数：

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `run` | 是 | 两个或更多已完成的候选 run。命令会要求每个 run 完整覆盖指定 benchmark，并至少保存到 `source-depth`。 |
| `--run-id ID` | 是 | 新的标准输出目录 `runs/ID/`。已经存在时不会覆盖。 |
| `--benchmark NAME` | 是 | 固定融合 cohort 和评估真值，例如 `triage_2025_test_all`。 |
| `--top-k N` | 否 | 最终候选预算，默认 20。 |
| `--source-depth N` | 否 | 每个输入来源最多读取到的名次，默认 50。它是内部检索池深度，不等于最终候选数。 |
| `--rank-constant FLOAT` | 否 | RRF 平滑常数，必须为正数，默认 60。 |
| `--weights W1 W2 ...` | 否 | 按 run 参数顺序提供的正权重；省略时所有来源权重均为 1。权重数量必须与 run 数相同。 |

计算公式：

```text
RRF(technique) = Σ_source weight_source / (rank_constant + rank_source)
```

没有进入某来源 `source-depth` 的 Technique 在该来源贡献为 0。最终按 RRF 分数排序并严格截取 `top-k`；同分时依次使用最佳来源名次、来源票数和 Technique ID 做不依赖标签的确定性 tie break。

标准无调参基线示例：

```bash
.venv/bin/python -m cve2attack fuse-rrf \
  --run-id triage_rrf_v1_v3a_d50_k60_top20 \
  --benchmark triage_2025_test_all \
  --top-k 20 \
  --source-depth 50 \
  --rank-constant 60 \
  runs/triage_diag_v1_fullranking \
  runs/triage_diag_v3a_fullranking
```

该命令不加载 embedding 模型、不调用 Ollama，也不使用 benchmark 标签决定融合分数。标签只在候选写出后用于常规评估。

### 11.12 `extract-graph-context`：提取第二阶段攻击图上下文

```bash
.venv/bin/python -m cve2attack extract-graph-context \
  --attack-graph PATH \
  --output PATH \
  [--max-graph-depth N] \
  [--force]
```

参数：

| 参数 | 必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--attack-graph PATH` | 是 | — | MulVAL `AttackGraph.xml`；相对路径从项目根目录解析。 |
| `--output PATH` | 是 | — | 上下文 JSON；相对路径从项目根目录解析。 |
| `--max-graph-depth N` | 否 | `2` | 上游证据展开深度，必须为非负整数。 |
| `--force` | 否 | false | 覆盖已有输出；默认遇到同名文件立即停止。 |

当前样例：

```bash
.venv/bin/python -m cve2attack extract-graph-context \
  --attack-graph tests/fixtures/mulval/AttackGraph.xml \
  --output stage2_runs/example/contexts.json
```

该命令不加载 embedding 模型，也不调用 Ollama。输出包含图统计、每个漏洞的直接利用条件和全部上游分支；候选字段保持为空，等待 `run-stage2` 接入第一阶段结果。

### 11.13 `run-stage2`：运行第一、第二阶段最小闭环

```bash
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack run-stage2 \
  --stage1-run PATH \
  --attack-graph PATH \
  --benchmark NAME \
  --run-id ID \
  [--output-root PATH] \
  [--scenario-kind NAME] \
  [--max-graph-depth N]
```

参数：

| 参数 | 必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--stage1-run PATH` | 是 | — | 已完成的第一阶段 run；可包含 `candidates/` 子目录或根目录年度 JSONL。 |
| `--attack-graph PATH` | 是 | — | 本次场景的 MulVAL `AttackGraph.xml`。 |
| `--benchmark NAME` | 是 | — | `data/benchmarks/` 下的公开标签目录，只在候选写出后用于评价。 |
| `--run-id ID` | 是 | — | `output-root` 下的新目录名；只能是单个安全目录名且不能重复。 |
| `--output-root PATH` | 否 | `stage2_runs` | 第二阶段运行根目录。 |
| `--scenario-kind NAME` | 否 | `synthetic_topology_smoke` | 写入 manifest/report 的场景来源标识。 |
| `--max-graph-depth N` | 否 | `2` | 上游图证据展开深度。 |

流程会依次输出上下文提取、候选加载、CVE 连接、规则重排、公开标签评价和 manifest
收口进度。它拒绝未完成的第一阶段 run、空 benchmark、零个可评价 CVE 和已存在的
输出目录。候选集合保持不变，`score` 保留第一阶段值；重排名次和证据写入 candidate
metadata 的 `stage2` 字段。

当前工程冒烟示例：

```bash
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack run-stage2 \
  --stage1-run /home/ghdemi/Code/cve2attack/runs/triage_rrf_v1_v3a_d50_k60_top20 \
  --attack-graph tests/fixtures/stage2/public_facing/AttackGraph.xml \
  --benchmark triage_2025_test_all \
  --run-id cve_2023_20887_public_facing_smoke \
  --scenario-kind synthetic_public_facing_smoke
```

该场景使用真实 CVE、真实第一阶段候选和公开标签，但攻击图拓扑是人工合成的，只用于
证明链路、确定性行为和报告正确，不能作为独立总体准确率证据。

输出目录：

```text
stage2_runs/<run_id>/
├── manifest.json
├── contexts.json
├── join_stats.json
├── joined_records.jsonl
├── reranked_records.jsonl
├── metrics.json
└── report.md
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

### 12.3 导入并运行 KEV 主基准

```bash
.venv/bin/python -m cve2attack import-kev

.venv/bin/python -m cve2attack inspect \
  experiments/v1_raw_attackbert.yaml \
  --benchmark ctid_kev_2025_02_13_all

.venv/bin/python -m cve2attack run \
  experiments/v1_raw_attackbert.yaml \
  --benchmark ctid_kev_2025_02_13_all
```

将 benchmark 名称分别换为 `ctid_kev_2025_02_13_exploitation` 和
`ctid_kev_2025_02_13_nonoverlap`，即可生成两个诊断结果。对于 V3a/V3b/V4，
必须先用同一 `--benchmark` 参数执行 `rewrite`，生成该 KEV 视图自己的 rewrite cache。

### 12.4 运行测试

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
- 当前 LLM 权重位于 `172.23.216.73` 的 Ollama 服务；`172.23.216.47` 只保存项目代码、输入数据和 rewrite cache。
- `run` 会加载 sentence-transformers 模型，但默认不访问网络下载；首次下载需要在实验 YAML 中显式关闭 `retrieval.local_files_only`。
- `--max-cves` 取排序后前 N 个 CVE，不是随机采样。
- `run` 和 `compare` 不覆盖已存在的目标目录。
- `classify-domain` 会重新写入所有年度 domain mapping。
- `--benchmark` 只覆盖本次命令使用的输入配置，不会修改实验 YAML。
- `archive/` 中的代码不参与当前 Python 包导入和运行。
- 历史混合实验结果保存在 `runs/legacy_import_20260628/`，仅用于追溯。
