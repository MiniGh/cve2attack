# AGENTS.md

本文件是面向维护本项目的 Agent 的代码导航和修改约定。开始实现功能或修改方法前，先阅读本文件；用户研究方案和明确要求始终优先于这里的默认约定。

## 1. 项目边界

本仓库实现 CVE → MITRE ATT&CK Technique 映射流程的第一阶段：为每个 CVE 生成排序后的 Technique 候选集。

需要明确区分：

- `main` 与 `new_method` 是第一阶段候选生成的两种独立方法。
- 当前代码属于 `new_method` 的嵌入检索方法，不是 `main` 中分层映射方法的后续阶段。
- 不要把 `main` 的分层 LLM 流程移植进本流水线，也不要在本项目中假设存在来自 `main` 的前置结果。
- 当前选定方案是 V3a：使用 `sec-i1` 将 CVE 描述改写为攻击者动作语言，再用 `basel/ATTACK-BERT` 在顶层 Technique 的名称和描述上检索。

## 2. 首先去哪里找代码

活动代码只在 `src/cve2attack/`。收到修改需求时，按下表定位：

| 需求 | 首要文件 | 相关文件 |
| --- | --- | --- |
| 修改或新增命令行命令 | `src/cve2attack/cli.py` | `src/cve2attack/__main__.py` |
| 修改完整运行流程、运行目录或 manifest | `src/cve2attack/pipeline.py` | `src/cve2attack/config.py` |
| 修改实验配置字段、默认值或策略校验 | `src/cve2attack/config.py` | `experiments/*.yaml` |
| 修改 CVE、benchmark 或旧候选结果的读取 | `src/cve2attack/data/loaders.py` | `src/cve2attack/schemas.py` |
| 修改候选输出格式或旧格式兼容 | `src/cve2attack/schemas.py` | `src/cve2attack/data/loaders.py`, `tests/test_schemas.py` |
| 修改 CVE → ATT&CK domain 分类 | `src/cve2attack/domain/classifier.py` | `data/derived/domain_mapping/` |
| 修改 LLM 重写提示词或 CWE 上下文 | `src/cve2attack/rewrite/pipeline.py` | `src/cve2attack/rewrite/ollama.py` |
| 修改 Ollama 地址调用、超时或重试 | `src/cve2attack/rewrite/ollama.py` | 对应实验 YAML 的 `query.llm` |
| 修改嵌入模型封装或向量归一化 | `src/cve2attack/retrieval/embedder.py` | `src/cve2attack/retrieval/generator.py` |
| 修改 Technique 文本、procedure 或子技术过滤 | `src/cve2attack/retrieval/technique_kb.py` | `tests/test_technique_kb.py` |
| 修改相似度、排序、Top-K 或嵌入缓存 | `src/cve2attack/retrieval/generator.py` | `src/cve2attack/retrieval/embedder.py`, `tests/test_retrieval.py` |
| 修改 V4 的结构化链融合 | `src/cve2attack/fusion/structured_chain.py` | `experiments/v4_rewrite_chain.yaml` |
| 修改 Recall、覆盖率或评估样本集合 | `src/cve2attack/evaluation/metrics.py` | `src/cve2attack/evaluation/report.py`, `tests/test_metrics.py` |
| 修改 Markdown 实验报告 | `src/cve2attack/evaluation/report.py` | `src/cve2attack/cli.py` |
| 增加一种已有组件的新组合方案 | 新建 `experiments/<name>.yaml` | 通常不需要复制 Python 代码 |
| 记录方案结论或废弃原因 | `docs/experiment_history.md` | 对应实验 YAML 的 `description` |

`archive/` 中是历史实现，只用于追溯。除非用户明确要求恢复或研究旧方法，否则不要从这里开始修改，也不要让活动代码 import `archive/`。

## 3. 运行流程

一次 `run` 的主流程在 `src/cve2attack/pipeline.py::run_experiment`：

1. `load_experiment` 读取 YAML 并合并稳定默认值。
2. `select_input_ids` 从指定 benchmark 或 Enterprise domain 映射选择 CVE。
3. `build_queries` 使用原始描述或已有 LLM rewrite cache 构建查询文本。
4. `load_technique_documents` 从 ATT&CK STIX bundle 构建 Technique 文本。
5. `SentenceTransformerEmbedder` 加载模型并生成向量。
6. `retrieve_candidates` 计算相似度并生成排序候选。
7. 如果 `fusion.strategy=structured_chain`，再执行结构化链融合。
8. 写入规范候选文件，并在指定 benchmark 的完整固定样本集合上评估。
9. 将 manifest、metrics 和报告写入独立的 `runs/<run_id>/`。

排查行为不符合预期时，沿这条链定位，避免把所有新逻辑继续堆进 `pipeline.py`。独立、可测试的策略应放入对应子包，`pipeline.py` 只负责组织流程。

## 4. 根目录结构

```text
.
├── AGENTS.md                 # 本文件：Agent 导航与修改约定
├── README.md                 # 面向使用者的项目说明和运行示例
├── pyproject.toml            # Python 包、依赖与命令行入口
├── experiments/              # 版本化实验定义，不保存运行结果
├── src/cve2attack/           # 唯一活动代码区
├── tests/                    # 快速单元测试
├── data/
│   ├── benchmarks/           # 两套相互独立的论文标注数据集
│   ├── knowledge/            # ATT&CK、CWE、CAPEC 知识源
│   ├── raw/                  # 原始 CVE 数据和下载包
│   └── derived/              # domain、rewrite、embedding、chain 中间数据
├── docs/                     # 研究方案与实验历史
├── runs/                     # 单次运行结果，Git 忽略
├── comparisons/              # 多次运行的固定 cohort 对比，Git 忽略
└── archive/                  # 不活动的历史代码与本地备份
```

### 数据目录约束

- `data/benchmarks/data_result/` 与 `data/benchmarks/cve2attack_result/` 来自不同论文，必须分别保存和评估。
- 绝对不要隐式合并两套 benchmark。若研究上需要联合数据集，必须新建有明确名称、来源和生成方式的版本化数据集。
- 每个 benchmark 的说明写在其 `dataset.yaml`；不要猜测论文引用信息。
- `data/raw/cve/` 保存按年份组织的原始 CVE JSON。
- `data/knowledge/enterprise-attack.json` 是 Technique 和 procedure 的来源。
- `data/derived/rewrite_cache/` 保存昂贵的 LLM 改写，并以 benchmark/模型或提示词版本区分。
- `data/derived/embedding_cache/` 是可重建的机器缓存，不提交 Git。
- `data/derived/structured_chain/` 只供显式启用该策略的方案（当前为 V4）使用。

## 5. 实验配置

实验配置是方法定义，运行目录是一次执行的结果，两者不要混在一起。

现有配置：

- `v1_raw_attackbert.yaml`：原始 CVE 描述 + Technique 名称/描述。
- `v2_raw_procedures.yaml`：V1 加入 procedure examples。
- `v3a_llm_rewrite.yaml`：LLM 改写 + 名称/描述；当前选定方案。
- `v3b_llm_rewrite_procedures.yaml`：V3a 加入 procedure examples。
- `v4_rewrite_chain.yaml`：改写检索结果再融合 CWE-CAPEC-ATT&CK 历史链。

核心字段：

```yaml
name: unique_experiment_name
input:
  mode: benchmark            # benchmark 或 full_enterprise
  benchmark: cve2attack_result
query:
  strategy: raw_description  # raw_description 或 rewrite_cache
technique_document:
  include_procedures: false
retrieval:
  model: basel/ATTACK-BERT
  top_k: 20
  batch_size: 32
fusion:
  strategy: none             # none 或 structured_chain
evaluation:
  benchmarks: [input]        # 对当前选择的 input benchmark 评估
```

如果新方案只是更换模型、Top-K、是否使用 procedure、是否重写或融合参数，复制最接近的 YAML 并修改即可，不要复制整套 Python 流程。

如果新增真正的新策略：

1. 在对应子包中新建可单测的实现。
2. 在 `config.py` 中注册并校验新的 strategy 值。
3. 在 `pipeline.py` 中只增加策略调度。
4. 新建唯一命名的实验 YAML。
5. 增加针对策略本身的测试。
6. 在 `docs/experiment_history.md` 记录目的和结果。

## 6. 候选输出格式

所有新代码必须写规范 schema，定义位于 `src/cve2attack/schemas.py`：

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
      "metadata": {}
    }
  ]
}
```

必须保持的约束：

- 新输出只能使用 `candidates`，不要重新输出旧字段 `techniques`。
- 旧的 `techniques: ["T..."]` 和 `techniques: [{"id": ..., "score": ...}]` 仅由共享 reader 兼容。
- 子技术 ID 默认通过 `parent_technique_id` 上卷到顶层 Technique；修改这一规则会改变 benchmark 口径，必须同时更新测试和实验说明。
- 候选必须保持排序，写入和读取过程中不能用无序集合代替最终列表。
- 新增字段优先放入 `metadata`，不要随意改变顶层 schema。

## 7. 评估不可破坏的约束

过去出现过输出格式不一致导致指标全部为 0，以及仅在“同时存在预测的 CVE”上计算指标的问题。修改评估时必须遵守：

- 分母是所选 benchmark 的完整固定 cohort，不是预测文件与 benchmark 的交集。
- 没有预测的 benchmark CVE 计为未命中。
- `coverage` 单独报告，不能用过滤缺失预测的方法提高 Recall。
- 两套论文 benchmark 分开报告。
- 比较多个 run 时，必须显式传入同一个 `--benchmark`。
- schema 解析统一走 `CandidateRecord.from_dict`/`candidate_records`，不要在评估脚本中再次手写格式判断。

任何评估修改至少要覆盖 `tests/test_metrics.py` 中“缺失预测降低覆盖率且计为 miss”的行为。

## 8. 运行结果

每次运行创建新的 `runs/<run_id>/`，不得覆盖已有目录。典型内容：

```text
runs/<run_id>/
├── manifest.json             # resolved config、commit、状态、覆盖信息
├── candidates/               # 按年份保存的规范候选 JSONL
├── metrics.json
└── report.md
```

- `runs/` 和 `comparisons/` 被 Git 忽略，不要把大批实验输出提交到源码历史。
- 需要长期保留的结论应整理进 `docs/experiment_history.md`；需要版本化的小型指标摘要应单独设计明确文件，而不是把整个 run 加入 Git。
- 历史混合结果保存在 `runs/legacy_import_20260628/`，不要把它当作新流水线的默认输出目录。
- 不要删除用户已有 run、cache 或 `archive/local_bak/`，除非用户明确要求。

## 9. 常用开发和验证命令

在仓库根目录运行：

```bash
.venv/bin/pip install -e . --no-deps

.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python -m cve2attack validate experiments/v3a_llm_rewrite.yaml

.venv/bin/python -m cve2attack inspect experiments/v3a_llm_rewrite.yaml

.venv/bin/python -m cve2attack run experiments/v3a_llm_rewrite.yaml --max-cves 2
```

比较两个已完成运行：

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark cve2attack_result \
  runs/<run-a> runs/<run-b>
```

建议的验证顺序：

1. `validate` 检查配置结构。
2. `inspect` 检查路径、查询覆盖和 Technique 数量，不加载嵌入模型。
3. 运行单元测试。
4. 使用 `--max-cves 2` 或另一小样本执行真实模型冒烟测试。
5. 只有在用户需要时再运行完整 benchmark。

`rewrite` 会调用实验 YAML 中配置的外部 Ollama 服务，并写入昂贵缓存。执行大规模重写前先检查服务、benchmark 和 cache 路径，优先用 `--max-cves` 小规模验证。不要因为缺少一个 rewrite 就无条件重建整个 cache。

## 10. 修改完成前检查

- 改动发生在活动目录，而不是 `archive/`。
- 配置路径相对项目根目录解析，没有写入个人绝对路径。
- 新方法有唯一 experiment name，不覆盖旧实验定义。
- 新 run 使用新目录，不覆盖旧结果。
- 两套 benchmark 没有被隐式合并。
- 输出仍能由 `CandidateRecord.from_dict` 读取。
- 固定 cohort、missing-as-miss 和 coverage 语义保持正确。
- 相关单元测试已增加或更新，并全部通过。
- 至少完成 `validate` 和 `inspect`；涉及模型流程时完成小样本冒烟测试。
- 没有提交 `runs/`、`comparisons/`、embedding cache、日志、虚拟环境或 `__pycache__`。
- 若实验含义或结论发生变化，同步更新 README/实验历史，而不是只改代码。
