# Stage 1 工作连续性文档

> 本文是持续维护的当前状态说明，不是迁移快照、聊天记录或完整研究日志。新接管者应先按“接管只读检查清单”核对实际工作区，再开展任何修改或实验。

## 0. 当前状态摘要

| 项目 | 当前值 |
| --- | --- |
| 最后内容更新时间 | `2026-07-29T17:38:31+08:00` |
| 最后事实核查时间 | `2026-07-29T17:38:31+08:00` |
| SSH 别名 | `sun_demi` |
| Stage 1 工作区 | `/home/ghdemi/Code/cve2attack` |
| Stage 1 冻结代码基线 | `5912587e1376825457239316faa8d42c6f11e07a` (`Freeze-Stage-1-action-retrieval`) |
| 最后核查时的分支 / HEAD | `refactor/new-method-stage1` / `5912587e1376825457239316faa8d42c6f11e07a` |
| 最后核查时的上游状态 | `origin/refactor/new-method-stage1`；ahead `0` / behind `0` |
| 文档生成 / 最后核查时的工作区状态 | 已跟踪文件无修改、无暂存；本文件尚未提交；另有两份本轮开始前已存在的未跟踪 rewrite cache，详见第 14 节 |
| Stage 1 状态 | 已冻结；正式方法是严格 LOO 的 V5c、固定 Top-20、label-free 候选生成 |
| 当前目标 | 保持冻结结果可追溯并向 Stage 2 提供受控候选；除非出现重新开启条件，否则不继续在 TRIAGE test 上调参 |
| 下一步 | 用户审核本文；若获授权，只显式暂存本文。研究默认转向 Stage 2 端到端验证 |
| 硬阻塞 | 无 |
| 当前风险 | 根 V5c 配置未显式固定 ATT&CK 15.1；正式 TRIAGE run manifest 记录的是冻结前提交；两套历史数据集引文仍缺失；无持久化测试日志 |

上述分支 HEAD、上游同步和工作区字段是 `2026-07-29T17:38:31+08:00` 的状态快照，不是自动更新的实时值。连续性文档自身的纯文档提交可能使实际分支 HEAD 和 ahead 数量高于这里记录的值，但不会因此改变 Stage 1 冻结代码基线 `5912587`。新任务接管时必须重新执行只读 Git 检查，并把“冻结代码基线”“文档提交”和“实时分支 HEAD”分开记录，不能只凭本文快照判断当前状态。

### 0.1 事实来源优先级

发生冲突时按以下顺序裁决：

1. 实际工作区、实际数据文件、run/comparison 结果文件；
2. Git 状态、Git 历史和实际代码；
3. 实验配置与 benchmark 元数据；
4. 项目文档；
5. 对话记忆。

本文用以下标签区分信息性质：

- **已确认事实**：可由当前文件、代码、Git 或结果直接复核。
- **研究决策**：已经作出的方法或论文口径选择，不等同于客观定理。
- **待复核**：证据不足或存在可复现性缺口，不允许靠猜测补全。
- **建议任务**：未来满足条件时可执行，不表示已获授权。

## 1. 文档用途与维护边界

本文为后续 Codex 任务和研究者提供以下连续性：

- 当前代码与 Git 基线；
- Stage 1 的职责、冻结方法和研究边界；
- 数据、配置、命令、结果与 Stage 2 接口地图；
- 已知风险、失败经验、受保护资产和下一步条件。

本文不是 `AGENTS.md` 的替代品。结构、完整命令参数和格式细节仍以 `AGENTS.md` 与代码为准；路线决策参考 `STAGE1_PLAN.md`；实验历史参考 `docs/experiment_history.md`。当这些文档与实际结果冲突时，遵循第 0.1 节。

## 2. Stage 1 在完整映射流程中的职责

Stage 1 只负责：对每个 CVE 生成一个排序稳定、候选预算受控、带可追溯证据的 ATT&CK **父 Technique** 候选列表。

Stage 1 不负责：

- 解析攻击图；
- 使用攻击图上下文重排候选；
- 在攻击图中作最终 Technique 判定；
- 把第二阶段上下文提前泄漏到第一阶段检索。

完整关系是：

```text
CVE 描述 -> Stage 1 候选生成（正式 Top-20 CandidateRecord）
         -> Stage 2 攻击图上下文连接与重排
         -> 图中最终映射与端到端评价
```

**已确认事实**：`main` 与 `new_method` 是 Stage 1 内的两种不同候选生成方法，不是 Stage 1 与 Stage 2 的关系。当前仓库仍有本地分支 `main`、`new_method` 和当前延续分支 `refactor/new-method-stage1`。Stage 2 使用独立工作树 `/home/ghdemi/Code/cve2attack-stage2` 和分支 `feat/full-pipeline-stage2`。

## 3. Git 状态快照与工作树基线

### 3.1 Stage 1

最后核查时间：`2026-07-29T17:38:31+08:00`。

```text
worktree:                   /home/ghdemi/Code/cve2attack
branch at last check:       refactor/new-method-stage1
branch HEAD at last check:  5912587e1376825457239316faa8d42c6f11e07a
frozen Stage-1 code base:   5912587e1376825457239316faa8d42c6f11e07a
upstream at last check:     origin/refactor/new-method-stage1
sync at last check:         ahead 0, behind 0
```

本文创建前，`git status --short --branch` 只有两份未跟踪 rewrite cache；已跟踪文件无修改。本文创建并完成本轮修订后额外出现 `?? docs/continuity/STAGE1_CONTINUITY.md`。当前连续性文档尚未提交，未经用户审核不得暂存、提交或推送。

本节记录的是指定时间点的 Git 快照。若之后只提交本文，分支 HEAD 和相对上游的 ahead 数会变化，提交哈希应作为“文档提交”单独记录；Stage 1 冻结代码基线仍保持 `5912587`。只有 Stage 1 方法或冻结代码正式改变时，才更新冻结代码基线。任何新任务都必须重新核对实时 Git 状态。

### 3.2 Stage 2（只读接口证据）

核查时的 Stage 2 工作树：

```text
worktree: /home/ghdemi/Code/cve2attack-stage2
branch:   feat/full-pipeline-stage2
HEAD:     be5ba41fac5231c0f277c09581c5f3812a52a4d8
upstream: origin/feat/full-pipeline-stage2
sync:     ahead 3
tracked working tree: clean
```

Stage 1 任务不得修改该工作树。第 12 节的接口结论来自这个具体 Stage 2 HEAD；如果 Stage 2 HEAD 改变，必须重新核查。

## 4. 正式冻结方案 V5c

### 4.1 方法身份与命名

正式 Top-20 配置是：

```text
experiments/v5c_raw_action_rank_rrf.yaml
```

冻结 ATT&CK 15.1 的跨基准配置是：

```text
experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml
```

TRIAGE 完整 202-parent-Technique 诊断 run 使用历史编号 V5k：

```text
config: experiments/diagnostics/v5k_raw_action_loo_rank_rrf_fullranking.yaml
run:    runs/triage_diag_v5k_raw_action_loo_rank_rrf_fullranking/
```

V5k 与正式 V5c 的核心算法相同：原始 CVE 查询、三类 action、严格 query-specific LOO、Top-3 action rank-RRF。主要区别是诊断配置保留 202 个父 Technique，而正式交付只保留 Top-20。后续论文与接口称其为“正式 V5c”；引用历史产物时同时写出 V5k run ID，避免名称混淆。

### 4.2 完整方法

**已确认事实**：

- 查询：`data/raw/cve/` 中的原始 CVE description，不使用 benchmark 标签、KEV comments 或 TRIAGE/SMET reference predictions。
- 检索模型：`basel/ATTACK-BERT`。
- action 语料类型：
  - `technique_description`：父 Technique 描述；
  - `subtechnique_description`：子技术描述；
  - `procedure`：ATT&CK `uses` relationship 的 procedure 文本。
- 文本过滤：最少 20 字符，最多 1200 字符。
- 子技术处理：子技术 action 可以提供细粒度检索证据，但在聚合时由 `Txxxx.xxx` 上卷到 `Txxxx`；正式输出只含父 Technique。
- ATT&CK 15.1 的正式 TRIAGE action corpus 有 14,121 个独立 action，覆盖 202 个有效父 Technique。
- 每个 action 独立编码；按 query 与 action 的余弦相似度获得 action 排名。
- 对每个父 Technique，最多取其排名最前的 3 个 action，累加 `1 / (60 + action_rank)`。
- Technique 排序依次按聚合分数、最佳 action 相似度、Technique ID 确定性打破平局。
- 正式输出 `top_k=20`；诊断配置 `top_k=202`，不能把完整排名当作实际候选预算。
- 这是单路 action 聚合，不是 V1/V3/V5 多来源之间的 RRF 融合；`fusion.strategy` 为 `none`。

主要实现证据：

- `src/cve2attack/retrieval/action_kb.py`
- `src/cve2attack/retrieval/action_generator.py`
- `src/cve2attack/pipeline.py`
- `src/cve2attack/schemas.py`

### 4.3 严格 LOO 与防泄漏机制

V5c 的 LOO 是 query-specific leave-one-CVE-out：

1. action 原文中的 `CVE-*` 和旧式 `CAN-*` 被抽取为规范化 CVE ID；
2. 送入嵌入模型前，精确编号替换为 `[VULNERABILITY]`；
3. 多条 action 在屏蔽编号后若变成同一文本，去重时合并其全部 vulnerability IDs，避免因只保留第一条而绕过排除；
4. 对当前 query CVE，聚合前排除所有原文曾提及该 CVE 的 action；
5. 被排除 action 不计入有效 action rank；
6. 每条 `CandidateRecord.metadata.excluded_query_cve_actions` 保存该 query 的排除数量。

这能阻止“相同 CVE 编号直接匹配”，但不能消除 ATT&CK procedure 中产品名、利用行为和上下文带来的合理语义信息。procedure 覆盖偏置必须继续披露。

### 4.4 ATT&CK 版本核查与当前配置风险

两个真实 STIX 文件**不一致**：

| 文件 | Collection 版本 | 大小 | SHA-256 | active parent / sub-technique |
| --- | ---: | ---: | --- | ---: |
| `data/knowledge/enterprise-attack.json` | 18.1 | 50,713,170 B | `f857d8f78f2f0c0b7db321a711a39fba98546c1e3076a657684850c83d0962fb` | 216 / 475 |
| `data/knowledge/enterprise-attack-15.1.json` | 15.1 | 43,474,692 B | `a57988bffe402bb3e19d92dbe80a12143e1970b814e013e080f9df2fa5a3f6bc` | 202 / 435 |

补充证据：18.1 bundle 的 collection modified 为 `2025-11-13T14:00:00.188Z`；15.1 为 `2024-05-02T14:00:00.188Z`。两者 Git blob 分别为 `c634454a...` 和 `190e688f...`。

**重要风险**：`experiments/v5c_raw_action_rank_rrf.yaml` 没有显式 `technique_document.attack_bundle`；其默认 benchmark `cve2attack_result/dataset.yaml` 也没有 `technique_corpus`。按 `resolve_attack_bundle()` 的实际代码，直接运行该根配置会回退到 `data/knowledge/enterprise-attack.json`，即当前 18.1，而不是 15.1。

**研究决策**：冻结评测口径是 ATT&CK Enterprise 15.1。复现正式跨基准结果必须使用 `experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml`，或使用在 `dataset.yaml` 中固定 15.1 的 KEV/TRIAGE benchmark。不得把根配置按 18.1 运行的结果与已冻结 15.1 表格混为同一实验条件。

**待复核/建议任务**：若未来要修复根配置，应在单独获批的代码轮次显式固定 15.1、验证配置解析与缓存键，并记录为新提交；本轮未修改配置。

## 5. 代码、配置与结果目录地图

| 内容 | 位置 | 作用 |
| --- | --- | --- |
| 项目结构和 CLI 总说明 | `AGENTS.md` | 人与 agent 的主要结构参考 |
| Stage 1 路线与边界 | `STAGE1_PLAN.md` | 研究决策、冻结条件、工作包 |
| 实验历史 | `docs/experiment_history.md` | 已完成实验与结果指针 |
| 正式方法配置 | `experiments/v5c_raw_action_rank_rrf.yaml` | 正式 Top-20 方法定义；当前有 ATT&CK 回退风险 |
| 15.1 验证配置 | `experiments/validation/` | 跨 benchmark 冻结版本运行 |
| 完整排名配置 | `experiments/diagnostics/` | Top-202 诊断，不是交付预算 |
| 语料消融配置 | `experiments/ablations/` | parent/subtechnique/procedure 消融 |
| CLI | `src/cve2attack/cli.py` | `validate`、`inspect`、`run`、`rewrite`、比较与诊断入口 |
| 配置解析 | `src/cve2attack/config.py` | 默认值、YAML 合并与策略校验 |
| 总流水线 | `src/cve2attack/pipeline.py` | 输入、ATT&CK corpus 解析、run manifest、候选与评测 |
| Benchmark/候选读取 | `src/cve2attack/data/loaders.py` | JSONL、truth、CandidateRecord 读取 |
| 候选 schema | `src/cve2attack/schemas.py` | `CandidateRecord`、父 Technique 上卷与历史兼容 |
| 父 Technique 语料 | `src/cve2attack/retrieval/technique_kb.py` | V1/V2 Technique 文档 |
| action 语料 | `src/cve2attack/retrieval/action_kb.py` | 三类 action、编号屏蔽、去重、上卷 |
| action 检索与聚合 | `src/cve2attack/retrieval/action_generator.py` | 缓存、批处理、LOO、max/rank-RRF、evidence |
| 模型封装 | `src/cve2attack/retrieval/embedder.py` | SentenceTransformer 与离线模型加载 |
| 多来源 RRF | `src/cve2attack/fusion/rrf.py` | 已完成但未选为正式 V5c 的跨 run 融合基线 |
| 常规指标 | `src/cve2attack/evaluation/metrics.py`、`ranking.py` | Recall/Hit/MAP 等 |
| TRIAGE 复评 | `src/cve2attack/evaluation/triage.py` | all/no-secondary 与公开预测统一口径 |
| 互补诊断 | `src/cve2attack/evaluation/diagnostics.py` | 曲线、rank、交集、oracle、分组 |
| 最终 action 审计 | `src/cve2attack/evaluation/action_final.py` | 消融、procedure 偏置、案例分类 |
| 单元测试 | `tests/` | 当前静态计数 37 个 `test_*` 方法 |
| 运行产物 | `runs/<run_id>/` | manifest、候选、指标、报告；被 Git 忽略 |
| 比较产物 | `comparisons/<comparison_id>/` | metrics、report、逐例诊断；被 Git 忽略 |
| Benchmark | `data/benchmarks/<name>/` | 年度 JSONL、dataset.yaml、可选 cohort |
| 原始 CVE | `data/raw/cve/` | 原始描述、CWE、CPE 等 |
| ATT&CK/CWE/CAPEC | `data/knowledge/` | 版本化知识源 |
| rewrite cache | `data/derived/rewrite_cache/` | 昂贵、可续跑的 LLM 改写 |
| embedding cache | `data/derived/embedding_cache/` | 可重建但昂贵的模型向量 |

## 6. CandidateRecord 输出契约

Schema 版本为 `1.0`。每个年度候选文件位于：

```text
runs/<run_id>/candidates/CVE-<year>.jsonl
```

每行至少包含：

```json
{
  "schema_version": "1.0",
  "cve_id": "CVE-2023-20887",
  "domain": "Enterprise",
  "candidates": [
    {
      "technique_id": "T1190",
      "score": 0.031234,
      "sources": ["action_embedding"],
      "metadata": {
        "name": "Exploit Public-Facing Application",
        "tactics": ["initial-access"],
        "retrieval_corpus": "action",
        "aggregation": "rank_rrf",
        "best_action_similarity": 0.712345,
        "corpus_action_count": 73,
        "action_evidence": []
      }
    }
  ],
  "metadata": {
    "excluded_query_cve_actions": 0
  }
}
```

字段约束：

- `cve_id`：Stage 2 连接键；`CAN-*` 在 Stage 2 上下文侧会规范为 `CVE-*`。
- `candidates`：有序数组；正式 V5c 至多 20 项。
- `technique_id`：规范化父 Technique；读取器会把历史 `Txxxx.xxx` 上卷并去重。
- `score`：可选；Stage 2 当前保留但不用于拓扑规则匹配。
- `sources`、candidate `metadata`：可追溯证据；Stage 2 当前保留。
- record `domain` 与 `metadata`：Stage 2 写入 `stage1` 区块供审计。

## 7. 数据集与评测用途

| Benchmark | 规模 | 用途与边界 |
| --- | ---: | --- |
| `cve2attack_result` | 1,661 CVE，2008–2022 | 历史论文数据集，完整保留；`citation: null`，不可把来源细节写得比元数据更确定 |
| `data_result` | 286,461 CVE，1999–2025 | 与前者不同的数据集，完整保留；引文和标签权威性尚未核实 |
| `data_result_hash_sample_2000` | 2,000 | 标签无关固定规模验证；不能冒充权威人工主基准 |
| `ctid_kev_2025_02_13_all` | 296 | CTID KEV exploitation + primary impact + secondary impact 并集 |
| `ctid_kev_2025_02_13_exploitation` | 284 | 只评显式 exploitation technique，最贴近 Stage 1 候选职责 |
| `ctid_kev_2025_02_13_nonoverlap` | 251 | 去除 `cve2attack_result` 重叠 CVE，严格外部视图 |
| `triage_2025_test_all` | 60 CVE / 143 父标签 | TRIAGE 公开固定 test split，同 cohort 对比 V1、V5c、SMET、TRIAGE |
| `triage_2025_test_no_secondary` | 同 60 CVE / 115 父标签 | 排除 secondary impact 的语义消融；不是独立测试集 |

`data_result_hash_sample_2000` 的成员选择为 `smallest_sha256(seed + NUL + cve_id)`，seed 为 `stage1-v5-multibench-20260727`；只使用 CVE ID，不看标签。

KEV 三个视图来自同一 CTID 2025-02-13、ATT&CK 15.1 快照，不是三份独立标注。原始 CSV SHA-256 为 `8f15aab468f17f9a1d655ef2db814b0323792cfa066373a02a0a1d7f4a8f6676`，来源记录为 Zenodo `16747173`。

TRIAGE 两个视图来自同一 296-CVE CTID 标签和同一 236/60 公开划分；其价值是与作者公开预测严格同 cohort，而不是提供新的独立人工标签源。TRIAGE 论文链接记录为 arXiv `2508.18439`，复现实验包为 Zenodo `17341504`。

## 8. 正式评测口径与防泄漏约束

### 8.1 指标定义

- `Micro Recall@K`：所有 CVE 的命中标签数之和 / 所有相关标签数之和；TRIAGE 论文主口径。
- `Macro Recall@K`：先算每个 CVE 的 Recall@K，再对固定 cohort 平均；跨 benchmark 配对表使用该口径。
- `Hit@K`：至少命中一个真值标签的 CVE 比例。
- `MAP`：使用已保存的全部候选深度计算；不同保存深度之间需要谨慎比较。
- `coverage`：固定 cohort 中存在候选预测的 CVE 比例；缺失预测保留在分母并计为 miss。
- 正式候选预算是 Top-20；Top-202、Top-50、union oracle 只作诊断。

Micro 与 Macro 不能混写。例如 TRIAGE V5c 的 Micro R@20 为 60.14%，同一运行的 Macro R@20 为 61.98%。

### 8.2 数据拆分与禁止事项

- TRIAGE 60-CVE test 是冻结测试集，不用于反复搜索 prompt、权重、source depth 或逐例规则。
- label-free 主方法不能读取 benchmark 真值、KEV comments、TRIAGE/SMET reference predictions。
- 子技术真值默认上卷到父 Technique；同一父 Technique 去重。
- 必须记录 ATT&CK corpus 的路径、版本和校验值。
- union oracle 的平均候选数大于 20，不能作为 Recall@20 主结果。
- 若未来训练 reranker，只能使用 TRIAGE 236-CVE train split 开发并明确标为 `label_efficient`；冻结 60-CVE test 只做最终评价。
- 不新增人工标注是当前约束。

## 9. 主要实验结果

### 9.1 TRIAGE all：Micro Recall

固定 60 CVE、143 个上卷父标签：

| 方法 | Micro R@10 | Micro R@20 |
| --- | ---: | ---: |
| V1 raw + ATTACK-BERT | 23.78% | 37.76% |
| 正式 V5c strict LOO | 44.06% | 60.14% |
| SMET 公开预测 | 37.76% | 52.45% |
| TRIAGE 公开监督预测 | 69.93% | 76.92% |

V5c 相对 V1 提升 22.38 个 Micro R@20 百分点，相对 SMET 高 7.69 点。TRIAGE 使用约 236 个标注训练/示例 CVE 和标签感知组件，是监督参考上界，不是同监督公平基线。

正式证据：

- run：`runs/triage_diag_v5k_raw_action_loo_rank_rrf_fullranking/`
- 统一比较：`comparisons/triage_action_v5_leave_one_cve_out/`
- 最终审计：`comparisons/triage_stage1_v5c_final_audit/`

### 9.2 TRIAGE no_secondary

**已确认存在统一正式结果**，不需要另建一份候选 run：同一 V5c 排名在排除 secondary impact 的 60-CVE、115-label 视图上复评。

| 方法 | Micro R@10 | Micro R@20 | Macro R@20 |
| --- | ---: | ---: | ---: |
| 正式 V5c / V5k run | 49.57% | 66.09% | 66.28% |
| SMET 公开预测 | 43.48% | 59.13% | 58.17% |
| TRIAGE 公开监督预测 | 72.17% | 76.52% | 78.17% |

对应文件：

- `comparisons/triage_action_v5_leave_one_cve_out/metrics.json`
- `comparisons/triage_action_v5_leave_one_cve_out/report.md`
- `comparisons/triage_action_v5_leave_one_cve_out/disagreements_v5k_raw_action_loo_rank_rrf_fullranking_triage_diag_v5k_raw_action_loo_rank_rrf_fullranking.jsonl`

`comparisons/triage_stage1_v5c_final_audit/` 主要针对 all 视图的最终语料与偏置审计，本身没有单独重复 no_secondary 表；no_secondary 的统一公开比较以上述 comparison 为准。

### 9.3 多基准配对 Macro Recall

所有正式运行 coverage 为 100%。区间为 10,000 次 CVE 级配对 bootstrap，seed `20260728`；下表所有 R@20 差值区间均完全高于 0。

| Benchmark | CVE | V1 R@10 / R@20 | V5c R@10 / R@20 | Delta R@20 (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| TRIAGE all | 60 | 26.40% / 39.43% | 45.49% / 61.98% | +22.55 [14.10, 31.32] pp |
| `cve2attack_result` | 1,661 | 35.44% / 47.83% | 44.83% / 58.53% | +10.70 [8.96, 12.47] pp |
| KEV all | 296 | 22.77% / 35.41% | 45.91% / 60.51% | +25.10 [21.12, 29.05] pp |
| KEV exploitation | 284 | 22.50% / 35.14% | 46.09% / 60.42% | +25.28 [21.34, 29.35] pp |
| KEV nonoverlap | 251 | 24.91% / 36.52% | 46.10% / 61.43% | +24.90 [20.59, 29.24] pp |
| `data_result` hash sample | 2,000 | 61.30% / 73.75% | 77.07% / 85.11% | +11.36 [9.79, 12.96] pp |

正式配对报告：

```text
comparisons/multibench_final_triage_v1_vs_v5c_paired/
comparisons/multibench_final_cve2attack_v1_vs_v5c_paired/
comparisons/multibench_final_kev_all_v1_vs_v5c_paired_retry/
comparisons/multibench_final_kev_exploitation_v1_vs_v5c_paired/
comparisons/multibench_final_kev_nonoverlap_v1_vs_v5c_paired/
comparisons/multibench_final_data_sample2000_v1_vs_v5c_paired/
```

主要 15.1 run：

```text
runs/multibench_cve2attack_v1_attack15_1_retry/
runs/multibench_cve2attack_v5c_action_attack15_1/
runs/kev_v1_raw_attackbert_15_1/
runs/multibench_kev_all_v5c_action_attack15_1/
runs/multibench_data_sample2000_v1_attack15_1/
runs/multibench_data_sample2000_v5c_action_attack15_1/
```

### 9.4 消融、偏置与案例诊断

固定 TRIAGE all 口径：

| 语料 | Micro R@10 | Micro R@20 | Macro R@20 |
| --- | ---: | ---: | ---: |
| V1 parent Technique 文档 | 23.78% | 37.76% | 39.43% |
| V5 parent 描述 only | 18.88% | 34.97% | 36.74% |
| V5 sub-technique 描述 only | 21.68% | 31.47% | 30.67% |
| V5 parent + sub-technique 描述 | 22.38% | 30.77% | 30.43% |
| procedure-only strict LOO | 46.15% | 61.54% | 64.06% |
| **正式 V5c 全 action strict LOO** | **44.06%** | **60.14%** | **61.98%** |

**研究决策**：procedure-only 略高，但它是查看冻结测试结果后的 post-hoc 消融，只用于解释增益来源，不能替换已经完成多基准验证的正式 V5c。

procedure 数量偏置：

- procedure count 与 V5c Top-20 exposure 的 Spearman rho：`0.596`；
- procedure count 与 false-positive exposure：`0.593`；
- procedure count 与有标签 Technique R@20：`0.314`。

逐标签案例：37 个 V5c 新命中，49 个 V1/V5c 共同命中，5 个 V1 命中被 V5c 丢失，19 个未解决标签位于 21–50，33 个位于 50 以后。

最终审计文件：

```text
comparisons/triage_stage1_v5c_final_audit/report.md
comparisons/triage_stage1_v5c_final_audit/summary.json
comparisons/triage_stage1_v5c_final_audit/cases.jsonl
comparisons/triage_stage1_v5c_final_audit/technique_bias.jsonl
```

### 9.5 未选用的融合结果

固定等权、`rank_constant=60`、`source_depth=50` 的跨来源 RRF 已完成，但 V1+V5k 的 Micro R@20 只有 48.25%，V1+V5e+V5k 为 46.15%，均低于 V5k 单路 60.14%。不再围绕冻结 TRIAGE test 搜索权重。

union oracle 可以展示互补覆盖，但平均候选数超出 20；不能当作正式受控候选结果。

## 10. 正式 run 的历史可复现性缺口

`runs/triage_diag_v5k_raw_action_loo_rank_rrf_fullranking/manifest.json` 记录：

```text
git_commit: aa0379a4d2031a1afe79f5407900dafb69507f20
status: complete
technique_corpus: data/knowledge/enterprise-attack-15.1.json
technique_count: 202
retrieval_document_count: 14121
candidate_records: 60
embedding_cache: data/derived/embedding_cache/actions_f0d54a569643e79e.npz
```

但 action retrieval 代码最终在 `4227472` 提交，冻结补丁在 `5912587`。该 run 是代码尚未提交时生成的，因此 manifest 的 `git_commit=aa0379a` 只反映当时 HEAD，不能完整表达脏工作区中的 action 代码。

**已确认事实**：后续提交保存并冻结了对应实现与测试，结果文件未被覆盖。

**待复核/建议任务**：若论文要求完全 clean-HEAD 的工件指纹，应在单独授权后，用冻结配置、冻结 ATT&CK 15.1 和独立新 run ID 做最小确定性核验或完整复跑，并对候选文件校验；不得覆盖原 run。本文不声称该 clean-HEAD 复跑已经完成。

## 11. 可复现命令与成功判断

以下命令只说明工作流；接管时不要未经授权直接运行正式实验。

### 11.1 只读检查

```bash
cd /home/ghdemi/Code/cve2attack
git branch --show-current
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
git rev-list --left-right --count HEAD...@'{upstream}'
git status --short --branch
```

### 11.2 配置验证、输入检查和测试

```bash
.venv/bin/python -m cve2attack validate \
  experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml

.venv/bin/python -m cve2attack inspect \
  experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml

.venv/bin/python -m unittest discover -s tests -v
```

成功判断：

- `validate` 输出配置有效；
- `inspect` 显示预期 cohort、ATT&CK 15.1 corpus、无缺失描述；
- 测试全部通过且无错误。

冻结前曾观察到 **37/37** 测试通过。本轮只读补查没有重跑测试；仓库没有持久化测试日志。当前静态代码中有 37 个 `test_*` 方法，这不等于本轮执行成功证明。

### 11.3 冒烟运行

```bash
.venv/bin/python -m cve2attack run \
  experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml \
  --max-cves 2 \
  --run-id <new_unique_smoke_run_id>
```

成功判断：新目录 `runs/<id>/manifest.json` 为 `status: complete`，有 2 条候选记录、每条不超过 20 个父 Technique，并写出 metrics/report。不得复用已有 run ID。

### 11.4 正式 15.1 候选运行

```bash
.venv/bin/python -m cve2attack run \
  experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml \
  --benchmark <benchmark_name> \
  --run-id <new_unique_run_id>
```

参数：

- `--benchmark`：覆盖输入 benchmark，同时保持 validation 配置显式的 ATT&CK 15.1。
- `--run-id`：在 `runs/` 下创建不可覆盖的新目录。
- 正式结果不要使用 `--max-cves`。

正式成功判断：manifest complete、input coverage 与候选 coverage 均为 100%、`technique_corpus` 指向 `enterprise-attack-15.1.json`、每条正式候选不超过 20。

### 11.5 统一比较

```bash
.venv/bin/python -m cve2attack compare \
  --benchmark <benchmark_name> \
  --comparison-id <new_unique_comparison_id> \
  runs/<v1_run> runs/<v5c_run>

.venv/bin/python -m cve2attack compare-triage \
  --comparison-id <new_unique_comparison_id> \
  runs/<fullranking_or_top20_run>
```

两个 run 的 `compare` 会输出 CVE 级配对 Recall 差值、10,000 次 bootstrap 区间以及 improved/same/worse。`compare-triage` 同时生成 all/no-secondary 与 mapping type 指标，不重新加载模型。

### 11.6 V3 rewrite（仅补充消融，不是冻结主方法）

```bash
.venv/bin/python -m cve2attack rewrite \
  experiments/v3a_llm_rewrite.yaml \
  --benchmark <benchmark_name> \
  --workers 1
```

- `--workers` 默认 4；最小有效值为 1。
- `--max-cves` 可用于小规模检查。
- 禁止随意使用 `--no-cache`，它会忽略已有昂贵缓存并重新生成目标内容。
- `run` 只读取 rewrite cache，不会自动调用 Ollama 补齐缺失项。

## 12. Stage 1 到 Stage 2 的实际接口

本节由 `/home/ghdemi/Code/cve2attack-stage2` 的实际代码在 HEAD `be5ba41` 上只读核查得到，主要证据：

- `src/cve2attack/cli.py`
- `src/cve2attack/data/loaders.py`
- `src/cve2attack/schemas.py`
- `src/cve2attack/stage2/candidate_joiner.py`
- `src/cve2attack/stage2/pipeline.py`
- `src/cve2attack/stage2/reranker.py`
- `docs/stage2_graph_context.md`

### 12.1 Stage 2 实际读取位置

`run-stage2 --stage1-run PATH` 接受 Stage 1 run 目录。读取器：

1. 如果 `PATH/candidates/` 存在，读取该目录；
2. 否则直接把 `PATH` 当候选目录；
3. 只读取匹配 `CVE-*.jsonl` 的年度文件；
4. 若 run 有 `manifest.json` 且 `status` 非空，只接受 `complete` 或 `imported`；无 manifest 时仍可读取，但来源追踪较弱。

### 12.2 是否严格要求 Top-20

**不严格要求**。代码没有断言候选数必须等于或不超过 20，也没有再次截断；它把 `CandidateRecord` 中的全部候选传入 Stage 2，并在当前 reranker 中“只改变顺序、不改变集合”。

研究计划和正式契约“通常为 Top-20”，因此正式交付仍必须使用 Top-20 run。若误把 Top-202 诊断 run 交给 Stage 2，代码会接受并重排全部 202 项，这会破坏论文候选预算但不一定报错。

### 12.3 Stage 2 使用的字段

- `CandidateRecord.cve_id`：与攻击图上下文规范化 CVE ID 连接；重复 CVE 会报错。
- `CandidateRecord.candidates`：完整有序候选列表。
- `CandidateRecord.domain` 和 record `metadata`：写入输出的 `stage1` 审计区块。
- candidate `technique_id`：拓扑规则匹配的实际字段。
- candidate `score`、`sources`、candidate `metadata`：原样保留；当前拓扑规则不依赖 score。
- Stage 2 在 candidate metadata 下新增 `metadata.stage2.original_rank`、`topology_match`、`matched_rules`、`reranked_rank` 等字段。

### 12.4 子技术与父 Technique

Stage 2 共享 `CandidateRecord.from_dict()` 和 `parent_technique_id()`。读取历史候选时会把 `Txxxx.xxx` 上卷到 `Txxxx` 并对父 ID 去重。当前拓扑规则使用父 Technique ID（如 `T1190`、`T1210`、`T1068`）。因此 Stage 1 正式输出父 Technique 与 Stage 2 当前实现一致。

## 13. 运行环境与模型依赖

### 13.1 主机职责

| 主机 | 职责 |
| --- | --- |
| `172.23.216.47`（SSH `sun_demi`） | 代码、Git 工作树、数据、ATTACK-BERT/HF 缓存、rewrite cache、runs、comparisons |
| `172.23.216.73:11434` | Ollama 服务与 `sec-i1` 权重；47 通过 HTTP 调用 |

不要把 73 上的模型误认为被复制到 47。仓库中的 `models/ollama/sec-i1-cve-rewrite-v1.Modelfile` 只是版本化运行模板和参数，不包含 GGUF 权重。

### 13.2 47 当前环境

核查值：

```text
hostname: ghdemi-virtual-machine
kernel: Linux 6.8.0-124-generic
Python: 3.10.12
NumPy: 2.2.6
PyYAML: 6.0.3
sentence-transformers: 5.5.1
torch: 2.12.0+cu130
```

47 的 `PATH` 中没有 `nvidia-smi` 或 `ollama` CLI。仅凭 torch 是 CUDA build 不能断言当前进程可使用 GPU；运行前应实际核查设备。此前 ATTACK-BERT 在 47 上表现为 CPU 路径，首次加载与大规模编码会慢。

HF 缓存：

```text
/home/ghdemi/.cache/huggingface/hub/models--basel--ATTACK-BERT
size: about 837 MB
```

`/home/ghdemi/.bashrc` 已设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

项目代码支持缓存模型离线加载；不要因外网慢而先删除缓存。

### 13.3 73 上的 rewrite 模型

当前 Ollama `/api/tags` 实际返回：

```text
name: sec-i1-cve-rewrite:v1
size: 4,921,467,478 bytes
parameter_size: 8.0B
quantization: Q4_K_M
digest: 27956b7739fa343a2c9fbe9fcb2f56a6c3a541b4cd24bc6fc42830d14a67916a
modified: 2026-07-20T10:44:18.738038181+08:00
```

源标签 `sec-i1:latest` 也约 4.92 GB。曾观察到约 38 GB 的运行占用不是模型文件大小；旧标签按 131,072 上下文分配 KV cache 可显著放大 GPU 内存。版本化标签通过 Modelfile 固定：

```text
FROM sec-i1:latest
Llama 3 system/user/assistant template
num_ctx=8192
num_predict=512
temperature=0
seed=42
```

### 13.4 V3 rewrite 实际参数

`experiments/v3a_llm_rewrite.yaml` 与 `v3b_llm_rewrite_procedures.yaml` 均配置：

```text
endpoint: http://172.23.216.73:11434/api/generate
model: sec-i1-cve-rewrite:v1
timeout_seconds: 300
max_retries: 3
cache: data/derived/rewrite_cache/{benchmark}_sec_i1_llama3_chat_v1.json
```

代码语义：`max_retries=3` 实际是**总共最多 3 次尝试**，不是首次之外再重试 3 次；失败后等待 1 秒、2 秒。CLI `--workers` 默认 4，YAML 不固定 workers。失败、空响应都会计入 failed；成功结果每 20 个完成项原子 checkpoint，已有非空 cache 默认跳过。

## 14. 用户数据、缓存与禁止操作

### 14.1 两份原有未跟踪 rewrite cache

这两份文件在本文创建前已存在，属于用户/实验数据，不得删除、覆盖、重命名、提交或重新生成：

| 路径 | 大小 | mtime | SHA-256 |
| --- | ---: | --- | --- |
| `data/derived/rewrite_cache/ctid_kev_2025_02_13_all_sec_i1.json` | 150,488 B | `2026-07-20 10:18:46.209863634 +0800` | `b3506aa8e3139a3226af2abfae0549da8ecdfb04b8a840ef325fee55991168d4` |
| `data/derived/rewrite_cache/ctid_kev_2025_02_13_all_sec_i1_llama3_chat_v1.json` | 179,005 B | `2026-07-20 11:45:35.721674961 +0800` | `e25aa2bb25c8257a21b336d917319523d62d8104e03b2f52c34262a3a8746de5` |

`.gitignore` 当前没有忽略 `data/derived/rewrite_cache/`，所以它们会持续显示为 `??`。这不是清理理由。

### 14.2 受保护实验资产

核查时体积：

```text
runs/                             about 403 MB
comparisons/                      about 8.7 MB
data/derived/embedding_cache/     about 82 MB
data/derived/rewrite_cache/       about 1.1 MB
```

- `runs/` 与 `comparisons/` 被 Git 忽略，但包含冻结结果和论文可追溯证据，不得随意清理。
- embedding cache 可理论重建，但成本高且与 corpus/model 键相关；不要为“干净”而删除。
- rewrite cache 昂贵且可能无法逐字重现；默认视为受保护用户数据。
- `data_result`、`cve2attack_result`、KEV、TRIAGE benchmark 和 raw 来源不得混合、覆盖或删除。
- 禁止使用宽泛的 `git add .`、`git add -A`。需要提交本文时只能显式：

```bash
git add docs/continuity/STAGE1_CONTINUITY.md
```

提交前必须再次确认 `git diff --cached --name-status` 只有获批文件。

## 15. 已遇到的失败、易误判与无效工作

1. **旧 rewrite 模板错误**：`sec-i1:latest` 导入模板只有原始 prompt，忽略 system 角色，可能直接输出 EOS/空文本。不要把旧 `*_sec_i1.json` cache 改名冒充 `*_sec_i1_llama3_chat_v1.json`。
2. **并发不是所有失败的根因**：把 workers 从 4 改到 1 不能修复模板错误、服务端状态或超时。先看逐 CVE 异常和 Ollama 服务。
3. **max_retries 名称易误读**：当前实现表示总尝试次数。
4. **V3 run 不生成 rewrite**：必须先运行 `rewrite`；缺失项会在 input coverage 中记录并跳过。
5. **首次模型/向量慢**：47 的 ATTACK-BERT 可能走 CPU；已有 HF 和 embedding cache 时不要删除重拉。
6. **旧输出全部为 0**：历史上非标准输出格式导致评估读取不到候选；现在统一使用 `CandidateRecord`。遇到全 0 先检查 schema、coverage 和候选读取，不要先判定模型完全失效。
7. **ATT&CK 版本静默漂移**：根 V5c 配置当前会在 `cve2attack_result` 上选 18.1；复现冻结结果必须显式 15.1。
8. **Top-202 被误作 Top-20**：Stage 2 不会主动截断诊断 run；交付前检查 manifest/config 的 `top_k`。
9. **跨来源 RRF 已经退化**：冻结 test 上继续搜索权重会引入过拟合；除非预注册新开发条件，不重复该无效工作。
10. **procedure-only 不能替换主方法**：它是 post-hoc 消融，即使单表略高也不能事后升级为正式方法。
11. **TRIAGE/KEV 视图并非独立来源**：不要把 all/no-secondary 或 KEV 三视图当多份独立标注证据。
12. **GPU 占用不等于模型文件大小**：上下文 KV cache、并发和多 GPU 映射会扩大运行显存。

## 16. 当前问题、过期文档与待复核事项

### 16.1 已确认但尚未修复

- 根 V5c 配置未显式固定 15.1，直接默认运行会选择 18.1；见第 4.4 节。
- `STAGE1_PLAN.md` 第 3 节第 4–5 条（当前约第 85–88 行）仍写“下一步做 procedure 偏置/消融，再决定 reranker”；这些工作实际已经完成，后文第 10 节又正确写明 Stage 1 已冻结。该内部矛盾是过期待办，本轮没有顺带修改。
- `STAGE1_PLAN.md` 冻结段写正式知识语料为 15.1，这是研究口径；但根配置回退风险没有在该段披露。
- 正式 TRIAGE run manifest 的 Git commit 与最终冻结提交不一致；见第 10 节。

### 16.2 尚未确认

- `data_result` 的可核实论文引文、构建方式和标签权威性；当前 `dataset.yaml` 为 `citation: null`。
- `cve2attack_result` 的完整规范引文；元数据写“CVE2ATT&CK paper dataset”，但 `citation: null`。
- 冻结 `5912587` clean HEAD 上是否存在与原 TRIAGE V5k 候选逐字完全一致的新 run；本轮没有复跑。
- 冻结前 37/37 测试通过的持久化日志；仓库中没有该日志，本轮也未重跑。
- Stage 2 分支尚未推送的 3 个提交未来是否改变接口；本文只保证 `be5ba41` 时的实际行为。

以上事项不阻塞本文创建，也不允许用猜测填补。

## 17. 当前研究决策、下一步与重新开启条件

### 17.1 当前决策

- Stage 1 已冻结，正式交付为 ATT&CK 15.1、strict LOO、V5c action Top-3 rank-RRF、Top-20。
- 主要论文定位是 label-free/label-efficient 的受控候选生成，不追求在不同监督条件下盲目追平 TRIAGE。
- SMET 是公开性能参照；TRIAGE 是监督参考上界。
- procedure 覆盖偏置必须主动披露。
- 下一阶段默认评估攻击图上下文能否在固定候选中提升最终映射。

### 17.2 下一步可直接执行的任务

在用户审核和授权后：

1. 只提交本文，保持两份 rewrite cache 未跟踪且不暂存；
2. Stage 2 使用正式 Top-20 V5c run 做端到端候选连接和上下文重排；
3. 记录 Stage 2 是否因真值不在 Top-20 而不可恢复，判断 Stage 1 是否仍为瓶颈；
4. 论文写作中使用已冻结比较表、消融和偏置结论，并补查两份历史数据集引文。

### 17.3 只有以下证据才重开 Stage 1

- 新增独立、可信、人工或权威 benchmark；
- 预先注册的新语料或新模型比较；
- Stage 2 端到端分析证明 Top-20 覆盖是主要不可恢复瓶颈。

重开时必须创建新配置、新 run ID、新 comparison ID，保留冻结结果；不能覆盖原产物，也不能在 60-CVE test 上反复搜索参数。

## 18. 最近重要变化

只保留高价值变更，不扩展为聊天时间线：

- `5912587`：冻结 Stage 1 action retrieval，加入最终语料消融、偏置和案例诊断。
- `e0bb8a6`：加入跨 benchmark V5 验证、2,000-CVE 标签无关样本和配对 bootstrap。
- `4227472`：实现 action-level ATT&CK 检索、严格 LOO、V5 配置与测试。
- `aa0379a`：加入 TRIAGE 候选互补诊断与无训练 RRF 基线。
- `b9f3ef6`：接入 TRIAGE 公开 benchmark 与统一公开预测比较。
- `076852b`：版本化 `sec-i1-cve-rewrite:v1` Llama 3 模板与上下文参数。
- `d7ae864`：长任务增加阶段、进度、耗时与 ETA 输出。
- `3b44154`：缓存 embedding 模型支持离线运行。

## 19. 新任务接管只读检查清单

接管者在任何修改、测试或运行前执行并记录：

- [ ] SSH 到 `sun_demi`，确认主机和 `/home/ghdemi/Code/cve2attack`。
- [ ] 重新确认实时分支、HEAD、上游、ahead/behind；本文顶部和第 3 节只提供最后核查时的快照。
- [ ] 区分 Stage 1 冻结代码基线、连续性文档提交和实时分支 HEAD；纯文档提交不得被误记为新的冻结代码基线。
- [ ] 运行 `git status --short --branch`，区分本文后续变化、用户变化和两份原有 cache。
- [ ] 核对两份未跟踪 rewrite cache 的大小与 SHA-256；不删除、不暂存。
- [ ] 阅读本文、`AGENTS.md`、`STAGE1_PLAN.md`、`docs/experiment_history.md`。
- [ ] 读取将要使用的实验 YAML 和对应 run manifest，不凭名称猜 ATT&CK 版本或 top_k。
- [ ] 核对 `technique_corpus` 是否确实为 `enterprise-attack-15.1.json`。
- [ ] 若要消费 Stage 2，重新核对 Stage 2 HEAD 和第 12 节接口代码是否变化。
- [ ] 若要引用指标，从具体 `metrics.json`/`summary.json` 重算或读取，不从对话复制。
- [ ] 若要运行测试，明确记录本次执行时间和通过数量；不要把静态测试数当执行结果。
- [ ] 若要生成新产物，使用唯一 ID，确认不会覆盖冻结 run/comparison。
- [ ] 任何暂存操作只列出明确文件，禁止 `git add .`。

## 20. 持续维护规则

### 20.1 何时必须更新本文

以下任一项发生实质变化时，在同一工作轮次更新本文：

- 方法定义、正式配置、ATT&CK 版本、LOO 或候选预算；
- Stage 1 冻结代码基线、实时分支 HEAD、文档提交、上游同步状态或工作树保护边界；
- 新 benchmark、数据拆分、标签来源或数据校验值；
- 被接受为论文证据的新 run、comparison、指标、消融或研究决策；
- Stage 1 -> Stage 2 接口；
- 模型标签、服务地址、缓存路径、关键依赖或主机职责；
- 当前目标、下一步、阻塞项或重新开启条件；
- 新的昂贵/不可重建用户数据或缓存。

### 20.2 如何更新

1. 先只读核查实际状态；不得直接沿用旧对话结论，也不得把本文中的 Git 快照当作实时状态。
2. 分别更新“最后内容更新时间”和“最后事实核查时间”。只检查未改内容时只更新核查时间；内容变更才更新内容时间。
3. 更新顶部的时间点快照，同时修正文中受影响的状态说明，不能只追加一段新时间线。
4. 始终区分“Stage 1 冻结代码基线”“连续性文档提交”和“实时分支 HEAD”。纯文档提交可以改变 HEAD/ahead，但不得据此改写冻结代码基线；文档提交哈希应单独记录。
5. 对正式实验记录 benchmark/cohort、配置、run/comparison ID、实验时 Git HEAD、clean/dirty、ATT&CK 版本、Macro/Micro 口径、结果路径和主结果/消融/post-hoc 身份。
6. 结果只在实际文件存在且成功状态可确认后写为“已确认”。
7. 最近变化最多保留约 5–8 个高价值条目；旧细节留在 Git 与实验历史。
8. 事实、研究决策、待复核和建议任务分开写。
9. 已解决问题从“待复核”移除，并在相应当前章节改成最终事实。
10. 更新本文时不要顺手清理缓存、覆盖 run 或修改无关文档。
11. 若提交，原则上与引发状态变化的代码/配置同一提交；若是纯连续性文档提交，则明确标记为文档提交。若实验产物被 Git 忽略，本文仍要记录不可覆盖路径和 manifest 指纹。

## 21. 证据索引

| 事实 | 主要证据 |
| --- | --- |
| Git/工作区状态 | `git status`、`git rev-parse`、`git rev-list`、`git worktree list` |
| 冻结提交 | Git `5912587`、`e0bb8a6`、`4227472`、`aa0379a` |
| V5c 方法 | V5c/V5k YAML、`action_kb.py`、`action_generator.py`、`pipeline.py` |
| ATT&CK 版本差异 | 两个 STIX collection 对象、文件大小、SHA-256、`resolve_attack_bundle()` |
| TRIAGE all/no-secondary | `comparisons/triage_action_v5_leave_one_cve_out/metrics.json` 与 `report.md` |
| 最终消融与偏置 | `comparisons/triage_stage1_v5c_final_audit/` |
| 多 benchmark 结果 | `comparisons/multibench_final_*` 与 `docs/experiment_history.md` |
| 数据规模与来源 | 各 `data/benchmarks/<name>/dataset.yaml`、TRIAGE/KEV source metadata |
| Stage 2 接口 | Stage 2 HEAD `be5ba41` 下的 loaders/schemas/candidate_joiner/pipeline/reranker |
| V3 rewrite 参数 | V3 YAML、`rewrite/ollama.py`、`rewrite/pipeline.py`、CLI、Modelfile、73 `/api/tags` |
| 测试数量 | `tests/` 静态 37 个 `test_*`；冻结前执行观察，无持久日志 |
| 受保护 cache | `git status`、`stat`、`sha256sum`、`.gitignore` |
