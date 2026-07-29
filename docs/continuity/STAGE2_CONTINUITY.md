# Stage 2 工作连续性文档

> 本文是持续维护的研究与工程状态入口，不是一次性迁移快照，也不是聊天记录。
> 接管 Stage 2 的人或 Agent 应先读本文，再进行只读核查，最后才决定是否修改代码。

## 顶部状态摘要

| 字段 | 当前值 |
| --- | --- |
| 最后内容更新时间 | 2026-07-30 00:03:12 +08:00 |
| 最后只读核查时间 | 2026-07-30 00:03:12 +08:00 |
| 当前唯一真实开发工作区 | `/home/ghdemi/Code/cve2attack-stage2` |
| 最后核查时的分支 | `feat/full-pipeline-stage2` |
| 最后核查时的 HEAD | `e3d095d1eb2573eb5bcd68a55cf6249bd2f69bda` |
| Stage 2 功能代码基线（最后核查时） | `be5ba41fac5231c0f277c09581c5f3812a52a4d8` |
| 最后核查时的上游分支 | `origin/feat/full-pipeline-stage2` |
| 最后核查时的上游实际 SHA | `e3d095d1eb2573eb5bcd68a55cf6249bd2f69bda` |
| 最后核查时的 ahead / behind | ahead 0 / behind 0 |
| 文档生成/最后核查时的工作区状态 | 测试、V3a 验证 run 和正式 V5c run 完成后 Git 可见状态干净；更新本文后仅本文为 Git 可见修改；另有被忽略的外部数据、缓存和实验结果 |
| 当前目标 | 以统一正式 V5c/ATT&CK 15.1/strict-LOO/父 Technique Top-20 为输入，完成可靠、可解释且不破坏正确候选的 Stage 2 毕设闭环 |
| 建议下一步 | 以本轮正式 V5c 基线为固定对照，设计不读取标签的“避免伤害”策略，并将 Tatsu Top-20 缺失继续作为 Stage 1 召回问题处理 |
| 当前阻塞/限制 | 只有 3 个公开轨迹案例；正式 V5c 基线仍是 1 个提升、1 个退化、1 个不可恢复；Top-1 总体不变，Sudo 退化和 Tatsu 缺失尚未解决，不能据此主张总体泛化 |

上述分支、功能代码基线、上游 SHA、ahead/behind 和工作区状态都是指定时间点的状态快照。
连续性文档自身的后续提交可能使当前分支 HEAD 和 ahead 数量高于这里记录的功能代码基线；这不表示
`be5ba41` 已失去作为当前 Stage 2 功能代码基线的意义。新对话接管时必须重新执行第 19 节的只读 Git 检查，
不得把本表当作实时 Git 状态。

## 1. 不可变的工作区边界

以下事实是后续工作的硬约束：

- Stage 2 从旧的 `~/Code/ldh_attackgraph/mapInGraph/` 整理和迁入当前仓库的操作已经完成。
- 当前唯一的真实 Stage 2 开发工作区是 `/home/ghdemi/Code/cve2attack-stage2`。
- 旧路径 `/home/ghdemi/Code/ldh_attackgraph/mapInGraph` 只能作为历史来源，不是待同步的副本。
- 禁止把“重新迁移、重新复制或重新组织 Stage 2 代码”列为待办，也不得执行相关操作。
- 本次更换本地电脑只迁移 Codex 对话和工作上下文；远程代码、数据、Git worktree、实验环境和结果不迁移。

迁移基础提交是 `66c3c510bc771b18fed5048c55308a6770a3e178`：
`refactor(stage2): migrate attack graph context extraction`。旧 `mapInGraph` 仓库当前位于
`main`、HEAD `fc2073c585c526f6449cb9227aa096dc7755bce7`，核查时工作区干净；这不改变其“仅供历史参考”的地位。

## 2. 本文如何维护

本文区分五类信息：

1. **长期稳定说明**：工作区边界、Stage 2 职责、数据接口和禁止事项。
2. **当前状态快照**：时间、Git SHA、工作区状态、环境和现存结果。
3. **已确认事实/研究决策**：有代码、测试、输入或结果文件支持的结论。
4. **待验证问题**：尚无足够证据，禁止写成既定结论。
5. **建议任务**：可以执行的下一步，但不是自动授权。

方法、schema、规则、数据源、实验结果、研究结论、当前目标或下一步发生实质变化时，应更新本文。
普通查看、一次性诊断命令和无结论变化的重复运行不需要记录。

## 3. Git、worktree 与需要保护的提交

Stage 1 和 Stage 2 是同一 Git 仓库的两个独立 worktree：

| worktree | 分支 | 用途 |
| --- | --- | --- |
| `/home/ghdemi/Code/cve2attack` | `refactor/new-method-stage1` | Stage 1 候选生成 |
| `/home/ghdemi/Code/cve2attack-stage2` | `feat/full-pipeline-stage2` | Stage 2 图上下文映射 |

两个 worktree 共用 `/home/ghdemi/Code/cve2attack/.git`。不得在一个 worktree 中重置、删除或切换另一个
worktree 正在使用的分支。最后核查时 Stage 1 工作树存在两个未跟踪的 rewrite cache；它们不属于
Stage 2，不得代为清理。

以下 Stage 2 功能提交和连续性文档提交均已进入实际 GitHub 上游：

| 提交 | 作用 |
| --- | --- |
| `a0c74893a3d113d871c76e949650b16aaaf0b9b5` | 建立外部场景数据源清单和 Git 忽略边界 |
| `8af99437f1a071d24815023c4775fc80d3a49d22` | 接入 M&NTIS Zerologon 轨迹派生场景与转换器 |
| `be5ba41fac5231c0f277c09581c5f3812a52a4d8` | 接入 Tatsu、Sudo，冻结候选并完成三案例回归与报告 |
| `e3d095d1eb2573eb5bcd68a55cf6249bd2f69bda` | 新增持续维护的 Stage 2 连续性文档 |

最后核查时本地 HEAD、上游跟踪引用和 `git ls-remote` 均为 `e3d095d`。后续仍不得自动 push。

## 4. Stage 2 在完整流程中的职责

完整流程为：

```text
Stage 1：CVE 文本/知识 → 有序 ATT&CK Top-K CandidateRecord
                              │
                              ▼
Stage 2：CandidateRecord + AttackGraph.xml → 图上下文重排 → 前后对照报告
```

Stage 2 不重新生成候选，也不补造 Stage 1 未召回的 Technique。当前基线只改变同一候选集合的顺序：

- 候选数量和 Technique 集合保持不变；
- 第一阶段 `score` 不被覆盖；
- 匹配图规则的候选移到非匹配候选之前；
- 两组内部都保留 Stage 1 原始顺序；
- 原名次、新名次、规则、证据和规则集版本写入 `metadata.stage2`。

第二阶段研究问题是：攻击图中的入口位置、跨主机关系、前置执行状态和后续权限状态，能否让
Stage 1 已召回的候选更符合当前攻击过程。

## 5. Stage 1 → Stage 2 的实际接口

规范对象定义在 `src/cve2attack/schemas.py`：

```json
{
  "schema_version": "1.0",
  "cve_id": "CVE-2021-XXXX",
  "candidates": [
    {
      "technique_id": "T1190",
      "score": 0.61,
      "sources": ["embedding"],
      "metadata": {}
    }
  ],
  "domain": "Enterprise"
}
```

接口行为：

- `candidate_records()` 从 run 根目录的 `candidates/` 或直接从目录读取年度 `CVE-*.jsonl`。
- `CandidateRecord.from_dict()` 兼容规范 `candidates` 和两种历史 `techniques` 格式。
- `parent_technique_id()` 将 Technique ID 规范为大写父技术；例如 `T1548.003 → T1548`。
- `normalize_cve_id()` 将历史 `CAN-...` 规范为 `CVE-...`。
- 规范化后重复的 Stage 1 CVE 会立即报错，不静默覆盖。
- 有效 CVE 图上下文但缺少候选时记录到 `missing_candidates`。
- 有候选但图中没有该 CVE 时记录到 `candidates_without_context`。
- `vulID` 等非 CVE 图标识记录到 `unresolved_context_ids`。
- 图上下文重复 CVE 同样报错。
- 只有匹配成功的记录进入重排；连接统计写入 `join_stats.json`。
- `benchmark_truth()` 默认把评价子技术上卷到父技术，因此 Sudo 的 `T1548.003` 以 `T1548` 评价。

### 5.1 候选数量预算与正式输入约束

- Stage 2 当前不会检查输入候选是否严格为 Top-20，也不会主动截断候选。
- 如果传入 Top-202 完整诊断 run，当前读取器和 reranker 会接受并重排全部 202 项，不一定报错。
- 这种行为保持了输入集合，但会破坏论文规定的候选预算；正式论文链路必须主动传入固定 Top-20 的
  Stage 1 run。
- 正式 V5c 端到端实验必须读取 Stage 1 `manifest.json` 和实际实验配置，核查 `top_k`、
  `technique_corpus`/ATT&CK 版本和 run 状态，不能只根据 run 名称判断。
- 与 Stage 1 冻结口径一致，正式 V5c 目标是 ATT&CK Enterprise 15.1、strict LOO、Top-20；
  Top-202 只用于完整排名诊断，不是 Stage 2 正式交付输入。

## 6. 实际代码地图

| 职责 | 文件 | 关键入口 |
| --- | --- | --- |
| MulVAL XML 加载、校验 | `src/cve2attack/stage2/graph_parser.py` | `parse_xml_to_graph()` |
| 明确反转为“条件→规则→结果” | `graph_parser.py` | `reverse_for_analysis()` |
| 保留全部上游 OR producer rules、循环和边界 | `path_expander.py` | `expand_upstream_evidence()` |
| 识别 `vulExists`，生成 local/graph context | `context_extractor.py` | `extract_cve_context()`、`extract_all_cve_contexts()` |
| Stage 1 候选与图 CVE 连接 | `candidate_joiner.py` | `join_contexts_with_candidates()` |
| topology-only 规则检测与重排 | `reranker.py` | `detect_topology_rules()`、`rerank_joined_record()` |
| 前后排名、MRR、提升/退化/不可恢复 | `evaluation.py` | `evaluate_reranking()` |
| 文件输出、manifest、进度、失败状态 | `pipeline.py` | `run_context_extraction()`、`run_stage2_experiment()` |
| 统一场景 YAML → MulVAL-compatible XML | `scenario_graph.py` | `build_attack_graph_from_scenario()` |
| Candidate schema 和父技术上卷 | `src/cve2attack/schemas.py` | `CandidateRecord`、`TechniqueCandidate` |
| 候选/benchmark 读取 | `src/cve2attack/data/loaders.py` | `candidate_records()`、`benchmark_truth()` |
| 命令行入口 | `src/cve2attack/cli.py` | `build-stage2-graph`、`extract-graph-context`、`run-stage2` |

测试代码：

- `tests/test_stage2_context.py`：固定 MulVAL 图、上下文和多分支证据。
- `tests/test_stage2_closed_loop.py`：候选连接、三条规则、集合保持和端到端输出。
- `tests/test_stage2_scenario_graph.py`：三个 M&NTIS 场景、图可重建、标签隔离和固定结果。

## 7. 当前重排方法：topology-rule-priority-v1

规则集版本：`topology-rule-priority-v1`。它不读取 benchmark 标签，也不读取
`remoteExploit`、`localExploit` 或 `expected_impact` 等目标语义字段，只根据图事实识别三种形状：

| 规则 | 触发证据 | 优先候选 |
| --- | --- | --- |
| `public_facing_service` | `attackerLocated(internet)`、internet→目标 `hacl`、目标网络服务 | `T1190` |
| `lateral_remote_service` | 另一主机已有 `execCode`，并通过 `hacl` 访问目标主机 | `T1210` |
| `local_privilege_transition` | 目标主机已有非 root `execCode`，直接后果为同主机 root `execCode` | `T1068` |

当前方法是确定性、无训练的毕设基线，不是最终学习模型。规则匹配候选不在 Top-K 时，Stage 2
不会插入它；因此第一阶段没有召回的正确答案不可恢复。

## 8. 标签隔离和信息泄漏防线

统一场景 YAML 明确分为：

- `source`：第三方数据集、步骤、文件和 CVE 解析依据；
- `context`：初始状态、网络可达、目标服务、漏洞事实和后果；
- `evaluation`：外部 ATT&CK 标签，仅用于事后评价。

`render_attack_graph_xml()` 在校验场景后只深拷贝 `context`，不读取 `evaluation`。benchmark 标签在
重排完成后的评价阶段才通过 `benchmark_truth()` 加载。测试会改变
`evaluation.expected_techniques` 并断言生成 XML 逐字节不变。

该边界防止把正确 ATT&CK Technique 写进图事实、规则或候选分数，再用同一个答案证明方法有效。
新增场景、规则和评价时必须保留此边界。

## 9. 外部数据、Git 归属和保护要求

### 9.1 M&NTIS

实际位置：

- 原始 ZIP：`data/stage2_sources/mantis/raw/`
- 解压内容：`data/stage2_sources/mantis/extracted/`
- 归一化并由本项目跟踪的输入：`data/stage2_scenarios/mantis/`
- 评价标签：`data/benchmarks/stage2_mantis_scenarios/`

raw/extracted 总量约 478 MB，均被外层 Git 忽略；归一化 YAML、生成图、冻结候选和 benchmark 标签由
`cve2attack` 仓库跟踪。

| 数据集 | 角色 | CVE | SHA256 |
| --- | --- | --- | --- |
| `099967fe-c11a-4bee-b4cc-916d96af5f3b` | Tatsu 公开服务 | CVE-2021-25094 | `10093bb162808fa2c5ed66cff033d40d51dcdfa42520f775c2d33a20dc05ab40` |
| `32133bd9-51dd-4fb3-8db2-e2e02323040f` | SQL injection→webshell，结构材料 | 无明确 CVE | `4d74c476f7d65d00c50093bcb0c35f6543224cee655ea873eff75826b47b2b9c` |
| `625f449f-e7f0-49a1-b0ce-030204be7545` | Zerologon 横向移动 | CVE-2020-1472 | `d21e0c4089c321a28fbedbd6416ae56a69336d8f7b939f0ab09523564fa145f2` |
| `d188aad5-a524-4255-82d3-5adab98715e1` | Sudo 本地提权 | CVE-2021-3156 | `80ed0b6e06788643572b7571feb9e477fed2c8ded826ce9a3ae3d9e4314bf7b7` |

M&NTIS 导出不是 MulVAL XML，必须先通过归一化场景和确定性转换器生成兼容图。三个案例只适合
轨迹派生案例研究，不能支持总体准确率主张。

### 9.2 AttackMate

- 位置：`data/stage2_sources/attackmate/repository`
- 外层 `cve2attack` 仓库将整个 repository 忽略。
- 它是独立嵌套 Git 仓库，origin 为 `git@github.com:ait-testbed/attackmate.git`。
- 分支 `main`，固定 commit `d2edd8bfbb4d18bf4788f222022fe8c73d8fb58f`，最后核查时干净。
- Zenodo v4（DOI `10.5281/zenodo.19810174`）实验包尚未下载。

现有仓库示例可以作为受控场景种子和标注分歧材料，但没有直接 MulVAL 图、缺少合适横向移动案例，
且不能作为最终外部 benchmark。

### 9.3 MulVAL fixture 与合成样例

- 跟踪样例：`tests/fixtures/mulval/AttackGraph.xml`
- 历史来源：`/home/ghdemi/Code/ldh_attackgraph/mulvalOutput/AttackGraph.xml`
- 两者 SHA256 均为 `5712ff4563c03faf7515faf98dd50f0176178c29e1a2341d3e56e8454b54a67d`。
- 固定图规模：44 节点、52 边、2 个 `vulExists`。
- 合成公开服务样例位于 `tests/fixtures/stage2/public_facing/`，只用于工程闭环，不是独立实验数据。

### 9.4 相邻攻击图生成仓库

位置：`/home/ghdemi/Code/ldh_attackgraph/ldh_attackgraph/attack_graph`。

- 独立仓库 origin：`git@github.com:shuoubin/attack_graph.git`
- 核查时 HEAD：`8bfc07dddbcdc52a9a6d07c2e2b67c7a63910b73`
- 当前存在大量用户修改、删除、未跟踪文件和生成物。

禁止对该仓库执行清理、reset、checkout 覆盖、批量删除或迁移。Stage 2 通过显式传入
`AttackGraph.xml` 与图生成项目解耦，不以修改该仓库为前提。

### 9.5 被忽略但必须保护的结果

- Stage 1 run：`runs/`
- Stage 2 run：`stage2_runs/`
- rewrite/embedding cache：`data/derived/`
- M&NTIS raw/extracted 和 AttackMate repository：`data/stage2_sources/`

这些目录不进入 Git，但仍是远端实验工作区的一部分。不得因 `git status` 看不到它们就清理。

## 10. 三个 M&NTIS 案例和已确认结果

### 10.1 历史 V3a 冻结基线

| 场景 | 输入与来源 | 图规模 | 外部标签 | Stage 1 最佳名次 | Stage 2 最佳名次 | 结果 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Zerologon | 数据集 `625f...`，攻击步骤 95，前序 18/89 | 8 节点/7 边 | T1210 | 2 | 1 | 提升 |
| Tatsu Builder RCE | 数据集 `0999...`，攻击步骤 17，前序 8/9/11 | 8 节点/7 边 | T1190 | 不在 Top-20 | 不在 Top-20 | 不可恢复 |
| Sudo Baron Samedit | 数据集 `d188...`，攻击步骤 17，前序 35 | 4 节点/3 边 | T1548.003→T1548 | 1 | 2 | 退化 |

历史保存运行目录：

- `stage2_runs/mantis_zerologon_v1`
- `stage2_runs/mantis_tatsu_rce_v1`
- `stage2_runs/mantis_sudo_cve_2021_3156_v1`

2026-07-29 从 `e3d095d` 使用三个单 CVE 冻结快照完成独立重现：

- `stage2_runs/verify_e3d095d_mantis_zerologon_v3a_20260729`
- `stage2_runs/verify_e3d095d_mantis_tatsu_v3a_20260729`
- `stage2_runs/verify_e3d095d_mantis_sudo_v3a_20260729`

三个验证 run 均为 `complete`，`candidate_records == matched == 1`，无缺失或未解析 ID，
候选集合保持，图哈希和原/新排名均与固定预期一致。

### 10.2 统一正式 V5c/ATT&CK 15.1 基线

统一 Stage 1 输入：

`/home/ghdemi/Code/cve2attack/runs/stage2_mantis_v5c_attack15_1_top20_20260729T2352`

只读验收确认：

- manifest `status=complete`，Stage 1 commit `259ce570b15518c7f56568ad78aa21bd61b74cef`；
- 配置为 `v5c_raw_action_rank_rrf_attack15_1.yaml`、`raw_description`、ATT&CK 15.1；
- action corpus 14121 条、202 个父 Technique、`exclude_query_cve_actions=true`；
- rank-RRF `top_m=3`、`rank_constant=60`、`top_k=20`，三个 CVE 覆盖 3/3；
- 每个 CVE 恰有 20 个互异父 Technique；
- `CVE-2020.jsonl` SHA256 为 `5b547b8841e953da16c412bd58a88e4d2c06e7e098c5915cd11843f52257ec66`；
- `CVE-2021.jsonl` SHA256 为 `0facfa934d4d265711e339884e1fe74f2e0729731f65e667f0c17e1989ba7c02`。

正式 Stage 2 运行目录：

- `stage2_runs/formal_v5c_attack15_1_e3d095d_zerologon_20260729`
- `stage2_runs/formal_v5c_attack15_1_e3d095d_tatsu_20260729`
- `stage2_runs/formal_v5c_attack15_1_e3d095d_sudo_20260729`

| 场景 | 外部标签 | 图规则 | Stage 1 最佳名次 | Stage 2 最佳名次 | 结果 |
| --- | --- | --- | ---: | ---: | --- |
| Zerologon | T1210 | `lateral_remote_service` | 13 | 1 | 提升 |
| Tatsu Builder RCE | T1190 | `public_facing_service` | 不在 Top-20 | 不在 Top-20 | 不可恢复 |
| Sudo Baron Samedit | T1548.003→T1548 | `local_privilege_transition` | 1 | 2 | 退化 |

三个正式 run 均满足：

- Stage 2 manifest `status=complete`、`git_commit=e3d095d...`，并记录同一 Stage 1 run ID 和 commit；
- 每张图 `matched=1`、`missing_candidates=[]`、`unresolved_context_ids=[]`；
- `candidate_records=3`，另外两个 CVE 出现在 `candidates_without_context`，这是单图逐例运行的预期连接统计；
- 候选集合全部保持，独立核对重排前后 Technique ID 集合一致；
- 图哈希分别为 Zerologon `6f3d0b0e...`、Tatsu `f7846de8...`、Sudo `3304f20c...`；
- 重排规则仍为 `topology-rule-priority-v1`，`uses_target_semantics=false`，没有修改规则或按标签调参。

研究意义：

- Zerologon 证明跨主机上下文可以纠正已召回候选的排序。
- Tatsu 证明 Stage 2 无法补回 Stage 1 Top-20 中不存在的正确 Technique。
- Sudo 证明粗粒度“非 root→root”规则可能覆盖 Stage 1 已经正确的、更具体机制判断。

Sudo 的退化不是测试错误。当前 topology-only 图只支持通用本地提权形状 `T1068`；M&NTIS 标签利用
Sudo 机制知识选择 `T1548.003`。尚未决定是否以及如何加入机制级证据，禁止为单个标签临时调规则。

## 11. 指标、口径差异和统计限制

### 11.1 历史 V3a 冻结基线

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Top-1 | 1/3 | 1/3 |
| Top-3 | 2/3 | 2/3 |
| MRR | 0.500 | 0.500 |

结果组成：1 个提升、1 个退化、1 个不可恢复；候选集合全部保持不变。

三个 V3a 快照并非完全来自同一 Stage 1 run：Zerologon 的 T1210 排名第 2，而统一
`stage2_mantis_v3a_top20` 中为第 5；Tatsu 的 T1190 缺失。该差异仍保留为 Stage 1 诊断问题。

### 11.2 统一正式 V5c 基线

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Top-1 | 1/3 | 1/3 |
| Top-3 | 1/3 | 2/3 |
| Top-5 | 1/3 | 2/3 |
| MRR | 0.359 | 0.500 |

逐例结果：

- Zerologon：T1210 从第 13 升到第 1，MRR 从 `1/13` 升到 `1`；
- Tatsu：T1190 不在统一正式 Top-20，前后均不可恢复；
- Sudo：T1548 从第 1 降到第 2，MRR 从 `1` 降到 `0.5`。

因此统一正式 V5c 输入下，Top-1 总体不变，Top-3 和 MRR 提升；案例组成仍是
1 个提升、1 个退化、1 个不可恢复。该结果是当前正式端到端小样本基线，不得省略退化和不可恢复案例。

必须同时记录以下限制：

1. 样本量仅 3，不能当作总体准确率或泛化结论。
2. 三个场景来自同一 M&NTIS 数据来源，属于公开轨迹派生案例研究，不是独立大样本 benchmark。
3. 候选集合保持不变；Tatsu 缺失属于 Stage 1 召回问题，不是 Stage 2 可以修复的排序问题。
4. Sudo 继续证明 topology-only 的通用 T1068 规则可能伤害机制更具体的正确 Top-1。
5. 当前没有正式的避免伤害策略、特征消融或 `max_graph_depth` 消融。
6. 不得在这三个案例上调规则后，再把同一案例结果描述为独立效果。

## 12. 实验环境、命令和成功判断

Stage 2 worktree 没有自己的 `.venv`。实际复用：

```text
/home/ghdemi/Code/cve2attack/.venv
```

核查版本：Python 3.10.12、NetworkX 3.4.2、PyYAML 6.0.3、pytest 9.1.1。

在 `/home/ghdemi/Code/cve2attack-stage2` 下使用统一前缀：

```bash
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack <command>
```

### 12.1 图上下文提取

```bash
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack extract-graph-context \
  --attack-graph PATH/AttackGraph.xml \
  --output stage2_runs/<new_run_id>/contexts.json \
  --max-graph-depth 2
```

成功判断：进度完成 5/5；输出 JSON schema 为 `1.0`；节点/边/CVE 数量合理；没有静默丢失的分支。
默认拒绝覆盖；不要为方便随意使用 `--force`。

### 12.2 从冻结输入重现一个案例

```bash
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack run-stage2 \
  --stage1-run data/stage2_scenarios/mantis/zerologon/stage1_snapshot \
  --attack-graph data/stage2_scenarios/mantis/zerologon/AttackGraph.xml \
  --benchmark stage2_mantis_scenarios \
  --run-id <new_unique_run_id> \
  --scenario-kind trace_derived_mantis_lateral_movement
```

Tatsu 和 Sudo 分别替换为其场景目录与对应 `scenario-kind`。输出位于
`stage2_runs/<new_unique_run_id>/`。

成功判断：

- `manifest.json.status == "complete"`；
- `join_stats.json` 中 `matched == 1`，没有该场景的缺失候选或未解析 ID；
- `metrics.json.candidate_sets_preserved == true`；
- `report.md` 的原排名、重排排名和场景预期一致；
- 命令拒绝覆盖同名 run，而不是混入旧结果。

### 12.3 快速测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ../cve2attack/.venv/bin/python -m pytest -q -p no:cacheprovider
```

2026-07-29 23:41 +08:00 在 `e3d095d` 上实际运行，结果为
`44 passed, 6 subtests passed in 5.14s`。运行使用 `PYTHONDONTWRITEBYTECODE=1` 和
`-p no:cacheprovider`，测试前后 `git status` 均无意外变化。

## 13. 已确认决策与已知问题

### 13.1 已确认决策

- 当前使用确定性、无训练的 topology-only v1 作为毕设基线。
- Stage 2 只重排同一候选集合，不在此阶段扩大候选集。
- benchmark 标签不参与图生成和规则检测。
- 不根据单个测试标签逐例修改规则。
- 合成场景只做工程验证；M&NTIS 三案例只做轨迹派生案例研究。
- 第一阶段 Top-K 缺失必须报告为不可恢复，不能伪装成第二阶段排序失败。

### 13.2 已知问题

- Sudo 中通用 T1068 规则把正确的 T1548 从第 1 降到第 2。
- Tatsu 正确标签 T1190 未进入 Stage 1 Top-20。
- 当前规则只有“命中即优先”，没有置信度或“保持原 Top-1”的保护机制。
- `max_graph_depth=2` 尚缺少系统验证和消融。
- 当前没有关闭单个上下文特征的正式消融开关。
- 历史 V3a 三案例冻结输入口径不完全一致；统一正式 V5c 输入已按同一配置生成并验证。
- 个别现有文档与 inventory 尾部仍包含过期状态，见第 17 节。

## 14. 待验证问题

以下问题尚未决定，不能写成已确认方案：

1. Zerologon 两次 V3a 候选排序不同的原因。
2. 是否采用“避免伤害”的重排保护策略，以及保护条件和评价方式。
3. 是否利用机制证据细化 T1068 与 T1548/T1548.003。
4. AttackMate Zenodo v4 完整实验包的最终价值。
5. 如何获得足够大的公开 Stage 2 样本集。
6. 如何设计不读取评价标签的上下文和深度消融，并避免在三个固定案例上过拟合。

## 15. Stage 1 与 Stage 2 的问题边界

应反馈给 Stage 1：

- 正确 Technique 未进入 Top-K，例如 Tatsu 的 T1190。
- rewrite 或检索条件导致同一 CVE 候选顺序不稳定。
- 候选深度、召回率、文本改写和 embedding retrieval 的改进。
- 后续候选实验继续维持同一配置、同一 ATT&CK 版本和 3/3 coverage 的统一输入契约。

应在 Stage 2 处理：

- 图上下文抽取是否完整。
- 多路径、边界和跨主机证据是否保留。
- 规则是否错误提升、降级或缺少解释。
- 避免伤害策略、上下文消融和重排评价。

不得为了弥补 Stage 1 缺失而在 Stage 2 偷偷插入 benchmark 正确答案。

## 16. 建议下一步任务

以下是建议顺序，不构成自动授权：

1. 将本轮统一正式 V5c 基线作为固定对照保留，不覆盖三个 run。
2. 在不读取评价标签的前提下，预先定义“避免伤害”保护条件，再评估其能否保留 Sudo 的 Stage 1 Top-1，
   同时不破坏 Zerologon 的提升。
3. 为 `no_context / local_context / full_graph_context` 和 `max_graph_depth` 增加正式消融设计。
4. 将 Tatsu 的 T1190 缺失继续反馈给 Stage 1；Stage 2 不插入正确答案。
5. 获得更多公开案例后再进行总体指标比较；不要在 3 个样本上优化权重。

“重新迁移/复制/整理 Stage 2 代码”不是下一步，也永远不应从本文派生为任务。

## 17. 已知过期文档内容

本轮只记录，不顺带修改：

- `docs/stage2_graph_context.md` 末尾仍只介绍 Zerologon，没有覆盖 Tatsu 和 Sudo。
- `STAGE2_PLAN.md` 前部已写三案例完成，但末尾仍把接入 Sudo 列为下一步。
- `data/stage2_sources/README.md` 仍写 M&NTIS 尚未下载和 `mantis/downloads/`；实际为 `raw/`、`extracted/`。
- `data/stage2_sources/source_inventory.yaml` 的 M&NTIS 主体信息基本正确，但
  `overall_assessment.next_actions` 和 `first_conversion_complete` 周边状态已过期。
- 现有部分复现文档使用 `.venv/bin/python`；Stage 2 实际应使用 `../cve2attack/.venv/bin/python`。
- `context_extractor.py` 中关于 Stage 1 “未来填充 candidates”的注释已落后于已经完成的候选连接实现。

处理这些过期内容时，应单独提交文档修正，不夹带方法或实验变化。

## 18. 最近的重要变化

- `66c3c51`：将旧 mapInGraph 的图上下文能力迁入当前 Stage 2 worktree，迁移完成。
- `9f19d34`：完成第一条 Stage 1→图上下文→确定性重排工程闭环，成为当前远端基线。
- `a0c7489`：盘点 AttackMate、M&NTIS 等外部来源并建立忽略/归属规则。
- `8af9943`：接入 Zerologon 轨迹派生场景，验证 T1210 从第 2 到第 1。
- `be5ba41`：加入 Tatsu 与 Sudo，形成提升、不可恢复、退化三种固定案例及回归测试。
- `e3d095d`：新增并推送持续维护的 Stage 2 连续性文档；功能代码基线仍为 `be5ba41`。
- 2026-07-29：在 `e3d095d` 上通过全部快速测试，并从三个冻结 V3a 快照创建全新验证 run。
- 2026-07-30：验收统一正式 V5c/ATT&CK 15.1/strict-LOO Top-20 输入，并用未修改的
  topology-only v1 完成三个正式端到端 run；结果为 1 个提升、1 个退化、1 个不可恢复。

这里只保留会改变接管判断的重要变化，不记录完整对话时间线。

## 19. 新对话接管后的只读核查清单

在修改任何内容前：

1. `pwd -P` 和 `git rev-parse --show-toplevel` 必须都指向
   `/home/ghdemi/Code/cve2attack-stage2`。
2. 核对当前分支、HEAD、上游、ahead/behind 和 `git status --porcelain=v1 --branch`。
3. 使用 `git ls-remote` 只读核对实际 GitHub 上游；未经授权不要先 fetch/pull。
4. 用 `git worktree list` 确认 Stage 1/Stage 2 worktree 边界。
5. 核对 `origin..HEAD` 的未推送提交，避免覆盖 `a0c7489`、`8af9943`、`be5ba41`。
6. 确认三个场景 YAML、AttackGraph、stage1 snapshot 和 benchmark 标签仍存在。
7. 确认 M&NTIS raw/extracted、AttackMate repository 和 ignored run 目录仍存在，不执行清理。
8. 阅读 `src/cve2attack/stage2/`、本文件、`docs/stage2_mantis_case_studies.md` 和对应测试。
9. 准备正式端到端输入时读取 Stage 1 manifest 和配置，核查 `top_k`、ATT&CK corpus/version 和
   `status`，不得根据 V5c、Top-20 等 run 名称猜测实验条件。
10. 查看正式 run 的 manifest/metrics，而不是只相信本文中的数字。
11. 检查相邻 attack_graph 仓库状态，但不得修改。
12. 只有核查结果与本文一致或差异已记录后，才制定修改计划。

## 20. 禁止操作清单

未经用户明确授权，禁止：

- 从旧 `mapInGraph` 重新迁移、复制或整理 Stage 2。
- 将本地电脑作为新的代码主工作区，或迁移远端代码/数据/环境。
- 对 Stage 1 worktree、旧 mapInGraph 或相邻 attack_graph 仓库执行 reset、clean、checkout 覆盖或批量删除。
- 清理被 Git 忽略的 M&NTIS、AttackMate、runs、stage2_runs、rewrite cache 或 embedding cache。
- 覆盖已有实验 run；必须使用新 run ID。
- 把 benchmark 标签写入图生成、上下文规则或候选分数。
- 为一个案例逐例调整规则并将其宣称为独立效果。
- 在未核对状态时 pull、rebase、merge、push 或删除分支/worktree。
- 把三个案例的结果描述为总体准确率。

## 21. 持续更新规则

每次实质更新本文时：

1. 更新“最后内容更新时间”和“最后核查时间”；两者可以不同。
2. 重新核对当前工作区、分支、完整 HEAD、实际上游 SHA、ahead/behind 和状态。
3. 新实验必须记录输入、run ID、manifest commit、参数、输出路径和成功判断。
4. 新数据必须记录来源、版本、路径、SHA256/commit、外层 Git 是否跟踪以及用途限制。
5. 新规则必须记录使用的事实、优先的 Technique、是否读取目标语义、候选集合是否保持。
6. 指标变化必须同时记录样本量、冻结输入口径、提升/退化/不可恢复案例，不能只写平均值。
7. 测试状态必须写实际命令和精确输出；未运行时明确写“未运行”。
8. 将已解决的待验证问题移入“已确认决策”，并留下证据路径或提交。
9. “建议任务”只有实际完成后才能改为“已完成事实”。
10. 最近变化保持简短；旧细节交给 Git 历史和实验 manifest。
11. 不记录密码、令牌、账号或其他秘密。
12. 更新后完整重读本文，并检查是否出现相互矛盾的当前状态与旧待办。
13. Stage 1 输入条件变化时，同时记录 manifest/config 的 `top_k`、ATT&CK 版本和冻结候选来源；
    不得只更新 run 名称。

本文本身应与相关实质变化一起提交；但创建或更新本文不自动授权 `git add`、commit 或 push。
