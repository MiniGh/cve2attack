# Stage 2 工作连续性文档

> 本文是持续维护的研究与工程状态入口，不是一次性迁移快照，也不是聊天记录。
> 接管 Stage 2 的人或 Agent 应先读本文，再进行只读核查，最后才决定是否修改代码。

## 顶部状态摘要

| 字段 | 当前值 |
| --- | --- |
| 最后内容更新时间 | 2026-08-08 13:41:43 +08:00 |
| 最后只读核查时间 | 2026-08-08 13:41:43 +08:00 |
| 当前唯一真实开发工作区 | `/home/ghdemi/Code/cve2attack-stage2`（现为完整流程整合分支，含已冻结的 Stage 1）|
| 最后核查时的分支 | `feat/full-pipeline-stage2` |
| 最后核查时的 HEAD | `9a94187228d75fce9de020fc1d6dd18009b435e4`（Stage 1 整合合并提交）|
| Stage 2 功能代码状态（最后核查时） | 四个必做工作包全部完成：v2、消融、PwnKit、扩展 LPE 冻结图与正式评价、统一收口报告均已提交并推送至 origin（`c62d54b`）|
| 最后核查时的上游分支 | `origin/feat/full-pipeline-stage2` |
| 最后核查时的上游实际 SHA | `9a94187228d75fce9de020fc1d6dd18009b435e4` |
| 最后核查时的 ahead / behind | ahead 0 / behind 0（本节提交前；分支已与 origin 同步）|
| 文档生成/最后核查时的工作区状态 | 更新本文前 Git 工作区干净；更新后仅本文有 Git 可见修改；被忽略的正式/消融/PwnKit run、扩展 Stage 1 run、外部数据和缓存均保留 |
| 当前目标 | Stage 1 整合已完成（§27）；剩余议题是按方案 B 把整合分支合并进 `main` |
| 建议下一步 | 合并 `main`：保留旧分层方法代码，剔除可重新下载的大数据（方案 B），处理 12 个目录重命名位置冲突 |
| 当前阻塞/限制 | CVE-2010-3856 缺少独立 ATT&CK 标签，只能作诊断例；Zenodo v4 档案不含本轮六个 CVE，未提供新证据（见 §23.3）|

上述分支、上游 SHA、ahead/behind 和工作区状态都是本次连续性文档提交前的状态快照。既有三个
提交前 v2 run 的 manifest 仍记录基础 HEAD `3f3f762...`，但其精确代码/测试补丁已进入
`729d45c...`。27 个正式消融 run 记录干净提交 `e8ca4ea...`，三个 PwnKit run 记录干净提交
`4b5ff98...`。扩展 LPE Stage 1 run 记录 Stage 1 提交 `9688a5b...`，来源/预注册和候选登记进入
Stage 2 提交 `a89d672...`、`30d8882...`，本连续性文档的扩展 LPE 记录进入 `79bac34...`；扩展图和 Stage 2 正式 run 不存在。新对话接管时必须
重新执行第 19 节的只读 Git 检查，不得把本表当作实时 Git 状态。

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

以下截至 `3f3f762` 的 Stage 2 功能与文档提交均已进入实际 GitHub 上游：

| 提交 | 作用 |
| --- | --- |
| `a0c74893a3d113d871c76e949650b16aaaf0b9b5` | 建立外部场景数据源清单和 Git 忽略边界 |
| `8af99437f1a071d24815023c4775fc80d3a49d22` | 接入 M&NTIS Zerologon 轨迹派生场景与转换器 |
| `be5ba41fac5231c0f277c09581c5f3812a52a4d8` | 接入 Tatsu、Sudo，冻结候选并完成三案例回归与报告 |
| `e3d095d1eb2573eb5bcd68a55cf6249bd2f69bda` | 新增持续维护的 Stage 2 连续性文档 |
| `3f3f762ae77ea4c7bc42b249e51a8c44053905c8` | 记录统一正式 V5c 端到端 v1 基线 |

以下提交此前为本地领先，已于 2026-08-08 经授权一并快进推送至 origin：

| 提交 | 作用 |
| --- | --- |
| `729d45c0bf25a7fced53e1989d7a59dd4917d91e` | topology-only v2 tactic 级避免伤害规则 |
| `ebf08fa` | 记录 v2 三案例结果 |
| `e8ca4ea9b8e0fe6a11b3648c74c5173c06b19049` | 增加 `no/local/full` 上下文与深度消融接口 |
| `4b5ff982e2ac08d2d1266ac2f2fabdae0b71def5` | 增加独立 AttackMate PwnKit 场景和回归测试 |
| `c51ea13` | 更新消融与 PwnKit 验证连续性事实 |
| `a89d6724218c5b69be8687477f40e7e369718861` | 冻结扩展 LPE 公开来源、CVE 角色、标签来源和评价规则 |
| `30d8882972a7bb570ed9a8cbebe76fe0b9a90356` | 登记冻结的六例 Stage 1 V5c Top-20 输入及哈希 |
| `79bac34b5b8834cf7e58782708717b9f6e832d78` | 在连续性文档记录扩展 LPE 来源与候选冻结 |
| `4d8e40b04374a2d2137091c8bd914ba1beb15c30` | 刷新连续性文档顶部状态块至 HEAD 79bac34 |

2026-08-08 经授权将上述提交快进推送，`git ls-remote` 现确认 origin/feat/full-pipeline-stage2 为 `4d8e40b`；本次 SHA256 登记提交为新的本地领先提交，未经授权不再 push。

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
- `tests/test_stage2_scenario_graph.py`：三个 M&NTIS 场景及 AttackMate PwnKit 场景、图可重建、标签隔离和固定结果。

## 7. 当前重排方法：topology-rule-priority-v2

当前规则集版本为 `topology-rule-priority-v2`，规则实现提交为 `729d45c...`，消融接口提交为
`e8ca4ea...`；二者已于 2026-08-08 随分支推送。v1 正式基线及其 run 保持不变。v2 不读取 benchmark 标签，也不读取
`remoteExploit`、`localExploit` 或 `expected_impact` 等目标语义字段。

| 规则 | 触发证据 | 匹配分辨率 | 优先候选 |
| --- | --- | --- | --- |
| `public_facing_service` | `attackerLocated(internet)`、internet→目标 `hacl`、目标网络服务 | Technique | `T1190` |
| `lateral_remote_service` | 另一主机已有 `execCode`，并通过 `hacl` 访问目标主机 | Technique | `T1210` |
| `local_privilege_transition` | 目标主机已有非 root `execCode`，直接后果为同主机 root `execCode` | tactic | metadata 中含 `privilege-escalation` 的候选稳定分组 |

v2 的变化只针对证据分辨率：非 root→root 拓扑能证明发生本地权限提升，但不能独立区分
`T1068`、`T1548` 等具体机制。因此所有 `privilege-escalation` 候选先于非匹配候选，同时组内继续保持
Stage 1 顺序；候选缺少 tactic metadata 时仅对 `T1068` 使用兼容回退。该设计不使用 CVE ID、场景名称
或评价答案，候选集合和原始分数仍保持不变。

第一阶段没有召回的正确答案仍不可恢复。v2 已在三个同源 M&NTIS 案例上完成设计回归，并在独立
AttackMate PwnKit 轨迹上证明不会破坏原有正确 Top-1；但后者没有产生排名提升，仍不能描述为总体效果。

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

- 位置：`data/stage2_sources/attackmate/repository`；外层仓库忽略该嵌套 Git 仓库。
- origin 为 `git@github.com:ait-testbed/attackmate.git`，分支 `main`，固定 commit
  `d2edd8bfbb4d18bf4788f222022fe8c73d8fb58f`，最后核查时干净。
- `playbook.yml` SHA256 为
  `91ce76fd0e09cf1cd50e899ae3b1ba3dedae7f089b2b6328cc23ed110f008970`；步骤 3 创建
  `foothold`，步骤 4 显式执行 CVE-2021-4034 PwnKit 本地模块并创建 `root`，步骤 6/7 在
  `root` 会话执行 `id`。
- 归一化场景位于
  `data/stage2_scenarios/attackmate/pwnkit_cve_2021_4034/`；生成图 SHA256 为
  `1e2d3a9c9b192761df1084dbef169e44262507194ce8f9596f2ec0d5e3977985`。
- 评价标签来自独立 CTID KEV 02.13.2025 exploitation benchmark，T1068；图生成器仍只读取
  `context`，不读取该标签。
- Zenodo v4（DOI `10.5281/zenodo.19810174`）官方 `playbooks.zip` 仍未下载：远端连接被拒绝，
  本机 DNS 解析也失败，且没有留下部分文件；不得将其标为已核验。

PwnKit 可作为独立来源的受控本地提权“避免伤害”案例，但它只有一个 CVE、Stage 1 已经 Top-1，
不能代替最终外部 benchmark，也不能证明规则能普遍提升排名。

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

Sudo 的 v1 退化不是测试错误：通用 topology-only 图只能证明本地权限变化，却被 v1 过度解释为
具体 `T1068`。这条失败分析是 v2 将本地规则降到 tactic 分辨率的依据；仍禁止为单个标签临时调规则。

### 10.3 topology-only v2 tactic guard 研究结果

2026-07-30 在基础 HEAD `3f3f762...` 的提交前 v2 工作树上，继续使用 10.2 的同一个统一正式 Stage 1
run 和三张冻结图；该工作树内容随后原样提交为 `729d45c...`。代码与两个测试文件的 diff SHA256 为
`d26c8869089af959ddeccbc803a3f89f29623676f25b76c0215a09024439bb53`。

新运行目录：

- `stage2_runs/v2_tactic_guard_v5c_attack15_1_zerologon_20260730T013916`
- `stage2_runs/v2_tactic_guard_v5c_attack15_1_tatsu_20260730T013916`
- `stage2_runs/v2_tactic_guard_v5c_attack15_1_sudo_20260730T013916`

| 场景 | 正确父 Technique | Stage 1 名次 | v1 名次 | v2 名次 | v2 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| Zerologon | T1210 | 13 | 1 | 1 | 提升保持 |
| Tatsu Builder RCE | T1190 | 不在 Top-20 | 不在 Top-20 | 不在 Top-20 | 不可恢复 |
| Sudo Baron Samedit | T1548 | 1 | 2 | 1 | 原 Top-1 保持 |

三个 v2 manifest 均为 `complete`、`reranker=topology-rule-priority-v2`、`uses_target_semantics=false`；
每例 `candidate_records=3`、`matched=1`、无缺失或未解析 ID，候选集合保持。图哈希与 v1 正式基线相同：
Zerologon `6f3d0b0e...`、Tatsu `f7846de8...`、Sudo `3304f20c...`。

注意：manifest 的 `git_commit=3f3f762...` 只表示运行时基础 HEAD，不表示该提交已包含 v2；上述 diff
已原样进入 `729d45c...`，因此该提交是后续复现 v2 的代码依据。原 run 仍不得误标为在干净提交上运行。

### 10.4 上下文与深度消融

消融接口已提交为 `e8ca4ea9b8e0fe6a11b3648c74c5173c06b19049`，没有改变 v2 规则：

- `no_context`：不应用图规则，直接保持 Stage 1 排名；
- `local_context`：只使用漏洞节点的直接条件、规则和后果；
- `full_graph_context`：在 local context 之外使用上游图证据；
- manifest 同时记录 `context_mode` 和 `max_graph_depth`，非法模式即使输入为空也会失败。

正式矩阵前缀为
`stage2_runs/ablation_e8ca4ea_v5c_20260730T080046_`，共 27 个不可覆盖 run：
3 种模式 × 深度 0/1/2 × Zerologon/Tatsu/Sudo。全部 manifest 为 `complete`，
`git_commit=e8ca4ea...`，并使用 10.2 的统一正式 Stage 1 run。每个 run 均为
`candidate_records=3`、`matched=1`、`missing_candidates=[]`，20 个互异父 Technique 集合保持。
三张图哈希仍为 Zerologon `6f3d0b0e...`、Tatsu `f7846de8...`、Sudo `3304f20c...`。

| 模式/深度 | Zerologon T1210 | Tatsu T1190 | Sudo T1548 | Top-1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_context`，d0/d1/d2 | 13→13 | 缺失→缺失 | 1→1 | 1/3 | 0.359 |
| `local_context`，d0/d1/d2 | 13→13 | 缺失→缺失 | 1→1 | 1/3 | 0.359 |
| `full_graph_context`，d0 | 13→13 | 缺失→缺失 | 1→1 | 1/3 | 0.359 |
| `full_graph_context`，d1/d2 | 13→1 | 缺失→缺失 | 1→1 | 2/3 | 0.667 |

规则审计显示：local context 只识别 Sudo 的 `local_privilege_transition`；完整图 d1/d2 才识别
Zerologon 的 `lateral_remote_service` 和 Tatsu 的 `public_facing_service`。d0/d1/d2 的上游事实数
分别为 Zerologon 2/5/5、Tatsu 2/5/5、Sudo 1/1/1。因此在这三张简单图上，深度 1 足够，深度 2
没有增加事实或改变结果；这不能外推到多跳、多分支攻击图。定向测试结果为
`22 passed, 6 subtests passed in 0.27s`，完整测试为
`48 passed, 6 subtests passed in 5.42s`。

### 10.5 独立 AttackMate PwnKit 避免伤害案例

场景与测试提交为 `4b5ff982e2ac08d2d1266ac2f2fabdae0b71def5`。Stage 1 输入为
`/home/ghdemi/Code/cve2attack/runs/multibench_kev_all_v5c_action_attack15_1`：manifest
`status=complete`、commit `422747270ff5867cf8bac9f2b6b38fc19210a952`、raw description、
ATT&CK 15.1、202 个父 Technique、14121 条 action 文档、strict LOO、rank-RRF top-m 3/k 60、
Top-20、296/296 coverage。其 `CVE-2021.jsonl` SHA256 为
`6d1c7b5297394a2b904a6b205a2c6819525a25e21ef99c1f0e7270ebdfc58c18`；CVE-2021-4034 有
20 个互异父 Technique，T1068 原排名第 1。

不可覆盖 run：

| run ID | 参数 | 规则 | T1068 名次 | manifest SHA256 |
| --- | --- | --- | ---: | --- |
| `attackmate_pwnkit_4b5ff98_20260730T233319_no_context_d0` | no context / d0 | 无 | 1→1 | `1a32820e3ff905b178870733e8df02030fee332206121626b178d70d664c1d7e` |
| `attackmate_pwnkit_4b5ff98_20260730T233319_local_context_d0` | local / d0 | local privilege | 1→1 | `463a386965bb68cc107d60a7756c37ed1658a7fe3c2b4c80258d78be0fd2e298` |
| `attackmate_pwnkit_4b5ff98_20260730T233319_full_graph_context_d1` | full graph / d1 | local privilege | 1→1 | `15f1a4601050b0b3bdedb371489b8ee6602a6c3fdb4e5473aa97a58b1c8b5441` |

三个 manifest 均为 `complete`，记录干净提交 `4b5ff98...`、同一 Stage 1 run 和图哈希
`1e2d3a9c...`。连接统计均为 `context_records=1`、`candidate_records=296`、`matched=1`、
`missing_candidates=[]`；其余 295 条候选无本图上下文是单图运行的预期结果。候选数 20、唯一数 20，
前后集合保持。local/full 模式稳定提升的 tactic 组为
`T1068,T1548,T1055,T1134,T1574,T1078,T1543,T1098,T1546`，组内保持 Stage 1 顺序；
因此正确 T1068 仍为 Top-1。单例前后 Top-1/Top-3/Top-5/MRR 都为 1.0，0 提升、1 保持、0 退化。

该案例与 Sudo 来源独立，支持“tactic guard 在另一本地提权轨迹上不破坏正确 Top-1”；但 Stage 1
本来已经正确，它不能证明 Stage 2 带来增益，也不能估计总体退化率。

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

1. M&NTIS 样本量仅 3，不能当作总体准确率或泛化结论。
2. 三个主案例来自同一 M&NTIS 数据来源，属于公开轨迹派生案例研究，不是独立大样本 benchmark。
3. 候选集合保持不变；Tatsu 缺失属于 Stage 1 召回问题，不是 Stage 2 可以修复的排序问题。
4. 27 个消融 run 复用了同三个样本，只能解释上下文来源和深度，不能增加统计样本量。
5. PwnKit 是独立来源，但只有一个 CVE 且 Stage 1 已为 Top-1，只能提供避免伤害证据。
6. 不得在这些案例上继续调规则后，再把相同案例结果描述为独立效果。

### 11.3 统一正式 V5c 输入上的 v2 研究指标

| 指标 | Stage 1 | v1 | v2 |
| --- | ---: | ---: | ---: |
| Top-1 | 1/3 | 1/3 | 2/3 |
| Top-3 | 1/3 | 2/3 | 2/3 |
| Top-5 | 1/3 | 2/3 | 2/3 |
| MRR | 0.359 | 0.500 | 0.667 |

v2 结果组成为 1 个提升、1 个保持、1 个不可恢复、0 个退化。Top-1 和 MRR 相对 v1 提升来自 Sudo
不再被粗粒度规则排坏；Zerologon 的提升保持，Tatsu 仍不可恢复。该结果同时用于设计反馈和评价，
只能证明三个固定案例上的回归改善，不能作为独立泛化证据。

### 11.4 消融和独立案例如何解释

- `no_context` 精确复现 Stage 1 排名，说明消融开关没有隐式重排。
- local context 足以识别同机权限变化，但不能恢复需要入口/跨主机证据的 Zerologon 和 Tatsu。
- full graph 深度 1 在三张现有图上达到深度 2 的全部效果；这只是现有图的最小充分深度。
- PwnKit 使独立本地提权案例数从 0 增加到 1，并保持正确 Top-1；它不改变三案例总体指标。
- 目前可主张“机制行为与设计一致、存在一个独立无伤害例”，不能主张总体准确率或显著性。

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

2026-07-30 的测试事实：

- v2 提交前完整快速测试：`45 passed, 6 subtests passed in 5.11s`；定向测试
  `19 passed, 6 subtests passed in 0.26s`。
- 消融接口提交前定向测试：`22 passed, 6 subtests passed in 0.27s`；完整测试
  `48 passed, 6 subtests passed in 5.42s`。
- PwnKit 场景提交前，`test_stage2_scenario_graph.py + test_stage2_closed_loop.py` 为
  `23 passed, 8 subtests passed in 0.29s`；完整测试为
  `49 passed, 8 subtests passed in 5.15s`。

各次测试后均执行 `git diff --check` 和状态核查，没有把缓存或运行结果加入 Git。

## 13. 已确认决策与已知问题

### 13.1 已确认决策

- 已提交的 topology-only v1、v2、正式 V5c run 和 27 个消融 run 均作为固定对照保留。
- 当前研究实现为确定性、无训练的 topology-only v2；本地权限变化只支持 tactic 级证据分辨率。
- `no_context / local_context / full_graph_context` 是正式消融接口；现有简单图上 full graph 深度 1
  足够，深度 2 不增加证据。
- Stage 2 只重排同一候选集合，不在此阶段扩大候选集。
- benchmark 标签不参与图生成和规则检测；v2 不读取 CVE ID、场景名或目标语义字段。
- PwnKit 是独立来源的避免伤害案例：规则触发但正确 T1068 保持 Top-1；不把保持描述为提升。
- 不根据单个测试标签逐例修改规则；合成场景只做工程验证，M&NTIS 和 AttackMate 均按案例研究解释。
- 第一阶段 Top-K 缺失必须报告为不可恢复，不能伪装成第二阶段排序失败。
- 扩展 LPE 主评价冻结为 CVE-2020-0787、CVE-2021-40449、CVE-2022-21999、CVE-2022-26904；
  CVE-2021-4034 只作既有桥接对照，CVE-2010-3856 在缺少独立标签时只作诊断。
- 六例候选必须使用冻结 Stage 1 V5c 方法；选择输入不含正确标签，候选均为 20 个互异父 Technique。
- 扩展图、来源行号和 SHA256 未经审阅冻结前，禁止运行正式 Stage 2 评价。

### 13.2 已知问题

- 只有一个独立本地提权案例，且其 Stage 1 已为 Top-1，尚不能证明 tactic guard 普遍安全或有效。
- Tatsu 正确标签 T1190 未进入 Stage 1 Top-20。
- 缺少 tactic metadata 的历史候选会回退到精确 T1068，不能获得完整的 tactic 级保护。
- 当前规则仍是“匹配组优先”，没有校准置信度；v2 只降低了本地规则的证据分辨率。
- 深度消融只覆盖三张简单图；尚无需要深度 2 以上或包含多条 producer path 的独立场景。
- 已有模式级消融，但还没有逐一关闭 hacl、入口位置、前置执行和后果等单个事实的细粒度开关。
- 截至 4d8e40b 的本地提交已于 2026-08-08 推送；三个早期 v2 run 的 manifest 仍只记录运行时基础 HEAD `3f3f762...`。
- 历史 V3a 三案例冻结输入口径不完全一致；统一正式 V5c 输入已按同一配置生成并验证。
- 个别现有文档与 inventory 尾部仍包含过期状态，见第 17 节。
- 扩展 LPE 五图已冻结，正式评价已完成（见 §23、§24）：四例主聚合 0 提升、4 保持、0 退化。
- CVE-2010-3856 的 AttackMate T1068 标注与同一攻击轨迹耦合，不能充当独立金标。
- AttackMate v4 档案已于 2026-08-08 人工交付到 ignored 目录（页面 MD5 一致、本地 SHA256 已登记，见 §22.2）；远程主机直连 Zenodo `:443` 仍被拒，只能人工下载。

## 14. 待验证问题

以下问题尚未决定，不能写成已确认方案：

1. Zerologon 两次 V3a 候选排序不同的原因。
2. v2 tactic 级保护在更多独立本地提权案例上的收益、无伤害率和失败模式。
3. 是否以及如何利用独立机制证据细化 T1068 与 T1548/T1548.003。
4. AttackMate Zenodo v4 完整实验包在网络可达后的最终价值。
5. 如何获得足够大的公开 Stage 2 样本集。
6. 深度 2 以上、多分支和循环图是否会改变当前“深度 1 足够”的案例结论。
7. 是否需要不读取评价标签的单事实级消融，以及怎样避免因开关过多而在小样本上过拟合。

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

1. 保留 v1/v2 正式基线、27 个消融 run 和 3 个 PwnKit run，不覆盖任何结果。
2. 仅依据已登记 Rapid7 模块/文档，label-blind 构建四个新主案例和 CVE-2010-3856 诊断图；
   不读取候选正确排名，不改变 v2 规则。
3. 为每张新图登记精确来源行号、转换假设和 SHA256，完成代码/数据审阅后冻结。
4. 图冻结闸门通过后，才用已冻结的六例 Stage 1 run 创建不可覆盖的正式 Stage 2 run；桥接例和
   诊断例不得混入四例主聚合指标。
5. AttackMate Zenodo v4 档案已于 2026-08-08 人工交付、校验并登记 SHA256（见 §22.2）；原始档案保持 ignored，仅按需解压到 ignored 目录读取。
6. 后续再寻找真正多跳或多 producer path 的独立图，验证深度 1 与深度 2 以上的差异。
7. 将 Tatsu 的 T1190 缺失继续反馈给 Stage 1；Stage 2 不插入正确答案。
8. 截至 4d8e40b 的提交已于 2026-08-08 经授权推送；此后的新提交（含本次 SHA256 登记）仍需获授权才 push。

“重新迁移/复制/整理 Stage 2 代码”不是下一步，也永远不应从本文派生为任务。

## 17. 已知过期文档内容

本轮只记录，不顺带修改：

- `docs/stage2_graph_context.md` 仍描述 v1 精确提升 T1068，且末尾只介绍 Zerologon；尚未覆盖 v2、Tatsu 和 Sudo。
- `STAGE2_PLAN.md` 前部已写三案例完成，但末尾仍把接入 Sudo 列为下一步。
- `data/stage2_sources/README.md` 仍写 M&NTIS 尚未下载和 `mantis/downloads/`；实际为 `raw/`、`extracted/`。
- `data/stage2_sources/source_inventory.yaml` 的扩展 LPE `next_actions` 已更新，但
  `first_conversion_complete` 仍是历史 M&NTIS 单例字段，不能理解为当前扩展图已经完成。
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
- `3f3f762`：记录并同步统一正式 V5c 端到端 v1 基线。
- `729d45c`：提交 topology-only v2 tactic 级避免伤害规则和对应测试，尚未推送。
- `e8ca4ea`：提交上下文模式和深度消融接口；27 个干净提交 run 证明现有三图上 full graph d1 足够。
- `4b5ff98`：提交独立 AttackMate PwnKit 场景和测试；三个新 run 保持 T1068 Top-1，尚未推送。
- `c51ea13`：连续性文档同步消融和 PwnKit 验证事实。
- `a89d672`：冻结六例扩展 LPE 的公开来源、案例角色、独立标签来源和正式评价闸门。
- Stage 1 `9688a5b`：提交不含正确标签的六例 selection-only cohort，并生成冻结 V5c Top-20 run。
- `30d8882`：在 Stage 2 登记该 run、manifest 和四个年度候选文件 SHA256；未运行 Stage 2。
- 2026-07-29：在 `e3d095d` 上通过全部快速测试，并从三个冻结 V3a 快照创建全新验证 run。
- 2026-07-30：验收统一正式 V5c/ATT&CK 15.1/strict-LOO Top-20 输入，并用未修改的
  topology-only v1 完成三个正式端到端 run；结果为 1 个提升、1 个退化、1 个不可恢复。
- 2026-07-30：v2 三案例结果为 1 个提升、1 个保持、1 个不可恢复；消融进一步定位了本地与上游
  证据作用，PwnKit 增加一个独立无伤害例，但仍无总体泛化证据。

这里只保留会改变接管判断的重要变化，不记录完整对话时间线。

## 19. 新对话接管后的只读核查清单

在修改任何内容前：

1. `pwd -P` 和 `git rev-parse --show-toplevel` 必须都指向
   `/home/ghdemi/Code/cve2attack-stage2`。
2. 核对当前分支、HEAD、上游、ahead/behind 和 `git status --porcelain=v1 --branch`。
3. 使用 `git ls-remote` 只读核对实际 GitHub 上游；未经授权不要先 fetch/pull。
4. 用 `git worktree list` 确认 Stage 1/Stage 2 worktree 边界。
5. 核对 `origin..HEAD` 的未推送提交，避免覆盖 `a0c7489`、`8af9943`、`be5ba41`。
6. 确认三个 M&NTIS 场景及 AttackMate PwnKit 场景的 YAML、AttackGraph、候选输入和 benchmark 标签仍存在。
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
- 把当前四个小样本案例的结果描述为总体准确率。

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

## 22. 扩展 LPE 来源、预注册和冻结 Stage 1 输入（2026-07-31）

本节记录公开检索和候选冻结后的实时事实。它不表示扩展 Stage 2 已运行；截至本节更新时间，
没有创建任何扩展 LPE Stage 2 正式 run，也没有查看正确 Technique 在六例候选中的排名。

### 22.1 公开执行证据与许可

Rapid7 Metasploit Framework 固定在 commit
`1816d9023b353800046567984f15b42d24bd334a`，ignored 只读来源位于
`data/stage2_sources/metasploit/repository/`。许可为 BSD-3-Clause（含第三方例外），`COPYING`
SHA256 为 `38f848ebdf03a4f7ce5a703b72e8be6bd724de17d7869d71f3df78734cb4e507`。
以下模块/文档提供可复核的本地提权步骤和低权限到 root/SYSTEM 的控制台证据：

| CVE | module SHA256 | documentation SHA256 |
| --- | --- | --- |
| CVE-2010-3856 | `f7126f6eb6ee87edd98de53d4f265b41d4386b11d11455ecfee918c28171fa40` | `0c72acdaa16bec8cb8934c3d0fa209859420f4c47bcba81e7b68f74d3faf2231` |
| CVE-2020-0787 | `afe494a1be326f819e0a514868664278c62de242c3f1a5bdb00786eb7373afd9` | `42ad0e058290155ee39eb82fdd905206f8a2480d4a2b8d743e326735d2f1d911` |
| CVE-2021-4034 | `7c238f16f477213d7f84c6d508bb08ce9a3df5ffcc49250c0b9065abbd5f410e` | `99ca8d99d1c45a0c968458241fbba446ab4e28cc172a8f0c251530dcaaab028e` |
| CVE-2021-40449 | `af7b23bbae6179c4c7ed3768ea3293170a2f653701e0e99edf9c26187dfccad9` | `c1d8b617acbbe4cd72bb35cc5d65b10efb69e3488d122de9d9f9e25aa8a94067` |
| CVE-2022-21999 | `bd86f2ac8cbcf4c3033bdf620b1e5b01a984e9aacc426ce00f590fc10d1e667f` | `b879679af021105cfe1554d6dc56d88acb177ba492d76688840690355e67d427` |
| CVE-2022-26904 | `fee4f463b59e53cd7ef58a16f3d9180fd3df852f1cf224cb999e6115cb8a96d2` | `353a80cc8bfdd231122939486e0505417da5ec9725cf40feb4b068ab1c6776ad` |

AttackMate repository 固定在 `d2edd8bfbb4d18bf4788f222022fe8c73d8fb58f`，许可 GPL-3.0。
`examples/http-put_example.yml` 使用 CVE-2010-3856 并标注 T1068，SHA256 为
`7d5a27eed6c931bdb57449b6b8b2e964d1d232bbfab9c62599ad5f5c4907b267`；但标签与同一攻击轨迹
耦合，只能用于诊断，不作为独立金标。既有 PwnKit `playbook.yml` SHA256 为
`91ce76fd0e09cf1cd50e899ae3b1ba3dedae7f089b2b6328cc23ed110f008970`。

独立标签使用 CTID KEV→ATT&CK 记录 `10.5281/zenodo.16747173`，版本 `02.13.2025`、
ATT&CK 15.1。本地冻结 CSV 位于
`/home/ghdemi/Code/cve2attack/data/raw/kev/kev-02.13.2025_attack-15.1-enterprise.csv`，SHA256
为 `8f15aab468f17f9a1d655ef2db814b0323792cfa066373a02a0a1d7f4a8f6676`。Zenodo 记录页未显示
明确许可；上游 CTID `mappings-explorer` 仓库是 Apache-2.0，二者不得混写。该 CSV 为五个
2020--2022 案例提供独立 T1068 标注；CVE-2010-3856 不在其中。

### 22.2 AttackMate Zenodo 人工下载缺口

AttackMate Zenodo v4 页面为 `https://zenodo.org/records/19810174`，DOI
`10.5281/zenodo.19810174`，许可 CC-BY-4.0。2026-07-31 远程主机直连 `zenodo.org:443` 被拒
（`curl` error 7）；2026-08-08 改为人工在 Windows 侧下载、经 Windows OpenSSH `scp.exe`
交付到 ignored 目录 `data/stage2_sources/attackmate/downloads/zenodo_v4_20260731/` 并完成校验：
三个文件页面 MD5 全部一致，本地 SHA256 已登记，`git check-ignore` 确认原始压缩包被忽略、
未进入 Git。

| 文件 | 大小(字节) | 页面 MD5 | 本地 SHA256 |
| --- | ---: | --- | --- |
| `playbooks.zip` | 6988 | `6a1dd5cf1a89d85915065124ab8ee08a` | `1e28d9ed0939009631e8c37b3c77213f4a89bac1feb3ffd4c64ba6ef4282abd7` |
| `privilege_escalation_attackmate.zip` | 66746187 | `f6c3d5b04f8855d2bd90be9d0143bf14` | `f595e38fc9e52c3f523f0245f674a94a3a0028fda27838d99225378984eb580a` |
| `privilege_escalation_atomic.zip` | 93843241 | `fdb102503ceb032375c5ebbfbe140e0f` | `7bd2193ce2901dfed033aa6347b471fbea138d3697f8e9c730be0ca42aebb8e5` |

用途：`playbooks.zip` 冻结案例编排与 Technique 声明；`privilege_escalation_attackmate.zip`
提供完整提权执行日志；可选 `privilege_escalation_atomic.zip` 为 Atomic Red Team 对照。
交付到位不改变标签隔离与建图闸门：图仍只能从已登记执行步骤 label-blind 构建。

### 22.3 冻结案例角色和 Stage 1 候选

主聚合只包含 CVE-2020-0787、CVE-2021-40449、CVE-2022-21999、CVE-2022-26904。
CVE-2021-4034 是已评价 PwnKit 的桥接对照，不计入新增主聚合；CVE-2010-3856 是强制诊断例，
在取得独立 ATT&CK 标签前不计主指标。完整六例按以下顺序生成候选：CVE-2010-3856、
CVE-2020-0787、CVE-2021-4034、CVE-2021-40449、CVE-2022-21999、CVE-2022-26904。

Stage 1 selection-only cohort 位于
`/home/ghdemi/Code/cve2attack/data/benchmarks/stage2_extended_lpe_selection_20260731/`，提交为
`9688a5b340dee0de3af4dd3ceaa48bf0267fc9d4`。输入不含正确标签；因当前选择器要求非空标签，
只使用无效哨兵 `T0000`，因此该 run 的 benchmark metrics 全部无效、不得报告。

冻结 run 绝对路径为
`/home/ghdemi/Code/cve2attack/runs/stage2_extended_lpe_v5c_attack15_1_top20_20260731T011812`。
manifest `status=complete`、覆盖 6/6、缺失描述 0；配置为
`experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml`：raw description、ATT&CK
Enterprise 15.1、202 个父 Technique、14,121 条 action、strict LOO、Top-3 rank-RRF、k=60、
Top-20。manifest SHA256 为 `228d4318a35e7a2a967c12b8959f633185a1fb5c689a520a4d14dfd172076753`。
六条记录均恰好包含 20 个互异父 Technique，无子技术 ID。

| 候选文件 | SHA256 |
| --- | --- |
| `CVE-2010.jsonl` | `9130243dac61b43656a4c728fb526e84db0eace744b38d6ca535bf9a15521e96` |
| `CVE-2020.jsonl` | `c8933f18cb852b701e5c0d8bb1360c3a88986c32f2b1cdf777ee7efe7636465b` |
| `CVE-2021.jsonl` | `33068234ec16c2f1070da7db07dde542fb6e08fcbba8b3f432dbbe5d1576fc6a` |
| `CVE-2022.jsonl` | `27ef57ab537c24199aeeaaf92c873f5288be880b3ef91daf16a63855916a0110` |

下一闸门是 label-blind 构建、审阅并冻结新图的来源行号、转换规则和 SHA256。在该闸门完成前，
不得检查正确标签的候选排名，不得运行新的正式 Stage 2 评价，不得为这些标签修改 v2 规则。

## 23. 扩展 LPE label-blind 图构建与冻结（2026-08-08）

### 23.1 已冻结的五张图

新增场景目录 `data/stage2_scenarios/extended_lpe/<cve>/`，各含 `scenario.yaml` 与
确定性生成的 `AttackGraph.xml`（均由 Git 跟踪）。构建过程只读取已登记的 Rapid7 模块文档中的
身份变化证据，未查看候选排名，未修改 `topology-rule-priority-v2`。

| CVE | 角色 | 组件 | 图 SHA256 |
| --- | --- | --- | --- |
| CVE-2020-0787 | 主 | `bits` | `a713a989dbea94741f9bf6d6e9fd86dd66b8606048b0d40c4cbf6733e6eb78b9` |
| CVE-2021-40449 | 主 | `win32k` | `dfe4eae077c177529d77488505aba8da5d399ed9b9b5fe013a288565120f7b36` |
| CVE-2022-21999 | 主 | `spooler` | `21742c98e2da07075b89fb1dc462caa1fcce6dad0fbd97071ffc64ef437e0237` |
| CVE-2022-26904 | 主 | `profsvc` | `691554323a8e6644605698ec5c07d5b489245ea78ba6fd953214131c0a89fbdf` |
| CVE-2010-3856 | 诊断 | `glibc_ld_audit` | `a30f11a363c8070a49cfb878bcc5a60922d3a62cf185bbf82073cfbbcdf356aa` |

场景 YAML 的 SHA256 与逐图来源行号见 `docs/stage2_extended_lpe_evaluation_protocol.md` §4.1。

### 23.2 权限令牌归一（研究决策）

四个 Windows 案例执行记录中的提权后身份为 `NT AUTHORITY\SYSTEM`。图中统一归一为通用最高本地
权限令牌 `root`，与既有 PwnKit 场景词汇一致。原因：`reranker.local_privilege_transition` 只识别
`root`（`reranker.py` 中后果 `execCode(*,root)` 与前置 `execCode(*,≠root)`），而预注册禁止为本批
案例修改 v2 规则。该归一在查看任何候选排名之前、对全部扩展 LPE 图统一作出，只编码“同主机
非特权执行变为最高特权执行”，不引入 OS 特定语义；代价是丢失 SYSTEM 与 root 的 OS 差别，已写入
每个场景的 `normalization_notes`。CVE-2010-3856 原生到达 `root`，未做归一。

### 23.3 Zenodo v4 档案的实际价值（解决第 14 节问题 4）

三个档案已交付并校验（§22.2）。实际内容核查结果：`playbooks.zip` 的提权场景是基于 cronjob
配置错误的提权，声明 `T1053`/`T1190`/`T1059`/`T1087`，不含任何 CVE，也不含 `T1068`；
`privilege_escalation_attackmate.zip` 是该同一场景的主机遥测/配置转储。两者都不含本轮六个 CVE。
因此 Zenodo v4 未为扩展 LPE 六例提供新证据；PwnKit 与 CVE-2010-3856 的证据仍分别来自 AttackMate
git 仓库 `d2edd8b` 的 `playbook.yml` 与 `examples/http-put_example.yml`。解压内容位于被忽略的
`data/stage2_sources/attackmate/extracted/zenodo_v4_20260731/`。

### 23.4 闸门核验与测试

- 五图均可从对应 `scenario.yaml` 逐字节确定性重建；
- 改写 `evaluation.expected_techniques` 不改变生成 XML（标签隔离成立）；
- 每图恰好解析出 1 个 CVE，`local_context.target_host=TARGET`、`exploit_type=localExploit`、
  `expected_impact=privilegeEscalation`；
- 每图只触发 `local_privilege_transition`，`match_scope=tactic`、
  `match_values=['privilege-escalation']`、`fallback_technique_ids=['T1068']`；
- 五图均不含 `attackerLocated`、`hacl`、`netAccess`、`networkServiceInfo`；
- 新增回归测试位于 `tests/test_stage2_scenario_graph.py::ExtendedLocalPrivilegeGraphTests`，
  并将五图纳入既有可重建性与标签隔离循环；
- 定向测试 `14 passed, 38 subtests passed in 0.38s`；完整测试
  `53 passed, 38 subtests passed in 5.19s`（此前基线 49 passed）。

### 23.5 仍未做的事

尚未运行任何扩展 LPE 正式 Stage 2 评价，也未查看 `T1068` 在六例候选中的排名。下一步须用冻结
Stage 1 run `stage2_extended_lpe_v5c_attack15_1_top20_20260731T011812` 与本节冻结图创建全新、
不可覆盖的 run；新增主聚合只含四个 Windows 主案例，CVE-2010-3856 与 PwnKit 分开报告。

## 24. 扩展 LPE 正式 Stage 2 评价结果（2026-08-08）

### 24.1 运行与完整性

18 个不可覆盖 run，前缀 `stage2_runs/extlpe_80383b0_20260808T0110_`，为 6 个案例 ×
`no_context / local_context / full_graph_context`，深度均为预注册的 1。Stage 1 输入为冻结 run
`stage2_extended_lpe_v5c_attack15_1_top20_20260731T011812`（Stage 1 commit `9688a5b...`）。
全部 run：`status=complete`、`git_commit=80383b0...`、`reranker=topology-rule-priority-v2`、
`uses_target_semantics=false`、`matched=1`、`missing_candidates=[]`、
`unresolved_context_ids=[]`、候选数 20、`candidate_sets_preserved=true`。每个 run 的
`candidate_records=6`，其余 5 例列入 `candidates_without_context`，是单图逐例运行的预期统计。

### 24.2 专用评价 benchmark

新增 `data/benchmarks/stage2_extended_lpe/`（Git 跟踪），每例真值严格为预注册的单一 `T1068`，
并登记 CTID mapping type 与 CSV 行号（2020-0787 行 83 exploitation_technique；40449 行 666、
21999 行 636、26904 行 629 均为 primary_impact；4034 行 442 exploitation_technique；
2010-3856 无独立标签，仅 AttackMate 作者标注）。不使用混池 `ctid_kev_2025_02_13_*`：那里
CVE-2021-40449 有 8 个真值（含 T1566、T1071、T1082），会让无关 technique 命中也算成功，且各例
真值数量不一致、跨例不可比。

### 24.3 结果

| 案例 | 角色 | `no_context` | `local_context` | `full_graph` d1 |
| --- | --- | --- | --- | --- |
| CVE-2020-0787 | 主 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2021-40449 | 主 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2022-21999 | 主 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2022-26904 | 主 | 2 → 2 | 2 → 2 | 2 → 2 |
| CVE-2021-4034 | 桥接 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2010-3856 | 诊断 | 7 → 7 | 7 → 5 | 7 → 5 |

四例主聚合（三种模式完全相同）：Top-1 `3/4 → 3/4`、Top-3 `4/4 → 4/4`、MRR `0.875 → 0.875`；
0 提升、4 保持、0 退化、0 不可恢复。

### 24.4 主聚合不变的机制解释（已逐候选核实）

规则确实触发且确实重排：CVE-2022-26904 中 `local_privilege_transition` 将全部
`privilege-escalation` 候选提到非匹配候选之前，T1078 6→5、T1574 7→6、T1098 8→7、T1546 10→8。
主聚合不变的原因是：

1. 四例中三例的 T1068 原本已是第 1，Stage 2 只能保持；
2. 唯一有空间的 CVE-2022-26904，其 Stage 1 第 1 名 T1548 的 metadata tactics 含
   `privilege-escalation`，与 T1068 同组，v2 在组内保持 Stage 1 原序，故 T1068 仍为第 2。

### 24.5 可主张与不可主张

可主张：四个独立 Windows 案例上 0 退化，独立避免伤害案例从 1 例扩大到 5 例；消融诚实
（`no_context` 精确复现 Stage 1）；`local_context` 在这些最小图上已足够。

不可主张：本轮**没有**准确率提升（Top-1/Top-3/MRR 全部持平，0 提升）；唯一名次上升的
CVE-2010-3856 是诊断例、标签与轨迹耦合、Top-1 仍为 T1574，不能作为效果证据；样本量 4+1+1，
不能作总体结论；四例图证据同源于 Rapid7 文档控制台记录这一种模态。

### 24.6 新发现的失败模式（属待验证问题）

当 Stage 1 的 Top-1 与正确答案属于同一 tactic 时（本轮 CVE-2022-26904 的 T1548 对 T1068），
当前 tactic 级 guard 结构上无法改善名次。要在这类情形取得提升需要能区分机制的额外证据，而不是
继续调 v2 权重；该方向必须先独立预注册并在另一批案例上验证，不得在本四例上调参后宣称独立效果。

## 25. 毕设闭环收口（2026-08-08）

统一收口报告为 `docs/stage2_closing_report.md`，汇总全部案例、分族聚合、v1→v2 修正动机、消融、
可主张与不可主张的结论、方法适用边界与已知失败模式。报告中的每个数字均直接来自 run 产物并已
逐项核对，不采用文档转述。

v2 主条件下共 8 个可用于主张的案例：1 提升（Zerologon T1210 13→1）、6 保持、0 退化、
1 不可恢复（Tatsu，正确标签不在 Stage 1 Top-20）。第 9 例 CVE-2010-3856 为诊断例（7→5），
标签与轨迹耦合，单独报告，不进入任何主指标。

分族聚合（不可合并，三族使用三个不同 Stage 1 run）：

- M&NTIS 三例：Stage 1 Top-1 1/3、MRR 0.359；v1 Top-1 1/3、MRR 0.500；v2 Top-1 2/3、MRR 0.667。
- 扩展 LPE 四例：Stage 1 与 Stage 2 均为 Top-1 3/4、Top-3 4/4、MRR 0.875。
- PwnKit 桥接：1→1 保持，在两个不同 Stage 1 run 上复算一致。

核心结论：图上下文的作用取决于拓扑是否具有区分力。外部入口与跨主机可达性与特定 Technique 近似
一一对应，因此能纠正排序；而"同主机非特权执行变为最高特权执行"同时兼容 T1068、T1548、T1055、
T1574 等多种机制，topology-only 证据在原理上无法区分，因此本地提权族的"0 提升、0 退化"是证据
能力边界的体现，不是实现缺陷。

至此 `STAGE2_PLAN.md` 的四个必做工作包全部完成。工作包 2 中此前未勾选的"支持关闭单个上下文特征
做消融"已由 `no_context / local_context / full_graph_context` 三模式接口实现；工作包 4 的复现、
测试、失败诊断与 Git 边界要求均已满足。

## 26. 合并进 main 的现状分析（2026-08-08，只读试合并）

本节记录一次性丢弃式试合并的结论。试合并在临时游离 HEAD worktree 中进行，随后已 `merge --abort`
并 `worktree remove`；两个真实 worktree 与全部分支均未被改动，核查后状态与试合并前一致。

### 26.1 分支拓扑

共同祖先为 `efc71f7`。自该点起：`main` 领先 5 个提交，`refactor/new-method-stage1` 领先 25 个，
`feat/full-pipeline-stage2` 领先 40 个。Stage 2 **不是** Stage 1 的后代：Stage 1 另有 6 个提交不在
Stage 2 中（`4227472`、`e0bb8a6`、`5912587`、`259ce57`、`0c74f25`、`9688a5b`），其中 `9688a5b`
正是扩展 LPE 使用的冻结候选队列。两者的共同祖先为 `aa0379a`。

`main` 是与新流水线**几乎完全不相交**的旧代码库：`stage2_cve_tactics/`、`stage3_cve_techniques/`、
`cve_to_attack_domain/`、`og_data/`、`Validate_data/`，自共同祖先起新增约 160 万行，并提交了
23 个 `.pyc`。

### 26.2 试合并结果一：Stage 1 → Stage 2

仅两个文件需要人工解决，且都是文档：

- `AGENTS.md`：内容冲突（两侧分别新增约 211 行和 169 行）；
- `data/benchmarks/stage2_mantis_scenarios/README.md`：add/add 冲突。

`src/cve2attack/cli.py` 与 `README.md` **自动合并成功**。两侧的
`stage2_mantis_scenarios/CVE-2020.jsonl` 与 `CVE-2021.jsonl` 内容逐字节相同，无数据分歧；
Stage 1 另有该目录下的 `dataset.yaml`，Stage 2 没有，合并时直接并入。

注意：`cli.py` 自动合并成功不等于语义正确，两侧都改过命令注册，合并后必须人工复核并跑全部测试。
Stage 1 还会带入 `retrieval/action_kb.py`、`retrieval/action_generator.py`、`evaluation/action_*.py`
及其测试，合并后测试总数应显著增加。

### 26.3 试合并结果二：Stage 2 → main

**没有任何内容冲突。** 12 个冲突全部是目录重命名导致的 file location 冲突：`main` 把新文件加进了
Stage 2 已重命名的目录。

- `Validate_data/*`（含 3 个 `.pyc`）→ Git 建议移到 `archive/legacy_tfidf/`；
- `og_data/ics-attack.json`、`og_data/mobile-attack.json` → Git 建议移到 `data/raw/`。

自共同祖先起，没有任何文件被 `main` 与 Stage 2 同时修改；Stage 2 只删除了 6 个 `.pyc`，`main`
未触碰其中任何一个。

### 26.4 体积影响

`main` 的树约 694.5 MB，Stage 2 的树约 721.8 MB，两者文件集几乎不相交，合并后仓库会显著增大。
`main` 独有的大文件包括 `og_data/enterprise-attack.json`（48.4 MB）、`og_data/cve/CVE-2025.json`
（34.5 MB）等，仅 `og_data/cve/` 一项即数百 MB。是否将这些原始数据一并并入，属于需要用户决定的
问题；`STAGE2_PLAN.md` §8 要求保留的是 `main` 中的旧分层**方法**，并未要求保留其原始数据副本。

### 26.5 合并前置条件

- Stage 2 worktree 干净，已与 `origin/feat/full-pipeline-stage2` 同步于 `c62d54b`；
- Stage 1 worktree 无未提交源码修改，但**领先 origin 1 个提交**（`9688a5b`，扩展 LPE 冻结队列），
  合并前应先推送或明确确认；其两个未跟踪的 rewrite cache 属于 Stage 1，不得代为清理。

## 27. Stage 1 整合完成（2026-08-08）

Stage 1 已并入本整合分支，合并提交 `9a94187228d75fce9de020fc1d6dd18009b435e4`，父提交为
`9ec1764`（本分支）与 `9688a5b`（Stage 1）。合并方向由两份计划书共同规定，且 Stage 1 已于
2026-07-28 冻结，不应被反向合并。冻结点已打 tag 并推送：`stage1-frozen-v5c` → `5912587`，
`stage1-final` → `9688a5b`。

合并解决的实质问题：此前本分支虽名为 full-pipeline，却无法复现 Stage 1 候选——
`experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml` 与
`retrieval/action_kb.py`、`action_generator.py` 只存在于 Stage 1 分支。合并后本分支可自洽
复现完整链路。

冲突两处，均取并集：

- `AGENTS.md`：两侧都占用小节编号 11.12–11.14。六个命令全部保留，重编号为 11.12–11.17
  （11.12–11.14 为 Stage 1 的 `audit-action-overlap`、`sample-benchmark`、
  `diagnose-action-final`；11.15–11.17 为 Stage 2 的 `build-stage2-graph`、
  `extract-graph-context`、`run-stage2`）；第 4 节同步重排，Stage 2 小节变为 4.13。
- `data/benchmarks/stage2_mantis_scenarios/README.md`：两侧记录同一交接契约的两端，合并为
  “Stage 1 生成义务”与“Stage 2 评价义务”两节，并保留 Stage 1 独有的 `dataset.yaml`。

`src/cve2attack/cli.py` 自动合并；合并后实际加载 argparse 核对，17 个子命令全部注册、无重复
无丢失。全部测试由 53 passed 增加到 `68 passed, 38 subtests`。Stage 1 工作树未被改动，其两个
未跟踪的 rewrite cache 保持原样。

文档组织保持按阶段分离：`STAGE1_PLAN.md` 与 `STAGE2_PLAN.md`、
`docs/continuity/STAGE1_CONTINUITY.md` 与 `STAGE2_CONTINUITY.md`、以及 `docs/stage2_*.md`
均为独立文件，合并没有把任一阶段文档并入另一份。仅 `AGENTS.md`、`README.md` 和上述 benchmark
README 这三个全仓库共享文件的内容被合并，这是它们本身的职责范围。
