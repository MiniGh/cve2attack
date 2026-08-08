# 上游攻击图生成管线只读评估（2026-08-08）

本文记录对 `/home/ghdemi/Code/ldh_attackgraph/ldh_attackgraph/attack_graph` 的一次只读评估，
用于判断"扩大到真实 MulVAL 大图"这条路线的可行性。评估全程未修改该仓库。

该仓库是 Stage 2 的前序工作：Stage 2 的图上下文提取能力由 `ldh_attackgraph/mapInGraph/`
迁入本仓库，而攻击图本身由本文评估的这条管线生成。二者通过 `AttackGraph.xml` 解耦。

## 1. 工具链状态：可用

| 组件 | 状态 |
| --- | --- |
| MulVAL | 已安装于 `/home/ghdemi/tools/mulval`，`utils/graph_gen.sh` 存在 |
| XSB | 已安装于 `/home/ghdemi/tools/xsb-git/bin/xsb` |
| 驱动脚本 | `scripts/mulval_graph_gen.sh`，固定 `MULVALROOT` 与 PATH |
| 代码 | `mulval/{runner,facts_builder,rules_builder,predicate_library}.py`、`models/topology.py`、语义抽取、CWE 映射、报告增强 |
| 端到端产物 | `outputs/` 下有 2026-04-21 的完整产物：`mulval_facts.P`、`AttackGraph.{xml,dot,eps,pdf}`、`custom_rules.P`、审计 JSON 与 Markdown 报告 |

## 2. 格式兼容性：Stage 2 无需改动即可消费

用 Stage 2 现有解析器直接读取该管线的真实产物
`outputs/AttackGraph.xml`（7 节点 / 6 边），结果：

- `parse_xml_to_graph` 与 `reverse_for_analysis` 正常；
- `extract_all_cve_contexts` 识别出 `CVE-2021-44228`，
  `local_context` 为 `app_server / http / remoteExploit / privEscalation`；
- 上游证据链完整还原为 `netAccess(app_server,tcp,0)` ←
  `RULE 6 (direct network access)` ← `hacl(internet,app_server,tcp,0)` +
  `attackerLocated(internet)`。

这验证了 Stage 2 "只通过 `AttackGraph.xml` 接收外部图"的解耦设计确实成立。

固定回归 fixture `tests/fixtures/mulval/AttackGraph.xml` 与
`/home/ghdemi/Code/ldh_attackgraph/mulvalOutput/AttackGraph.xml` 的 SHA256 均为
`5712ff4563c03faf7515faf98dd50f0176178c29e1a2341d3e56e8454b54a67d`，确认 fixture 就是本管线产物。

## 3. MulVAL 输入是可程序化生成的谓词

`outputs/mulval_facts.P` 的实际内容形如：

```prolog
attackerLocated(internet).
hacl(internet, app_server, tcp, 0).
networkServiceInfo(app_server, http, tcp, 80, http).
vulExists(app_server, 'CVE-2021-44228', http, remoteExploit, privEscalation).
attackGoal(execCode(app_server, root)).
```

全部为结构化谓词。"同一 CVE 放置在不同拓扑位置"的受控实验，在输入侧只是改动若干条
`hacl` / `networkServiceInfo` / `attackerLocated` 事实，不需要新工具。

## 4. 两个必须先处理的发现

### 4.1 自定义 MulVAL 规则会改变进入推导链的事实，进而影响 Stage 2 规则触发

在 7 节点产物上，`public_facing_service` **没有触发**。原因不是缺陷：该规则要求
`attackerLocated(internet)`、`hacl(internet→目标)` 和 `networkServiceInfo(目标)` 三项同时在证据中，
而该次运行使用自定义规则 `RULE 25 (Log4j2 JNDI remote code execution)`，其推导链没有引入
`networkServiceInfo`——该事实存在于 `mulval_facts.P`，却不在该 CVE 的证据子图内。

对比：在 44 节点标准 fixture 上，`full_graph_context` 下 `public_facing_service` 正常触发。

含义：**Stage 2 规则的触发率依赖于上游使用了哪些 MulVAL 交互规则**。扩大实验前必须固定上游
规则集，并把"证据是否进入推导链"作为显式变量记录，否则触发率差异会被误读为方法差异。

### 4.2 真实图会让多条规则同时命中，这是手工场景从未出现过的情形

在 44 节点 fixture 上，`CVE-2002-0392` 同时触发 `public_facing_service`（T1190）与
`lateral_remote_service`（T1210）。三个 M&NTIS 场景和 AttackMate 场景每例都只触发一条规则，
因此该情形此前没有被评估过。

用一个构造候选集验证当前重排行为：两条规则命中的候选都被提到未命中候选之前，组内保持
Stage 1 原序，结果为 `T1210` 2→1、`T1190` 4→2、`T1059` 1→3。行为是确定性的，也没有崩溃，
但"多条技术级规则同时命中时如何排序"目前没有预注册的裁决规则，当前实现等价于"并集提升、
组内保持原序"。

含义：真实图上这会成为常态而非例外。扩大实验前需要先就多规则裁决形成独立假设与预注册，
不得看到结果后再定规则。

## 5. 该仓库自身的状态风险

核查时该仓库位于 `main`，HEAD `8bfc07d`（`success version1.1`），但工作区有 **114 项改动**：
46 项删除、29 项修改、39 项未跟踪。因此磁盘上的代码与最后一次提交已显著脱节，
`outputs/` 中 2026-04-21 的产物由哪一版代码生成无法确定。

在把该管线用于正式实验前，必须先由用户决定如何固化其状态（提交、打 tag 或另开分支）。
按既有约定，本仓库的任务不得对该仓库执行清理、reset、checkout 覆盖或批量删除。

## 6. 结论

工具链可用、格式天然兼容、输入可程序化生成，因此"用真实 MulVAL 大图扩大样本"不需要重建工具，
主要工作量在批量拓扑生成、CVE 放置策略、跑批与结果归集，以及尚不存在的对照基线。

推进前的前置条件，按顺序：

1. 固化上游仓库状态，明确正式实验使用的代码版本与 MulVAL 交互规则集；
2. 就多规则同时命中的裁决方式形成独立预注册；
3. 明确 `networkServiceInfo` 等事实是否应从全图事实而非证据子图读取，同样先预注册再改规则。

以上三项都必须在查看任何新的正确标签排名之前完成。
