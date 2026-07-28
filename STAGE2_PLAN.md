# 第二阶段行动指南：攻击图上下文映射

本文档是第二阶段的总路线和验收清单。进入
`feat/full-pipeline-stage2` 分支后，应先读本文，再根据具体任务阅读
`docs/stage2_graph_context.md` 和代码注释。

第二阶段当前坚持“先完成毕设闭环，再增加论文增强项”。没有完成本文第 4 节的
必做工作前，不接入 LLM、GNN 或大规模全图优化。

## 1. 研究目标和边界

第一阶段为每个 CVE 生成受控的 ATT&CK Technique Top-K 候选。第二阶段利用攻击图
中的环境上下文，对同一批候选重新排序，从而回答：

> 同一个 CVE 出现在不同攻击位置、前置权限和后续结果中时，图上下文能否把更符合
> 当前攻击过程的 Technique 提到更靠前的位置？

第二阶段的核心输入：

1. 第一阶段标准 `CandidateRecord`，通常为 Top-20。
2. MulVAL `AttackGraph.xml`。

核心输出：

1. 每个图中 CVE 的局部条件和上游路径证据。
2. 原候选名次、上下文分数和重排后名次。
3. “第一阶段原始排名 vs 图上下文重排”的统一评价报告。

当前不负责：

- 重新生成第一阶段候选。
- 训练新的文本模型或图神经网络。
- 修改 MulVAL、XSB 或攻击图生成器。
- 为整个公开 CVE 空间人工建立新标签集。

攻击图生成项目保持外部独立。本仓库只通过 `AttackGraph.xml` 接收结果。

## 2. 当前已经完成的内容

| 状态 | 内容 | 位置 |
| --- | --- | --- |
| 已完成 | 从 `AttackGraph.xml` 解析 NetworkX 有向图 | `stage2/graph_parser.py` |
| 已完成 | 显式反转为“条件 → 规则 → 结果”方向 | `stage2/graph_parser.py` |
| 已完成 | 识别所有 `vulExists` 节点 | `stage2/context_extractor.py` |
| 已完成 | 分离 `local_context` 与 `graph_context` | `stage2/context_extractor.py` |
| 已完成 | 保留全部上游 producer rules，不任意取第一条 | `stage2/path_expander.py` |
| 已完成 | 标记当前利用边界、循环和深度截断 | `stage2/path_expander.py` |
| 已完成 | 版本化 JSON、原子写入和终端进度 | `stage2/pipeline.py` |
| 已完成 | 44 节点、52 边、2 CVE 的固定回归样例 | `tests/fixtures/mulval/` |
| 已完成（最小闭环） | 接入第一阶段 `CandidateRecord`，报告缺失、未解析和重复输入 | `stage2/candidate_joiner.py` |
| 已完成（v1 基线） | topology-only 确定性重排序，候选集合和原始分数不变 | `stage2/reranker.py` |
| 已完成（案例闭环） | M&NTIS 公开服务、Zerologon 横向移动和 Sudo 本地提权三场景 | 工作包 3 |
| 已完成（案例报告） | 冻结 Top-20、逐场景报告、聚合结果和回归测试 | `docs/stage2_mantis_case_studies.md` |

现有提取命令：

```bash
.venv/bin/python -m cve2attack extract-graph-context \
  --attack-graph tests/fixtures/mulval/AttackGraph.xml \
  --output stage2_runs/example/contexts.json \
  --max-graph-depth 2
```

详细参数和 JSON 字段见 `docs/stage2_graph_context.md`。

## 3. 最终最小闭环

```text
第一阶段已完成 run
└── runs/<stage1_run>/candidates/*.jsonl
                    │
                    ├──────────────┐
                    ▼              ▼
             CandidateRecord   AttackGraph.xml
                    │              │
                    └──────┬───────┘
                           ▼
                 CVE ID 对齐与输入校验
                           │
                           ▼
                 确定性图上下文重排序
                           │
                           ▼
                  stage2_runs/<run_id>/
                  ├── manifest.json
                  ├── contexts.json
                  ├── reranked_candidates.jsonl
                  ├── metrics.json
                  └── report.md
```

闭环的主要对照只有两组：

- Baseline：第一阶段候选的原始排名。
- Stage 2：候选集合不变，仅使用图上下文重新排序。

候选集合保持不变非常重要。否则无法区分提升来自第一阶段召回，还是来自第二阶段
上下文判断。

## 4. 必做工作包和验收标准

### 工作包 1：接入第一阶段候选（最小闭环已完成）

目标：把图中的 CVE 与第一阶段 `CandidateRecord` 一一对齐，形成正式第二阶段输入。

计划实现：

- 已新增 `src/cve2attack/stage2/candidate_joiner.py`。
- 复用 `src/cve2attack/schemas.py`，不得建立不兼容的候选格式。
- 支持读取标准 run 目录中的年度候选 JSONL。
- 将历史 `CAN-...` 规范为 `CVE-...` 后再连接。
- 明确报告 matched、missing candidate、missing graph context 和 duplicate CVE。
- 当前由 `run-stage2` 在完整流程内调用并输出统计；只有出现独立复用需求时才拆出 `build-stage2-input`。

验收标准：

- [x] 固定样例可以接入一个小型候选文件，真实冒烟使用完整 Top-20。
- [x] 每个匹配 CVE 同时包含 context 和原始有序 candidates。
- [x] 缺失、未解析或重复输入不会被静默忽略。
- [x] 候选 ID、原始分数、来源和 metadata 不丢失。
- [x] 测试覆盖标准格式、历史 CAN 规范化、缺失候选和重复记录。

### 工作包 2：确定性上下文重排序基线（v1 已完成）

目标：先证明图上下文在不训练模型的情况下能够改变并解释候选顺序。

计划实现：

- 已新增 `src/cve2attack/stage2/reranker.py`，规则集版本为 `topology-rule-priority-v1`。
- 保留第一阶段原始 rank 和 score。
- 从图中提取可解释特征，例如：
  - 攻击者是否从外部网络进入；
  - 当前动作是跨主机还是同主机；
  - 是否依赖已有代码执行、账户或权限；
  - 后果是代码执行、权限变化、文件访问还是网络可达；
  - 当前节点处于初始进入、横向移动还是本地后续动作。
- 为每个候选记录各特征贡献和最终上下文分数。
- 分数相同时使用原始名次和 Technique ID 做确定性 tie break。

首个基线不训练权重。规则和权重必须写在版本化配置中，不能根据测试标签逐例修改。

验收标准：

- [x] 相同输入多次运行得到完全相同的排名。
- [x] 候选数量和候选集合与第一阶段一致，原始 score 不覆盖。
- [x] 每个名次变化都有机器可读的原名次、新名次、规则、证据和理由。
- [ ] 支持关闭单个上下文特征做消融。
- [x] 单元测试覆盖外部进入、横向移动和本地权限三类拓扑规则。

### 工作包 3：毕设闭环场景和统一评价

目标：回答第二阶段是否比第一阶段原始 Top-1 更适合当前攻击图上下文。

最小场景集：

1. 外部攻击者直接利用公开服务。
2. 已控制一台主机后跨主机横向移动。
3. 已在本机获得低权限后进行本地权限提升。

优先使用公开 benchmark 中已有标签的 CVE，不为评价结果临时创造新标签。每个场景都
保存攻击图输入、第一阶段候选 run、预期标签来源和生成命令。

统一报告：

- Stage-1 Top-1 命中率。
- Stage-2 Top-1 命中率。
- Stage-1 与 Stage-2 的 Top-3、MRR。
- 正确标签上升、下降、不变的 CVE 数量。
- 因第一阶段 Top-20 不含正确标签而无法挽救的数量。
- 每个场景单独结果，不只报告混合平均值。

验收标准：

- [x] 至少三个场景可以从命令行端到端重现。
- [x] 原始排名与重排结果使用完全相同的候选集合。
- [x] 报告同时展示提升案例和退化案例。
- [x] 不把 Top-20 召回不足错误归因于第二阶段。
- [x] 测试和评价命令都不需要调用 LLM。

当前已完成第一条工程纵向闭环：

- CVE：`CVE-2023-20887`。
- 第一阶段输入：`triage_rrf_v1_v3a_d50_k60_top20` 的真实 Top-20。
- 标签：公开 `triage_2025_test_all` 中的 `T1059`、`T1190`。
- 图：`tests/fixtures/stage2/public_facing/AttackGraph.xml`，明确标记为合成公开服务拓扑。
- 行为：`public_facing_service` 只依据互联网入口、目标网络服务和主机关系优先 `T1190`。

该案例用于工程验收，不进入独立总体准确率主张。横向移动场景已在下方接入公开执行
轨迹；本地提权仍需要完整攻击图、真实 CVE 候选和公开标签。

当前已完成第一条公开执行轨迹派生闭环：

- CVE：`CVE-2020-1472`（Zerologon）。
- 来源：M&NTIS 数据集 `625f449f-e7f0-49a1-b0ce-030204be7545`，攻击步骤 `95`。
- 统一描述：`data/stage2_scenarios/mantis/zerologon/scenario.yaml`。
- 转换图：`data/stage2_scenarios/mantis/zerologon/AttackGraph.xml`，由
  `build-stage2-graph` 确定性生成。
- 第一阶段输入：真实 `kev_v3a_llm_rewrite_15_1` 中该 CVE 的冻结 Top-20；`T1210`
  原始排名第 2。
- 标签：M&NTIS worker 的 `T1210`，单独保存在 `stage2_mantis_scenarios`，不进入图生成。
- 行为：已有普通用户会话和跨主机可达性触发 `lateral_remote_service`，把 `T1210`
  提升到第 1。

该案例是公开轨迹派生的可复现案例研究，不等同于大样本独立准确率。

工作包 3 的三场景现已完成。新增的 Tatsu Builder RCE 场景中，标签 `T1190` 未进入
第一阶段 Top-20，因此被明确报告为 `unrecoverable`；Sudo CVE-2021-3156 场景中，第一阶段
父 Technique `T1548` 原本排名第 1，而通用本地提权规则将 `T1068` 提升到第 1，形成固定的
退化案例。完整来源、命令、逐场景结果和聚合指标见
`docs/stage2_mantis_case_studies.md`。

### 工作包 4：运行目录、报告和文档收口

目标：让毕设实验可以由其他人或 Agent 独立复现。

计划实现：

- 每次运行写入独立 `stage2_runs/<run_id>/`，默认不覆盖。
- `manifest.json` 保存 Git commit、输入攻击图哈希、第一阶段 run、配置和状态。
- `metrics.json` 保存原始数值，`report.md` 保存可读表格和案例。
- 更新 `AGENTS.md`、`README.md` 和本指南中的最终命令。

验收标准：

- [ ] 从干净 checkout 按文档可以重现样例和报告。
- [ ] 全部第一、第二阶段快速测试通过。
- [ ] 失败运行留下可诊断的 manifest，不产生看似完成的半成品。
- [ ] 结果目录不进入 Git，实验配置和固定测试输入进入 Git。

## 5. 目标数据契约

上下文提取记录已经包含预留的 `candidates` 数组。工作包 1 接入后，每个 CVE 的
核心结构保持为：

```json
{
  "schema_version": "1.0",
  "cve_id": "CVE-2021-XXXX",
  "local_context": {},
  "graph_context": {},
  "candidates": [
    {
      "technique_id": "T1190",
      "score": 0.61,
      "sources": ["embedding"],
      "metadata": {}
    }
  ]
}
```

重排结果不能覆盖原始字段，应增加：

```json
{
  "original_rank": 4,
  "original_score": 0.61,
  "context_score": 0.35,
  "final_score": 0.72,
  "reranked_rank": 1,
  "context_evidence": []
}
```

具体 schema 在实现前用单元测试固定。后续修改字段时必须升级版本或保持向后兼容。

## 6. 评价循环和标签泄漏防线

攻击图生成过程可能根据 CVE 描述、CWE 或默认规则产生
`remoteExploit`、`localExploit`、`privEscalation` 等字段。如果重排序直接使用这些
字段，再用相同来源构建的 CVE → ATT&CK 标签评价，可能形成循环证据。

因此结果至少分成两组：

1. `topology_only`：只使用主机关系、端口、已有状态、跨主机路径和结果节点。
2. `with_target_semantics`：额外使用 exploit type 和 expected impact，作为消融结果。

论文主张优先建立在 `topology_only` 上。`with_target_semantics` 只能说明加入目标语义
后的变化，不能被描述为完全独立的图证据。

同样禁止：

- 使用 benchmark 真实 Technique 调整单个 CVE 的规则或权重。
- 把攻击图生成器的默认 `remoteExploit + privEscalation` 当成人工真值。
- 只展示提升样例而删除退化样例。
- 将第一阶段没有召回正确答案的案例计为重排序模型可以解决的问题。

## 7. 暂缓的论文增强项

完成四个必做工作包后，才按证据决定是否增加：

- LLM 对 Top-20 候选做上下文判断。
- 学习排序权重。
- GNN 或路径编码模型。
- 更多攻击图和更深路径。
- 与其他第二阶段方法进行大规模比较。

增加条件：确定性基线已经稳定、评价数据足够、增强项能回答明确研究问题。不能只因
方法复杂或“看起来像论文”而加入。

## 8. Git 和 worktree 规则

当前两个工作目录属于同一个 Git 仓库：

| 工作目录 | 分支 | 用途 |
| --- | --- | --- |
| `~/Code/cve2attack` | `refactor/new-method-stage1` | 第一阶段继续实验 |
| `~/Code/cve2attack-stage2` | `feat/full-pipeline-stage2` | 第二阶段和完整流程整合 |

执行第二阶段任务时只修改 `~/Code/cve2attack-stage2`。不要切换第一阶段工作树的
分支，不要移动或清理它的 rewrite cache。

如果第一阶段分支产生新提交：

1. 先确认两个 worktree 都没有未提交源码修改。
2. 在第二阶段分支合并 `refactor/new-method-stage1`。
3. 解决公共文件冲突，重点检查 `schemas.py`、`cli.py`、`pyproject.toml` 和文档。
4. 运行全部测试后再继续第二阶段开发。

最终合并到 `main` 前，先在完整整合分支保留并验证 `main` 中的旧分层第一阶段方法，
不能用新方法直接覆盖它。

旧仓库 `/home/ghdemi/Code/ldh_attackgraph/mapInGraph` 暂时只作为历史参考。新代码验证
完成前不删除；后续只添加“已迁移”说明，不在两个仓库同时维护实现。

## 9. 文件修改导航

| 要修改的功能 | 主要文件 |
| --- | --- |
| 公开场景 schema 与攻击图转换 | `src/cve2attack/stage2/scenario_graph.py`、`data/stage2_scenarios/` |
| XML 节点、边或方向解析 | `src/cve2attack/stage2/graph_parser.py` |
| 上游分支、循环和边界行为 | `src/cve2attack/stage2/path_expander.py` |
| local/graph context 字段 | `src/cve2attack/stage2/context_extractor.py` |
| 上下文输出与进度 | `src/cve2attack/stage2/pipeline.py` |
| 第一阶段候选格式 | `src/cve2attack/schemas.py` |
| 第一、第二阶段连接 | `src/cve2attack/stage2/candidate_joiner.py` |
| 上下文重排序 | `src/cve2attack/stage2/reranker.py` |
| 重排前后评价 | `src/cve2attack/stage2/evaluation.py` |
| 命令行参数 | `src/cve2attack/cli.py` |
| 固定图回归 | `tests/test_stage2_context.py`、`tests/fixtures/mulval/` |
| 第二阶段结构和格式说明 | `docs/stage2_graph_context.md` |
| 总体进度和下一步 | 本文档 `STAGE2_PLAN.md` |

## 10. 下一次开发从哪里开始

候选接入、合成公开服务闭环和 M&NTIS Zerologon 横向移动闭环已经完成。下一步是：

1. 使用同一统一场景 schema 接入 M&NTIS `CVE-2021-3156` 本地提权案例。
2. 先生成或冻结该 CVE 的真实 Top-20，确认 `T1548` 与 `T1068` 是否都被召回。
3. 根据普通用户会话到 root 会话的公开轨迹生成攻击图并增加图级回归。
4. 修正当前把全部本地权限变化固定优先 `T1068` 的粗粒度规则，且不得根据单例答案调参。
5. 加入 `no_context / local_context / full_graph_context` 消融开关。
6. 三场景通过后再汇总毕设闭环报告，并同时展示提升、退化和不可挽救案例。

三场景和消融完成前，不开始设计 LLM prompt，也不根据测试标签调重排序规则。
