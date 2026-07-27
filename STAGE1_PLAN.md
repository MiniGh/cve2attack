# 第一阶段行动指南：CVE → ATT&CK 候选生成与诊断

本文档是 `refactor/new-method-stage1` 分支的研究路线、实验边界和验收清单。
它回答“下一步做什么、为什么做、做到什么程度才继续”。项目结构、全部命令参数和
数据格式以 `AGENTS.md` 为准；已经完成的历史实验见
`docs/experiment_history.md`。进入第一阶段任务后，应先读这三个文件。

第一阶段只负责为每个 CVE 生成受控的 ATT&CK Technique 候选集。攻击图解析、图上下文
重排序和最终图中映射属于第二阶段，不在本分支修改。

## 1. 研究目标与论文定位

第一阶段要证明的不是“只根据 CVE 文本一次性给出唯一正确 Technique”，而是：

> 在不依赖新增人工标注，或只使用少量公开训练标签的条件下，能否用固定候选预算
> 覆盖尽可能多的人工认可 Technique，并为第二阶段保留可解释、可复现的候选证据？

当前主要约束：

- 最终候选预算固定为 Top-20；内部检索可以保存完整父 Technique 排名。
- 主路线首先保持 label-free：不使用 benchmark 真值训练、选权重或逐例调参数。
- 可选 label-efficient 路线只能使用 TRIAGE 公开的 236 个训练 CVE 开发，最终只在
  冻结的 60-CVE test split 上报告测试结果。
- 不新增人工标注，不把第二阶段攻击图信息提前用于第一阶段检索。
- 第一阶段输出必须使用统一 `CandidateRecord`，供第二阶段直接读取。

论文比较关系：

- V1/V2/V3a/V3b：项目内部消融和候选来源。
- SMET：无须盲目复现其全部流程，但其公开结果是当前最低性能参照。
- TRIAGE：使用约 235 个公开标注示例和标签感知组件，是监督参考上界，不是与
  label-free 方法完全公平的同监督基线。
- `data_result`、`cve2attack_result`：两个不同论文来源的数据集，分别保留和报告。

## 2. 当前已经完成的内容

| 状态 | 内容 | 位置 |
| --- | --- | --- |
| 已完成 | V1/V2/V3a/V3b/V4 可配置候选流水线 | `experiments/`、`src/cve2attack/pipeline.py` |
| 已完成 | 统一 `CandidateRecord` 与历史格式兼容 | `src/cve2attack/schemas.py` |
| 已完成 | 运行 manifest、候选、指标和报告 | `runs/<run_id>/` |
| 已完成 | KEV 三个公开评测视图 | `data/benchmarks/ctid_kev_*` |
| 已完成 | TRIAGE 60-CVE 固定测试视图和公开预测复评 | `evaluation/triage.py` |
| 已完成 | V1/V2/V3a/V3b 全父 Technique 排名 | `experiments/diagnostics/` |
| 已完成 | Recall 曲线、rank 分布、独有命中和 oracle 诊断 | `evaluation/diagnostics.py` |
| 已完成 | 无训练 RRF 受控 Top-20 基线 | `fusion/rrf.py` |
| 已完成 | 动作级描述、子技术和 procedure 检索来源 | `retrieval/action_kb.py`、`action_generator.py` |
| 已完成 | 动作级来源互补性、直接 CVE 重叠和严格 LOO 诊断 | 工作包 2 |
| 基线已完成 | 动作级来源与 V1 的固定等权受控融合 | 工作包 3；简单 RRF 退化，暂不选用 |
| 条件执行 | 使用公开训练 split 的轻量 reranker | 工作包 4 |
| 未完成 | 第一阶段最终多基准报告和论文表格 | 工作包 5 |

## 3. 已有证据与当前判断

统一口径为 TRIAGE 公开 test split：60 个 CVE、143 个上卷后的父 Technique 标签。

| 方法 | Micro R@10 | Micro R@20 |
| --- | ---: | ---: |
| V1 raw + ATTACK-BERT | 23.78% | 37.76% |
| V2 raw + procedures | 23.08% | 37.76% |
| V3a rewrite | 20.28% | 35.66% |
| V3b rewrite + procedures | 20.98% | 34.97% |
| 当前最佳 RRF | 20.28% | 39.86% |
| V5e raw + action descriptions（无 procedure） | 24.48% | 40.56% |
| **V5k raw + action procedures，严格 LOO** | **44.06%** | **60.14%** |
| SMET 公开结果 | 37.76% | 52.45% |
| TRIAGE 公开结果 | 69.93% | 76.92% |

关键诊断：

- 加入严格动作级来源后，项目六路共有 97/143 个标签进入至少一路 Top-20。
- 25/143 个标签的项目最佳排名在 21–50；另有 21/143 个标签仍在全部项目来源 Top-50 之外。
- 六路 Top-20 并集 oracle 为 67.83%，但平均需要 39.0 个候选，不能当作受控 Top-20。
- V5k 自身在固定 Top-20 下达到 60.14%，并贡献 21 个其他五路都没有命中的正确标签。
- 加入公开 SMET 后，来源级 Top-20 并集 oracle 为 72.73%，平均需要 44.5 个候选。
- ATT&CK procedure 只直接提及 8/60 个测试 CVE、形成 5/143 个直接真值对；对每个查询严格
  排除所有提及该 CVE 的 action 后，V5k 仍为 60.14%，直接编号重叠不是主要增益来源。

因此当前结论是：

1. 动作级检索是有效的新来源：严格 V5k 比 V1 高 22.38 个百分点，比 SMET 高 7.69 个百分点。
2. 固定参数等权 RRF 会稀释强动作级排序：V1+V5k 只有 48.25%，加入 V5e 后为 46.15%。
   这两个负结果必须保留，不再围绕冻结测试集搜索权重。
3. 剩余 25 个 21–50 位标签仍属于排序问题；21 个 Top-50 外标签仍属于候选覆盖问题。
4. 下一步先冻结严格 LOO 的动作语料和聚合定义，并做多 benchmark 验证与 procedure 数量偏置
   诊断；只有证据表明有必要时，才在公开 236-CVE 训练 split 上开发融合或 reranker。

当前不再把 V3a 视为已经胜出的最终方案。V3a 是有价值的查询视角和消融项；V1 是
TRIAGE 同口径下最强的现有单路 Top-20 基线。

## 4. 目标最小闭环

```text
CVE 原始描述 ───────────────┐
                            ├── Technique 级检索（现有 V1）
可选 attacker-action rewrite ┘

CVE 原始描述 / rewrite ─────── 动作级 ATT&CK 语料检索
                                   │
                                   ▼
                         action hit → parent Technique
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
          来源独立完整排名                    候选互补性诊断
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                       label-free 受控 Top-20 融合
                                   │
                                   ▼
                       标准 CandidateRecord + 报告
                                   │
                                   ▼
                              交给第二阶段
```

第一阶段最终必须同时保留：

- 各个候选来源的独立结果，便于消融和追溯。
- 固定 Top-20 的融合结果，作为第二阶段正式输入。
- 完整父 Technique 排名，仅用于诊断，不冒充实际候选预算。

## 5. 必做工作包与验收标准

### 工作包 1：实现动作级 ATT&CK 检索来源（已完成）

目标：避免把某个 Technique 的名称、长描述和全部 procedures 拼成一个大文档，而是将
ATT&CK 中描述攻击行为的细粒度文本作为独立检索单元，再把命中的 action 映射回父
Technique。

首版动作语料包括：

1. Technique 描述中可独立表达攻击者行为的句子或段落。
2. ATT&CK procedure examples 中的单条行为描述。
3. 每个 action 保存 `technique_id`、action 类型、原始 STIX 对象/relationship ID、
   ATT&CK 版本和原文来源。

首版保持无训练：

- 分别支持原始 CVE 描述和已有 attacker-action rewrite 作为查询。
- 对 action 文本使用现有 ATTACK-BERT 编码和余弦相似度。
- 将 action 命中聚合为父 Technique 分数；至少实现并固定测试 `max` 与确定性的
  rank-based aggregation，不根据测试标签逐例选择。
- 对同一 Technique 的重复 action 去重，并保存最佳 action、支持 action 数和原始名次。
- 完整诊断运行保存全部 143/202 个有效父 Technique 排名，正式运行严格截取 Top-20。

标签泄漏防线：

- action 文本中的 `CVE-*`、`CAN-*` 等漏洞编号必须移除或占位化，不能通过相同 CVE ID
  直接命中 benchmark。
- benchmark 标签、KEV comments 和 TRIAGE reference predictions 不能进入查询或语料。
- procedure 语料与仅 Technique 描述语料必须分别报告，不能把 procedure 带来的变化
  描述成独立人工证据。

验收标准：

- [x] 新来源可以从实验 YAML 独立运行，不破坏 V1–V4。
- [x] 构建 action 语料和检索过程有数量、批次、耗时与 ETA 输出。
- [x] 输出仍是标准 `CandidateRecord`，候选 metadata 可以追溯到 action 证据。
- [x] 相同输入重复运行得到完全相同的排序。
- [x] 测试覆盖 CVE ID 清洗、子技术上卷、重复 action、聚合和 tie break。
- [x] 先产生 TRIAGE 60-CVE 的完整排名 run，再进行任何融合。

### 工作包 2：动作级来源互补性诊断

状态：已完成。正式解释以严格 leave-one-CVE-out 的 V5k 为准；未排除直接 procedure 的
V5a–V5d full-ranking 运行只保留为泄漏敏感性消融。

目标：先回答新来源是否找到了现有 V1–V3 找不到的正确标签，而不是看到单一 Recall 后
立即调参数。

统一运行现有 `diagnose-triage`，加入动作级来源并报告：

- Recall@1/3/5/10/20/30/50。
- 正确标签 rank 分布。
- 与 V1/V3a/SMET 的候选 Jaccard 和独有正确命中。
- 项目来源 Top-20、Top-50 并集 oracle 及实际并集预算。
- 按 mapping type、年份、标签频率、CWE 和描述长度分组。
- exploitation、primary impact、secondary impact 分开解释。

决策门：

- 新来源没有增加独有正确命中，也没有提高项目并集 oracle：停止融合调参，重新检查
  action 语料或寻找另一种检索来源。
- 新来源提高 oracle，但受控 Top-20 未提高：进入工作包 3，问题主要在融合排序。
- 新来源自身和融合结果均提高：冻结该来源定义，再做多基准验证。

### 工作包 3：受控 Top-20 融合

状态：固定 `rank_constant=60`、`source_depth=50`、等权且不调参的基线已完成。V1+V5k
与 V1+V5e+V5k 均显著低于 V5k 单路，因此当前正式 label-free 候选输出使用 V5k 单路
Top-20；evidence-aware 融合只作为后续条件工作，不因存在 union oracle 就默认执行。

目标：在候选数固定为 20 的情况下，将 Technique 级和动作级来源的互补覆盖转化为实际
Recall，而不是报告候选数超过 20 的 union oracle。

第一条基线继续使用无训练 RRF：

- `rank_constant=60` 作为固定基线。
- `source_depth` 是内部候选池深度，必须与最终 `top_k=20` 分开报告。
- 所有来源等权作为主 label-free 结果。
- 可以做事先规定的深度消融，但不能根据 60-CVE test 标签挑选最好的一组冒充主结果。

如果普通 RRF 无法把新增覆盖压入 Top-20，再实现可解释、无训练的 evidence-aware 排序，
特征只来自候选生成过程，例如：来源一致性、最佳 action rank、独立 action 支持数和
Technique 级原始名次。每个最终候选必须保存逐来源贡献。

验收标准：

- [ ] 最终每个 CVE 恰好不超过 20 个候选，候选去重且顺序确定。
- [ ] manifest 保存全部输入 run、Git commit、语料版本、权重和深度参数。
- [ ] 同时报告 V1、动作级单路、旧 RRF、新融合和 union oracle。
- [ ] 主结果不是在冻结 test split 上搜索大量参数后的最优点。
- [ ] 提升和退化案例都进入报告。

### 工作包 4：条件执行的 label-efficient reranker

只有满足以下条件才执行：动作级加入后，正确标签明显进入候选池，但 label-free 方法仍
无法稳定将它们排入 Top-20 或 Top-10。

可使用 TRIAGE 公开的 236 个训练 CVE，不需要新增人工标注。模型只能对固定候选集合
重排序，不能在测试标签上增加候选。候选特征可以包括：

- Technique 级相似度和名次。
- 动作级最佳相似度、支持 action 数和 rank 聚合分数。
- rewrite/raw 两个查询视角的一致性。
- Technique tactic、CWE 等公开非标签特征。

实验必须明确标为 `label_efficient`，与完全 label-free 主结果分表报告。236 个训练 CVE
用于训练和参数选择，冻结的 60 个测试 CVE只运行一次最终评价。

验收标准：

- [ ] train/test CVE 严格不重叠并由代码验证。
- [ ] 数据划分和随机种子写入 manifest。
- [ ] 与相同候选池的无训练融合比较，不能把候选覆盖变化算作 reranker 提升。
- [ ] 报告训练标签频率分组，特别关注 rare/unseen Technique。

### 工作包 5：多基准收口与论文报告

最终报告按用途分层：

1. `triage_2025_test_all`：主同口径比较，报告 V1、最终 label-free、SMET 和 TRIAGE。
2. `triage_2025_test_no_secondary`：排除 secondary impact 的语义消融。
3. `ctid_kev_2025_02_13_exploitation`：专门检查第一阶段最直接负责的利用动作。
4. `ctid_kev_2025_02_13_nonoverlap`：与旧 `cve2attack_result` 去重后的严格外部视图。
5. `data_result` 与 `cve2attack_result`：两个历史论文数据集，分别报告，不合并。

统一指标：

- 覆盖率和缺失预测数。
- Micro Recall@1/3/5/10/20；必要时同时保留项目旧的 Macro Recall。
- Hit@10/20、MRR 或 MAP，但必须注明保存排名深度。
- 平均候选数、最大候选数和实际 Top-K 预算。
- mapping type、年份、标签频率和文本特征分组。

最终产物应包含机器可读 `metrics.json`、可读 `report.md`、逐 CVE rank/分歧记录和运行
manifest。论文表格中的每个数值必须可以追溯到一个不可覆盖的 run/comparison 目录。

## 6. 实验口径与防止测试集过拟合

冻结的 60-CVE TRIAGE test split 是最终测试集，不承担反复选参数的开发集角色。

label-free 主路线允许：

- 使用 ATT&CK、CVE 描述、CWE 和已有 rewrite 等公开无标签信息。
- 使用固定、事先说明的聚合公式和等权融合。
- 在 test split 上做一次诊断来判断失败类型，但每次尝试都必须保留，不能只报告最好值。

label-free 主路线禁止：

- 根据真实 Technique 为单个 CVE 调 prompt、权重、来源深度或候选顺序。
- 从 TRIAGE/SMET reference predictions 复制候选或将其作为融合输入。
- 用 union oracle 的超预算结果冒充 Recall@20。
- 混用 ATT&CK 版本而不记录映射和上卷规则。

如果必须通过标签选择参数，应切换到 label-efficient 路线，只用 236-CVE train split，
并在实验名称、配置和论文表格中明确标识监督条件。

## 7. 配置、运行与产物规则

- 新方法或新消融先增加 `experiments/*.yaml`，不要把研究参数散落在脚本常量里。
- 完整排名诊断配置放入 `experiments/diagnostics/`，正式 Top-20 配置放在
  `experiments/` 根目录。
- 实验定义进入 Git；`runs/`、`comparisons/`、embedding cache 不进入 Git。
- 昂贵且不可轻易重建的 rewrite cache 保存在 `data/derived/rewrite_cache/`，不得覆盖或
  重命名以混淆模型模板版本。
- 每个长任务必须输出当前阶段、完成数/总数、批次、耗时和 ETA；失败记录需指出 CVE
  或语料项及具体异常。
- 新代码需要模块 docstring、公共函数 docstring，以及解释研究口径或非显然算法选择的
  行内注释；不要求给显然赋值逐行写注释。
- 新命令默认不覆盖已有 run 或 comparison。

标准验证顺序：

```bash
.venv/bin/python -m cve2attack validate experiments/<new_method>.yaml
.venv/bin/python -m cve2attack inspect experiments/<new_method>.yaml
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m cve2attack run experiments/<new_method>.yaml --max-cves 2
```

冒烟测试通过后，才运行 60-CVE 完整排名和诊断。

## 8. 文件修改导航

| 要修改的内容 | 主要位置 |
| --- | --- |
| 实验方法和参数 | `experiments/`、`experiments/diagnostics/` |
| 命令入口与参数 | `src/cve2attack/cli.py` |
| 总流水线和 run manifest | `src/cve2attack/pipeline.py` |
| CVE、benchmark 和候选文件读取 | `src/cve2attack/data/loaders.py` |
| 候选 schema | `src/cve2attack/schemas.py` |
| Technique 级语料 | `src/cve2attack/retrieval/technique_kb.py` |
| 动作级语料 | `src/cve2attack/retrieval/action_kb.py` |
| 动作级检索与 Technique 聚合 | `src/cve2attack/retrieval/action_generator.py` |
| procedure/CVE 直接重叠审计 | `src/cve2attack/evaluation/action_overlap.py` |
| 向量模型与批处理 | `src/cve2attack/retrieval/embedder.py` |
| RRF | `src/cve2attack/fusion/rrf.py` |
| 新的无训练证据融合 | 计划新增 `src/cve2attack/fusion/evidence_ranker.py` |
| 常规指标 | `src/cve2attack/evaluation/metrics.py`、`ranking.py` |
| TRIAGE 统一复评 | `src/cve2attack/evaluation/triage.py` |
| 互补性和失败诊断 | `src/cve2attack/evaluation/diagnostics.py` |
| 结构、格式与全部命令 | `AGENTS.md` |
| 研究历史 | `docs/experiment_history.md` |
| 当前路线与下一步 | 本文档 `STAGE1_PLAN.md` |

## 9. Git 与 worktree 规则

| 工作目录 | 分支 | 用途 |
| --- | --- | --- |
| `~/Code/cve2attack` | `refactor/new-method-stage1` | 第一阶段候选生成与实验 |
| `~/Code/cve2attack-stage2` | `feat/full-pipeline-stage2` | 第二阶段和完整流程整合 |

本任务只修改 `~/Code/cve2attack`。不得切换或修改第二阶段工作树，不得移动或清理用户的
rewrite cache。

第一阶段产生新提交后，由第二阶段整合任务决定何时把该提交合入
`feat/full-pipeline-stage2`。第一阶段不能为了方便直接把第二阶段代码反向合入本分支。

提交前必须：

1. 核对分支和 `git status`。
2. 区分本次修改与用户已有的未跟踪 cache。
3. 运行与修改范围相称的测试。
4. 检查 `git diff --check` 和提交差异。
5. 只有用户明确要求时才 push。

## 10. 下一次开发从哪里开始

下一工作包固定为“冻结动作级定义并做多基准验证”，执行顺序如下：

1. 以正式 `experiments/v5c_raw_action_rank_rrf.yaml` 为严格 LOO Top-20 配置，不再根据
   60-CVE test 标签修改 `aggregation_top_m=3` 或 `rank_constant=60`。
2. 在 `data_result`、`cve2attack_result` 和 KEV 的 all/exploitation/nonoverlap 视图分别运行，
   检查提升是否跨数据来源成立，而不是 TRIAGE 切片特例。
3. 量化每个 Technique 的 procedure 数量与 V5c 排名/命中之间的关系，明确 Top-3 action
   累加是否偏向 procedure 丰富或训练标签高频的 Technique。
4. 逐例检查 V5c 新增命中和退化案例，特别是 secondary impact、medium/rare label 与
   仍在 Top-50 外的 21 个标签。
5. 多基准结果稳定后冻结 V5c 语料版本；若仍需融合，只用公开 236-CVE 训练 split 开发
   evidence-aware 方法，60-CVE test 仅做一次最终评价。

这一工作包不继续修改 V3 prompt，也不在冻结 test split 上搜索动作聚合参数或融合权重。
