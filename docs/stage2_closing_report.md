# Stage 2 毕设闭环报告：攻击图上下文重排

本文是 Stage 2 的统一收口报告，汇总全部案例、指标、消融和失败案例。它取代分散在
`docs/stage2_mantis_case_studies.md` 与
`docs/stage2_extended_lpe_evaluation_protocol.md` 中的逐族结论，但不替代它们的来源登记。

**本文的定位是毕业设计的 Stage 2 闭环交付。** 它只记录已完成并经核验的工作与结论，
不包含面向后续扩大验证或论文投稿的规划；此类前瞻性内容不在本报告范围内，单独维护。

生成时间：2026-08-08。重排实现：`topology-rule-priority-v2`（未修改）。

## 1. 研究问题

Stage 1 只读 CVE 文本，为每个 CVE 产生 20 个候选 ATT&CK 父 Technique 的有序列表，
它不知道该漏洞在真实攻击中处于什么位置。Stage 2 提出的问题是：

> 同一个 CVE 出现在不同攻击位置、前置权限和后续结果中时，攻击图的环境上下文能否把更符合当前
> 攻击过程的 Technique 提到更靠前的位置？

Stage 2 只重排同一候选集合：候选数量与集合不变，Stage 1 原始分数不被覆盖，因此任何指标变化都
只能来自上下文判断，不可能来自召回改善。

## 2. 实验条件

| 项目 | 取值 |
| --- | --- |
| Stage 1 方法 | V5c：raw description、ATT&CK Enterprise 15.1、202 个父 Technique、14121 条 action、strict LOO、Top-3 rank-RRF、k=60、Top-20 |
| Stage 2 重排 | `topology-rule-priority-v2`，确定性、无训练、不读取 benchmark 标签 |
| 主条件 | `full_graph_context`，预注册深度 1（M&NTIS 族历史 run 为深度 2） |
| 消融 | `no_context`、`local_context` |
| 标签隔离 | 图生成器只深拷贝场景 `context`，从不读取 `evaluation`；标签仅在重排完成后用于评价 |

三个场景族使用三个不同的冻结 Stage 1 run，因此**不能把三族简单合并成一个总体指标**：

- M&NTIS 族：`stage2_mantis_v5c_attack15_1_top20_20260729T2352`
- PwnKit 桥接：`multibench_kev_all_v5c_action_attack15_1`（另在扩展族 run 中复算一次）
- 扩展 LPE 族：`stage2_extended_lpe_v5c_attack15_1_top20_20260731T011812`

## 3. 全部案例总表（v2 主条件）

| 案例 | CVE | 场景类型 | 标签 | Stage 1 名次 | Stage 2 名次 | 结果 |
| --- | --- | --- | --- | ---: | ---: | --- |
| Zerologon | CVE-2020-1472 | 跨主机横向移动 | T1210 | 13 | **1** | 提升 |
| Tatsu Builder RCE | CVE-2021-25094 | 外部利用公开服务 | T1190 | 不在 Top-20 | 不在 Top-20 | 不可恢复 |
| Sudo Baron Samedit | CVE-2021-3156 | 本地提权 | T1548 | 1 | 1 | 保持 |
| PwnKit | CVE-2021-4034 | 本地提权 | T1068 | 1 | 1 | 保持 |
| BITS | CVE-2020-0787 | 本地提权 | T1068 | 1 | 1 | 保持 |
| Win32k | CVE-2021-40449 | 本地提权 | T1068 | 1 | 1 | 保持 |
| SpoolFool | CVE-2022-21999 | 本地提权 | T1068 | 1 | 1 | 保持 |
| User Profile Service | CVE-2022-26904 | 本地提权 | T1068 | 2 | 2 | 保持 |
| glibc LD_AUDIT（诊断） | CVE-2010-3856 | 本地提权 | T1068（非独立） | 7 | 5 | 提升（不计主指标） |

合计 8 个可用于主张的案例：**1 提升、6 保持、0 退化、1 不可恢复**。第 9 例为诊断例，其标签与
同一攻击轨迹耦合，不是独立金标，单独报告。

## 4. 分族聚合指标

### 4.1 M&NTIS 三例（v1 与 v2 对照）

| 指标 | Stage 1 | Stage 2 v1 | Stage 2 v2 |
| --- | ---: | ---: | ---: |
| Top-1 | 1/3 | 1/3 | **2/3** |
| Top-3 | 1/3 | 2/3 | 2/3 |
| MRR | 0.359 | 0.500 | **0.667** |

v1 组成：1 提升、1 退化、1 不可恢复。v2 组成：1 提升、1 保持、0 退化、1 不可恢复。

### 4.2 扩展 LPE 四例主聚合

三种上下文模式结果完全相同：

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Top-1 | 3/4 | 3/4 |
| Top-3 | 4/4 | 4/4 |
| MRR | 0.875 | 0.875 |

组成：0 提升、4 保持、0 退化、0 不可恢复。

### 4.3 单独报告的案例

- PwnKit（桥接对照）：1 → 1，保持。在两个不同 Stage 1 run 上复算，结论一致。
- CVE-2010-3856（诊断）：7 → 5，Top-1 仍为 T1574。标签来自 AttackMate 作者在同一轨迹中的标注，
  不是独立金标，不进入任何主指标。

## 5. v1 到 v2：一次由失败案例驱动的修正

Sudo 案例暴露了 v1 的过度解释：图中"非 root 执行先于同主机 root 执行"这一拓扑只能证明**发生了
本地权限提升**，却被 v1 直接解释为具体机制 `T1068`，因而把 Stage 1 已经正确的 `T1548` 从第 1 挤到
第 2，形成退化。

v2 的修正只改变证据分辨率，不改变证据本身：本地提权规则降到 **tactic 级**，把全部
`privilege-escalation` 候选整组提到非匹配候选之前，组内继续保持 Stage 1 原序，不再点名具体
Technique。Sudo 因此从"退化"变为"保持"，Zerologon 的提升不受影响。

方法论约束：v2 是**看着 Sudo 案例设计**的，因此不能用 Sudo 证明 v2 有效。这正是扩展 LPE 验证
存在的原因——必须在规则没有调过参的独立案例上检验"这个 guard 会不会把本来对的搞坏"。

## 6. 扩展验证回答了什么

预注册问题：**v2 tactic guard 在未调参的独立案例上是否造成伤害？**

答案：**否。四个独立 Windows 本地提权案例 0 退化**，加上 PwnKit，独立无伤害案例由 1 例扩大到
5 例。

主聚合未提升，原因经逐候选核实，不是规则未触发：

1. 四例中三例的 T1068 原本已是 Stage 1 第 1，Stage 2 只能保持，没有改进空间；
2. 唯一有空间的 CVE-2022-26904，其 Stage 1 第 1 名 `T1548` 的 tactics 同样包含
   `privilege-escalation`，与正确答案同组；v2 在组内保持 Stage 1 原序，故 T1068 停在第 2。

规则确实触发且确实重排：同一 run 中 T1078 由 6 升到 5、T1574 由 7 升到 6、T1098 由 8 升到 7、
T1546 由 10 升到 8。

## 7. 消融结论

| 模式 | M&NTIS 三例 | 扩展 LPE 四例 |
| --- | --- | --- |
| `no_context` | 精确复现 Stage 1 排名 | 精确复现 Stage 1 排名 |
| `local_context` | 只识别同主机权限变化，无法恢复 Zerologon 与 Tatsu | 与主条件结果相同 |
| `full_graph_context` d1 | 识别跨主机与入口证据，Zerologon 13→1 | 与 `local_context` 相同 |

`no_context` 精确复现 Stage 1 证明消融开关没有隐式重排。本地提权图的全部判据都在漏洞节点的直接
邻域内，因此 `local_context` 已足够；跨主机与外部入口证据只有在完整图上才能取得。在现有全部图
上深度 1 已达到深度 2 的全部效果，但这些图都很浅，该结论不可外推到多跳、多分支图。

## 8. 可以主张与不可主张的结论

可以主张：

1. **图上下文可以纠正 Stage 1 的排序错误**：Zerologon 中跨主机可达性把 T1210 从第 13 提到第 1，
   MRR 由 1/13 升到 1。
2. **v2 的避免伤害设计在独立案例上成立**：5 个独立本地提权案例 0 退化。
3. **消融行为诚实**，且方法完全确定性、无训练、不读取评价标签。
4. **失败被如实归因**：Tatsu 的正确标签不在 Stage 1 Top-20，属 Stage 1 召回问题，Stage 2 结构上
   无法补回，报告为不可恢复而非 Stage 2 排序失败。

不可主张：

1. **不能主张总体准确率或统计显著性**：可用案例仅 8 个，来自三个场景族、三个不同 Stage 1 run。
2. **不能主张本地提权族有提升**：扩展四例 Top-1、Top-3、MRR 全部持平，提升数为 0。
3. **不能用 CVE-2010-3856 作为效果证据**：标签与轨迹耦合，且其 Top-1 未改变。
4. **来源多样性有限**：扩展四例的图证据同源于 Rapid7 文档控制台记录这一种模态；M&NTIS 三例同源
   于一个数据集。
5. **不能把设计用案例当作独立验证**：Sudo 用于设计 v2，其改善不构成独立证据。

## 9. 方法的适用边界（本工作的核心结论）

综合全部案例，图上下文的作用取决于**拓扑本身是否具有区分力**：

- **拓扑有区分力时，Stage 2 有效**。外部入口指向 T1190，跨主机可达性指向 T1210，这些拓扑事实与
  特定 Technique 之间存在近似一一对应，因此能纠正排序。
- **拓扑只能确定 tactic 时，Stage 2 最多做到不添乱**。"同主机非特权执行变为最高特权执行"这一拓扑
  同时兼容 T1068、T1548、T1055、T1574 等多种机制，topology-only 证据在**原理上**无法区分它们。
  强行区分正是 v1 的错误来源。

因此本地提权族的"0 提升、0 退化"不是实现缺陷，而是 topology-only 证据能力边界的直接体现。

## 10. 已知失败模式

**同 tactic 竞争失败模式**（本轮首次观察到）：当 Stage 1 的 Top-1 与正确答案属于同一 tactic 时
（CVE-2022-26904 的 T1548 对 T1068），当前 tactic 级 guard 结构上无法改善名次。这与第 9 节的
适用边界一致——拓扑能确定发生了权限提升（tactic），却无法区分 T1068 与 T1548 等具体机制
（technique）；当正确答案与 Stage 1 Top-1 同处该 tactic 组时，组内保持 Stage 1 原序，正确答案
无法上移。该现象是 topology-only 证据能力边界在排序层面的直接体现，不是实现缺陷。

## 11. 复现

全部 run 目录被 Git 忽略但保留在实验主机上；场景 YAML、生成图、评价标签和测试均由 Git 跟踪。

```bash
# 单个案例（以扩展 LPE 主条件为例）
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack run-stage2 \
  --stage1-run /home/ghdemi/Code/cve2attack/runs/stage2_extended_lpe_v5c_attack15_1_top20_20260731T011812 \
  --attack-graph data/stage2_scenarios/extended_lpe/cve_2022_26904/AttackGraph.xml \
  --benchmark stage2_extended_lpe \
  --run-id <新的唯一 run id> \
  --scenario-kind trace_derived_local_privilege \
  --context-mode full_graph_context --max-graph-depth 1

# 从场景确定性重建攻击图
PYTHONPATH=src ../cve2attack/.venv/bin/python -m cve2attack build-stage2-graph \
  --scenario data/stage2_scenarios/extended_lpe/cve_2022_26904/scenario.yaml \
  --output /tmp/AttackGraph.xml

# 全部测试
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ../cve2attack/.venv/bin/python -m pytest -q -p no:cacheprovider
```

run 目录前缀：M&NTIS v1 `formal_v5c_attack15_1_e3d095d_*`；M&NTIS v2
`v2_tactic_guard_v5c_attack15_1_*`；M&NTIS 消融矩阵 `ablation_e8ca4ea_v5c_*`；PwnKit
`attackmate_pwnkit_4b5ff98_*`；扩展 LPE `extlpe_80383b0_20260808T0110_*`。
