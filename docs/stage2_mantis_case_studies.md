# M&NTIS 第二阶段三场景验证

本文记录毕设最小闭环使用的三个公开执行轨迹场景。它们用于验证工程链路和暴露规则边界，
不是大样本总体准确率 benchmark。

## 数据与隔离原则

每个场景目录都保存四类可复现输入：

- `scenario.yaml`：从 M&NTIS manifest 和 attack report 归一化的上下文事实。
- `AttackGraph.xml`：由 `build-stage2-graph` 确定性生成的 MulVAL-compatible 攻击图。
- `stage1_snapshot/`：从真实 V3a 运行冻结的单 CVE Top-20，不需要再次调用 LLM。
- `data/benchmarks/stage2_mantis_scenarios/`：仅在重排结束后加载的 M&NTIS 标签。

`evaluation.expected_techniques` 不参与攻击图生成。测试会修改该字段并验证输出 XML 不变，
用于防止标签泄漏。

## 结果

| 场景 | CVE | M&NTIS 标签 | 图规则 | Stage 1 最佳名次 | Stage 2 最佳名次 | 结论 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 横向移动：Zerologon | CVE-2020-1472 | T1210 | `lateral_remote_service` | 2 | 1 | 提升 |
| 公开服务：Tatsu Builder RCE | CVE-2021-25094 | T1190 | `public_facing_service` | 不在 Top-20 | 不在 Top-20 | 第一阶段不可挽救 |
| 本地提权：Sudo Baron Samedit | CVE-2021-3156 | T1548.003（评价上卷为 T1548） | `local_privilege_transition` | 1 | 2 | 退化 |

三个场景的候选集合在重排前后完全相同。按各场景冻结输入汇总：

- Stage-1 Top-1：1/3；Stage-2 Top-1：1/3。
- Stage-1 Top-3：2/3；Stage-2 Top-3：2/3。
- Stage-1 MRR：0.500；Stage-2 MRR：0.500。
- 1 个提升、1 个退化、1 个因正确标签不在 Top-20 而不可挽救。

聚合指标没有提升，但三个案例分别验证了第二阶段的三条重要边界：图上下文可以纠正候选
顺序；无法补回第一阶段没有召回的 Technique；过于粗粒度的拓扑规则可能覆盖第一阶段已经
正确的、更具体的方法判断。

Sudo 结果不是测试错误。攻击图只表达“普通用户到 root”的状态变化，足以支持通用的
`T1068 Exploitation for Privilege Escalation`；M&NTIS 标签则利用了漏洞机制知识，将它标成更
具体的 `T1548.003 Sudo and Sudo Caching`。当前 topology-only v1 没有机制级证据，因此不应
根据这一条标签临时改规则。它应作为后续重排保护策略和机制特征设计的固定退化案例。

## 从冻结输入复现

以下命令不会加载 ATTACK-BERT，也不会调用 sec-i1：

```bash
PYTHONPATH=src .venv/bin/python -m cve2attack run-stage2 \
  --stage1-run data/stage2_scenarios/mantis/zerologon/stage1_snapshot \
  --attack-graph data/stage2_scenarios/mantis/zerologon/AttackGraph.xml \
  --benchmark stage2_mantis_scenarios \
  --run-id mantis_zerologon_reproduction \
  --scenario-kind trace_derived_mantis_lateral_movement

PYTHONPATH=src .venv/bin/python -m cve2attack run-stage2 \
  --stage1-run data/stage2_scenarios/mantis/tatsu_rce/stage1_snapshot \
  --attack-graph data/stage2_scenarios/mantis/tatsu_rce/AttackGraph.xml \
  --benchmark stage2_mantis_scenarios \
  --run-id mantis_tatsu_reproduction \
  --scenario-kind trace_derived_public_facing

PYTHONPATH=src .venv/bin/python -m cve2attack run-stage2 \
  --stage1-run data/stage2_scenarios/mantis/sudo_cve_2021_3156/stage1_snapshot \
  --attack-graph data/stage2_scenarios/mantis/sudo_cve_2021_3156/AttackGraph.xml \
  --benchmark stage2_mantis_scenarios \
  --run-id mantis_sudo_reproduction \
  --scenario-kind trace_derived_local_privilege
```

运行结果写入 `stage2_runs/<run_id>/`，该目录不进入 Git。固定输入、预期行为和防泄漏检查
由 `tests/test_stage2_scenario_graph.py` 覆盖。
