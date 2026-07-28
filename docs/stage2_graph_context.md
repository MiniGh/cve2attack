# 第二阶段：攻击图上下文提取

第二阶段的正式图输入仍是 MulVAL 兼容的 `AttackGraph.xml`。当前实现可以把图中的
每个 `vulExists` 节点转换为带版本号的上下文 JSON，并接入第一阶段候选、执行确定性
重排序和统一评价。

M&NTIS 等公开场景包提供 `topology.yaml`、`attacks.json` 和执行证据，但它们不是
MulVAL XML。此类数据先整理成版本化场景 YAML，再由 `build-stage2-graph` 确定性生成
`AttackGraph.xml`。场景的 `context` 与 `evaluation` 必须分离：前者用于生成图，后者
只保存评价标签，转换器不会把 ATT&CK 答案写入图事实。

## 代码位置

```text
src/cve2attack/stage2/
├── scenario_graph.py     # 规范化场景 YAML → MulVAL-compatible AttackGraph.xml
├── graph_parser.py       # AttackGraph.xml → NetworkX，并显式反转边方向
├── path_expander.py      # 保留 OR 状态的全部上游规则与证据分支
├── context_extractor.py  # 生成 local_context 与 graph_context
└── pipeline.py           # 文件读写、进度输出和版本化 JSON
```

对应测试在 `tests/test_stage2_context.py`，固定样例在
`tests/fixtures/mulval/AttackGraph.xml`。正式运行不依赖旧的
`ldh_attackgraph/mulvalOutput` 目录，任意攻击图都通过命令行参数传入。

## 为什么不再选择一条“主路径”

MulVAL 中 `vulExists` 是利用规则的一个前提，不是攻击事件节点。同一个 OR 状态
可能由多条规则产生。旧原型会任意选择其中一条，并在现有样例中把第二个漏洞的
网络前提缩减成 `fileServer → fileServer` 自访问分支。

新提取器执行以下规则：

1. `local_context` 保留当前利用规则直接要求的状态和直接后果。
2. `graph_context` 保留每个状态的全部上游 producer rules。
3. 当前 CVE、当前规则和当前后果仍出现在证据中，但标为 boundary，不继续递归。
4. 深度达到上限的非叶节点标记为 `truncated`，不会伪装成完整证据。

因此上下文提取只负责保存证据，不在这一阶段偷偷替重排序器做路径选择。

## 命令行

公开执行场景转换：

```bash
.venv/bin/python -m cve2attack build-stage2-graph \
  --scenario data/stage2_scenarios/mantis/zerologon/scenario.yaml \
  --output data/stage2_scenarios/mantis/zerologon/AttackGraph.xml \
  --force
```

| 参数 | 必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--scenario PATH` | 是 | — | schema 1.0 的统一场景 YAML。 |
| `--output PATH` | 是 | — | 生成的 MulVAL-compatible `AttackGraph.xml`。 |
| `--force` | 否 | false | 允许覆盖已有生成图。 |

统一场景至少包含 `source`、`context` 和 `evaluation`。`source` 记录原始数据集、
攻击步骤和 CVE 解析依据；`context` 只保存初始状态、主机关系、目标服务、漏洞和
后果；`evaluation.expected_techniques` 是隔离的事后答案。测试会修改答案并断言生成
XML 完全不变，以防标签泄漏。

上下文提取：

```bash
.venv/bin/python -m cve2attack extract-graph-context \
  --attack-graph tests/fixtures/mulval/AttackGraph.xml \
  --output stage2_runs/example/contexts.json \
  --max-graph-depth 2
```

参数：

| 参数 | 必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--attack-graph PATH` | 是 | — | MulVAL `AttackGraph.xml`。相对路径从项目根目录解析。 |
| `--output PATH` | 是 | — | 上下文 JSON 输出路径。相对路径从项目根目录解析。 |
| `--max-graph-depth N` | 否 | `2` | 上游证据展开深度，必须为非负整数。 |
| `--force` | 否 | false | 允许覆盖已经存在的输出文件。 |

运行时依次输出解析、反转、CVE 识别、逐 CVE 提取和结果写入进度。默认拒绝覆盖
旧文件，避免把不同攻击图或参数的结果混在一起。

## 上下文 JSON

顶层结构：

```json
{
  "schema_version": "1.0",
  "attack_graph": {
    "source": "/absolute/path/AttackGraph.xml",
    "node_count": 44,
    "edge_count": 52,
    "type_counts": {"AND": 17, "LEAF": 19, "OR": 8},
    "edge_direction": "requirement_to_effect"
  },
  "contexts": []
}
```

每个 `contexts` 记录包含：

- `cve_id`：用于连接第一阶段候选；历史 `CAN-...` 自动规范为 `CVE-...`。
- `vulnerability_id_raw`：攻击图中的原始漏洞标识。
- `local_context`：目标主机、服务、利用类型、影响、直接前提、规则和后果。
- `graph_context`：直接前提的全部上游证据分支。
- `candidates`：单独执行 `extract-graph-context` 时为空；`run-stage2` 会填入第一阶段 `CandidateRecord`。

`exploit_type` 和 `expected_impact` 来自攻击图生成时的漏洞语义。在最终实验中需要
与纯拓扑上下文分别做消融，避免把由 CVE 描述生成的标签重新当作独立证据。

## 最小闭环

`run-stage2` 会读取一个已有第一阶段 run，用规范化 CVE ID 填充 `candidates`，然后
执行 `topology-rule-priority-v1`。该规则集当前识别：

- `public_facing_service` → 优先 `T1190`；
- `lateral_remote_service` → 优先 `T1210`；
- `local_privilege_transition` → 优先 `T1068`。

这些规则只读取拓扑事实，不读取 `exploit_type`、`expected_impact` 或 benchmark 标签。
匹配候选被移到非匹配候选之前，两组内部都保留第一阶段顺序。原候选 `score` 不覆盖，
原名次、新名次、规则和证据写入 `metadata.stage2`。

当前固定公开服务场景位于：

```text
tests/fixtures/stage2/public_facing/
├── AttackGraph.xml
└── scenario.yaml
```

它使用真实 `CVE-2023-20887`，但拓扑是工程合成输入；只能验证链路，不能直接作为
独立实验准确率证据。

首个公开执行轨迹派生场景位于：

```text
data/stage2_scenarios/mantis/zerologon/
├── scenario.yaml
├── AttackGraph.xml
└── stage1_snapshot/
```

它将 M&NTIS Zerologon 场景标准化为“已有普通用户会话 → 跨主机访问域控 →
获得管理员凭据”。第一阶段真实 V3a Top-20 中 `T1210` 原排第 2，拓扑规则将其提升
到第 1。该结果是可复现案例研究，不是大样本总体准确率结论。
