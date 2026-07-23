# 第二阶段：攻击图上下文提取

第二阶段接收 MulVAL 生成的 `AttackGraph.xml`。当前已实现的工作是把图中的每个
`vulExists` 节点转换为可测试、带版本号的上下文 JSON；第一阶段候选接入、确定性
重排序和最终评价将在后续模块中实现。

## 代码位置

```text
src/cve2attack/stage2/
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
- `candidates`：预留给下一工作包接入第一阶段 `CandidateRecord`，当前为空数组。

`exploit_type` 和 `expected_impact` 来自攻击图生成时的漏洞语义。在最终实验中需要
与纯拓扑上下文分别做消融，避免把由 CVE 描述生成的标签重新当作独立证据。
