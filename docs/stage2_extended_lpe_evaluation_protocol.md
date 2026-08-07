# Stage 2 扩展本地提权评测预注册

冻结时间：2026-07-31（Asia/Shanghai）
状态：来源、清单、标签、规则、Stage 1 候选和五张新图均已冻结；正式 Stage 2 评价已于 2026-08-08
在 18 个不可覆盖 run 上完成（见 4.1、6 节）。

## 1. 目的与闸门

本轮扩大 topology-only v2 的独立本地提权验证范围，不修改 Stage 1 V5c，不修改 Stage 2
规则，也不根据正确标签挑选或调整案例。来源和标签先冻结，候选随后由冻结 Stage 1 生成；
攻击图只能从已登记的执行步骤与权限变化证据构建。

在以下条件全部满足前，不运行新的正式 Stage 2 评价：

1. 每张新图完成 label-blind 规范化、测试和人工审阅；
2. 图文件 SHA256、场景 YAML 和来源行号冻结；
3. 正式 Stage 1 run 为 `complete`，六个 CVE 均恰好有 20 个互异父 Technique；
4. 评价前不查看候选中的正确标签名次，不修改 v2 规则或权重；
5. 正式 run ID 全新且不可覆盖。

## 2. 冻结 CVE 清单与角色

| CVE | 角色 | 轨迹/日志证据 | 独立 ATT&CK 标签 | 是否进入新增主聚合 |
| --- | --- | --- | --- | --- |
| CVE-2010-3856 | 必选诊断案例 | AttackMate 步骤 + Rapid7 root 会话记录 | AttackMate 作者标注 T1068；未找到与轨迹独立的 CTID 标签 | 否 |
| CVE-2020-0787 | 新主案例 | Rapid7 normal→SYSTEM 完整控制台记录 | CTID KEV T1068（exploitation technique） | 是 |
| CVE-2021-4034 | 桥接/既有对照 | AttackMate + Rapid7 msfuser→root 记录 | CTID KEV T1068（exploitation technique） | 否 |
| CVE-2021-40449 | 新主案例 | Rapid7 normal→SYSTEM 完整控制台记录 | CTID KEV T1068（primary impact） | 是 |
| CVE-2022-21999 | 新主案例 | Rapid7 本地用户→SYSTEM 完整控制台记录 | CTID KEV T1068（primary impact） | 是 |
| CVE-2022-26904 | 新主案例 | Rapid7 normal non-admin→SYSTEM 完整控制台记录 | CTID KEV T1068（primary impact） | 是 |

新增主聚合固定为 4 个 CVE。PwnKit 只用于连接既有结果，CVE-2010-3856 单独报告，不因其
标签证据较弱而删除，也不把它混入独立标签主指标。

## 3. 冻结来源

### 3.1 Rapid7 Metasploit Framework

- 仓库：`https://github.com/rapid7/metasploit-framework`
- 固定提交：`1816d9023b353800046567984f15b42d24bd334a`
- 本地 ignored 路径：`data/stage2_sources/metasploit/repository`
- 获取方式：GitHub SSH 浅层、blob-filter、sparse checkout
- 许可：BSD-3-Clause；第三方组件例外以该提交 `COPYING` 为准
- `COPYING` SHA256：`38f848ebdf03a4f7ce5a703b72e8be6bd724de17d7869d71f3df78734cb4e507`

固定文件哈希：

| CVE | 模块 SHA256 | 执行记录 SHA256 |
| --- | --- | --- |
| CVE-2010-3856 | `f7126f6eb6ee87edd98de53d4f265b41d4386b11d11455ecfee918c28171fa40` | `0c72acdaa16bec8cb8934c3d0fa209859420f4c47bcba81e7b68f74d3faf2231` |
| CVE-2020-0787 | `afe494a1be326f819e0a514868664278c62de242c3f1a5bdb00786eb7373afd9` | `42ad0e058290155ee39eb82fdd905206f8a2480d4a2b8d743e326735d2f1d911` |
| CVE-2021-4034 | `7c238f16f477213d7f84c6d508bb08ce9a3df5ffcc49250c0b9065abbd5f410e` | `99ca8d99d1c45a0c968458241fbba446ab4e28cc172a8f0c251530dcaaab028e` |
| CVE-2021-40449 | `af7b23bbae6179c4c7ed3768ea3293170a2f653701e0e99edf9c26187dfccad9` | `c1d8b617acbbe4cd72bb35cc5d65b10efb69e3488d122de9d9f9e25aa8a94067` |
| CVE-2022-21999 | `bd86f2ac8cbcf4c3033bdf620b1e5b01a984e9aacc426ce00f590fc10d1e667f` | `b879679af021105cfe1554d6dc56d88acb177ba492d76688840690355e67d427` |
| CVE-2022-26904 | `fee4f463b59e53cd7ef58a16f3d9180fd3df852f1cf224cb999e6115cb8a96d2` | `353a80cc8bfdd231122939486e0505417da5ec9725cf40feb4b068ab1c6776ad` |

执行记录中的身份检查和新会话输出是场景事实来源；模块或文档中的 ATT&CK 推断不作为主标签。

### 3.2 AttackMate

- 仓库：`https://github.com/ait-testbed/attackmate`
- 固定提交：`d2edd8bfbb4d18bf4788f222022fe8c73d8fb58f`
- 许可：GPL-3.0
- `examples/http-put_example.yml` SHA256：
  `7d5a27eed6c931bdb57449b6b8b2e964d1d232bbfab9c62599ad5f5c4907b267`
- `playbook.yml`（PwnKit）SHA256：
  `91ce76fd0e09cf1cd50e899ae3b1ba3dedae7f089b2b6328cc23ed110f008970`

Zenodo v4（DOI `10.5281/zenodo.19810174`）为 CC-BY-4.0。2026-07-31 从 `pri_sun`
访问文件下载端点时，`zenodo.org:443` 立即拒绝连接；没有绕过，也没有留下部分文件。
2026-08-08 改为人工在 Windows 侧下载、经 Windows OpenSSH `scp.exe` 交付到下列 ignored 目录，
并完成校验（页面 MD5 全部一致，本地 SHA256 见表末登记）：

`data/stage2_sources/attackmate/downloads/zenodo_v4_20260731/`

| 文件 | Zenodo MD5 | 用途 | 优先级 |
| --- | --- | --- | --- |
| `playbooks.zip` | `6a1dd5cf1a89d85915065124ab8ee08a` | 固定数据集实际 playbook 与场景关联 | 必需 |
| `privilege_escalation_attackmate.zip` | `f6c3d5b04f8855d2bd90be9d0143bf14` | AttackMate 执行日志和主机/网络遥测 | 必需 |
| `privilege_escalation_atomic.zip` | `fdb102503ceb032375c5ebbfbe140e0f` | Atomic Red Team 对照执行日志 | 可选增强 |

交付登记（2026-08-08，本地 SHA256）：`playbooks.zip` = `1e28d9ed0939009631e8c37b3c77213f4a89bac1feb3ffd4c64ba6ef4282abd7`；`privilege_escalation_attackmate.zip` = `f595e38fc9e52c3f523f0245f674a94a3a0028fda27838d99225378984eb580a`；`privilege_escalation_atomic.zip` = `7bd2193ce2901dfed033aa6347b471fbea138d3697f8e9c730be0ca42aebb8e5`。原始压缩包保持 ignored；交付到位不改变标签隔离与建图闸门。

### 3.3 CTID KEV→ATT&CK

- 页面/DOI：`https://zenodo.org/records/16747173`，`10.5281/zenodo.16747173`
- 版本：`02.13.2025`，ATT&CK Enterprise `15.1`
- 文件：`kev-02.13.2025_attack-15.1-enterprise.csv`
- MD5：`21338a56761278482dc5f169638414ca`
- SHA256：`8f15aab468f17f9a1d655ef2db814b0323792cfa066373a02a0a1d7f4a8f6676`
- 本地冻结文件：
  `/home/ghdemi/Code/cve2attack/data/raw/kev/kev-02.13.2025_attack-15.1-enterprise.csv`
- 上游 Mappings Explorer 仓库为 Apache-2.0；Zenodo 该记录页面没有显示许可证，报告时保留此差异。

CTID 文件只在候选生成完成后用于评价。它的 comments、references 和 Technique 字段均不进入
Stage 1 查询、action 语料或攻击图构建。

## 4. Stage 1 候选生成预注册

选择 cohort 固定为以下排序：

1. CVE-2010-3856
2. CVE-2020-0787
3. CVE-2021-4034
4. CVE-2021-40449
5. CVE-2022-21999
6. CVE-2022-26904

输入目录只负责传递 CVE ID，使用明确的非 ATT&CK 哨兵 `T0000`，不保存任何正确标签；其
Stage 1 metrics 必须忽略。正式候选固定使用
`experiments/validation/v5c_raw_action_rank_rrf_attack15_1.yaml`：raw description、ATT&CK
15.1、完整 action 语料、CVE/CAN 屏蔽、查询级 strict LOO、Top-3 rank-RRF、k=60、Top-20。
不得修改方法、参数、模型或候选预算。

验收要求：manifest `status=complete`；覆盖 6/6；每条恰好 20 个互异父 Technique；候选文件
逐文件计算 SHA256；记录绝对 run 路径、完整 cohort、Git commit、ATT&CK 版本、202 个父
Technique 和 action 数。候选生成程序不得读取本文件第 2、3.3 节的正确标签。

实际冻结 run：

- 绝对路径：`/home/ghdemi/Code/cve2attack/runs/stage2_extended_lpe_v5c_attack15_1_top20_20260731T011812`
- manifest SHA256：`228d4318a35e7a2a967c12b8959f633185a1fb5c689a520a4d14dfd172076753`
- Stage 1 commit：`9688a5b340dee0de3af4dd3ceaa48bf0267fc9d4`
- 状态/覆盖：`complete`，6/6，缺失描述 0
- 语料：ATT&CK Enterprise 15.1，202 个父 Technique，14,121 条 action
- 方法：raw description、strict LOO、Top-3 rank-RRF、k=60、Top-20
- 结构验收：六条记录均为 20 个互异父 Technique，没有子技术 ID

候选文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `CVE-2010.jsonl` | `9130243dac61b43656a4c728fb526e84db0eace744b38d6ca535bf9a15521e96` |
| `CVE-2020.jsonl` | `c8933f18cb852b701e5c0d8bb1360c3a88986c32f2b1cdf777ee7efe7636465b` |
| `CVE-2021.jsonl` | `33068234ec16c2f1070da7db07dde542fb6e08fcbba8b3f432dbbe5d1576fc6a` |
| `CVE-2022.jsonl` | `27ef57ab537c24199aeeaaf92c873f5288be880b3ef91daf16a63855916a0110` |

## 4.1 Label-blind 图构建与冻结（2026-08-08）

五张新图已按第 1 节闸门 label-blind 构建、测试并冻结。构建只读取已登记的 Rapid7 执行记录中的
身份变化证据；未查看任何候选排名，未修改 `topology-rule-priority-v2`。

统一转换：每图 4 顶点 / 3 边，`execCode(TARGET,user)` 与
`vulExists(TARGET,'<CVE>',<组件>,localExploit,privilegeEscalation)` 经一条 trace-derived 规则
产生 `execCode(TARGET,root)`。图中不含 `attackerLocated`、`hacl`、`netAccess`、
`networkServiceInfo`，因此只可能触发本地提权规则。

权限令牌归一：四个 Windows 案例的实际身份为 `NT AUTHORITY\SYSTEM`，统一归一为通用最高本地
权限令牌 `root`，与既有 PwnKit 场景词汇一致。该决定在查看任何候选排名之前、对全部扩展 LPE 图
统一作出，只编码“同主机非特权执行变为最高特权执行”，不引入 OS 特定语义，也不修改重排规则
（`reranker.local_privilege_transition` 只识别 `root`）。CVE-2010-3856 原生到达 `root`，不需归一。

| CVE | 组件 | 场景 YAML SHA256 | 图 SHA256 |
| --- | --- | --- | --- |
| CVE-2020-0787 | `bits` | `4b3236d37f35942120fbb524e45808dba1777d48c8e79ba7a6c610c6e8f0c7c3` | `a713a989dbea94741f9bf6d6e9fd86dd66b8606048b0d40c4cbf6733e6eb78b9` |
| CVE-2021-40449 | `win32k` | `ecedbc59f82d4e631506fea26488a9921be43024b78c1e6173a5f774cb18e4fc` | `dfe4eae077c177529d77488505aba8da5d399ed9b9b5fe013a288565120f7b36` |
| CVE-2022-21999 | `spooler` | `4fc4c7e4505d28427d1c7e05749daf6d4fff1a48e2ea9317d3682b328dd077f3` | `21742c98e2da07075b89fb1dc462caa1fcce6dad0fbd97071ffc64ef437e0237` |
| CVE-2022-26904 | `profsvc` | `c7cb68b17c22fccecc8f5d1b18042a2b2bd5367a06c2c17c5f8f60de7c59506e` | `691554323a8e6644605698ec5c07d5b489245ea78ba6fd953214131c0a89fbdf` |
| CVE-2010-3856 | `glibc_ld_audit` | `bdd48c213bf994c7b0077bd780288ac1a866fe9961a153d4f638b6daa9bdd434` | `a30f11a363c8070a49cfb878bcc5a60922d3a62cf185bbf82073cfbbcdf356aa` |

每张图的精确来源行号记录在对应 `scenario.yaml` 的 `source.predecessor_step`、
`source.attack_step` 和 `source.confirmation_steps` 中，均指向冻结提交
`1816d9023b353800046567984f15b42d24bd334a` 的模块文档；CVE-2010-3856 另引用 AttackMate
`examples/http-put_example.yml` 第 103-113、116-118 行。

闸门核验（2026-08-08）：五图均可从 YAML 逐字节确定性重建；改写
`evaluation.expected_techniques` 不改变生成 XML；五图各恰好解析出 1 个 CVE 且只触发
`local_privilege_transition`（tactic 级、`privilege-escalation`、回退 `T1068`）；定向测试
`14 passed, 38 subtests`，完整测试 `53 passed, 38 subtests`。

Zenodo v4 档案的实际价值：`playbooks.zip` 与 `privilege_escalation_attackmate.zip` 的提权场景是
基于 cronjob 配置错误的提权（声明 `T1053`/`T1190`/`T1059`/`T1087`），不含本轮六个 CVE 中的任何
一个，也不含 `T1068`；日志包为该场景的主机遥测转储。因此这些档案未为六例提供新证据，六例证据
仍全部来自 Rapid7 冻结提交与 AttackMate git 仓库。

下一步仍受闸门约束：正式 Stage 2 评价只能在本节冻结的图与既有冻结 Stage 1 run 上运行，
run ID 必须全新且不可覆盖，且新增主聚合只含四个 Windows 主案例。

## 5. 后续 Stage 2 正式评价规则

- 实现固定为当前未修改的 `topology-rule-priority-v2`。
- 主条件为 `full_graph_context`，预注册深度为 1；`no_context` 和 `local_context` 作为消融。
- 候选集合必须前后完全一致；Stage 2 只重排，不补入正确答案。
- 每例报告原/新排名、Top-1、Top-3、MRR、提升/保持/退化/不可恢复、规则和上下文证据。
- 每个 run 保存 `join_stats`、图 SHA256、Stage 1 manifest 路径与 SHA256、Stage 2 manifest。
- 新增主指标只计算 4 个新主案例；CVE-2010-3856、PwnKit、三个 M&NTIS 案例分别报告。
- 如果正确标签不在 Stage 1 Top-20，状态固定为不可恢复，归因于 Stage 1 召回。
- 不以这 4 个新案例调规则；任何未来规则变化必须先形成独立假设和新版本，再使用另一批案例。

本预注册不授权或触发 Stage 2 运行。下一步只能进行 label-blind 图构建、测试与审阅；
图和来源行号冻结前仍不得进行正式评价。

## 6. 正式评价结果（2026-08-08）

评价在冻结图、冻结 Stage 1 run 和未修改的 `topology-rule-priority-v2` 上运行，共 18 个不可覆盖
run，前缀 `stage2_runs/extlpe_80383b0_20260808T0110_`，涵盖 6 个案例 × 3 种上下文模式，
预注册主条件为 `full_graph_context` 深度 1。全部 run 的 `status=complete`、
`git_commit=80383b0...`、`reranker=topology-rule-priority-v2`、`uses_target_semantics=false`、
`matched=1`、`missing_candidates=[]`、`unresolved_context_ids=[]`、候选数 20、
`candidate_sets_preserved=true`。

评价标签使用新增的专用 benchmark `data/benchmarks/stage2_extended_lpe/`，每例真值严格为预注册的
单一 `T1068`。不使用 `ctid_kev_2025_02_13_*` 混池 benchmark：那些数据把全部 mapping type 合并，
CVE-2021-40449 会得到 8 个真值（含 T1566 Phishing、T1071、T1082），CVE-2022-21999 得到 5 个，
CVE-2020-0787 得到 2 个；用它评价会让与本地提权无关的 technique 命中也算成功，且各例真值数量不
一致、跨例不可比。

### 6.1 T1068 名次（Stage 1 → Stage 2）

| 案例 | 角色 | `no_context` | `local_context` | `full_graph_context` d1 |
| --- | --- | --- | --- | --- |
| CVE-2020-0787 | 主 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2021-40449 | 主 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2022-21999 | 主 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2022-26904 | 主 | 2 → 2 | 2 → 2 | 2 → 2 |
| CVE-2021-4034 | 桥接 | 1 → 1 | 1 → 1 | 1 → 1 |
| CVE-2010-3856 | 诊断 | 7 → 7 | 7 → 5 | 7 → 5 |

### 6.2 四例主聚合

三种模式结果完全相同：

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Top-1 | 3/4 | 3/4 |
| Top-3 | 4/4 | 4/4 |
| MRR | 0.875 | 0.875 |

结果组成：0 提升、4 保持、0 退化、0 不可恢复。

### 6.3 为什么主聚合没有变化

这不是规则未触发，也不是消融开关失效。逐候选记录显示规则确实触发并确实改变了排序：在
CVE-2022-26904 中 `local_privilege_transition` 把全部 `privilege-escalation` 候选提到非匹配候选
之前，T1078 由 6 升到 5、T1574 由 7 升到 6、T1098 由 8 升到 7、T1546 由 10 升到 8。

主聚合不变有两个原因：

1. **Stage 1 几乎没有留下改进空间。** 四例中有三例的 T1068 原本已是第 1，Stage 2 只能保持。
2. **唯一有空间的一例被同 tactic 竞争者挡住。** CVE-2022-26904 的 Stage 1 第 1 名是 T1548
   （Abuse Elevation Control Mechanism），其 metadata tactics 含 `privilege-escalation`，与 T1068
   属于同一个 tactic 分组；v2 在组内保持 Stage 1 原序，因此 T1068 仍为第 2。这正是 v2 的设计
   意图：非 root→root 拓扑能证明发生了本地权限提升，但不能独立区分 T1068 与 T1548 等具体机制，
   因此规则拒绝在机制层面下判断。

### 6.4 可以主张与不可主张的结论

可以主张：

- 在四个来源独立、Stage 1 冻结的 Windows 本地提权案例上，topology-only v2 **没有造成任何退化**
  （0 losses），把独立避免伤害案例数从 1（PwnKit）扩大到 5。
- 消融行为诚实：`no_context` 精确复现 Stage 1 排名；`local_context` 已足以识别同主机权限变化，
  在这些最小图上与 `full_graph_context` 深度 1 结果相同。
- v2 的 tactic 级证据分辨率在真实同 tactic 竞争情形（CVE-2022-26904 的 T1548 对 T1068）下按设计
  行为执行，没有把拓扑证据过度解释为机制判断。

不可主张：

- **不能主张本轮带来准确率提升。** 四例主聚合 Top-1、Top-3、MRR 全部持平，提升数为 0。
- 唯一出现名次上升的 CVE-2010-3856（7 → 5）是诊断例，其 T1068 标注与同一 AttackMate 轨迹耦合，
  不是独立金标，且其 Top-1 仍为 T1574，因此不能作为效果证据。
- 样本量仍为 4（主）+1（桥接）+1（诊断），不能作为总体准确率或显著性结论。
- 四个主案例的图证据同源于 Rapid7 文档控制台记录这一种模态，来源多样性有限。

### 6.5 由结果引出的新问题

CVE-2022-26904 给出了一个此前没有观察到的具体失败模式：当 Stage 1 的 Top-1 与正确答案属于同一
tactic 时，当前 tactic 级 guard 结构上无法改善名次。要在这类情形上取得提升，需要能区分机制的
额外证据（例如漏洞组件与 Technique 机制的对应关系），而不是继续调 v2 权重。是否引入该证据、
以及如何避免引入目标语义泄漏，属于新假设，必须先形成独立预注册并使用另一批案例验证，不得在本
四例上调参后再宣称独立效果。
