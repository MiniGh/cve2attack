# Stage 2 扩展本地提权评测预注册

冻结时间：2026-07-31（Asia/Shanghai）
状态：来源、清单、标签、规则和 Stage 1 候选已冻结；新图尚未构建，禁止运行正式 Stage 2 评价。

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
