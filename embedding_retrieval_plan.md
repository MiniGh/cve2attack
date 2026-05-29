# 阶段一 · 1b 嵌入检索原型 · 实现 Plan

> 给写代码 agent 的任务说明。**本次只实现嵌入检索这一条路**,不要做结构化链、不要做 LLM 精修、不要碰阶段二、不要做 sub-technique。
> 工作目录为 `./`。

---

## 0. 目标与产出

**目标:** 搭一个纯嵌入检索流水线。对每个 Enterprise 域的 CVE,用安全领域嵌入模型把 CVE 描述编码,和 ATT&CK technique 知识库做余弦相似度匹配,产出 **top-20 候选 technique(带分数)**。

**要产出两样东西:**

1. **候选集文件**(机器用):每个 CVE 一条记录,含其 top-20 候选 technique 及余弦分数。
2. **抽样对照文件**(人眼看效果用):随机抽 ~30 个 CVE,把"CVE 描述 + top-10 候选的 technique 名称和分数"排版成可读文本,供人工判断检索结果靠不靠谱。

**本次明确不产出 recall@k 数字**(原因见 §6:当前数据无 ground truth)。

---

## 1. 输入文件

| 文件 / 配置 | 内容 | 用途 |
|---|---|---|
| `./cve_to_attack_domain/result/CVE-{year}.jsonl` | 每行 `{"cve_id": ..., "domain": ...}` | 决定处理哪些 CVE(**本次只处理 domain == "Enterprise" 的**) |
| `./og_data/cve/CVE-{year}.json` | dict,key 是 cve_id,value 含 `description` / `cwes` / `cpes` | 取 CVE 的 `description` 作查询文本 |
| `./og_data/enterprise-attack.json` | MITRE ATT&CK 完整 STIX bundle(所有对象都在里面) | 从中抽取 technique 知识库 |
| 环境变量 `SILICONFLOW_API_KEY` | 硅基流动 API Key | 调用嵌入 API 鉴权，**不要硬编码进代码** |

**数据 join 逻辑:** 遍历某年的 domain jsonl → 对 domain == "Enterprise" 的 cve_id → 从同年的 `./og_data/cve/CVE-{year}.json` 里按 cve_id 取出 `description`。`year` 从 cve_id 解析(`CVE-2016-1499` → `2016`)。

---

## 2. 输出格式

**候选集:** `./output/retrieval/candidates.jsonl`,每行:

```json
{
  "cve_id": "CVE-2016-1499",
  "domain": "Enterprise",
  "query_text": "ownCloud Server before 8.0.10 ...",
  "candidates": [
    {"tech_id": "T1190", "name": "Exploit Public-Facing Application", "score": 0.81, "tactics": ["initial-access"]},
    {"tech_id": "T1083", "name": "File and Directory Discovery", "score": 0.77, "tactics": ["discovery"]}
  ]
}
```

- `candidates` 按 `score` 降序,长度 20。
- `score` 是余弦相似度原值,**不要在这一步跨 CVE 归一化**(后续融合再处理)。
- `tactics` 先存着(阶段二要用),本次不参与逻辑。

**抽样对照:** `./output/retrieval/inspect_sample.md`,人类可读,每个抽样 CVE 一段:CVE id + 完整描述 + top-10 候选(序号、tech_id、name、score)。

---

## 3. 实现步骤

### Step A · 从 enterprise-attack.json 抽取 technique 知识库

这是最容易出错的一步,严格按下面来。

1. 读 bundle,取 `data["objects"]`。
2. **筛选 technique**:保留满足以下**全部**条件的对象:
   - `type == "attack-pattern"`
   - `x_mitre_is_subtechnique` 不为 `true`(**只要顶层 technique,丢弃所有子技术**)
   - `revoked` 不为 `true`
   - `x_mitre_deprecated` 不为 `true`(字段可能不存在,不存在按 false 处理)
3. 对每个保留的 technique 抽:
   - `tech_id`:在 `external_references` 里找 `source_name == "mitre-attack"` 的那条,取它的 `external_id`(形如 `T1190`)。
   - `name`、`description`。
   - `tactics`:把 `kill_chain_phases` 里每个的 `phase_name` 收集成列表(形如 `["initial-access"]`)。
   - `stix_id`:对象的 `id`(后面挂 procedure 用)。
4. **抽取 procedure examples**(信息量比纯定义大,要加上):
   - 扫一遍所有 `type == "relationship"` 且 `relationship_type == "uses"` 且 `target_ref` 指向某个 technique 的对象,把它的 `description` 收集到 `target_ref → [描述...]` 的映射里。
   - relationship 的 description 里含 `[名字](链接)` 这种引用标记,**清洗成纯文本**(把 `[xxx](yyy)` 替换成 `xxx`)。
   - 每个 technique 的 procedure 文本做长度上限(比如总长截到 ~1500 字符),避免个别 technique 文档过长拉偏向量。
5. **拼成知识库文档**:每个 technique 一条文档 = `name + "。" + description + " " + 拼接后的 procedure 文本`。

最终得到一个列表,每项:`{tech_id, name, tactics, doc}`,预计 ~200 条。

### Step B · 调用 API 编码 technique 知识库（并缓存）

**模型：`BAAI/bge-m3`，通过硅基流动 API 调用。**

API 调用方式（OpenAI 兼容格式）：
- Endpoint：`https://api.siliconflow.cn/v1/embeddings`
- 鉴权：`Authorization: Bearer {SILICONFLOW_API_KEY}`
- 请求体：`{"model": "BAAI/bge-m3", "input": ["文本1", "文本2", ...]}`
- 返回：`response["data"][i]["embedding"]` 是第 i 条文本的向量（float 列表）
- 可直接用 `openai` Python 包：`client = OpenAI(api_key=..., base_url="https://api.siliconflow.cn/v1")`，然后 `client.embeddings.create(model="BAAI/bge-m3", input=[...])`

**批次大小：每次 API 请求传入不超过 32 条文本。** ~200 个 technique 分几批即可。

**缓存机制（重要，节省 API 费用）：**
- technique 向量编码完后，**立即保存到 `./output/retrieval/tech_embeddings_cache.npz`**（numpy 格式，存向量矩阵 + tech_id 列表）。
- 下次运行时先检查缓存文件是否存在，存在则直接加载，**不重复调用 API**。
- technique 知识库不会变，这个缓存长期有效。

向量归一化：从 API 拿回的向量做 L2 归一化（除以模长），方便后续用点积代替余弦计算。

**不需要 FAISS / Chroma**——~200 个向量，用 numpy 矩阵乘法暴力算即可。

### Step C · 逐 CVE 检索

- 对每个 Enterprise CVE：`query_text = description`（**首版只用描述，不拼 CWE**，原因见 §5）。
- 单条调用 API：`client.embeddings.create(model="BAAI/bge-m3", input=[query_text])`，取 `data[0].embedding`，做 L2 归一化。
- 和 technique 向量矩阵做点积（即归一化后的余弦相似度），取 top-20，记录每个的 `tech_id / name / score / tactics`。
- **限速处理**：CVE 量大时在请求之间加短暂等待（如每 10 条 sleep 0.5s），避免触发硅基流动的速率限制。如遇 429 错误，需实现指数退避重试（最多 3 次）。

### Step D · 写候选集 + 抽样对照

- 全部写入 `candidates.jsonl`。
- 随机抽 ~30 个 CVE 生成 `inspect_sample.md`。

---

## 4. 关键细节与坑

1. **子技术过滤要彻底**:Step A 第 2 步丢掉所有 `x_mitre_is_subtechnique == true`。`external_id` 带小数点的(如 `T1190.001`)也是子技术,正常情况上一步已滤掉,可作为兜底校验。
2. **STIX 字段缺省**:`x_mitre_deprecated`、`revoked` 可能不存在,用 `obj.get(field, False)` 安全取。
3. **relationship 引用标记清洗**:见 Step A 第 4 步,不洗会把一堆 URL 编进向量。
4. **CVE 原始文件是 dict**:按 cve_id 取值,不是数组。
5. **description 缺失/为空的 CVE**:跳过并记一条日志,不要让它进检索(空文本会产出噪声向量)。
6. **score 保留原值**:不要跨 CVE 做 min-max 之类的归一化,后续融合阶段才处理。
7. **只处理 Enterprise**:domain 是 ICS / Mobile 的本次直接跳过(没有对应矩阵 bundle,见 §6)。
8. **technique 向量只调一次 API 并缓存**：缓存存在就直接加载，缓存不存在才调 API 编码并保存。CVE query 的 API 调用量大，注意限速。

---

## 5. 明确不要做(scope 纪律)

- **不要**实现结构化链(CWE→CAPEC→ATT&CK)。
- **不要**接任何 LLM。
- **不要**做 sub-technique 粒度。
- **不要**搭向量数据库基础设施(200 个向量用不上)。
- **首版不要拼 CWE 文本进 query**:先要一个最干净的"纯描述检索"基线。CWE 增强是后面单独的对比实验(开关式加),现在加进去会让基线不纯、说不清提升来自哪。
- **不要**在没有 ground truth 的情况下编造/计算 recall —— 本次就是产出候选 + 人工看。

---

## 6. 需要用户拍板 / 补充的输入

写代码 agent 遇到下面这些**先按括号里的默认做法跑,并在产出里标注**,等用户确认:

1. **ground truth(最关键)**:当前数据没有"CVE→正确 technique"标注,所以**这次只能定性看候选**。要得到 V1 的 recall@k 数字,需要带标注的数据集(CVE2ATT&CK,约 1813 条)——它的 CVE 和用户自己的不是同一批,但这正是 V1 该用的评测集。(默认:本次不算 recall,只产出候选 + 抽样对照。)
   - 提醒:评测时若标注的是子技术(如 T1059.001),要先 roll-up 到父技术(T1059)再判命中,否则对"只做父级"的我们不公平。
2. **CWE 增强所需的 CWE 目录**:若之后要拼 CWE 文本,需要 CWE id → 名称/描述的查表(MITRE CWEC 列表)。(默认:首版不用。)
3. **ICS / Mobile 矩阵**:若要处理非 Enterprise 域,需要 `ics-attack.json` / `mobile-attack.json`。(默认:本次只做 Enterprise。)
