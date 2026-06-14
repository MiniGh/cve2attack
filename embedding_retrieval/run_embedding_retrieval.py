"""
Embedding-only CVE to ATT&CK retrieval pipeline (v4 — description only).

This script implements phase-1 retrieval with these constraints:
- Enterprise domain only.
- Parent techniques only (no sub-techniques).
- No structured chain (CWE→CAPEC→ATT&CK), no LLM reranking.
- No Procedure Examples in technique documents; only technique name + description.
- Output top-k technique IDs per CVE.

Change log:
  v4 (description_only): Removed Procedure Examples from technique embedding
      documents.  Reverted to the approach from commit cf3b2bb where only
      the technique's own name and description are used.  Rationale: adding
      Procedure Examples produced no measurable improvement (commit e295530)
      and inflates the embedding input without benefit.

  Usage: python run_embedding_retrieval.py [--args...]
"""

from __future__ import annotations

import argparse
import json
import random           # reservoir sampling 随机采样
import re               # Markdown 链接清洗
import time             # 请求限速等待
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: sentence-transformers/torch. "
        "Install with: pip install sentence-transformers torch numpy"
    ) from exc


# ─────────────────────────────────────────────────────────────────
#  路径常量：项目根目录下各数据目录
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN_DIR = PROJECT_ROOT / "cve_to_attack_domain" / "result"
DEFAULT_CVE_DIR = PROJECT_ROOT / "og_data" / "cve"
DEFAULT_ATTACK_BUNDLE = PROJECT_ROOT / "og_data" / "enterprise-attack.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "retrieval"

# ─────────────────────────────────────────────────────────────────
#  嵌入模型 & 缓存版本
# ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "basel/ATTACK-BERT"
# CACHE_VERSION 决定了缓存文件名；修改 technique 文档内容时必须变更版本号，
# 否则会误用旧向量导致结果不一致。
CACHE_VERSION = "v4_description_only"


@dataclass
class TechniqueDoc:
    """一条 ATT&CK technique 的知识库记录。

    Attributes:
        tech_id: ATT&CK 技术 ID（如 T1190）。
        name: 技术名称（如 "Exploit Public-Facing Application"）。
        tactics: 所属战术阶段列表（如 ["initial-access"]）。
        stix_id: STIX 对象 ID（用于关联 relationship）。
        doc: 拼接后用于嵌入检索的纯文本。
    """

    tech_id: str
    name: str
    tactics: List[str]
    stix_id: str
    doc: str


# ─────────────────────────────────────────────────────────────────
#  命令行参数解析
# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 argparse.Namespace。"""
    parser = argparse.ArgumentParser(
        description="Run embedding retrieval for Enterprise CVEs (v4 description-only)."
    )
    parser.add_argument(
        "--domain-dir", type=Path, default=DEFAULT_DOMAIN_DIR,
        help="Directory containing CVE-{year}.jsonl domain files."
    )
    parser.add_argument(
        "--cve-dir", type=Path, default=DEFAULT_CVE_DIR,
        help="Directory containing yearly raw CVE JSON dict files."
    )
    parser.add_argument(
        "--attack-bundle", type=Path, default=DEFAULT_ATTACK_BUNDLE,
        help="Path to enterprise-attack.json STIX bundle."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for yearly results and inspect file."
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of technique candidates per CVE."
    )
    parser.add_argument(
        "--sample-size", type=int, default=30,
        help="Number of CVEs for inspect_sample markdown."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Embedding batch size for technique docs."
    )
    parser.add_argument(
        "--query-sleep-every", type=int, default=10,
        help="Sleep interval in number of CVE queries (rate limiting)."
    )
    parser.add_argument(
        "--query-sleep-seconds", type=float, default=0.5,
        help="Sleep duration in seconds for rate limiting."
    )
    parser.add_argument(
        "--max-cves", type=int, default=None,
        help="Optional cap for number of Enterprise CVEs to process."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for inspect sample reservoir sampling."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  文本清洗工具
# ─────────────────────────────────────────────────────────────────
def clean_markdown_links(text: str) -> str:
    """将 Markdown 链接 [显示的文本](url) 转为纯文本（只保留显示文本）。"""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text or "")


def normalize_text_for_doc(text: str) -> str:
    """对文档文本做基础清洗：去掉 Markdown 链接 + 合并多余空白。"""
    cleaned = clean_markdown_links(text)
    return re.sub(r"\s+", " ", cleaned).strip()


# ─────────────────────────────────────────────────────────────────
#  CVE ID 工具
# ─────────────────────────────────────────────────────────────────
def get_year_from_cve_id(cve_id: str) -> str:
    """从 CVE 编号（如 "CVE-2024-1234"）中提取年份 "2024"。"""
    parts = cve_id.split("-")
    if len(parts) < 3 or not parts[1].isdigit():
        raise ValueError(f"Unexpected CVE format: {cve_id}")
    return parts[1]


# ─────────────────────────────────────────────────────────────────
#  ATT&CK technique 知识库提取
# ─────────────────────────────────────────────────────────────────
def extract_tech_id(external_references: Sequence[dict]) -> str | None:
    """从 external_references 中读取 ATT&CK 的 external_id（如 T1190）。

    ATT&CK STIX 把 technique ID 存在 external_references 数组中，
    需要找到 source_name == "mitre-attack" 的那条取 external_id。
    """
    for ref in external_references or []:
        if ref.get("source_name") == "mitre-attack":
            ext_id = ref.get("external_id")
            if isinstance(ext_id, str) and ext_id.strip():
                return ext_id.strip()
    return None


def compose_technique_doc(name: str, description: str) -> str:
    """为一条 technique 构造嵌入检索文本。

    v4 (description_only): 只用 name + description，不再拼接 Procedure Examples。
    格式：
        Technique Name: <name>
        Technique Description: <description>
    """
    parts: List[str] = []

    if name:
        parts.append(f"Technique Name: {name}")
    if description:
        parts.append(f"Technique Description: {description}")

    return "\n\n".join(parts)


def extract_technique_kb(attack_bundle_path: Path) -> List[TechniqueDoc]:
    """从 MITRE ATT&CK STIX bundle 中提取顶层 (parent-only) technique 知识库。

    过滤规则：
      - 只保留 type == "attack-pattern"
      - 排除子技术 (x_mitre_is_subtechnique == true)
      - 排除已撤销 (revoked == true)
      - 排除已废弃 (x_mitre_deprecated == true)
      - external_id 含 "." 的兜底排除（子技术的特征）
    """
    with attack_bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    techniques: List[TechniqueDoc] = []

    for obj in objects:
        # ── 类型过滤 ──
        if obj.get("type") != "attack-pattern":
            continue

        # ── 子技术过滤 ──
        if obj.get("x_mitre_is_subtechnique", False):
            continue

        # ── 状态过滤 ──
        if obj.get("revoked", False):
            continue
        if obj.get("x_mitre_deprecated", False):
            continue

        # ── 提取 technique ID ──
        tech_id = extract_tech_id(obj.get("external_references", []))
        if not tech_id or "." in tech_id:
            # "." 表示子技术 (如 T1190.001)，兜底排除
            continue

        # ── 提取名称和描述 ──
        name = normalize_text_for_doc(obj.get("name", ""))
        description = normalize_text_for_doc(obj.get("description", ""))
        stix_id = obj.get("id", "")

        # ── 提取所属战术阶段 ──
        tactics = [
            phase.get("phase_name")
            for phase in obj.get("kill_chain_phases", [])
            if isinstance(phase, dict) and isinstance(phase.get("phase_name"), str)
        ]

        # ── 构造嵌入文档 (v4: name + description only) ──
        doc = compose_technique_doc(name=name, description=description)
        if not doc:
            continue

        techniques.append(
            TechniqueDoc(
                tech_id=tech_id,
                name=name or tech_id,
                tactics=tactics,
                stix_id=stix_id,
                doc=doc.strip(),
            )
        )

    # 按 technique ID 排序，保证可复现性
    techniques.sort(key=lambda x: x.tech_id)
    return techniques


def write_technique_export(techniques: Sequence[TechniqueDoc], output_path: Path) -> None:
    """把实际用于嵌入的 technique 完整文本导出为 JSONL，方便后续人工核对。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    for technique in techniques:
        lines.append(
            json.dumps(
                {
                    "tech_id": technique.tech_id,
                    "name": technique.name,
                    "tactics": technique.tactics,
                    "stix_id": technique.stix_id,
                    "doc": technique.doc,
                },
                ensure_ascii=False,
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────
#  向量工具
# ─────────────────────────────────────────────────────────────────
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """按最后一维对向量做 L2 归一化。

    归一化后：||v|| = 1，点积等价于余弦相似度，省去除法运算。
    零向量特殊处理：赋范数为 1，避免除零错误。
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """使用 sentence-transformers 对批量的文本进行向量化。"""
    if not texts:
        return np.asarray([], dtype=np.float32)

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  缓存读写（避免重复调用模型编码 technique 向量）
# ─────────────────────────────────────────────────────────────────
def save_tech_cache(
    cache_path: Path,
    embeddings: np.ndarray,
    techniques: Sequence[TechniqueDoc],
) -> None:
    """将归一化后的 technique 向量及元数据保存为 .npz 压缩文件。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        tech_ids=np.asarray([t.tech_id for t in techniques], dtype=object),
        names=np.asarray([t.name for t in techniques], dtype=object),
        tactics=np.asarray(
            [json.dumps(t.tactics, ensure_ascii=False) for t in techniques],
            dtype=object,
        ),
        stix_ids=np.asarray([t.stix_id for t in techniques], dtype=object),
        docs=np.asarray([t.doc for t in techniques], dtype=object),
    )


def load_tech_cache(cache_path: Path) -> Tuple[np.ndarray, List[TechniqueDoc]]:
    """从 .npz 缓存加载 technique 向量及元数据。

    返回:
        (embeddings, techniques): 归一化后的向量矩阵和技术列表。
    """
    loaded = np.load(cache_path, allow_pickle=True)
    embeddings = loaded["embeddings"].astype(np.float32)
    tech_ids = loaded["tech_ids"].tolist()
    names = loaded["names"].tolist()
    tactics_json = loaded["tactics"].tolist()
    stix_ids = (
        loaded["stix_ids"].tolist()
        if "stix_ids" in loaded.files
        else ["" for _ in tech_ids]
    )
    docs = (
        loaded["docs"].tolist()
        if "docs" in loaded.files
        else ["" for _ in tech_ids]
    )

    techniques: List[TechniqueDoc] = []
    for tech_id, name, tactics_s, stix_id, doc in zip(
        tech_ids, names, tactics_json, stix_ids, docs
    ):
        techniques.append(
            TechniqueDoc(
                tech_id=str(tech_id),
                name=str(name),
                tactics=list(json.loads(str(tactics_s))),
                stix_id=str(stix_id),
                doc=str(doc),
            )
        )

    return embeddings, techniques


# ─────────────────────────────────────────────────────────────────
#  CVE 数据收集与检索
# ─────────────────────────────────────────────────────────────────
def collect_enterprise_cve_ids(domain_dir: Path) -> List[str]:
    """读取各年份 domain JSONL 文件，返回去重后的 Enterprise CVE ID 列表。

    只保留 domain == "Enterprise" 的 CVE，ICS/Mobile 域的不处理。
    """
    ids: List[str] = []
    seen = set()
    for domain_file in sorted(domain_dir.glob("CVE-*.jsonl")):
        with domain_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cve_id = item.get("cve_id")
                if not isinstance(cve_id, str) or item.get("domain") != "Enterprise":
                    continue
                if cve_id not in seen:
                    ids.append(cve_id)
                    seen.add(cve_id)
    return ids


def build_query_records(
    cve_ids: Sequence[str],
    cve_dir: Path,
) -> Tuple[List[dict], int, int]:
    """将 Enterprise CVE ID 列表与原始 CVE 描述文件关联，生成查询记录。

    返回:
        (query_records, missing_count, empty_description_count)
        - query_records: [{"cve_id", "domain", "query_text"}, ...]
        - missing_count: 原始 JSON 中找不到的 CVE 数量
        - empty_description_count: description 为空的 CVE 数量
    """
    ids_by_year: Dict[str, List[str]] = {}
    for cve_id in cve_ids:
        ids_by_year.setdefault(get_year_from_cve_id(cve_id), []).append(cve_id)

    query_records: List[dict] = []
    missing_count = 0
    empty_description_count = 0

    for year, year_ids in sorted(ids_by_year.items()):
        year_file = cve_dir / f"CVE-{year}.json"
        if not year_file.exists():
            missing_count += len(year_ids)
            continue

        with year_file.open("r", encoding="utf-8") as f:
            year_data = json.load(f)

        if not isinstance(year_data, dict):
            missing_count += len(year_ids)
            continue

        for cve_id in year_ids:
            record = year_data.get(cve_id)
            if not isinstance(record, dict):
                missing_count += 1
                continue

            description = (record.get("description") or "").strip()
            if not description:
                empty_description_count += 1
                continue

            query_records.append({
                "cve_id": cve_id,
                "domain": "Enterprise",
                "query_text": description,
            })

    query_records.sort(key=lambda x: x["cve_id"])
    return query_records, missing_count, empty_description_count


def top_k_candidates(
    query_embedding: np.ndarray,
    tech_embeddings: np.ndarray,
    techniques: Sequence[TechniqueDoc],
    top_k: int,
) -> List[str]:
    """计算查询向量与所有 technique 向量的余弦相似度，返回 top-k technique ID 列表。

    前提条件：所有向量均已 L2 归一化，此时点积等价于余弦相似度。
    """
    scores = tech_embeddings @ query_embedding
    k = min(top_k, len(scores))
    if k <= 0:
        return []

    # argpartition 只保证前 k 个最大值的索引，不保证顺序
    top_idx = np.argpartition(scores, -k)[-k:]
    # 排序使结果按分数降序
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [techniques[int(idx)].tech_id for idx in top_idx]


# ─────────────────────────────────────────────────────────────────
#  抽样 & 输出
# ─────────────────────────────────────────────────────────────────
def reservoir_sample_push(
    rng: random.Random,
    samples: List[dict],
    candidate: dict,
    seen_count: int,
    sample_size: int,
) -> None:
    """蓄水池抽样：以均等概率维护一个固定大小的随机样本。

    使用算法的 R 变体，每个元素有 sample_size / seen_count 的概率被选中。
    """
    if sample_size <= 0:
        return

    if len(samples) < sample_size:
        samples.append(candidate)
        return

    j = rng.randint(1, seen_count)
    if j <= sample_size:
        samples[j - 1] = candidate


def write_inspect_markdown(
    sample_records: Sequence[dict],
    output_path: Path,
) -> None:
    """生成可读的 markdown 抽样报告，供人工判断检索结果质量。"""
    lines: List[str] = [
        "# Embedding Retrieval Inspection Sample",
        "",
        f"Sample size: {len(sample_records)}",
        "",
    ]

    for i, item in enumerate(sample_records, start=1):
        lines.extend([
            f"## {i}. {item['cve_id']}",
            "",
            "**Description**",
            "",
            item["query_text"],
            "",
            "**Top-10 Techniques**",
            "",
        ])

        for rank, tech_id in enumerate(item["techniques"][:10], start=1):
            lines.append(f"{rank}. {tech_id}")

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_yearly_outputs(
    records: Sequence[dict],
    output_dir: Path,
) -> List[Path]:
    """按年份分组写入结果 JSONL 文件，每条记录包含 cve_id 和 techniques 列表。"""
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        grouped.setdefault(
            get_year_from_cve_id(record["cve_id"]), []
        ).append(record)

    written_paths: List[Path] = []
    for year in sorted(grouped):
        year_path = output_dir / f"CVE-{year}.jsonl"
        with year_path.open("w", encoding="utf-8") as f:
            for record in sorted(
                grouped[year], key=lambda item: item["cve_id"]
            ):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_paths.append(year_path)

    return written_paths


def ensure_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    """根据参数计算所有输出文件路径，创建必要目录。"""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 缓存文件名包含版本号和模型名，确保不同配置的缓存不会互相覆盖
    cache_name = (
        f"tech_embeddings_cache_{CACHE_VERSION}_"
        f"{EMBED_MODEL.replace('/', '_')}.npz"
    )
    technique_export_path = output_dir / "techniques_for_embedding.jsonl"
    return (
        output_dir / cache_name,
        output_dir / "candidates.jsonl",
        output_dir / "inspect_sample.md",
        technique_export_path,
    )


def render_progress(current: int, total: int, prefix: str, bar_len: int = 30) -> str:
    """生成简洁的文本进度条字符串，用于终端输出。"""
    if total <= 0:
        return f"{prefix} [no items]"
    pct = current / total * 100
    filled = int(bar_len * current / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    return f"{prefix} [{bar}] {current}/{total} ({pct:.1f}%)"


# ─────────────────────────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    """嵌入检索主入口：加载/构建技术知识库 → 逐 CVE 检索 → 输出结果。"""
    args = parse_args()
    cache_path, candidates_path, inspect_path, technique_export_path = ensure_paths(
        args
    )

    # ── 加载嵌入模型 ──
    model = SentenceTransformer(EMBED_MODEL)

    # ── 校验输入文件存在 ──
    if not args.attack_bundle.exists():
        raise SystemExit(f"Missing ATT&CK bundle: {args.attack_bundle}")
    if not args.domain_dir.exists():
        raise SystemExit(f"Missing domain directory: {args.domain_dir}")
    if not args.cve_dir.exists():
        raise SystemExit(f"Missing CVE directory: {args.cve_dir}")

    print(f"[INFO] Using embed model: {EMBED_MODEL}")
    print(f"[INFO] Output directory: {args.output_dir}")

    # ── 步骤 A: 获取 technique 向量 ──
    # 优先加载缓存；缓存不存在或版本不匹配则重新提取 technique 文本并编码
    if cache_path.exists():
        tech_embeddings, techniques = load_tech_cache(cache_path)
        print(f"[INFO] Loaded technique cache: {cache_path}")
        print(f"[INFO] Technique vectors: {tech_embeddings.shape[0]}")
        # [注释] 不再导出 techniques_for_embedding.jsonl
        # if techniques and techniques[0].doc:
        #     write_technique_export(techniques, technique_export_path)
    else:
        # ── 提取 and ATT&CK technique 知识库 ──
        # v4: 只用 name + description，不再收集 Procedure Examples
        techniques = extract_technique_kb(args.attack_bundle)
        if not techniques:
            raise SystemExit(
                "No technique docs extracted from enterprise-attack.json"
            )

        # [注释] 不再导出 techniques_for_embedding.jsonl
        # write_technique_export(techniques, technique_export_path)

        # 批量编码 technique 文档
        tech_docs = [t.doc for t in techniques]
        raw_vectors = embed_texts(
            model=model,
            texts=tech_docs,
            batch_size=args.batch_size,
        )
        tech_embeddings = l2_normalize(raw_vectors).astype(np.float32)

        # 保存缓存，下次运行时无需重复编码
        save_tech_cache(cache_path, tech_embeddings, techniques)

        print(f"[INFO] Extracted techniques: {len(techniques)}")
        # [注释] 不再打印 technique export 路径
        print(f"[INFO] Saved technique cache: {cache_path}")

    # ── 步骤 B: 收集 Enterprise CVE ──
    enterprise_ids = collect_enterprise_cve_ids(args.domain_dir)
    if args.max_cves is not None:
        enterprise_ids = enterprise_ids[: max(0, args.max_cves)]

    query_records, missing_count, empty_desc_count = build_query_records(
        enterprise_ids, args.cve_dir
    )

    print(f"[INFO] Enterprise CVE IDs from mapping: {len(enterprise_ids)}")
    print(f"[INFO] Query-ready CVEs: {len(query_records)}")
    print(f"[INFO] Missing raw CVE records: {missing_count}")
    print(f"[INFO] Empty descriptions skipped: {empty_desc_count}")

    # ── 步骤 C: 逐 CVE 检索 ──
    total_queries = len(query_records)
    print(f"[INFO] Starting retrieval for {total_queries} CVEs")

    records_by_year: Dict[str, List[dict]] = {}
    for record in query_records:
        year = get_year_from_cve_id(record["cve_id"])
        records_by_year.setdefault(year, []).append(record)

    rng = random.Random(args.seed)
    inspect_samples: List[dict] = []
    written_paths: List[Path] = []
    seen_count = 0

    for year in sorted(records_by_year):
        year_records = records_by_year[year]
        total_year = len(year_records)
        print(f"[INFO] Processing year {year} with {total_year} CVEs")
        year_output_records: List[dict] = []

        for idx, query in enumerate(year_records, start=1):
            # 编码 CVE description → 归一化向量
            q_vec_raw = embed_texts(
                model=model,
                texts=[query["query_text"]],
                batch_size=1,
            )
            q_vec = l2_normalize(q_vec_raw)[0]

            # 与 technique 向量矩阵做点积得到 top-k 候选
            techniques_only = top_k_candidates(
                q_vec, tech_embeddings, techniques, args.top_k
            )
            year_output_records.append({
                "cve_id": query["cve_id"],
                "techniques": techniques_only,
            })

            # 蓄水池抽样（用于生成的人工检查报告）
            seen_count += 1
            reservoir_sample_push(
                rng=rng,
                samples=inspect_samples,
                candidate={
                    "cve_id": query["cve_id"],
                    "query_text": query["query_text"],
                    "techniques": techniques_only,
                },
                seen_count=seen_count,
                sample_size=args.sample_size,
            )

            # 限速：每 N 条请求后等待一小段时间，避免触发 API 速率限制
            if args.query_sleep_every > 0 and seen_count % args.query_sleep_every == 0:
                time.sleep(args.query_sleep_seconds)

            # 进度条（每 20 条或达到总量时刷新）
            if idx % 20 == 0 or idx == total_year:
                progress = render_progress(
                    idx, total_year, prefix=f"[{year}]"
                )
                print(f"\r{progress}", end="", flush=True)

        if total_year:
            print()

        if year_output_records:
            written_paths.extend(
                write_yearly_outputs(year_output_records, candidates_path.parent)
            )

    # ── 步骤 D: 写入输出 ──
    # [注释] 不再生成 inspect_sample.md
    # write_inspect_markdown(inspect_samples, inspect_path)

    print(
        f"[INFO] Wrote yearly outputs: {len(written_paths)} files under "
        f"{candidates_path.parent}"
    )
    # [注释] 不再打印 inspect sample 路径


if __name__ == "__main__":
    main()
