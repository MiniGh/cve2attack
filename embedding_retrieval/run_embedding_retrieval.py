"""Embedding-only CVE to ATT&CK retrieval pipeline.

This script implements phase-1 retrieval with these constraints:
- Enterprise domain only.
- Parent techniques only (no sub-techniques).
- No structured chain, no LLM reranking.
- Output top-k technique IDs per CVE.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: sentence-transformers/torch. Install with: pip install sentence-transformers torch numpy"
    ) from exc



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN_DIR = PROJECT_ROOT / "cve_to_attack_domain" / "result"
DEFAULT_CVE_DIR = PROJECT_ROOT / "og_data" / "cve"
DEFAULT_ATTACK_BUNDLE = PROJECT_ROOT / "og_data" / "enterprise-attack.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "retrieval"

EMBED_MODEL = "basel/ATTACK-BERT"


@dataclass
class TechniqueDoc:
    """Knowledge base record for one ATT&CK technique."""

    tech_id: str
    name: str
    tactics: List[str]
    stix_id: str
    doc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding retrieval for Enterprise CVEs.")
    parser.add_argument("--domain-dir", type=Path, default=DEFAULT_DOMAIN_DIR, help="Directory containing CVE-{year}.jsonl domain files.")
    parser.add_argument("--cve-dir", type=Path, default=DEFAULT_CVE_DIR, help="Directory containing yearly raw CVE JSON dict files.")
    parser.add_argument("--attack-bundle", type=Path, default=DEFAULT_ATTACK_BUNDLE, help="Path to enterprise-attack.json STIX bundle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for yearly results and inspect file.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of technique candidates per CVE.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of CVEs for inspect_sample markdown.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding API batch size for technique docs.")
    parser.add_argument("--procedure-char-limit", type=int, default=1500, help="Max procedure text length per technique.")
    parser.add_argument("--query-sleep-every", type=int, default=10, help="Sleep interval in number of CVE queries.")
    parser.add_argument("--query-sleep-seconds", type=float, default=0.5, help="Sleep duration in seconds for rate limiting.")
    parser.add_argument("--max-cves", type=int, default=None, help="Optional cap for number of Enterprise CVEs to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for inspect sample reservoir sampling.")
    return parser.parse_args()


def clean_markdown_links(text: str) -> str:
    """Convert markdown links like [x](y) into x."""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text or "")


def normalize_text_for_doc(text: str) -> str:
    """Simple whitespace cleanup for document composition."""
    cleaned = clean_markdown_links(text)
    return re.sub(r"\s+", " ", cleaned).strip()


def get_year_from_cve_id(cve_id: str) -> str:
    """Extract year from CVE ID like CVE-2024-1234."""
    parts = cve_id.split("-")
    if len(parts) < 3 or not parts[1].isdigit():
        raise ValueError(f"Unexpected CVE format: {cve_id}")
    return parts[1]


def extract_tech_id(external_references: Sequence[dict]) -> str | None:
    """Read ATT&CK external ID from external_references."""
    for ref in external_references or []:
        if ref.get("source_name") == "mitre-attack":
            ext_id = ref.get("external_id")
            if isinstance(ext_id, str) and ext_id.strip():
                return ext_id.strip()
    return None


def build_procedure_map(objects: Sequence[dict]) -> Dict[str, List[str]]:
    """Collect relationship descriptions for uses->technique references."""
    mapping: Dict[str, List[str]] = {}
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "uses":
            continue

        target_ref = obj.get("target_ref")
        if not isinstance(target_ref, str):
            continue

        desc = normalize_text_for_doc(obj.get("description", ""))
        if not desc:
            continue

        mapping.setdefault(target_ref, []).append(desc)

    return mapping


def extract_technique_kb(attack_bundle_path: Path, procedure_char_limit: int) -> List[TechniqueDoc]:
    """Extract top-level ATT&CK techniques and compose retrieval documents."""
    with attack_bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    procedure_map = build_procedure_map(objects)

    techniques: List[TechniqueDoc] = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("x_mitre_is_subtechnique", False):
            continue
        if obj.get("revoked", False):
            continue
        if obj.get("x_mitre_deprecated", False):
            continue

        tech_id = extract_tech_id(obj.get("external_references", []))
        if not tech_id or "." in tech_id:
            continue

        name = normalize_text_for_doc(obj.get("name", ""))
        description = normalize_text_for_doc(obj.get("description", ""))
        stix_id = obj.get("id", "")
        tactics = [
            phase.get("phase_name")
            for phase in obj.get("kill_chain_phases", [])
            if isinstance(phase, dict) and isinstance(phase.get("phase_name"), str)
        ]

        procedure_chunks = procedure_map.get(stix_id, [])
        procedure_text = " ".join(chunk for chunk in procedure_chunks if chunk).strip()
        if len(procedure_text) > procedure_char_limit:
            procedure_text = procedure_text[:procedure_char_limit].rstrip()

        doc_parts = [part for part in (name, description, procedure_text) if part]
        if not doc_parts:
            continue

        doc = "。".join(doc_parts[:1])
        if len(doc_parts) > 1:
            doc = doc + " " + " ".join(doc_parts[1:])

        techniques.append(
            TechniqueDoc(
                tech_id=tech_id,
                name=name or tech_id,
                tactics=tactics,
                stix_id=stix_id,
                doc=doc.strip(),
            )
        )

    techniques.sort(key=lambda x: x.tech_id)
    return techniques


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors along the last dimension."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Embed texts with sentence-transformers."""
    if not texts:
        return np.asarray([], dtype=np.float32)

    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_tech_cache(cache_path: Path, embeddings: np.ndarray, techniques: Sequence[TechniqueDoc]) -> None:
    """Persist normalized technique embeddings and metadata."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        tech_ids=np.asarray([t.tech_id for t in techniques], dtype=object),
        names=np.asarray([t.name for t in techniques], dtype=object),
        tactics=np.asarray([json.dumps(t.tactics, ensure_ascii=False) for t in techniques], dtype=object),
    )


def load_tech_cache(cache_path: Path) -> Tuple[np.ndarray, List[TechniqueDoc]]:
    """Load technique embeddings and metadata from cache."""
    loaded = np.load(cache_path, allow_pickle=True)
    embeddings = loaded["embeddings"].astype(np.float32)
    tech_ids = loaded["tech_ids"].tolist()
    names = loaded["names"].tolist()
    tactics_json = loaded["tactics"].tolist()

    techniques: List[TechniqueDoc] = []
    for tech_id, name, tactics_s in zip(tech_ids, names, tactics_json):
        techniques.append(
            TechniqueDoc(
                tech_id=str(tech_id),
                name=str(name),
                tactics=list(json.loads(str(tactics_s))),
                stix_id="",
                doc="",
            )
        )

    return embeddings, techniques


def collect_enterprise_cve_ids(domain_dir: Path) -> List[str]:
    """Read all yearly mapping files and return unique Enterprise CVE IDs."""
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


def build_query_records(cve_ids: Sequence[str], cve_dir: Path) -> Tuple[List[dict], int, int]:
    """Join Enterprise CVE IDs with raw descriptions from yearly CVE dict files."""
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

            query_records.append({"cve_id": cve_id, "domain": "Enterprise", "query_text": description})

    query_records.sort(key=lambda x: x["cve_id"])
    return query_records, missing_count, empty_description_count


def top_k_candidates(query_embedding: np.ndarray, tech_embeddings: np.ndarray, techniques: Sequence[TechniqueDoc], top_k: int) -> List[str]:
    """Compute top-k cosine candidates using normalized dot product."""
    scores = tech_embeddings @ query_embedding
    k = min(top_k, len(scores))
    if k <= 0:
        return []

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [techniques[int(idx)].tech_id for idx in top_idx]


def reservoir_sample_push(rng: random.Random, samples: List[dict], candidate: dict, seen_count: int, sample_size: int) -> None:
    """Maintain a fixed-size random sample via reservoir sampling."""
    if sample_size <= 0:
        return

    if len(samples) < sample_size:
        samples.append(candidate)
        return

    j = rng.randint(1, seen_count)
    if j <= sample_size:
        samples[j - 1] = candidate


def write_inspect_markdown(sample_records: Sequence[dict], output_path: Path) -> None:
    """Write a human-readable markdown report for random CVE samples."""
    lines: List[str] = ["# Embedding Retrieval Inspection Sample", "", f"Sample size: {len(sample_records)}", ""]

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


def write_yearly_outputs(records: Sequence[dict], output_dir: Path) -> List[Path]:
    """Write one JSONL file per CVE year."""
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        grouped.setdefault(get_year_from_cve_id(record["cve_id"]), []).append(record)

    written_paths: List[Path] = []
    for year in sorted(grouped):
        year_path = output_dir / f"CVE-{year}.jsonl"
        with year_path.open("w", encoding="utf-8") as f:
            for record in sorted(grouped[year], key=lambda item: item["cve_id"]):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_paths.append(year_path)

    return written_paths

def ensure_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Resolve output paths from args."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"tech_embeddings_cache_{EMBED_MODEL.replace('/', '_')}.npz"
    return output_dir / cache_name, output_dir / "candidates.jsonl", output_dir / "inspect_sample.md"

def render_progress(current: int, total: int, prefix: str, bar_len: int = 30) -> str:
    """Render a compact progress bar string."""
    if total <= 0:
        return f"{prefix} [no items]"
    pct = current / total * 100
    filled = int(bar_len * current / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    return f"{prefix} [{bar}] {current}/{total} ({pct:.1f}%)"

def main() -> None:
    args = parse_args()
    cache_path, candidates_path, inspect_path = ensure_paths(args)

    model = SentenceTransformer(EMBED_MODEL)

    if not args.attack_bundle.exists():
        raise SystemExit(f"Missing ATT&CK bundle: {args.attack_bundle}")
    if not args.domain_dir.exists():
        raise SystemExit(f"Missing domain directory: {args.domain_dir}")
    if not args.cve_dir.exists():
        raise SystemExit(f"Missing CVE directory: {args.cve_dir}")

    print(f"[INFO] Using embed model: {EMBED_MODEL}")
    print(f"[INFO] Output directory: {args.output_dir}")

    if cache_path.exists():
        tech_embeddings, techniques = load_tech_cache(cache_path)
        print(f"[INFO] Loaded technique cache: {cache_path}")
        print(f"[INFO] Technique vectors: {tech_embeddings.shape[0]}")
    else:
        techniques = extract_technique_kb(args.attack_bundle, args.procedure_char_limit)
        if not techniques:
            raise SystemExit("No technique docs extracted from enterprise-attack.json")

        tech_docs = [t.doc for t in techniques]
        raw_vectors = embed_texts(
            model=model,
            texts=tech_docs,
            batch_size=args.batch_size,
        )
        tech_embeddings = l2_normalize(raw_vectors).astype(np.float32)
        save_tech_cache(cache_path, tech_embeddings, techniques)

        print(f"[INFO] Extracted techniques: {len(techniques)}")
        print(f"[INFO] Saved technique cache: {cache_path}")

    enterprise_ids = collect_enterprise_cve_ids(args.domain_dir)
    if args.max_cves is not None:
        enterprise_ids = enterprise_ids[: max(0, args.max_cves)]

    query_records, missing_count, empty_desc_count = build_query_records(enterprise_ids, args.cve_dir)

    print(f"[INFO] Enterprise CVE IDs from mapping: {len(enterprise_ids)}")
    print(f"[INFO] Query-ready CVEs: {len(query_records)}")
    print(f"[INFO] Missing raw CVE records: {missing_count}")
    print(f"[INFO] Empty descriptions skipped: {empty_desc_count}")

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
            q_vec_raw = embed_texts(
                model=model,
                texts=[query["query_text"]],
                batch_size=1,
            )
            q_vec = l2_normalize(q_vec_raw)[0]

            techniques_only = top_k_candidates(q_vec, tech_embeddings, techniques, args.top_k)
            year_output_records.append({"cve_id": query["cve_id"], "techniques": techniques_only})

            seen_count += 1
            reservoir_sample_push(
                rng=rng,
                samples=inspect_samples,
                candidate={"cve_id": query["cve_id"], "query_text": query["query_text"], "techniques": techniques_only},
                seen_count=seen_count,
                sample_size=args.sample_size,
            )

            if args.query_sleep_every > 0 and seen_count % args.query_sleep_every == 0:
                time.sleep(args.query_sleep_seconds)

            if idx % 20 == 0 or idx == total_year:
                progress = render_progress(idx, total_year, prefix=f"[{year}]")
                print(f"\r{progress}", end="", flush=True)

        if total_year:
            print()

        if year_output_records:
            written_paths.extend(write_yearly_outputs(year_output_records, candidates_path.parent))

    write_inspect_markdown(inspect_samples, inspect_path)

    print(f"[INFO] Wrote yearly outputs: {len(written_paths)} files under {candidates_path.parent}")
    print(f"[INFO] Wrote inspect sample: {inspect_path}")


if __name__ == "__main__":
    main()
