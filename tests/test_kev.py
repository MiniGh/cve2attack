import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cve2attack.data.kev import KEV_VIEW_NAMES, import_kev_benchmarks
from cve2attack.data.loaders import iter_jsonl


class KEVImportTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "kev.csv"
        columns = [
            "mapping_framework",
            "mapping_framework_version",
            "capability_id",
            "mapping_type",
            "attack_object_id",
            "attack_object_name",
            "attack_version",
            "technology_domain",
            "references",
            "comments",
        ]
        rows = [
            [
                "kev",
                "02/13/2025",
                "CVE-2024-0001",
                "exploitation_technique",
                "T1059.004",
                "Unix Shell",
                "15.1",
                "enterprise",
                "https://example.test/one",
                "Execution through a shell.",
            ],
            [
                "kev",
                "02/13/2025",
                "CVE-2024-0001",
                "secondary_impact",
                "T1202",
                "Indirect Command Execution",
                "15.1",
                "enterprise",
                "https://example.test/one",
                "Execution through a shell.",
            ],
            [
                "kev",
                "02/13/2025",
                "CVE-2024-0002",
                "primary_impact",
                "T1190",
                "Exploit Public-Facing Application",
                "15.1",
                "enterprise",
                "https://example.test/two",
                "Remote exploitation.",
            ],
            [
                "kev",
                "02/13/2025",
                "CVE-2024-0003",
                "exploitation_technique",
                "T1505.003",
                "Web Shell",
                "15.1",
                "enterprise",
                "https://example.test/three",
                "Web shell deployment.",
            ],
        ]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

        cve2attack = root / "cve2attack_result"
        cve2attack.mkdir()
        (cve2attack / "CVE-2024.jsonl").write_text(
            json.dumps({"cve_id": "CVE-2024-0001", "techniques": ["T1059"]}) + "\n",
            encoding="utf-8",
        )
        return source, cve2attack

    @staticmethod
    def _records(directory: Path) -> dict[str, dict]:
        return {
            record["cve_id"]: record
            for path in directory.glob("CVE-*.jsonl")
            for record in iter_jsonl(path)
        }

    def test_import_creates_semantic_views_and_preserves_raw_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cve2attack = self._write_fixture(root)
            benchmark_root = root / "benchmarks"

            stats = import_kev_benchmarks(
                source=source,
                benchmark_root=benchmark_root,
                cve2attack_benchmark=cve2attack,
            )

            self.assertEqual(stats["source_stats"]["mapping_rows"], 4)
            self.assertEqual(stats["cve2attack_overlap"], 1)
            self.assertEqual(
                stats["views"]["ctid_kev_2025_02_13_all"]["annotated_cves"], 3
            )
            self.assertEqual(
                stats["views"]["ctid_kev_2025_02_13_exploitation"]["annotated_cves"], 2
            )
            self.assertEqual(
                stats["views"]["ctid_kev_2025_02_13_nonoverlap"]["annotated_cves"], 2
            )

            all_records = self._records(benchmark_root / KEV_VIEW_NAMES[0])
            first = all_records["CVE-2024-0001"]
            self.assertEqual(first["techniques"], ["T1059", "T1202"])
            self.assertEqual(first["techniques_raw"], ["T1059.004", "T1202"])
            self.assertEqual(
                first["labels_by_mapping_type"]["exploitation_technique"]["techniques_raw"],
                ["T1059.004"],
            )

            exploitation_records = self._records(benchmark_root / KEV_VIEW_NAMES[1])
            self.assertEqual(set(exploitation_records), {"CVE-2024-0001", "CVE-2024-0003"})
            nonoverlap_records = self._records(benchmark_root / KEV_VIEW_NAMES[2])
            self.assertEqual(set(nonoverlap_records), {"CVE-2024-0002", "CVE-2024-0003"})

            metadata = yaml.safe_load(
                (benchmark_root / KEV_VIEW_NAMES[0] / "dataset.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["attack_version"], "15.1")
            self.assertEqual(metadata["label_policy"]["raw_label_field"], "techniques_raw")
            self.assertEqual(metadata["technique_corpus"]["version"], "15.1")

    def test_import_refuses_to_overwrite_an_existing_frozen_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cve2attack = self._write_fixture(root)
            benchmark_root = root / "benchmarks"
            import_kev_benchmarks(
                source=source,
                benchmark_root=benchmark_root,
                cve2attack_benchmark=cve2attack,
            )
            with self.assertRaises(FileExistsError):
                import_kev_benchmarks(
                    source=source,
                    benchmark_root=benchmark_root,
                    cve2attack_benchmark=cve2attack,
                )


if __name__ == "__main__":
    unittest.main()
