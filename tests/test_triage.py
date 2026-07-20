import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cve2attack.data.loaders import benchmark_truth, iter_jsonl
from cve2attack.data.triage import TRIAGE_VIEW_NAMES, import_triage_benchmarks
from cve2attack.evaluation.triage import load_reference_history


class TRIAGEImportTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        source = root / "triage"
        source.mkdir()
        with (source / "cves_train.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["CVE ID"])
            writer.writeheader()
            writer.writerow({"CVE ID": "CVE-2024-0001"})
        with (source / "cves_test.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["CVE ID"])
            writer.writeheader()
            writer.writerows(
                [{"CVE ID": "CVE-2024-0002"}, {"CVE ID": "CVE-2024-0003"}]
            )

        columns = ["CVE ID", "mapping_type", "attack_id", "attack_name"]
        rows = [
            ["CVE-2024-0001", "primary_impact", "T1001", "Train label"],
            ["CVE-2024-0002", "exploitation_technique", "T1059.004", "Unix Shell"],
            ["CVE-2024-0002", "secondary_impact", "T1202", "Secondary"],
            ["CVE-2024-0003", "primary_impact", "T1190", "Primary"],
        ]
        with (source / "labeled_cve_to_attack.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        (source / "source.yaml").write_text(
            yaml.safe_dump(
                {
                    "expected_counts": {
                        "train_cves": 1,
                        "test_cves": 2,
                        "labeled_cves": 3,
                        "label_rows": 4,
                    }
                }
            ),
            encoding="utf-8",
        )
        return source

    @staticmethod
    def _records(directory: Path) -> dict[str, dict]:
        return {
            record["cve_id"]: record
            for path in directory.glob("CVE-*.jsonl")
            for record in iter_jsonl(path)
        }

    def test_import_preserves_split_mapping_types_and_raw_subtechniques(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_fixture(root)
            benchmark_root = root / "benchmarks"
            stats = import_triage_benchmarks(
                source_dir=source, benchmark_root=benchmark_root
            )

            self.assertEqual(stats["source"]["test_cves"], 2)
            all_records = self._records(benchmark_root / TRIAGE_VIEW_NAMES[0])
            no_secondary = self._records(benchmark_root / TRIAGE_VIEW_NAMES[1])
            self.assertEqual(all_records["CVE-2024-0002"]["techniques"], ["T1059", "T1202"])
            self.assertEqual(
                all_records["CVE-2024-0002"]["techniques_raw"],
                ["T1059.004", "T1202"],
            )
            self.assertEqual(no_secondary["CVE-2024-0002"]["techniques"], ["T1059"])

    def test_import_refuses_to_overwrite_frozen_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_fixture(root)
            benchmark_root = root / "benchmarks"
            import_triage_benchmarks(source_dir=source, benchmark_root=benchmark_root)
            with self.assertRaises(FileExistsError):
                import_triage_benchmarks(source_dir=source, benchmark_root=benchmark_root)

    def test_reference_history_must_match_benchmark_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_fixture(root)
            benchmark_root = root / "benchmarks"
            import_triage_benchmarks(source_dir=source, benchmark_root=benchmark_root)
            truth = benchmark_truth(benchmark_root / TRIAGE_VIEW_NAMES[0])
            history = [
                {
                    "target_cve": cve_id,
                    "predictions": sorted(labels),
                    "true labels": sorted(labels),
                }
                for cve_id, labels in truth.items()
            ]
            history_path = root / "history.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")
            loaded = load_reference_history(history_path, expected_truth=truth)
            self.assertEqual(set(loaded), set(truth))


if __name__ == "__main__":
    unittest.main()
