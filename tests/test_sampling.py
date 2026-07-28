"""Tests for deterministic label-independent benchmark sampling."""

import json
import tempfile
import unittest
from pathlib import Path

from cve2attack.data.loaders import benchmark_truth
from cve2attack.data.sampling import sample_benchmark


class BenchmarkSamplingTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "CVE-2024.jsonl").write_text(
            "\n".join(
                json.dumps({"cve_id": f"CVE-2024-{index}", "techniques": [f"T{index:04d}"]})
                for index in range(1, 7)
            )
            + "\n",
            encoding="utf-8",
        )
        (source / "dataset.yaml").write_text(
            "name: source\nannotated_cves: 6\n",
            encoding="utf-8",
        )
        return source

    def test_hash_sample_is_reproducible_and_preserves_selected_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            first = root / "first"
            second = root / "second"
            sample_benchmark(source_dir=source, output_dir=first, sample_size=3, seed="v1")
            sample_benchmark(source_dir=source, output_dir=second, sample_size=3, seed="v1")

            first_truth = benchmark_truth(first)
            second_truth = benchmark_truth(second)

        self.assertEqual(first_truth, second_truth)
        self.assertEqual(len(first_truth), 3)
        self.assertTrue(all(first_truth[cve_id] for cve_id in first_truth))

    def test_refuses_to_overwrite_frozen_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            output = root / "sample"
            sample_benchmark(source_dir=source, output_dir=output, sample_size=2, seed="v1")
            with self.assertRaises(FileExistsError):
                sample_benchmark(source_dir=source, output_dir=output, sample_size=2, seed="v1")


if __name__ == "__main__":
    unittest.main()
