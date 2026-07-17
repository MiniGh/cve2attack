import tempfile
import unittest
from pathlib import Path

import yaml

from cve2attack.pipeline import resolve_attack_bundle


class TechniqueCorpusResolutionTests(unittest.TestCase):
    def test_benchmark_metadata_selects_its_frozen_technique_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "data" / "knowledge" / "enterprise-attack-15.1.json"
            corpus.parent.mkdir(parents=True)
            corpus.write_text("{}", encoding="utf-8")
            benchmark = root / "data" / "benchmarks" / "ctid_kev"
            benchmark.mkdir(parents=True)
            (benchmark / "dataset.yaml").write_text(
                yaml.safe_dump(
                    {
                        "technique_corpus": {
                            "path": "data/knowledge/enterprise-attack-15.1.json",
                            "version": "15.1",
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "input": {"mode": "benchmark", "benchmark": "ctid_kev"},
                "technique_document": {},
            }

            self.assertEqual(resolve_attack_bundle(config, root), corpus)

    def test_explicit_experiment_corpus_overrides_benchmark_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            explicit = root / "data" / "knowledge" / "migration.json"
            explicit.parent.mkdir(parents=True)
            explicit.write_text("{}", encoding="utf-8")
            config = {
                "input": {"mode": "benchmark", "benchmark": "missing_metadata"},
                "technique_document": {"attack_bundle": "data/knowledge/migration.json"},
            }

            self.assertEqual(resolve_attack_bundle(config, root), explicit)


if __name__ == "__main__":
    unittest.main()
