import json
import tempfile
import unittest
from pathlib import Path

from cve2attack.retrieval.technique_kb import load_technique_documents


class TechniqueKnowledgeBaseTests(unittest.TestCase):
    def test_filters_subtechniques_and_adds_procedures_only_when_enabled(self):
        bundle = {
            "objects": [
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--parent",
                    "name": "Parent",
                    "description": "Description",
                    "external_references": [{"source_name": "mitre-attack", "external_id": "T1000"}],
                },
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--child",
                    "name": "Child",
                    "x_mitre_is_subtechnique": True,
                    "external_references": [{"source_name": "mitre-attack", "external_id": "T1000.001"}],
                },
                {
                    "type": "relationship",
                    "relationship_type": "uses",
                    "target_ref": "attack-pattern--parent",
                    "description": "[Actor](https://example.invalid) used it.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            plain = load_technique_documents(path, include_procedures=False)
            enhanced = load_technique_documents(path, include_procedures=True)
        self.assertEqual(len(plain), 1)
        self.assertNotIn("Procedure Examples", plain[0].text)
        self.assertIn("Actor used it", enhanced[0].text)


if __name__ == "__main__":
    unittest.main()
