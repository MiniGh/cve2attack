import unittest

import numpy as np

from cve2attack.retrieval.generator import retrieve_candidates
from cve2attack.retrieval.technique_kb import TechniqueDocument


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts, batch_size):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "web" in text else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def test_returns_ranked_canonical_candidates(self):
        techniques = [
            TechniqueDocument("T1190", "Exploit Web", (), "a", "web"),
            TechniqueDocument("T1059", "Command", (), "b", "command"),
        ]
        records = retrieve_candidates(
            queries={"CVE-2024-1": "web vulnerability"},
            techniques=techniques,
            technique_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            embedder=FakeEmbedder(),
            top_k=2,
            batch_size=2,
        )
        self.assertEqual(records[0].candidates[0].technique_id, "T1190")
        self.assertEqual(records[0].to_dict()["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
