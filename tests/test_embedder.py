import os
import unittest
from unittest.mock import patch

from cve2attack.retrieval.embedder import SentenceTransformerEmbedder


class SentenceTransformerEmbedderTests(unittest.TestCase):
    @patch("sentence_transformers.SentenceTransformer")
    def test_passes_local_files_only_to_sentence_transformers(self, factory):
        with patch.dict(os.environ, {}, clear=True):
            SentenceTransformerEmbedder("cached-model", local_files_only=True)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        factory.assert_called_once_with(
            "cached-model",
            local_files_only=True,
            model_kwargs={"use_safetensors": False},
        )

    @patch("sentence_transformers.SentenceTransformer", side_effect=OSError("not cached"))
    def test_missing_offline_model_has_actionable_error(self, _factory):
        with self.assertRaisesRegex(RuntimeError, "local cache"):
            SentenceTransformerEmbedder("missing-model", local_files_only=True)


if __name__ == "__main__":
    unittest.main()
