import json
import math
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.embeddings import cluster_embedding_jsonl, cosine_similarity, greedy_cosine_clusters


class EmbeddingsTest(unittest.TestCase):
    def test_cosine_similarity_and_greedy_clusters(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

        clusters = greedy_cosine_clusters(
            [
                {"index": 0, "source_item_id": "a", "embedding": [1.0, 0.0]},
                {"index": 1, "source_item_id": "b", "embedding": [0.99, 0.01]},
                {"index": 2, "source_item_id": "c", "embedding": [0.0, 1.0]},
            ],
            threshold=0.95,
        )

        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]["members"]), 2)
        self.assertTrue(math.isclose(clusters[0]["members"][1]["similarity_to_centroid"], 0.999949, rel_tol=1e-5))

    def test_cluster_embedding_jsonl_writes_output(self):
        temp_dir = Path(".test_tmp") / f"embeddings_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        embeddings = temp_dir / "embeddings.jsonl"
        output = temp_dir / "clusters.json"
        rows = [
            {"index": 0, "source_item_id": "a", "embedding": [1.0, 0.0]},
            {"index": 1, "source_item_id": "b", "embedding": [0.0, 1.0]},
        ]
        embeddings.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        result = cluster_embedding_jsonl(embeddings, output, threshold=0.95)
        written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["rows"], 2)
        self.assertEqual(len(written["clusters"]), 2)


if __name__ == "__main__":
    unittest.main()
