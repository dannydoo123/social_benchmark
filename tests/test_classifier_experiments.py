import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.classifier_experiments import EmbeddingVectorCache, _encode_with_cache, grouped_holdout_indexes


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class ClassifierExperimentTest(unittest.TestCase):
    def test_grouped_holdout_keeps_source_items_on_one_side(self):
        examples = [
            {"source_item_id": "a", "text": "one"},
            {"source_item_id": "a", "text": "two"},
            {"source_item_id": "b", "text": "three"},
            {"source_item_id": "c", "text": "four"},
        ]

        for train, test in grouped_holdout_indexes(examples, runs=3, test_fraction=0.34):
            self.assertFalse(set(train) & set(test))
            train_groups = {examples[index]["source_item_id"] for index in train}
            test_groups = {examples[index]["source_item_id"] for index in test}
            self.assertFalse(train_groups & test_groups)

    def test_grouped_holdout_can_group_by_thread(self):
        examples = [
            {"thread_id": "t1", "source_item_id": "a"},
            {"thread_id": "t1", "source_item_id": "b"},
            {"thread_id": "t2", "source_item_id": "c"},
        ]

        train, test = grouped_holdout_indexes(examples, runs=1, test_fraction=0.5, group_field="thread_id")[0]

        train_threads = {examples[index]["thread_id"] for index in train}
        test_threads = {examples[index]["thread_id"] for index in test}
        self.assertFalse(train_threads & test_threads)

    def test_embedding_cache_deduplicates_texts_before_encoding(self):
        embedder = FakeEmbedder()

        vectors = _encode_with_cache(
            ["same", "same", "different"],
            embedding_model="fake-model",
            embedder=embedder,
        )

        self.assertEqual(embedder.calls, [["same", "different"]])
        self.assertEqual(vectors[0], vectors[1])

    def test_embedding_cache_reuses_vectors_from_disk(self):
        temp_dir = Path(".test_tmp") / f"embedding_cache_{uuid.uuid4().hex}"
        embedder = FakeEmbedder()
        cache = EmbeddingVectorCache(temp_dir)

        first = _encode_with_cache(
            ["cached", "new"],
            embedding_model="fake/model",
            embedder=embedder,
            cache=cache,
        )
        second_embedder = FakeEmbedder()
        second = _encode_with_cache(
            ["cached", "new", "cached"],
            embedding_model="fake/model",
            embedder=second_embedder,
            cache=EmbeddingVectorCache(temp_dir),
        )

        self.assertEqual(embedder.calls, [["cached", "new"]])
        self.assertEqual(second_embedder.calls, [])
        self.assertEqual(second, [first[0], first[1], first[0]])


if __name__ == "__main__":
    unittest.main()
