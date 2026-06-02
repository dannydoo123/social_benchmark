from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

from social_benchmark.pipeline.text_features import model_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class HuggingFaceTextEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        backend: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.batch_size = batch_size
        self.normalize = normalize
        self._sentence_model: Any | None = None
        self._tokenizer: Any | None = None
        self._transformer_model: Any | None = None
        self._torch: Any | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend in {"auto", "sentence-transformers"}:
            try:
                return self._encode_sentence_transformers(texts)
            except ImportError:
                if self.backend == "sentence-transformers":
                    raise
        return self._encode_transformers(texts)

    def _encode_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install sentence-transformers to use the sentence-transformers embedding backend.") from exc
            self._sentence_model = SentenceTransformer(self.model_name)
        vectors = self._sentence_model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _encode_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._tokenizer is None or self._transformer_model is None or self._torch is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers, or install transformers and torch, to use local Hugging Face embeddings."
                ) from exc
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._transformer_model = AutoModel.from_pretrained(self.model_name)
            self._transformer_model.eval()

        output: list[list[float]] = []
        torch = self._torch
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                model_output = self._transformer_model(**encoded)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            vectors = summed / counts
            if self.normalize:
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            output.extend(vectors.cpu().tolist())
        return output


def write_embeddings_jsonl(
    input_jsonl: str | Path,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    backend: str = "auto",
    batch_size: int = 32,
) -> int:
    rows = _read_jsonl(input_jsonl)
    embedder = HuggingFaceTextEmbedder(model_name=model_name, backend=backend, batch_size=batch_size)
    texts = [model_text(row) for row in rows]
    vectors = embedder.encode(texts)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, (row, text, vector) in enumerate(zip(rows, texts, vectors)):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "embedding_model": model_name,
                        "text_hash": _text_hash(text),
                        "source_platform": row.get("source_platform", ""),
                        "source_item_id": row.get("source_item_id", ""),
                        "model_id": row.get("model_id", ""),
                        "embedding": vector,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
    return len(rows)


def cluster_embedding_jsonl(
    embeddings_jsonl: str | Path,
    output_path: str | Path,
    *,
    threshold: float = 0.92,
) -> dict[str, Any]:
    rows = _read_jsonl(embeddings_jsonl)
    clusters = greedy_cosine_clusters(rows, threshold=threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"threshold": threshold, "clusters": clusters}, indent=2, sort_keys=True), encoding="utf-8")
    return {"rows": len(rows), "clusters": len(clusters), "out": str(output)}


def greedy_cosine_clusters(rows: list[dict[str, Any]], *, threshold: float = 0.92) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    centroids: list[list[float]] = []
    for row in rows:
        vector = [float(value) for value in row.get("embedding", [])]
        if not vector:
            continue
        best_index = -1
        best_score = -1.0
        for index, centroid in enumerate(centroids):
            score = cosine_similarity(vector, centroid)
            if score > best_score:
                best_index = index
                best_score = score
        member = {
            "index": row.get("index"),
            "source_item_id": row.get("source_item_id", ""),
            "model_id": row.get("model_id", ""),
            "similarity_to_centroid": round(best_score, 6) if best_index >= 0 else 1.0,
        }
        if best_index >= 0 and best_score >= threshold:
            clusters[best_index]["members"].append(member)
            centroids[best_index] = _updated_centroid(
                centroids[best_index],
                vector,
                len(clusters[best_index]["members"]),
            )
        else:
            clusters.append({"cluster_id": f"emb_{len(clusters) + 1}", "members": [member]})
            centroids.append(vector)
    return clusters


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _updated_centroid(current: list[float], vector: list[float], count: int) -> list[float]:
    previous_count = count - 1
    return [((value * previous_count) + new_value) / count for value, new_value in zip(current, vector)]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
