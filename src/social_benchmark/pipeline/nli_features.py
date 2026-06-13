from __future__ import annotations

from typing import Any

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class NliRubricScorer:
    """Scores text against fixed label-rubric hypotheses with an NLI cross-encoder.

    `encode(texts)` returns one entailment-probability vector per text, with one
    dimension per hypothesis, so the scorer can be used anywhere an embedder is
    expected (including the bake-off vector cache).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        hypotheses: list[str] | None = None,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.hypotheses = list(hypotheses or [])
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._entailment_index: int | None = None
        self._device = "cpu"

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.hypotheses:
            raise ValueError("NliRubricScorer requires at least one hypothesis.")
        self._ensure_model()
        import torch

        vectors: list[list[float]] = []
        pairs = [(text, hypothesis) for text in texts for hypothesis in self.hypotheses]
        probabilities: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            batch_probabilities = torch.softmax(logits, dim=-1)[:, self._entailment_index]
            probabilities.extend(batch_probabilities.cpu().tolist())
        width = len(self.hypotheses)
        for index in range(len(texts)):
            vectors.append(probabilities[index * width : (index + 1) * width])
        return vectors

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        label_map = {
            str(label).lower(): int(index)
            for index, label in (self._model.config.id2label or {}).items()
        }
        self._entailment_index = next(
            (index for label, index in label_map.items() if "entail" in label),
            None,
        )
        if self._entailment_index is None:
            raise ValueError(f"Model {self.model_name} has no entailment label: {label_map}")
