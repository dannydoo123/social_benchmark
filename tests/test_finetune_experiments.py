import json
import tempfile
import unittest
import uuid
from pathlib import Path

from social_benchmark.pipeline.finetune_experiments import (
    _accuracy,
    _freeze_lower_layers,
    _macro_f1,
    run_finetuned_encoder_bakeoff,
)


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeLayer:
    def __init__(self) -> None:
        self._parameters = [_FakeParameter(), _FakeParameter()]

    def parameters(self):
        return self._parameters


class _FakeEncoder:
    def __init__(self, layer_count: int) -> None:
        class _Inner:
            pass

        self.encoder = _Inner()
        self.encoder.layer = [_FakeLayer() for _ in range(layer_count)]
        self.pooler = None

    def parameters(self):
        for layer in self.encoder.layer:
            yield from layer.parameters()


class FinetuneExperimentTest(unittest.TestCase):
    def test_metrics_helpers(self):
        self.assertEqual(_accuracy(["a", "b"], ["a", "a"]), 0.5)
        self.assertAlmostEqual(_macro_f1(["a", "b"], ["a", "b"]), 1.0)

    def test_freeze_keeps_only_top_layers_trainable(self):
        encoder = _FakeEncoder(layer_count=4)
        _freeze_lower_layers(encoder, unfrozen_layers=2)
        trainable = [
            any(p.requires_grad for p in layer.parameters()) for layer in encoder.encoder.layer
        ]
        self.assertEqual(trainable, [False, False, True, True])

    def test_bakeoff_requires_training_file(self):
        missing = Path(tempfile.gettempdir()) / f"missing_{uuid.uuid4().hex}.jsonl"
        with self.assertRaises(OSError):
            run_finetuned_encoder_bakeoff(missing, missing.with_suffix(".json"))


if __name__ == "__main__":
    unittest.main()
