from types import SimpleNamespace

import numpy as np
import pytest

from raggy import reranker


class FakeEncoding:
    def __init__(self, ids, attention_mask, type_ids):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


def test_default_onnx_file_arm64(monkeypatch):
    monkeypatch.setattr(reranker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(reranker.platform, "machine", lambda: "arm64")
    assert reranker.default_onnx_file() == "onnx/model_qint8_arm64.onnx"


def test_default_onnx_file_non_arm64(monkeypatch):
    monkeypatch.setattr(reranker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(reranker.platform, "machine", lambda: "x86_64")
    assert reranker.default_onnx_file() == "onnx/model.onnx"


def test_sigmoid_bounded():
    assert 0.0 < reranker._sigmoid(-20) < 1.0
    assert 0.0 < reranker._sigmoid(20) < 1.0
    assert reranker._sigmoid(0) == pytest.approx(0.5)


def test_score_pads_batch_and_returns_sigmoid_logits():
    enc = reranker.OnnxCrossEncoder.__new__(reranker.OnnxCrossEncoder)
    enc._pad_id = 0
    enc.max_length = 512
    enc._tokenizer = SimpleNamespace(
        encode_batch=lambda pairs: [
            FakeEncoding([1, 2, 3], [1, 1, 1], [0, 0, 0]),
            FakeEncoding([1, 2, 3, 4], [1, 1, 1, 1], [0, 0, 0, 0]),
        ]
    )

    def fake_run(_outputs, feeds):
        assert feeds["input_ids"].shape == (2, 4)
        assert feeds["attention_mask"].shape == (2, 4)
        assert feeds["token_type_ids"].shape == (2, 4)
        assert feeds["input_ids"].dtype == np.int64
        return [np.array([[0.0], [2.0]], dtype=np.float32)]

    enc._session = SimpleNamespace(run=fake_run)

    scores = enc.score([("q", "a"), ("q", "b")])

    assert scores == pytest.approx([0.5, reranker._sigmoid(2.0)])


def test_get_cross_encoder_caches_shared_instance(monkeypatch):
    reranker._CACHE.clear()
    calls = []
    monkeypatch.setattr(reranker, "default_onnx_file", lambda: "onnx/model.onnx")
    monkeypatch.setattr(
        reranker,
        "OnnxCrossEncoder",
        lambda model, onnx_file: (calls.append((model, onnx_file)), object())[1],
    )

    first = reranker.get_cross_encoder("model-a")
    second = reranker.get_cross_encoder("model-a")

    assert first is second
    assert calls == [("model-a", "onnx/model.onnx")]
