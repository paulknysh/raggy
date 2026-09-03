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
    enc._input_names = {"input_ids", "attention_mask", "token_type_ids"}

    scores = enc.score([("q", "a"), ("q", "b")])

    assert scores == pytest.approx([0.5, reranker._sigmoid(2.0)])


def test_score_omits_token_type_ids_when_model_has_no_such_input():
    enc = reranker.OnnxCrossEncoder.__new__(reranker.OnnxCrossEncoder)
    enc._pad_id = 0
    enc.max_length = 512
    enc._tokenizer = SimpleNamespace(
        encode_batch=lambda pairs: [
            FakeEncoding([1, 2, 3], [1, 1, 1], [0, 0, 0]),
        ]
    )

    def fake_run(_outputs, feeds):
        assert "token_type_ids" not in feeds
        return [np.array([[1.0]], dtype=np.float32)]

    enc._session = SimpleNamespace(run=fake_run)
    enc._input_names = {"input_ids", "attention_mask"}

    enc.score([("q", "a")])


def test_get_cross_encoder_caches_shared_instance(monkeypatch):
    reranker._CACHE.clear()
    calls = []
    monkeypatch.setattr(
        reranker, "_resolve_onnx_file", lambda model, pref: "onnx/model.onnx"
    )
    monkeypatch.setattr(
        reranker,
        "OnnxCrossEncoder",
        lambda model, onnx_file: (calls.append((model, onnx_file)), object())[1],
    )

    first = reranker.get_cross_encoder("model-a")
    second = reranker.get_cross_encoder("model-a")

    assert first is second
    assert calls == [("model-a", "onnx/model.onnx")]


def test_resolve_onnx_file_uses_preferred_when_present(monkeypatch):
    reranker._MODEL_FILES_CACHE.clear()
    monkeypatch.setattr(
        reranker._HF_API,
        "list_repo_files",
        lambda repo_id: ["onnx/model_qint8_arm64.onnx", "tokenizer.json"],
    )
    assert (
        reranker._resolve_onnx_file("repo", "onnx/model_qint8_arm64.onnx")
        == "onnx/model_qint8_arm64.onnx"
    )


def test_resolve_onnx_file_falls_back_when_preferred_missing(monkeypatch):
    reranker._MODEL_FILES_CACHE.clear()
    monkeypatch.setattr(
        reranker._HF_API,
        "list_repo_files",
        lambda repo_id: ["onnx/model_quantized.onnx", "tokenizer.json"],
    )
    assert (
        reranker._resolve_onnx_file("repo", "onnx/model_qint8_arm64.onnx")
        == "onnx/model_quantized.onnx"
    )


def test_resolve_onnx_file_raises_when_no_onnx_export(monkeypatch):
    reranker._MODEL_FILES_CACHE.clear()
    monkeypatch.setattr(
        reranker._HF_API,
        "list_repo_files",
        lambda repo_id: ["model.safetensors", "tokenizer.json"],
    )
    with pytest.raises(RuntimeError, match="No ONNX export found"):
        reranker._resolve_onnx_file("repo", "onnx/model_qint8_arm64.onnx")


def test_resolve_onnx_file_caches_repo_files():
    reranker._MODEL_FILES_CACHE.clear()
    reranker._MODEL_FILES_CACHE["cached-repo"] = ("onnx/model.onnx",)
    assert (
        reranker._resolve_onnx_file("cached-repo", "onnx/model_qint8_arm64.onnx")
        == "onnx/model.onnx"
    )


def test_ensure_reranker_model_downloads_weights_and_tokenizer(monkeypatch):
    downloads = []
    monkeypatch.setattr(
        reranker, "_resolve_onnx_file", lambda model, pref: "onnx/model.onnx"
    )
    monkeypatch.setattr(
        reranker,
        "hf_hub_download",
        lambda repo_id, filename, token, tqdm_class: downloads.append(
            (repo_id, filename, tqdm_class)
        ),
    )

    reranker.ensure_reranker_model("model-a")

    assert downloads == [
        ("model-a", "onnx/model.onnx", None),
        ("model-a", "tokenizer.json", None),
    ]


def test_ensure_reranker_model_reports_bytes_to_progress(monkeypatch):
    reported = []
    monkeypatch.setattr(
        reranker, "_resolve_onnx_file", lambda model, pref: "onnx/model.onnx"
    )

    def fake_download(repo_id, filename, token, tqdm_class):
        with tqdm_class(total=100, unit="B", desc=f"{filename}: downloading") as bar:
            bar.update(40)
            bar.update_transfer(40)  # transfer bytes are not the file's bytes
            bar.set_postfix_str("1.0MB/s")
            bar.update(60)

    monkeypatch.setattr(reranker, "hf_hub_download", fake_download)

    reranker.ensure_reranker_model(
        "model-a", progress=lambda *args: reported.append(args)
    )

    assert (
        reported
        == [
            ("downloading model-a ...", 40, 100),
            ("downloading model-a ...", 100, 100),
        ]
        * 2
    )


def test_progress_tqdm_rewinds_on_resumed_download():
    reported = []
    bar = reranker._progress_tqdm_class(
        lambda *args: reported.append(args), "downloading ..."
    )(total=100)

    bar.update(30)
    bar.update(-30)  # huggingface-hub rewinds the bar when a download restarts

    assert [n for _, n, _ in reported] == [30, 0]


def test_progress_tqdm_stays_silent_without_a_total():
    reported = []
    bar = reranker._progress_tqdm_class(
        lambda *args: reported.append(args), "downloading ..."
    )()

    bar.update(30)

    assert reported == []
