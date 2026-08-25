"""Cross-encoder re-ranking backed by a quantized ONNX model.

Implements LangChain's ``BaseCrossEncoder`` interface so it plugs directly
into ``CrossEncoderReranker`` and ``ContextualCompressionRetriever``. The
model runs on onnxruntime (already a dependency), so no PyTorch install is
required. On Apple Silicon the ARM64 int8-quantized weights are selected
automatically, keeping the download small and inference fast on the CPU.
"""

from __future__ import annotations

import platform
import threading
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from langchain_core.cross_encoders import BaseCrossEncoder
from tokenizers import Tokenizer

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

_ONNX_VARIANTS = {
    "arm64": "onnx/model_qint8_arm64.onnx",
    "default": "onnx/model.onnx",
}


def default_onnx_file() -> str:
    """Pick an ONNX variant suited to the current machine.

    Apple Silicon uses the int8-quantized ARM64 weights (~22 MB vs ~86 MB);
    everything else falls back to the fp32 export.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return _ONNX_VARIANTS["arm64"]
    return _ONNX_VARIANTS["default"]


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(logit)))


class OnnxCrossEncoder(BaseCrossEncoder):
    """Score (query, document) pairs with an ONNX cross-encoder.

    A BERT-style cross-encoder feeds the query and document into the model
    together and emits a single relevance logit per pair, giving much finer
    discrimination than cosine similarity between separately embedded texts.
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANKER_MODEL,
        onnx_file: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.max_length = max_length

        onnx_file = onnx_file or default_onnx_file()
        self.onnx_file = onnx_file
        model_path = hf_hub_download(repo_id=model, filename=onnx_file)
        tokenizer_path = hf_hub_download(repo_id=model, filename="tokenizer.json")

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def score(self, text_pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return a relevance probability in (0, 1) for each text pair.

        ``text_pairs`` is a sequence of ``(query, document)`` tuples. All pairs
        are tokenized and scored in a single batch; results stay in input order.
        """
        encodings = self._tokenizer.encode_batch(
            [(query, doc) for query, doc in text_pairs]
        )

        max_len = min(max(len(enc.ids) for enc in encodings), self.max_length)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        token_type_ids: list[list[int]] = []
        for enc in encodings:
            ids = enc.ids[:max_len]
            mask = enc.attention_mask[:max_len]
            padding = max_len - len(ids)
            input_ids.append(ids + [self._pad_id] * padding)
            attention_mask.append(mask + [0] * padding)
            token_type_ids.append(enc.type_ids[:max_len] + [0] * padding)

        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
        }
        logits = self._session.run(None, feeds)[0]
        return [_sigmoid(float(v)) for v in logits[:, 0]]


_CACHE: dict[tuple[str, str], OnnxCrossEncoder] = {}
_CACHE_LOCK = threading.Lock()


def get_cross_encoder(
    model: str = DEFAULT_RERANKER_MODEL,
    onnx_file: str | None = None,
) -> OnnxCrossEncoder:
    """Return a shared ``OnnxCrossEncoder`` instance.

    Model files are cached on disk by huggingface-hub and the loaded session
    is reused across calls, so long-running processes (e.g. the eval harness)
    only initialize the model once.
    """
    onnx_file = onnx_file or default_onnx_file()
    key = (model, onnx_file)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = OnnxCrossEncoder(model=model, onnx_file=onnx_file)
        return _CACHE[key]
