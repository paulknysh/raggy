"""Cross-encoder re-ranking backed by a quantized ONNX model.

Implements LangChain's ``BaseCrossEncoder`` interface so it plugs directly
into ``CrossEncoderReranker`` and ``ContextualCompressionRetriever``. The
model runs on onnxruntime (already a dependency), so no PyTorch install is
required. On Apple Silicon the ARM64 int8-quantized weights are selected
automatically, keeping the download small and inference fast on the CPU.
"""

from __future__ import annotations

import logging
import platform
import threading
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import HfApi, hf_hub_download
from langchain_core.cross_encoders import BaseCrossEncoder
from tokenizers import Tokenizer

_HF_HTTP_LOGGER = logging.getLogger("huggingface_hub.utils._http")


class _UntauthAdvisoryFilter(logging.Filter):
    """Drop the HF server's benign unauthenticated-download advisory line.

    The Hub responds to anonymous downloads with an ``X-HF-Warning`` header
    ("You are sending unauthenticated requests to the HF Hub. Please set a
    HF_TOKEN ..."). huggingface-hub re-emits it at WARNING level on the first
    cold download. It's pure noise for public model files, so we strip only
    that specific line while letting genuine ERROR/rate-limit messages through.
    """

    _ADVISORY = "unauthenticated requests to the HF Hub"

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        return self._ADVISORY not in record.getMessage()


_HF_HTTP_LOGGER.addFilter(_UntauthAdvisoryFilter())

_ONNX_VARIANTS = {
    "arm64": "onnx/model_qint8_arm64.onnx",
    "default": "onnx/model.onnx",
}

# Candidate ONNX filenames tried in order when the platform-preferred variant
# is not present in a model repo. This lets models that only publish community
# exports (e.g. ``onnx-community/bge-reranker-v2-m3-ONNX``) still work.
_ONNX_FALLBACKS = (
    "onnx/model_quantized.onnx",
    "onnx/model_int8.onnx",
    "onnx/model_q8.onnx",
    "onnx/model.onnx",
)

_MODEL_FILES_CACHE: dict[str, tuple[str, ...]] = {}
_HF_API = HfApi(token=False)


def default_onnx_file() -> str:
    """Pick an ONNX variant suited to the current machine.

    Apple Silicon uses the int8-quantized ARM64 weights (~22 MB vs ~86 MB);
    everything else falls back to the fp32 export.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return _ONNX_VARIANTS["arm64"]
    return _ONNX_VARIANTS["default"]


def _resolve_onnx_file(repo_id: str, preferred: str) -> str:
    """Pick an ONNX filename that actually exists in ``repo_id``.

    The platform-preferred variant comes first; if the repo doesn't carry it
    (many cross-encoders ship only safetensors weights or community ONNX
    exports under a different name), fall back to known variant names. When no
    ONNX export exists at all, raise an actionable error.
    """
    files = _MODEL_FILES_CACHE.get(repo_id)
    if files is None:
        files = tuple(_HF_API.list_repo_files(repo_id))
        _MODEL_FILES_CACHE[repo_id] = files

    candidates = (preferred,) + _ONNX_FALLBACKS
    for candidate in candidates:
        if candidate in files:
            # if candidate != preferred:
            #     print(
            #         f"reranker: '{preferred}' not found for {repo_id}, "
            #         f"using '{candidate}' instead"
            #     )
            return candidate

    available = ", ".join(f for f in files if f.endswith(".onnx"))
    raise RuntimeError(
        f"No ONNX export found for reranker model '{repo_id}'. "
        f"This reranker requires a model repo that ships pre-exported ONNX "
        f"weights (e.g. 'onnx-community/{repo_id.split('/')[-1]}-ONNX'). "
        f"Available .onnx files: {available or 'none'}"
    )


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
        model: str,
        onnx_file: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.max_length = max_length

        onnx_file = onnx_file or _resolve_onnx_file(model, default_onnx_file())
        self.onnx_file = onnx_file
        model_path = hf_hub_download(repo_id=model, filename=onnx_file, token=False)
        tokenizer_path = hf_hub_download(
            repo_id=model, filename="tokenizer.json", token=False
        )

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

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
        }
        # RoBERTa/XLMRoberta-family rerankers take no token-type ids; only feed
        # them to BERT-style models whose ONNX graph declares the input.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int64)
        logits = self._session.run(None, feeds)[0]
        return [_sigmoid(float(v)) for v in logits[:, 0]]


_CACHE: dict[tuple[str, str], OnnxCrossEncoder] = {}
_CACHE_LOCK = threading.Lock()


def get_cross_encoder(
    model: str,
    onnx_file: str | None = None,
) -> OnnxCrossEncoder:
    """Return a shared ``OnnxCrossEncoder`` instance.

    Model files are cached on disk by huggingface-hub and the loaded session
    is reused across calls, so long-running processes (e.g. the eval harness)
    only initialize the model once.
    """
    onnx_file = onnx_file or _resolve_onnx_file(model, default_onnx_file())
    key = (model, onnx_file)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = OnnxCrossEncoder(model=model, onnx_file=onnx_file)
        return _CACHE[key]
