"""Provider-agnostic LLM construction.

Generation can run locally via Ollama (default) or remotely via a cloud
provider. API keys are read from the provider's standard environment
variable at construction time (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
``GEMINI_API_KEY``); they are never stored in config.yaml.
"""

import logging
import shutil
import subprocess

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "google")

# Anthropic requires an explicit max_tokens; the others are fine with a sane
# default. Kept as a constant so remotes behave consistently out of the box.
# A value is only mandatory for anthropic, but setting it uniformly avoids
# surprise across providers.
_DEFAULT_MAX_TOKENS = 1024


def ensure_ollama_model(model: str) -> bool:
    """Pull ``model`` into the local Ollama instance if it isn't already present.

    Runs ``ollama pull <model>``, which is idempotent: when the model is already
    downloaded Ollama pulls nothing. Returns True if a pull was performed (model
    was missing), False if the model was already present. Raises ``RuntimeError``
    if the ``ollama`` CLI cannot be found or the pull itself fails.
    """
    exe = shutil.which("ollama")
    if exe is None:
        raise RuntimeError(
            "ollama CLI not found on PATH; install Ollama and start `ollama serve`."
        )

    # `ollama list` cheaply reports downloaded models without contacting the
    # network, so we skip the pull for models that are already present. Its
    # output is tab-delimited with the model name in the first column. Names
    # carry a ``:tag`` suffix (e.g. ``llama3.2:latest``); a bare name in config
    # implicitly means the ``:latest`` tag, so compare with that normalization.
    try:
        listed = subprocess.run(
            [exe, "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        raise RuntimeError(f"Failed to run ollama CLI: {e}") from e

    requested = model if ":" in model else f"{model}:latest"
    for line in listed.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == requested:
            return False

    logger.info("Pulling Ollama model '%s' (first run may take a while)...", model)
    try:
        subprocess.run([exe, "pull", model], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to pull Ollama model '{model}': {e.stderr.strip() or e}"
        ) from e
    return True


def get_llm(
    provider: str,
    model_name: str,
    temperature: float,
) -> BaseChatModel:
    """Initialize and return a chat language model for ``provider``.

    All supported providers implement LangChain's ``BaseChatModel`` interface,
    so callers use ``.invoke()`` / ``.stream()`` identically regardless of
    which one is selected; only construction differs.
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=None,  # falls back to OPENAI_API_KEY env var
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=None,  # falls back to ANTHROPIC_API_KEY env var
            max_tokens=_DEFAULT_MAX_TOKENS,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=None,  # falls back to GEMINI_API_KEY env var
        )

    raise ValueError(
        f"Unknown llm_provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}"
    )
