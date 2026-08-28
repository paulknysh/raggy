"""Provider-agnostic LLM construction.

Generation can run locally via Ollama (default) or remotely via a cloud
provider. API keys are read from the provider's standard environment
variable at construction time (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
``GEMINI_API_KEY``); they are never stored in config.yaml.
"""

from langchain_core.language_models import BaseChatModel

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "google")

# Anthropic requires an explicit max_tokens; the others are fine with a sane
# default. Kept as a constant so remotes behave consistently out of the box.
# A value is only mandatory for anthropic, but setting it uniformly avoids
# surprise across providers.
_DEFAULT_MAX_TOKENS = 1024


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
