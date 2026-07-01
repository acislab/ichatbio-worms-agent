import os
from contextvars import ContextVar
from typing import Any

temporary_llm_key: ContextVar[str | None] = ContextVar(
    "temporary_llm_key",
    default=None,
)


def update_llm_credentials(metadata: dict[str, Any] | None):
    match metadata:
        case {"https://ichatbio.org/a2a/v1": {"temporary_llm_key": llm_key}}:
            temporary_llm_key.set(llm_key)
        case _:
            temporary_llm_key.set(None)


def get_llm_client_kwargs() -> dict[str, str]:
    metadata_llm_key = temporary_llm_key.get()
    use_proxy = os.getenv("USE_LLM_PROXY") == "true" or metadata_llm_key is not None

    if use_proxy:
        assert metadata_llm_key is not None, "Temporary LLM key is required for proxy mode"
        proxy_base_url = os.getenv("PROXY_OPENAI_BASE_URL")
        assert proxy_base_url is not None, "PROXY_OPENAI_BASE_URL environment variable must be set"
        return {"api_key": metadata_llm_key, "base_url": proxy_base_url}

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    assert openai_api_key is not None, "OPENAI_API_KEY environment variable must be set"
    assert openai_base_url is not None, "OPENAI_BASE_URL environment variable must be set"
    return {"api_key": openai_api_key, "base_url": openai_base_url}
