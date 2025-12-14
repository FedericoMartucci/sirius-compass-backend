import os
from typing import Callable, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

LLMBuilder = Callable[[float, bool], BaseChatModel]

def _build_openai(temperature: float, streaming: bool = False) -> BaseChatModel:
    """Builder for OpenAI (GPT) models."""
    print(f"Connecting with OpenAI...")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=temperature,
        streaming=streaming,
    )

def _build_gemini(temperature: float, streaming: bool = False) -> BaseChatModel:
    """Builder for Google Gemini models."""
    print(f"Connecting with Google Gemini...")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        temperature=temperature,
        streaming=streaming,
        convert_system_message_to_human=True
    )

_PROVIDER_REGISTRY: Dict[str, LLMBuilder] = {
    "openai": _build_openai,
    "gemini": _build_gemini
}


def get_llm(temperature: float = 0, streaming: bool = False) -> BaseChatModel:
    """
    Factory Pattern to instantiate the LLM using a dynamic registry.
    
    To add a new provider:
    1. Create a `_build_provider` function.
    2. Add it to `_PROVIDER_REGISTRY`.
    
    No modification to this function is needed (OCP Compliant).
    """
    provider_key = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    builder = _PROVIDER_REGISTRY.get(provider_key)
    
    if not builder:
        valid_keys = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unsupported LLM provider: '{provider_key}'. Supported: {valid_keys}")

    return builder(temperature, streaming)
