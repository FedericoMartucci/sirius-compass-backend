from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from langchain_core.callbacks.base import AsyncCallbackHandler


def _coerce_stream_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)

    if isinstance(value, dict):
        for key in ("text", "content", "value", "token"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        return str(value)

    if isinstance(value, list):
        return "".join(_coerce_stream_value(part) for part in value)

    if hasattr(value, "content"):
        return _coerce_stream_value(getattr(value, "content"))

    if hasattr(value, "text"):
        return _coerce_stream_value(getattr(value, "text"))

    return str(value)


class TokenStreamHandler(AsyncCallbackHandler):
    """
    Async callback handler that collects streamed LLM tokens into an asyncio.Queue.

    This enables SSE streaming even when the surrounding orchestration is executed
    via a single async call (e.g., LangGraph `ainvoke`).
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.error: Optional[BaseException] = None
        self._saw_llm_tokens: bool = False

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def on_chat_model_end(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def on_chat_model_stream(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if self._saw_llm_tokens:
            return None

        chunk = kwargs.get("chunk")
        if chunk is None and args:
            chunk = args[0]

        text = _coerce_stream_value(chunk)
        if text:
            await self.queue.put(text)
        return None

    async def on_llm_new_token(self, token: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._saw_llm_tokens = True
        text = _coerce_stream_value(token)
        if text:
            await self.queue.put(text)

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:  # noqa: ARG002
        self.error = error


def sse_data(payload: Dict[str, Any]) -> str:
    """
    Format a payload as a Server-Sent Events data message.
    """

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
