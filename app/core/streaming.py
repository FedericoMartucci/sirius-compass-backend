from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from langchain_core.callbacks.base import AsyncCallbackHandler


class TokenStreamHandler(AsyncCallbackHandler):
    """
    Async callback handler that collects streamed LLM tokens into an asyncio.Queue.

    This enables SSE streaming even when the surrounding orchestration is executed
    via a single async call (e.g., LangGraph `ainvoke`).
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.error: Optional[BaseException] = None

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # noqa: ARG002
        await self.queue.put(token)

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:  # noqa: ARG002
        self.error = error


def sse_data(payload: Dict[str, Any]) -> str:
    """
    Format a payload as a Server-Sent Events data message.
    """

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

