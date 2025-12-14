from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from app.core.database.models import ChatMessage, ChatThread


def coerce_content_to_text(content: Any) -> str:
    """
    Normalize model/tool message content to a plain string.

    Some providers may return rich content (e.g., list-based parts). For persistence
    and chat UI rendering we store a single text field.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def get_or_create_thread(
    session: Session,
    external_thread_id: str,
    owner_id: Optional[str] = None,
) -> ChatThread:
    thread = session.exec(
        select(ChatThread).where(ChatThread.external_thread_id == external_thread_id)
    ).first()

    if thread:
        if owner_id and thread.owner_id != owner_id:
            thread.owner_id = owner_id
            thread.updated_at = datetime.utcnow()
            session.add(thread)
            session.commit()
        return thread

    thread = ChatThread(
        external_thread_id=external_thread_id,
        owner_id=owner_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(thread)
    session.commit()
    session.refresh(thread)
    return thread


def load_thread_messages(
    session: Session,
    chat_thread_id: int,
    limit: int = 50,
) -> List[Tuple[str, str]]:
    rows = session.exec(
        select(ChatMessage)
        .where(ChatMessage.chat_thread_id == chat_thread_id)
        .order_by(ChatMessage.id.asc())
        .limit(limit)
    ).all()
    return [(m.role, m.content) for m in rows]


def append_message(
    session: Session,
    chat_thread_id: int,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ChatMessage:
    message = ChatMessage(
        chat_thread_id=chat_thread_id,
        role=role,
        content=content,
        message_metadata=metadata or {},
        created_at=datetime.utcnow(),
    )
    session.add(message)

    thread = session.get(ChatThread, chat_thread_id)
    if thread:
        thread.updated_at = datetime.utcnow()
        session.add(thread)

    session.commit()
    session.refresh(message)
    return message

