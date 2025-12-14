from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken


def _get_master_key() -> bytes:
    key = os.getenv("CREDENTIALS_MASTER_KEY")
    if not key:
        raise ValueError("CREDENTIALS_MASTER_KEY is not set")
    return key.encode("utf-8")


def encrypt_secret(secret: str) -> str:
    """
    Encrypt a plaintext secret using the application master key.
    """
    fernet = Fernet(_get_master_key())
    return fernet.encrypt(secret.encode("utf-8")).decode("utf-8")


def decrypt_secret(encrypted_secret: str) -> str:
    """
    Decrypt an encrypted secret using the application master key.
    """
    fernet = Fernet(_get_master_key())
    try:
        return fernet.decrypt(encrypted_secret.encode("utf-8")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Invalid encryption token or wrong CREDENTIALS_MASTER_KEY") from e

