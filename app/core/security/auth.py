"""
Auth0 JWT Authentication and Authorization for FastAPI.

Validates JWT tokens from Auth0 and extracts user identity for multi-tenancy.
"""
import os
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import requests
from functools import lru_cache
from dotenv import load_dotenv
from app.core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
AUTH0_ALGORITHMS = ["RS256"]

security = HTTPBearer()


class AuthError(Exception):
    """Custom exception for authentication errors."""
    def __init__(self, error: dict, status_code: int):
        self.error = error
        self.status_code = status_code


@lru_cache(maxsize=1)
def get_jwks() -> dict:
    """
    Fetches the JSON Web Key Set (JWKS) from Auth0.
    Cached to avoid repeated calls.
    """
    if not AUTH0_DOMAIN:
        raise ValueError("AUTH0_DOMAIN environment variable is not set")
    
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise AuthError(
            {"code": "jwks_fetch_failed", "description": "Unable to fetch JWKS"},
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def verify_token(token: str) -> dict:
    """
    Verifies and decodes a JWT token from Auth0.
    
    Returns the decoded token payload containing user information.
    """
    if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
        raise ValueError("AUTH0_DOMAIN and AUTH0_AUDIENCE must be set")
    
    try:
        # Get the key ID from the token header
        unverified_header = jwt.get_unverified_header(token)
        
        # Find the matching key in JWKS
        jwks = get_jwks()
        rsa_key = {}
        for key in jwks.get("keys", []):
            if key.get("kid") == unverified_header.get("kid"):
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
        
        if not rsa_key:
            raise AuthError(
                {"code": "invalid_header", "description": "Unable to find appropriate key"},
                status.HTTP_401_UNAUTHORIZED
            )
        
        # Verify and decode the token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=AUTH0_ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )
        
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise AuthError(
            {"code": "invalid_token", "description": "Token is invalid or expired"},
            status.HTTP_401_UNAUTHORIZED
        )
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}")
        raise AuthError(
            {"code": "verification_failed", "description": "Token verification failed due to an internal error"},
            status.HTTP_401_UNAUTHORIZED
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    FastAPI dependency that extracts and validates the current user from JWT token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            user_id = user["sub"]  # Auth0 subject (user ID)
            ...
    
    Returns:
        dict: Decoded JWT payload with keys:
            - sub: User ID (Auth0 subject)
            - email: User email (if available)
            - name: User name (if available)
            - permissions: List of permissions (if using Auth0 RBAC)
    """
    token = credentials.credentials
    
    try:
        payload = verify_token(token)
        return payload
    except AuthError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.error,
            headers={"WWW-Authenticate": "Bearer"}
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service misconfigured"
        )


def get_user_id(user: dict = Depends(get_current_user)) -> str:
    """
    Extracts the user_id (sub claim) from the authenticated user.
    
    This is the primary dependency for multi-tenancy scoping.
    
    Usage:
        @app.get("/my-projects")
        async def my_projects(user_id: str = Depends(get_user_id)):
            # Query only projects owned by user_id
            ...
    """
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    return user_id


# Optional: Dependency for endpoints that MAY have auth but don't require it
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[dict]:
    """
    Optional authentication - returns None if no token provided.
    Use for endpoints that behave differently with/without auth.
    """
    if not credentials:
        return None
    
    try:
        return verify_token(credentials.credentials)
    except AuthError:
        return None
