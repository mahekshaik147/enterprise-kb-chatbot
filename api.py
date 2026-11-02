import logging
import os
import re
import secrets
import sys
import threading
import time
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="TechNova Knowledge Base Chatbot")

bearer = HTTPBearer()

AUTH_ENV_VARS = (
    "TECHNOVA_EMPLOYEE_USERNAME",
    "TECHNOVA_EMPLOYEE_PASSWORD",
    "TECHNOVA_HR_USERNAME",
    "TECHNOVA_HR_PASSWORD",
)

SESSIONS = {}  # token -> role
CACHE = {}     # (role, question) -> response
_cache_sizes = {}
_cache_bytes = 0
_session_lock = threading.Lock()
_cache_lock = threading.Lock()
_ask_semaphore = threading.BoundedSemaphore(value=1)

# Bound retained state and expensive concurrent work for the Render Free tier.
CACHE_MAX_ENTRIES = 100
CACHE_MAX_ENTRY_BYTES = 32 * 1024
CACHE_MAX_BYTES = 1024 * 1024
SESSION_MAX_ENTRIES = 1000


def _object_size(value, seen=None):
    """Estimate retained Python object memory for the cache's plain data."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _object_size(key, seen) + _object_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple)):
        size += sum(_object_size(item, seen) for item in value)
    return size


def cache_put(key, value):
    global _cache_bytes
    entry_size = _object_size(key) + _object_size(value)

    if entry_size > CACHE_MAX_ENTRY_BYTES:
        return

    with _cache_lock:
        if key in CACHE:
            del CACHE[key]
            _cache_bytes -= _cache_sizes.pop(key)
        while CACHE and (
            len(CACHE) >= CACHE_MAX_ENTRIES
            or _cache_bytes + entry_size > CACHE_MAX_BYTES
        ):
            oldest = next(iter(CACHE))
            CACHE.pop(oldest)  # FIFO eviction
            _cache_bytes -= _cache_sizes.pop(oldest)
        CACHE[key] = value
        _cache_sizes[key] = entry_size
        _cache_bytes += entry_size


def cache_get(key):
    with _cache_lock:
        return CACHE.get(key)


def _store_session(token, role):
    with _session_lock:
        while len(SESSIONS) >= SESSION_MAX_ENTRIES:
            SESSIONS.pop(next(iter(SESSIONS)))  # FIFO eviction
        SESSIONS[token] = role

BUSY_ERRORS = ["503", "504", "UNAVAILABLE", "DEADLINE"]


class LoginRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)


def current_role(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
):
    with _session_lock:
        role = SESSIONS.get(creds.credentials)

    if role is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first",
        )

    return role


def _configured_users():
    """Return configured role credentials, or None if configuration is unsafe."""
    values = {name: os.environ.get(name) for name in AUTH_ENV_VARS}
    if any(value is None or not value.strip() for value in values.values()):
        return None

    employee_username = values["TECHNOVA_EMPLOYEE_USERNAME"]
    hr_username = values["TECHNOVA_HR_USERNAME"]
    if employee_username == hr_username:
        return None

    return {
        employee_username: {
            "password": values["TECHNOVA_EMPLOYEE_PASSWORD"],
            "role": "employee",
        },
        hr_username: {
            "password": values["TECHNOVA_HR_PASSWORD"],
            "role": "hr",
        },
    }

def ask_with_retry(question, role, tries=3):
    from chat import answer
    for attempt in range(tries):
        try:
            return answer(question, role=role)

        except Exception as e:
            msg = str(e)

            safe_msg = re.sub(
                r"(?i)(api[_ -]?key|access[_ -]?token)(\s*[:=]\s*)\S+",
                r"\1\2[REDACTED]",
                msg,
            )

            logger.error(
                "AI request failed on attempt %s: %s: %s",
                attempt + 1,
                type(e).__name__,
                safe_msg[:500],
            )


            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise HTTPException(
                    status_code=429,
                    detail="The AI service is temporarily rate-limited or its quota is exhausted. Please try again later.",
                )

            if any(error in msg for error in BUSY_ERRORS):
                if attempt < tries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue

                raise HTTPException(
                    status_code=503,
                    detail="The AI service is busy. Please try again in a minute.",
                )

            raise HTTPException(
                status_code=500,
                detail="Unable to process your question right now.",
            )

    raise HTTPException(
        status_code=503,
        detail="The AI service is temporarily unavailable.",
    )


def get_cited_sources(reply, chunks):
    """Return only chunks referenced by [1], [2], etc. in the answer."""
    cited_numbers = list(
        dict.fromkeys(
            int(number)
            for number in re.findall(r"\[(\d+)\]", reply)
        )
    )

    sources = []

    for number in cited_numbers:
        # Citation numbers are 1-based.
        if 1 <= number <= len(chunks):
            chunk = chunks[number - 1]

            sources.append({
                "file": chunk["source"],
                "page": chunk["page"] + 1,
                "distance": round(chunk["distance"], 2),
            })

    return sources


@app.post("/login")
def login(req: LoginRequest):
    users = _configured_users()
    if users is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured.",
        )

    user = users.get(req.username)

    if not user or user["password"] != req.password:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    token = secrets.token_hex(16)
    _store_session(token, user["role"])

    return {
        "token": token,
        "role": user["role"],
    }


@app.post("/logout")
def logout(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    role: str = Depends(current_role),
):
    # Revoke the bearer token on the server, not only in the browser.
    with _session_lock:
        SESSIONS.pop(creds.credentials, None)
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest, role: str = Depends(current_role)):
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    key = (role, question.lower())

    cached = cache_get(key)
    if cached is not None:
        return {**cached, "cached": True}

    if not _ask_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The chatbot is busy. Please try again shortly.",
        )

    try:
        reply, chunks = ask_with_retry(question, role)

        result = {
            "answer": reply,
            "sources": get_cited_sources(reply, chunks),
        }
        cache_put(key, result)

        return {**result, "cached": False}
    finally:
        _ask_semaphore.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse("index.html")
