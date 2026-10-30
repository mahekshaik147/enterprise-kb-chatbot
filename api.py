import logging
import re
import secrets
import time

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse

from pydantic import BaseModel


logger = logging.getLogger(__name__)

app = FastAPI(title="TechNova Knowledge Base Chatbot")

bearer = HTTPBearer()

# Demo users only. Do not use these credentials for real company data.
USERS = {
    "asha": {"password": "employee123", "role": "employee"},
    "ravi": {"password": "hr123", "role": "hr"},
}

SESSIONS = {}  # token -> role
CACHE = {}     # (role, question) -> response

# Bound the response cache so it cannot grow without limit
# (a memory-leak / DoS vector on small free-tier hosts).
CACHE_MAX_ENTRIES = 500


def cache_put(key, value):
    while len(CACHE) >= CACHE_MAX_ENTRIES:
        CACHE.pop(next(iter(CACHE)))  # FIFO eviction
    CACHE[key] = value

BUSY_ERRORS = ["503", "504", "UNAVAILABLE", "DEADLINE"]


class LoginRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str


def current_role(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
):
    role = SESSIONS.get(creds.credentials)

    if role is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first",
        )

    return role



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
    user = USERS.get(req.username)

    if not user or user["password"] != req.password:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    token = secrets.token_hex(16)
    SESSIONS[token] = user["role"]

    return {
        "token": token,
        "role": user["role"],
    }


@app.post("/ask")
def ask(req: AskRequest, role: str = Depends(current_role)):
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    key = (role, question.lower())

    if key in CACHE:
        return {**CACHE[key], "cached": True}

    reply, chunks = ask_with_retry(question, role)

    result = {
        "answer": reply,
        "sources": get_cited_sources(reply, chunks),
    }

    cache_put(key, result)

    return {**result, "cached": False}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse("index.html")
