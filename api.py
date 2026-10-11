import secrets
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from chat import answer

app = FastAPI(title="TechNova Knowledge Base Chatbot")
bearer = HTTPBearer()

# Demo users only. A real system would use hashed passwords and a database.
USERS = {
    "asha": {"password": "employee123", "role": "employee"},
    "ravi": {"password": "hr123", "role": "hr"},
}
SESSIONS = {}  # token -> role
CACHE = {}     # (role, question) -> response, saves Gemini calls

BUSY_ERRORS = ["503", "504", "UNAVAILABLE", "DEADLINE"]

class LoginRequest(BaseModel):
    username: str
    password: str

class AskRequest(BaseModel):
    question: str

def current_role(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    role = SESSIONS.get(creds.credentials)
    if role is None:
        raise HTTPException(status_code=401, detail="Please log in first")
    return role

def ask_with_retry(question, role, tries=3):
    for attempt in range(tries):
        try:
            return answer(question, role=role)
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in BUSY_ERRORS):
                if attempt < tries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise HTTPException(status_code=503, detail="The AI service is busy. Please try again in a minute.")
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise HTTPException(status_code=429, detail="Daily AI quota reached. Please try again later.")
            raise

@app.post("/login")
def login(req: LoginRequest):
    user = USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_hex(16)
    SESSIONS[token] = user["role"]
    return {"token": token, "role": user["role"]}

@app.post("/ask")
def ask(req: AskRequest, role: str = Depends(current_role)):
    key = (role, req.question.strip().lower())
    if key in CACHE:
        return {**CACHE[key], "cached": True}

    reply, chunks = ask_with_retry(req.question, role)
    result = {
        "answer": reply,
        "sources": [
            {"file": c["source"], "page": c["page"] + 1, "distance": round(c["distance"], 2)}
            for c in chunks
        ],
    }
    CACHE[key] = result  # only successful answers are cached
    return {**result, "cached": False}

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("index.html")