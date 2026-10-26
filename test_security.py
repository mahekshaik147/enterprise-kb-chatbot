"""Security tests for the TechNova knowledge base chatbot.

Run with: python -m pytest test_security.py -v

These tests assert the core security rules of the project. They never
log or print confidential document text, only file names and distances.
"""

from fastapi.testclient import TestClient

import api
import retrieve


def test_employee_cannot_get_hr_confidential_chunks():
    """Employee retrieval must never return hr_confidential.pdf."""
    chunks = retrieve.retrieve(
        "What are the senior engineer salary bands?",
        k=5,
        role="employee",
    )

    assert all(c["source"] != "hr_confidential.pdf" for c in chunks)


def test_hr_can_get_hr_confidential_chunks():
    """Authorized HR users must still reach confidential documents."""
    chunks = retrieve.retrieve(
        "What are the senior engineer salary bands?",
        k=5,
        role="hr",
    )

    assert any(c["source"] == "hr_confidential.pdf" for c in chunks)


def test_invalid_role_is_rejected():
    try:
        retrieve.retrieve("hello", k=1, role="admin")
    except ValueError:
        return  # expected

    raise AssertionError("Invalid role was accepted")


def test_relevance_cutoff_blocks_irrelevant_question():
    """An off-topic question must return no chunks (fallback path)."""
    chunks = retrieve.retrieve(
        "What is the capital of France?",
        k=3,
        role="employee",
    )

    assert chunks == []


def test_relevance_cutoff_keeps_relevant_question():
    """A genuinely on-topic question must still retrieve context."""
    chunks = retrieve.retrieve(
        "How many sick leaves do I get?",
        k=3,
        role="employee",
    )

    assert len(chunks) > 0


def test_login_rejects_wrong_password():
    client = TestClient(api.app)

    res = client.post(
        "/login",
        json={"username": "asha", "password": "wrong"},
    )

    assert res.status_code == 401


def test_ask_requires_token():
    client = TestClient(api.app)

    res = client.post("/ask", json={"question": "hi"})

    assert res.status_code == 401


def test_ask_rejects_blank_question():
    client = TestClient(api.app)

    login = client.post(
        "/login",
        json={"username": "asha", "password": "employee123"},
    )
    assert login.status_code == 200

    token = login.json()["token"]
    res = client.post(
        "/ask",
        json={"question": "   "},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 400


def test_cache_is_bounded():
    """The response cache must never exceed CACHE_MAX_ENTRIES."""
    for i in range(api.CACHE_MAX_ENTRIES + 50):
        api.cache_put(f"k{i}", {"answer": "stub"})

    assert len(api.CACHE) <= api.CACHE_MAX_ENTRIES


def test_health_is_open_but_ask_is_not():
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"question": "x"}).status_code == 401
