"""Security tests for the TechNova knowledge base chatbot.

Run with: python -m pytest test_security.py -v

These tests assert the core security rules of the project. They never
log or print confidential document text, only file names and distances.
"""

from fastapi.testclient import TestClient
import pytest

import api
import retrieve


@pytest.fixture(autouse=True)
def configured_auth_environment(monkeypatch):
    """Provide isolated test credentials without relying on host settings."""
    monkeypatch.setenv("TECHNOVA_EMPLOYEE_USERNAME", "employee-test-user")
    monkeypatch.setenv("TECHNOVA_EMPLOYEE_PASSWORD", "employee-test-password")
    monkeypatch.setenv("TECHNOVA_HR_USERNAME", "hr-test-user")
    monkeypatch.setenv("TECHNOVA_HR_PASSWORD", "hr-test-password")


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
        json={"username": "employee-test-user", "password": "wrong"},
    )

    assert res.status_code == 401


def test_employee_login_returns_employee_session():
    client = TestClient(api.app)

    res = client.post(
        "/login",
        json={
            "username": "employee-test-user",
            "password": "employee-test-password",
        },
    )

    assert res.status_code == 200
    assert res.json()["role"] == "employee"
    assert res.json()["token"]


def test_hr_login_returns_hr_session():
    client = TestClient(api.app)

    res = client.post(
        "/login",
        json={"username": "hr-test-user", "password": "hr-test-password"},
    )

    assert res.status_code == 200
    assert res.json()["role"] == "hr"
    assert res.json()["token"]


def test_login_fails_safely_when_auth_configuration_is_missing(monkeypatch):
    monkeypatch.delenv("TECHNOVA_HR_PASSWORD")
    client = TestClient(api.app)

    res = client.post(
        "/login",
        json={
            "username": "employee-test-user",
            "password": "employee-test-password",
        },
    )

    assert res.status_code == 503
    assert res.json()["detail"] == "Authentication is not configured."


def test_ask_requires_token():
    client = TestClient(api.app)

    res = client.post("/ask", json={"question": "hi"})

    assert res.status_code == 401


def test_ask_rejects_blank_question():
    client = TestClient(api.app)

    login = client.post(
        "/login",
        json={
            "username": "employee-test-user",
            "password": "employee-test-password",
        },
    )
    assert login.status_code == 200

    token = login.json()["token"]
    res = client.post(
        "/ask",
        json={"question": "   "},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 400


def test_ask_rejects_oversized_question():
    client = TestClient(api.app)
    login = client.post(
        "/login",
        json={
            "username": "employee-test-user",
            "password": "employee-test-password",
        },
    )
    res = client.post(
        "/ask",
        json={"question": "x" * 4001},
        headers={"Authorization": f"Bearer {login.json()['token']}"},
    )

    assert res.status_code == 422


def test_cache_is_bounded():
    """The response cache must never exceed CACHE_MAX_ENTRIES."""
    for i in range(api.CACHE_MAX_ENTRIES + 50):
        api.cache_put(f"k{i}", {"answer": "stub"})

    assert len(api.CACHE) <= api.CACHE_MAX_ENTRIES
    assert api._cache_bytes <= api.CACHE_MAX_BYTES


def test_cache_skips_oversized_entries():
    key = ("employee", "oversized-answer")
    api.cache_put(key, {"answer": "x" * (api.CACHE_MAX_ENTRY_BYTES + 1)})

    assert api.cache_get(key) is None


def test_sessions_are_bounded():
    """Login sessions cannot grow without limit."""
    for i in range(api.SESSION_MAX_ENTRIES + 50):
        api._store_session(f"token-{i}", "employee")

    assert len(api.SESSIONS) <= api.SESSION_MAX_ENTRIES


def test_session_eviction_preserves_other_role_sessions(monkeypatch):
    original_sessions = dict(api.SESSIONS)
    with api._session_lock:
        api.SESSIONS.clear()
    monkeypatch.setattr(api, "SESSION_MAX_ENTRIES", 2)

    try:
        api._store_session("employee-old", "employee")
        api._store_session("hr-current", "hr")
        api._store_session("employee-new", "employee")

        assert api.SESSIONS == {
            "hr-current": "hr",
            "employee-new": "employee",
        }
    finally:
        with api._session_lock:
            api.SESSIONS.clear()
            api.SESSIONS.update(original_sessions)


def test_ask_rejects_concurrent_expensive_work():
    client = TestClient(api.app)
    login = client.post(
        "/login",
        json={
            "username": "employee-test-user",
            "password": "employee-test-password",
        },
    )
    token = login.json()["token"]

    assert api._ask_semaphore.acquire(blocking=False)
    try:
        res = client.post(
            "/ask",
            json={"question": "How many sick days do I get?"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        api._ask_semaphore.release()

    assert res.status_code == 503
    assert res.json()["detail"] == "The chatbot is busy. Please try again shortly."


def test_health_does_not_initialize_embedding_model(monkeypatch):
    """Health stays usable without invoking the ONNX model pipeline."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError("health must not load or run embeddings")

    monkeypatch.setattr(retrieve, "get_embedding_model", fail_if_called)
    monkeypatch.setattr(retrieve, "encode_question", fail_if_called)
    client = TestClient(api.app)

    res = client.get("/health")

    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_health_is_open_but_ask_is_not():
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"question": "x"}).status_code == 401
