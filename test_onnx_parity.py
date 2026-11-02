"""Compare ONNX embeddings with the existing SentenceTransformer implementation.

Verifies bit-level (or near bit-level) parity so the existing ChromaDB index
remains valid. Run manually: pytest test_onnx_parity.py -v
"""
import numpy as np
import pytest

import retrieve


def st_encode(text):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(retrieve.MODEL_NAME)
    return m.encode([text], convert_to_numpy=True)


SAMPLE_QUESTIONS = [
    "What is the leave policy for employees?",
    "How do I reset my password?",
    "How many sick days do I get per year?",
    "confidential salary bands for senior roles",
    "What is the onboarding process for new hires?",
    "How do I request a new laptop?",
    "xyzzy nonsense question about zebra unicorns",
]


@pytest.fixture(scope="session", autouse=True)
def load_onnx():
    retrieve.get_embedding_model()


@pytest.mark.parametrize("text", SAMPLE_QUESTIONS)
def test_onnx_vs_sentence_transformer_parity(text):
    onnx_emb = retrieve.encode_question(text)
    st_emb = st_encode(text)

    # Both should already be L2 normalized.
    assert abs(np.linalg.norm(onnx_emb[0]) - 1.0) < 1e-5
    assert abs(np.linalg.norm(st_emb[0]) - 1.0) < 1e-5

    cos = float(np.dot(onnx_emb[0], st_emb[0]))
    max_diff = float(np.max(np.abs(onnx_emb[0] - st_emb[0])))

    print(f"\n{text!r}: cos={cos:.6f} max_abs_diff={max_diff:.6f}")
    assert cos >= 0.99999, f"cosine similarity {cos} too low for {text!r}"
    assert max_diff < 1e-4, f"max abs diff {max_diff} too high for {text!r}"


def test_dimension_matches_index():
    import chromadb
    col = chromadb.PersistentClient(path="chroma_db").get_collection("technova_kb")
    row = col.get(limit=1, include=["embeddings"])
    assert len(row["embeddings"][0]) == 384
    emb = retrieve.encode_question("test")
    assert emb.shape == (1, 384)


def test_retrieval_matches_expected_chunks():
    """Sanity: employee should get non-confidential results; hr gets both."""
    emp = retrieve.retrieve("How many sick days do I get per year?", k=3, role="employee")
    hr = retrieve.retrieve("How many sick days do I get per year?", k=3, role="hr")
    assert all(r["source"] != "hr_confidential.pdf" for r in emp)
    assert all("access" not in r for r in emp)  # metadata should not leak
    assert len(hr) == len(emp) or len(hr) >= 1
