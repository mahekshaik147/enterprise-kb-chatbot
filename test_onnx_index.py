"""Focused tests for the ONNX indexing path in build_index.py.

Verifies the ONNX indexing path produces embeddings that are 384-dim,
L2-normalized, metadata/access-role-preserving, and compatible with the
existing ChromaDB collection — WITHOUT actually rebuilding the index.
"""
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from unittest.mock import Mock

import retrieve
from ingest import load_documents, split_documents


@pytest.fixture(scope="session", autouse=True)
def load_onnx():
    retrieve.get_embedding_model()


def test_build_index_pipeline_dimension_and_normalization():
    """Embeddings produced by the ONNX path must be 384-dim and unit-norm."""
    chunks = split_documents(load_documents())
    texts = [c.page_content for c in chunks]
    assert len(texts) > 0

    for text in texts[:5]:
        emb = retrieve.encode_question(text)
        assert emb.shape == (1, 384)
        assert abs(np.linalg.norm(emb[0]) - 1.0) < 1e-5


def test_embedding_models_initialize_once_under_concurrency(monkeypatch):
    """Parallel first requests must share one tokenizer and ONNX session."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = object()
    session = object()

    def load_tokenizer(*args, **kwargs):
        sleep(0.05)
        return tokenizer

    tokenizer_loader = Mock(side_effect=load_tokenizer)
    model_downloader = Mock(return_value="mock-model.onnx")
    session_loader = Mock(return_value=session)

    monkeypatch.setattr(retrieve, "_tokenizer", None)
    monkeypatch.setattr(retrieve, "_session", None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", model_downloader)
    monkeypatch.setattr(retrieve.ort, "InferenceSession", session_loader)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: retrieve.get_embedding_model(), range(16)))

    assert all(results)
    assert tokenizer_loader.call_count == 1
    assert model_downloader.call_count == 1
    assert session_loader.call_count == 1
    options = session_loader.call_args.kwargs["sess_options"]
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == retrieve.ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_embedding_initialization_can_retry_after_exception(monkeypatch):
    """Failed initialization releases the lock and can be retried."""
    from transformers import AutoTokenizer

    tokenizer = object()
    session = object()
    tokenizer_loader = Mock(side_effect=[RuntimeError("temporary"), tokenizer])
    session_loader = Mock(side_effect=[RuntimeError("temporary"), session])

    monkeypatch.setattr(retrieve, "_tokenizer", None)
    monkeypatch.setattr(retrieve, "_session", None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", Mock(return_value="mock-model.onnx"))
    monkeypatch.setattr(retrieve.ort, "InferenceSession", session_loader)

    with pytest.raises(RuntimeError, match="temporary"):
        retrieve.get_embedding_model()

    assert retrieve._tokenizer is None
    assert retrieve._session is None

    with pytest.raises(RuntimeError, match="temporary"):
        retrieve.get_embedding_model()

    assert retrieve._tokenizer is tokenizer
    assert retrieve._session is None

    assert retrieve.get_embedding_model() is True
    assert retrieve._tokenizer is tokenizer
    assert retrieve._session is session
    assert tokenizer_loader.call_count == 2
    assert session_loader.call_count == 2


def test_chunk_metadata_and_access_roles_preserved():
    """Chunking and metadata must be identical to the pre-ONNX path."""
    chunks = split_documents(load_documents())
    assert len(chunks) == 16, f"expected 16 chunks, got {len(chunks)}"

    sources = {c.metadata["source"] for c in chunks}
    assert sources == {
        "hr_policy.pdf",
        "it_support_guide.pdf",
        "onboarding_manual.pdf",
        "hr_confidential.pdf",
    }

    # Access roles must still be present and correct.
    for c in chunks:
        expected = "hr_only" if "confidential" in c.metadata["source"] else "all"
        assert c.metadata["access"] == expected
        assert isinstance(c.metadata.get("page", 0), int)


def test_index_compatibility_without_rebuild():
    """The existing chroma_db collection stays valid.

    - collection name, count, dim, metadata must be unchanged
    - ONNX-encoded queries must still retrieve expected chunks
    - the ONNX query embedding must be compatible with stored 384-dim vectors
    """
    import chromadb
    col = chromadb.PersistentClient(path="chroma_db").get_collection("technova_kb")

    assert col.count() == 16, f"index must stay at 16 chunks, got {col.count()}"

    row = col.get(limit=1, include=["embeddings", "metadatas"])
    assert len(row["embeddings"][0]) == 384
    assert set(row["metadatas"][0].keys()) == {"source", "page", "access"}

    # Employee role must never see the confidential chunk.
    emp = retrieve.retrieve("How many sick days do I get per year?", k=3, role="employee")
    assert all(r["source"] != "hr_confidential.pdf" for r in emp)

    # HR role should be able to see confidential chunks when on-topic.
    hr = retrieve.retrieve("What are the salary bands for senior roles?", k=3, role="hr")
    assert any(r["source"] == "hr_confidential.pdf" for r in hr)

    # ONNX query embedding must be same dim as stored vectors.
    q = retrieve.encode_question("test query")
    assert q.shape == (1, 384)
