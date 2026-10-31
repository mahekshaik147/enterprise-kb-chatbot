
import logging
import os
import threading
import time

import chromadb
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ONNX model and tokenizer locations, matching what the official
# sentence-transformers/all-MiniLM-L6-v2 repo ships.
ONNX_MODEL_FILE = "onnx/model.onnx"
MAX_SEQ_LENGTH = 256   # same truncation length SentenceTransformer used

# Models are loaded only when required.
_tokenizer = None
_session = None
_reranker = None
_rerank_tokenizer = None
_embedding_init_lock = threading.Lock()

# Keep one client alive for this process and reuse its collection handle.
_chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = _chroma_client.get_collection(COLLECTION)

ROLE_ACCESS = {
    "employee": ["all"],
    "hr": ["all", "hr_only"],
}


# Cosine distance threshold: chunks farther than this are considered
# irrelevant to the question. Calibrated against eval_results.csv:
# on-topic hits cluster at distance <= ~1.30, off-topic noise at >= ~1.53.
MAX_DISTANCE = 1.45


def _get_tokenizer():
    """Load the tokenizer that ships with all-MiniLM-L6-v2 (BERT wordpiece)."""
    global _tokenizer
    if _tokenizer is None:
        with _embedding_init_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def _get_onnx_session():
    """Load the official all-MiniLM-L6-v2 ONNX model, CPU only."""
    global _session
    if _session is None:
        with _embedding_init_lock:
            if _session is None:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(MODEL_NAME, ONNX_MODEL_FILE)
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.enable_cpu_mem_arena = False
                so.enable_mem_pattern = False
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                _session = ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
    return _session


def encode_question(text):
    """Encode a question with the ONNX pipeline.

    Replicates the SentenceTransformer pipeline exactly:
    BERT wordpiece tokenizer (truncation 256, padding)
    -> forward pass
    -> mean pooling over token embeddings using the attention mask
    -> L2 normalization.
    """
    tokenizer = _get_tokenizer()
    session = _get_onnx_session()

    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )

    feed = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc["token_type_ids"].astype(np.int64),
    }

    out = session.run(None, feed)
    hidden = out[0]  # (batch, seq_len, 384)

    # Mean pooling over real (non-padding) tokens.
    mask = enc["attention_mask"][:, :, None].astype(np.float32)
    summed = (hidden.astype(np.float32, copy=False) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    mean = summed / counts

    # L2 normalize.
    norm = np.linalg.norm(mean, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    emb = mean / norm

    return emb


def get_embedding_model():
    """Backwards-compatible hook: triggers ONNX session + tokenizer load."""
    _get_tokenizer()
    _get_onnx_session()
    return True


def get_reranker():
    """Development-only reranker.

    Uses PyTorch via sentence-transformers and is only imported when
    rerank=True is explicitly requested. Production /ask never sets this
    flag, so PyTorch is not loaded in the normal retrieval path.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def retrieve(
    question,
    k=3,
    role="employee",
    rerank=False,
    candidates=8,
):
    if role not in ROLE_ACCESS:
        raise ValueError("Invalid role")

    question = question.strip()

    if not question:
        return []

    allowed = ROLE_ACCESS[role]
    retrieval_started = time.perf_counter()

    embedding = encode_question(question)
    q_emb = embedding.tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=candidates if rerank else k,
        where={"access": {"$in": allowed}},
    )

    results = []

    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        # Filter out chunks too far from the question so irrelevant
        # questions do not return random passages.
        if dist <= MAX_DISTANCE:
            results.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "distance": dist,
            })

    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    logger.info(
        "timing role=%s step=retrieval duration_ms=%.1f results=%d status=%s",
        role,
        retrieval_ms,
        len(results),
        "ok" if results else "empty",
    )

    if rerank and results:
        rerank_started = time.perf_counter()
        reranker_model = get_reranker()
        scores = reranker_model.predict(
            [(question, item["text"]) for item in results]
        )

        for item, score in zip(results, scores):
            item["score"] = float(score)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        results = results[:k]
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        logger.info(
            "timing role=%s step=rerank duration_ms=%.1f results=%d status=ok",
            role,
            rerank_ms,
            len(results),
        )

    return results
