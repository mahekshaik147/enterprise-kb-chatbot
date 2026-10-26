
import logging
import time

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"

MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Models are loaded only when required.
model = None
reranker = None

collection = chromadb.PersistentClient(
    path=DB_PATH
).get_collection(COLLECTION)

ROLE_ACCESS = {
    "employee": ["all"],
    "hr": ["all", "hr_only"],
}


# Cosine distance threshold: chunks farther than this are considered
# irrelevant to the question. Calibrated against eval_results.csv:
# on-topic hits cluster at distance <= ~1.30, off-topic noise at >= ~1.53.
MAX_DISTANCE = 1.45


def get_embedding_model():
    global model

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    return model


def get_reranker():
    global reranker

    if reranker is None:
        reranker = CrossEncoder(RERANK_MODEL)

    return reranker


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

    embedding_model = get_embedding_model()
    q_emb = embedding_model.encode(
        [question],
        convert_to_numpy=True,
    ).tolist()

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
