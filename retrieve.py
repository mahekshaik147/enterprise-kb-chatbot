import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"
MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL)
collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)

ROLE_ACCESS = {
    "employee": ["all"],
    "hr": ["all", "hr_only"],
}

def retrieve(question, k=3, role="employee", rerank=False, candidates=8):
    allowed = ROLE_ACCESS[role]
    q_emb = model.encode([question]).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=candidates if rerank else k,
        where={"access": {"$in": allowed}},
    )
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})

    if rerank:
        scores = reranker.predict([(question, r["text"]) for r in results])
        for r, s in zip(results, scores):
            r["score"] = float(s)
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:k]
    return results

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        for r in retrieve(q):
            print(f"\n[{r['source']} | page {r['page']} | distance {r['distance']:.3f}]")
            print(r["text"][:300])