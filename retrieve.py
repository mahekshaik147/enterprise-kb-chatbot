import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)

def retrieve(question, k=3):
    q_emb = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})
    return results

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        for r in retrieve(q):
            print(f"\n[{r['source']} | page {r['page']} | distance {r['distance']:.3f}]")
            print(r["text"][:300])