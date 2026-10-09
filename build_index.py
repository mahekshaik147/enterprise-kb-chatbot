import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_documents, split_documents

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"
MODEL_NAME = "all-MiniLM-L6-v2"

def build_index():
    chunks = split_documents(load_documents())
    texts = [c.page_content for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": c.metadata["source"],
                "page": c.metadata.get("page", 0),
                "access": c.metadata["access"],
            }
            for c in chunks
        ],
    )
    print(f"Stored {collection.count()} chunks in '{COLLECTION}'")

if __name__ == "__main__":
    build_index()