import chromadb
from ingest import load_documents, split_documents
import retrieve

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"


def build_index():
    chunks = split_documents(load_documents())
    texts = [c.page_content for c in chunks]

    # ONNX embedding pipeline (same tokenizer, mean pooling, L2
    # normalization, and 256-token truncation as retrieve.py).
    retrieve.get_embedding_model()
    embeddings = []
    for text in texts:
        embeddings.append(retrieve.encode_question(text)[0].tolist())

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
