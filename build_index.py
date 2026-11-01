import chromadb
from ingest import load_documents, split_documents

DB_PATH = "chroma_db"
COLLECTION = "technova_kb"


def build_index():
    chunks = split_documents(load_documents())
    texts = [c.page_content for c in chunks]

    # Create the (empty) collection first. retrieve.py looks the collection
    # up as soon as it is imported, so it must already exist before that import.
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    # Imported here, not at the top of the file, for the reason above.
    import retrieve

    # ONNX embedding pipeline (same tokenizer, mean pooling, L2
    # normalization, and 256-token truncation as retrieve.py).
    retrieve.get_embedding_model()
    embeddings = []
    for text in texts:
        embeddings.append(retrieve.encode_question(text)[0].tolist())

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