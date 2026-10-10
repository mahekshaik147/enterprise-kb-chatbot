from langchain_core.documents import Document
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")

def load_documents():
    docs = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf)).load()
        for page in pages:
            page.page_content = re.sub(r"\s+", " ", page.page_content).strip()
            page.metadata["source"] = pdf.name
            # used later for access control
            page.metadata["access"] = "hr_only" if "confidential" in pdf.name else "all"
        docs.extend(pages)
    return docs

def split_documents(docs, chunk_size=250, chunk_overlap=40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for doc in docs:
        # split before numbered headings like "2. Leave Policy"
        sections = re.split(r"\s(?=\d\.\s[A-Z])", doc.page_content)
        for sec in sections:
            piece = Document(page_content=sec, metadata=dict(doc.metadata))
            chunks.extend(splitter.split_documents([piece]))
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages from {DATA_DIR}")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks\n")

    for chunk in chunks[:3]:
        print("SOURCE:", chunk.metadata["source"], "| PAGE:", chunk.metadata.get("page"))
        print(chunk.page_content)
        print("-" * 50)