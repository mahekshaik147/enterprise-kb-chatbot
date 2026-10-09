# Enterprise Knowledge Base Chatbot

A RAG-based chatbot that answers employee questions from company documents, with source citations and "I don't know" when the answer isn't found.

## Status
Work in progress (Week 1: setup and prototype)

## Tech Stack
Python, LangChain, ChromaDB, sentence-transformers, Gemini API

## Setup
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install packages: `pip install -r requirements.txt`
5. Create a `.env` file with `GOOGLE_API_KEY=your_key`
6. Run: `python test_setup.py`

## Planned Features
- Document ingestion (PDF) with chunking
- Hybrid retrieval with source citations
- Role-based access control
- Evaluation metrics (retrieval accuracy, hallucination rate)
- FastAPI backend, Docker, live demo