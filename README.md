# TechNova Enterprise Knowledge Base Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that helps employees find answers from company documents. It uses semantic search and Google's Gemini API to generate context-based answers with source citations.

The system includes role-based document access, ensuring employees can only retrieve information they are authorized to view.

**Live Demo:** https://enterprise-kb-chatbot.onrender.com

## Project Status

- [x] PDF ingestion and document processing
- [x] Semantic search using ChromaDB
- [x] Gemini-powered question answering
- [x] Source citations
- [x] Role-based access control
- [x] FastAPI backend and login
- [x] Security testing
- [x] Docker-based deployment
- [x] Live deployment on Render
- [ ] Final live chatbot response verification

## Key Features

- **Document-based Q&A:** Answers employee questions using company PDF documents.
- **RAG Architecture:** Retrieves relevant document sections before generating an answer.
- **Role-based Access Control:** Supports employee and HR roles.
- **Secure Retrieval:** Restricts confidential document retrieval based on user roles.
- **Source Citations:** Provides document and page references for answers.
- **Context-aware Responses:** Uses retrieved company information rather than relying on general knowledge.
- **Fallback Responses:** Handles questions when relevant information is unavailable.
- **API Backend:** FastAPI endpoints for authentication and chatbot interactions.
- **Response Caching and Retry Handling:** Helps manage repeated requests and temporary AI service errors.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | FastAPI |
| AI / LLM | Google Gemini API |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Document Processing | PyPDF |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker, Render |
| Testing | Pytest |

## Architecture

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Ingestion:** `ingest.py` loads company PDF documents, extracts text, and splits it into sections.
2. **Index Building:** `build_index.py` generates embeddings and stores document chunks in ChromaDB.
3. **Query Processing:** The user submits a question through the frontend.
4. **Secure Retrieval:** `retrieve.py` searches for relevant chunks while applying role-based access filters.
5. **Answer Generation:** `chat.py` sends the retrieved context and question to Gemini.
6. **Response Delivery:** The API returns the generated answer with source information.

## Sample Company Documents

The project uses fictional company documents from **TechNova Solutions**:

- HR Policy
- IT Support Guide
- Onboarding Manual
- Confidential HR Salary Document

The confidential HR document is restricted to users with the HR role.

## Local Setup

### Prerequisites

- Python 3.10 or later
- Git
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/mahekshaik147/enterprise-kb-chatbot.git
cd enterprise-kb-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Never commit your actual API key to GitHub.

### 5. Build the Vector Index

```bash
python build_index.py
```

### 6. Start the Application

```bash
uvicorn api:app --reload
```

Open the application at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## Demo Credentials

| Role | Username | Password |
|---|---|---|
| Employee | asha | employee123 |
| HR | ravi | hr123 |

These are demonstration credentials for the fictional company environment. Do not use them in a production system.

## Evaluation and Testing

The retrieval pipeline was evaluated using a manually prepared question set.

| Metric | Result |
|---|---:|
| Top-1 Retrieval Accuracy | 15/15 |
| Top-3 Retrieval Accuracy | 15/15 |
| Answer Accuracy (LLM) | 11/11 |
| Correct Refusals (LLM) | 4/4 |
| Held-out Top-1 Retrieval | 10/10 |
| Held-out Top-3 Retrieval | 10/10 |

A separate security test suite was executed:

- **10 security tests passed.**
- Employee access to HR-only documents was restricted.
- HR users could retrieve authorized confidential information.

**Evaluation note:** These results come from a small, manually prepared dataset and are not a guarantee of real-world accuracy. The evaluation also used more than one Gemini model after free-tier quota limits were reached.

## Security and Access Control

Documents are assigned access levels:

- `all` — accessible to authorized employee and HR roles.
- `hr_only` — accessible only to HR users.

Access filtering is applied during vector database retrieval, preventing unauthorized document chunks from being passed to the language model.

The application also uses environment variables for API key configuration.

## Experiments and Improvements

### Chunk Size Optimization

A chunk size of 500 sometimes mixed unrelated topics and reduced retrieval accuracy. Reducing it to 250 improved retrieval for the test questions.

### Section-aware Chunking

Splitting documents at numbered headings before applying chunk-size limits improved Top-1 retrieval accuracy from 14/15 to 15/15 in the evaluation.

### Cross-encoder Reranking

The cross-encoder reranker did not improve retrieval accuracy consistently. It is therefore disabled by default.

### Retrieval Threshold

Vector distance alone was not reliable enough to identify every question outside the document collection. The system uses context-based answer generation and fallback handling.

## Limitations and Future Improvements

- The document collection is currently small and uses fictional company data.
- Demo credentials and in-memory sessions are intended for demonstration, not production authentication.
- The Gemini free-tier API has usage limits that can temporarily prevent answer generation.
- Larger evaluation datasets are needed to measure real-world performance.

Potential future improvements:

- Hybrid keyword and semantic search.
- Multi-turn conversational memory.
- Persistent user authentication and session storage.
- Larger document collections and evaluation datasets.
- Improved monitoring and production deployment configuration.

## Deployment

The application is containerized using Docker and deployed on Render.

**Live Application:** https://enterprise-kb-chatbot.onrender.com

The application uses environment variables configured in the deployment environment for sensitive credentials.

## Repository

GitHub: https://github.com/mahekshaik147/enterprise-kb-chatbot

## Note

All company names, users, and company documents used in this project are fictional and created for demonstration purposes.