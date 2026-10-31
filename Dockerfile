
FROM python:3.13-slim

WORKDIR /app

# Share one Hugging Face cache between image build and runtime.
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index inside the image
RUN python build_index.py

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api:app --workers 1 --host 0.0.0.0 --port ${PORT:-10000}"]
