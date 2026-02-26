# RAG Chatbot Backend for Published Books

A production-grade RAG (Retrieval-Augmented Generation) chatbot backend that enables readers to ask questions about published books and receive accurate, citation-backed answers.

**Features:**
- Ingest book content (markdown/text) into searchable chunks
- Full-book RAG mode for general questions
- Selected-text mode for focused passage Q&A
- Strict grounding with zero hallucination tolerance
- Cohere-only LLM integration (via OpenAI Compatibility API)
- Free-tier architecture (Qdrant + Neon + Cohere)

## Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **Vector Store**: Qdrant Cloud
- **Metadata Store**: Neon Serverless Postgres
- **LLM**: Cohere (command-r-plus) via OpenAI Compatibility API
- **Agents**: OpenAI Agents SDK

## Quick Start

### Prerequisites

1. Python 3.11+ installed
2. Neon Postgres database ([Create free account](https://neon.tech))
3. Qdrant Cloud cluster ([Create free account](https://cloud.qdrant.io))
4. Cohere API key ([Get API key](https://cohere.com/api-keys))

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-native-book/backend

# Install dependencies (using pip)
pip install -r requirements.txt

# OR using uv (recommended)
pip install uv
uv pip install -e .
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# - NEON_DATABASE_URL: Your Neon Postgres connection string
# - QDRANT_CLUSTER_ID: Your Qdrant cluster ID
# - QDRANT_API_ENDPOINT: Your Qdrant API endpoint
# - QDRANT_API_KEY: Your Qdrant API key
# - COHERE_API_KEY: Your Cohere API key
```

### Database Setup

```bash
# Run migrations to create database schema
python -m app.db.migrations
```

### Ingest Book Content

```bash
# Ingest a book from markdown/text files
python scripts/ingest_book.py --path /path/to/book/folder

# Example: Ingest sample book
python scripts/ingest_book.py --path ./sample_book
```

### Start Server

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Start all services (API + Qdrant)
docker-compose up --build

# API will be available at http://localhost:8000
# Qdrant dashboard at http://localhost:6333/dashboard
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "neon_connected": true,
  "cohere_connected": true,
  "collection_name": "book_chunks"
}
```

#### 2. Ingest Book Content
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "book_path": "/path/to/your/book",
    "format": "md"
  }'
```

**Response**:
```json
{
  "book_id": "abc123...",
  "chunks_created": 150,
  "status": "success",
  "message": "Ingestion complete",
  "processing_time_seconds": 12.34
}
```

#### 3. Chat (Full-Book RAG Mode)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "max_chunks": 5
  }'
```

**Response**:
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "citations": [
    {
      "chunk_id": "abc123",
      "text": "RAG stands for Retrieval-Augmented...",
      "source": "chapter1.md",
      "chapter": "Introduction to RAG",
      "section": null,
      "score": 0.85
    }
  ],
  "mode": "full_book",
  "chunks_retrieved": 3,
  "latency_ms": 1234,
  "model_used": "command-r-plus"
}
```

#### 4. Chat with Selected Text
```bash
curl -X POST http://localhost:8000/api/v1/chat/selected \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this passage in detail",
    "selected_text": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with generative AI...",
    "max_chunks": 3
  }'
```

**Response**:
```json
{
  "answer": "The passage explains that RAG combines retrieval with generation...",
  "citations": [
    {
      "chunk_id": "selected",
      "text": "RAG stands for Retrieval-Augmented...",
      "source": "User Selection",
      "chapter": null,
      "section": null,
      "score": 1.0
    },
    {
      "chunk_id": "def456",
      "text": "Additional context...",
      "source": "chapter2.md",
      "chapter": "How RAG Works",
      "section": "Process",
      "score": 0.72
    }
  ],
  "mode": "selected_text",
  "chunks_retrieved": 2,
  "latency_ms": 987,
  "model_used": "command-r-plus"
}
```

#### 5. List Chunks (Debugging)
```bash
curl "http://localhost:8000/api/v1/chunks?book_id=abc123&limit=10&offset=0"
```

**Response**:
```json
{
  "chunks": [
    {
      "chunk_id": "abc123",
      "source_file": "chapter1.md",
      "chapter": "Introduction to RAG",
      "position": 0
    }
  ],
  "count": 10
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_chunking.py

# Run with coverage
pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy app/

# Linting and formatting
ruff check app/
ruff format app/
```

## Deployment

### Railway (Recommended - Easiest)

1. **Create accounts** (if you haven't already):
   - [Neon](https://neon.tech) - Create project, get connection string
   - [Qdrant Cloud](https://cloud.qdrant.io) - Create free cluster, get API key
   - [Cohere](https://cohere.com) - Create API key

2. **Deploy to Railway**:
   ```bash
   # Install Railway CLI
   npm install -g railway

   # Login
   railway login

   # Link your repo
   railway link

   # Deploy
   railway up
   ```

3. **Set environment variables** in Railway dashboard:
   - `NEON_DATABASE_URL`: Your Neon connection string
   - `QDRANT_CLUSTER_ID`: Your Qdrant cluster ID
   - `QDRANT_API_ENDPOINT`: Your Qdrant API endpoint
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `COHERE_API_KEY`: Your Cohere API key

4. **Deploy!** Your API will be live at: `https://your-app.railway.app`

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Create new app
flyctl launch

# Set environment variables
flyctl secrets set NEON_DATABASE_URL="..."
flyctl secrets set COHERE_API_KEY="..."
flyctl secrets set QDRANT_API_KEY="..."

# Deploy
flyctl deploy
```

### Render Free Tier

1. Push code to GitHub
2. Log into [Render](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Select `backend/` directory
6. Set environment variables
7. Click "Deploy Web Service"

### Environment Variables Reference

Copy these values from your respective service dashboards:

**Neon Postgres**:
- Dashboard → Projects → Your Project → Connection Details → Connection String
- Format: `postgresql://user:password@ep-xxx.aws.neon.tech/neondb`

**Qdrant Cloud**:
- Dashboard → Your Cluster → API Key
- Cluster ID: Visible in cluster URL
- API Endpoint: `https://<cluster-id>.qdrant.io`

**Cohere**:
- Dashboard → API Keys
- Create new key or use existing key

### Docker Deployment

```bash
# Build and run locally with Docker
docker-compose up --build

# Or build container
docker build -t rag-chatbot-backend .

# Run container
docker run -p 8000:8000 \
  -e NEON_DATABASE_URL="..." \
  -e QDRANT_API_KEY="..." \
  -e COHERE_API_KEY="..." \
  rag-chatbot-backend
```

## Architecture

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config/              # Configuration
│   ├── models/              # Pydantic schemas
│   ├── agents/              # OpenAI Agents SDK integration
│   ├── services/            # Qdrant, Neon, Cohere clients
│   ├── api/                 # FastAPI routes
│   └── db/                  # Database connections
├── scripts/
│   └── ingest_book.py       # CLI ingestion script
└── tests/                   # Contract, integration, unit tests
```

## Constraints & Compliance

- ✅ **Cohere-Only**: All LLM calls via Cohere Compatibility API
- ✅ **Free-Tier**: Qdrant ≤ 1GB, Neon free compute, Cohere free tier
- ✅ **Strict Grounding**: Zero hallucination tolerance
- ✅ **Type Safety**: 100% type hints with mypy strict mode
- ✅ **Async-First**: All I/O operations use async/await

## License

MIT

## Support

For issues and questions, please open a GitHub issue.
