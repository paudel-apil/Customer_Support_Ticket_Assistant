# Support Ticket Intelligence Assistant

An unsupervised ML system that automatically classifies support tickets into meaningful categories using semantic embeddings and vector similarity search — no labeled training data required.

---

## Features

| Feature | Description |
|---|---|
| Unsupervised Classification | Assigns tickets to categories via UMAP + HDBSCAN clustering — no labels needed |
| Semantic Similarity Search | Finds similar past tickets using Qdrant vector search |
| FastAPI Backend | Clean REST API with auto-generated Swagger docs |
| PostgreSQL | Persistent relational ticket storage |
| Qdrant | High-performance vector database for similarity search |
| Streamlit UI | Simple interface for creating and browsing tickets |
| Docker | Fully containerized — one command to run everything |

---

## Quick Start
### Run with Docker
```bash
git clone https://github.com/yourusername/support-ticket-assistant.git
cd support-ticket-assistant

cp .env.example .env   # fill in your values

docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## 🛠️ Local Development (without Docker)

**1. Start Qdrant and PostgreSQL**
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
docker run -d -p 5432:5432 --name postgres -e POSTGRES_PASSWORD=yourpassword postgres:16
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start FastAPI**
```bash
python -m uvicorn app.main:app --reload
```

**4. Start Streamlit**
```bash
streamlit run app/streamlit_app/app.py
```

---

## Project Structure

```text
support-ticket-assistant/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── core/                      # Config, Qdrant client
│   ├── db/                        # SQLAlchemy models & session
│   ├── schemas/                   # Pydantic schemas
│   ├── services/
│   │   ├── ticket_services.py     # Business logic, ML inference
│   │   └── preprocessing.py       # Text cleaning
│   └── api/v1/tickets.py          # Route handlers
├── ml/
│   └── artifacts/                 # Model files (not tracked in git)
│       ├── umap_surrogate.joblib  # MLP surrogate for UMAP inference
│       ├── reduced_embeddings.npy # 5D UMAP-reduced training embeddings
│       ├── embeddings.npy         # Original 1024D embeddings
│       ├── meta_cluster_ids.npy   # HDBSCAN cluster assignments
│       ├── final_cluster_names.json
│       ├── desc_cats.csv
│       ├── multi_lin_prio_model.joblib
│       └── multi_lin_tfidf_vec.joblib
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/tickets/` | Create and classify a new ticket |
| `GET` | `/api/v1/tickets/` | List all tickets (filterable by category/priority) |
| `GET` | `/api/v1/tickets/{id}/similar` | Find semantically similar tickets |
| `POST` | `/api/v1/tickets/classify` | Classify text without saving |

Full interactive docs at **http://localhost:8000/docs**

---

## Ticket Categories

The system automatically classifies tickets into these categories:

- Data Security & Compliance
- Financial Analytics & Investment Insights
- Billing, Payments & Subscription Management
- Platform Operations & System Performance
- Platform Integration & Workflow Optimization
- Security & Data Breach Concerns
- Network & Connectivity Operations
- Hardware & End-User Device Support
- User Onboarding & Training
- General Operational Support

---

## How It Works

### Training (Google Colab)
```
Raw ticket descriptions
        │
        ▼
Text Cleaning & Preprocessing
        │
        ▼
Embedding Generation (BAAI/bge-large-en-v1.5) → 1024D vectors
        │
        ▼
UMAP Dimensionality Reduction (cuML GPU) → 5D vectors
        │
        ▼
HDBSCAN Clustering → cluster labels
        │
        ▼
MLP Surrogate trained to approximate UMAP (1024D → 5D)
        │
        ▼
Artifacts saved: surrogate model, reduced embeddings, cluster labels
```

### Inference (Production)
```
New Ticket
        │
        ▼
Text Cleaning & Preprocessing
        │
        ▼
Embedding Generation (BAAI/bge-large-en-v1.5) → 1024D
        │
        ▼
MLP Surrogate → 5D vector (same space as training clusters)
        │
        ▼
Qdrant Similarity Search → top 3 nearest neighbors
        │
        ▼
Majority vote → Category assigned
        │
        ▼
TF-IDF + Linear Model → Priority predicted
        │
        ▼
Ticket + 5D embedding stored for future similarity searches
```

---

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Database**: PostgreSQL + SQLAlchemy
- **Vector DB**: Qdrant
- **Embeddings**: `BAAI/bge-large-en-v1.5` (SentenceTransformers)
- **Clustering**: cuML UMAP + HDBSCAN (Colab/GPU)
- **Inference**: MLP Surrogate (scikit-learn) + TF-IDF + Logistic Regression
- **Frontend**: Streamlit
- **Deployment**: Docker + Docker Compose

---

## Docker Commands

```bash
# Start all services
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f

# View logs for a specific service
docker compose logs -f backend

# Stop all services
docker compose down

# Stop and remove volumes (resets DB + Qdrant)
docker compose down -v
```

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=tickets
SECRET_KEY=your-secret-key
```
