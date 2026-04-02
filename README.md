#  Support Ticket Intelligence Assistant

An unsupervised ML system that automatically classifies support tickets into meaningful categories using semantic embeddings and vector similarity search no labeled training data required.

---

##  Features

| Feature | Description |
|---|---|
|  Unsupervised Classification | Assigns tickets to categories via embedding similarity — no labels needed |
|  Semantic Search | Finds similar past tickets using Qdrant vector search |
|  FastAPI Backend | Clean REST API with auto-generated Swagger docs |
|  PostgreSQL | Persistent relational ticket storage |
|  Qdrant | High-performance vector database for similarity search |
|  Streamlit UI | Simple interface for creating and browsing tickets |
|  Docker | Fully containerized — one command to run everything |

---

##  Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed

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

**1. Start Qdrant**
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start FastAPI**
```bash
uvicorn app.main:app --reload --port 8000
```

**4. Start Streamlit**
```bash
streamlit run streamlit_app/app.py
```

---

##  Project Structure
```text
support-ticket-assistant/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── core/                    # Config, Qdrant client
│   ├── db/                      # SQLAlchemy models & session
│   ├── schemas/                 # Pydantic schemas
│   ├── services/                # Business logic, ML inference
│   └── api/v1/tickets.py        # Route handlers
├── ml/
│   └── artifacts/               # Embeddings, cluster names, models (not tracked in git)
├── streamlit_app/
│   └── app.py                   # Streamlit frontend
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/tickets/` | Create and classify a new ticket |
| `GET` | `/api/v1/tickets/` | List all tickets (filterable by category/priority) |
| `GET` | `/api/v1/tickets/{id}/similar` | Find semantically similar tickets |
| `POST` | `/api/v1/tickets/classify` | Classify text without saving |

Full interactive docs at **http://localhost:8000/docs**

---

##  Ticket Categories

The system automatically classifies tickets into 10 categories:

-  Data Security & Compliance
-  Financial Analytics & Investment Insights
-  Billing, Payments & Subscription Management
-  Platform Operations & System Performance
-  Platform Integration & Workflow Optimization
-  Security & Data Breach Concerns
-  Network & Connectivity Operations
-  Hardware & End-User Device Support
-  User Onboarding & Training
-  General Operational Support

---

##  How It Works
```
New Ticket
    │
    ▼
Text Cleaning & Preprocessing
    │
    ▼
Embedding Generation (BAAI/bge-large-en-v1.5)
    │
    ▼
Qdrant Similarity Search (top-3 nearest neighbors)
    │
    ▼
Category assigned from nearest neighbor cluster
    │
    ▼
Priority predicted (TF-IDF + Linear model)
    │
    ▼
Ticket + Embedding stored → available for future searches
```

---

##  Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Database**: PostgreSQL + SQLAlchemy
- **Vector DB**: Qdrant
- **Embeddings**: `BAAI/bge-large-en-v1.5` (SentenceTransformers)
- **Priority Model**: TF-IDF + Logistic Regression
- **Frontend**: Streamlit
- **Deployment**: Docker + Docker Compose

---

##  Docker Commands
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

##  Environment Variables

Create a `.env` file based on `.env.example`:
```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=tickets
SECRET_KEY=your-secret-key
```

---

