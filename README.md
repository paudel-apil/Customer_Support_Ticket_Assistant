# Support Ticket Intelligence Assistant

A modern, unsupervised support ticket system that automatically organizes tickets into meaningful categories using embeddings and similarity search.

## Features

- **Unsupervised Classification**: Automatically assigns tickets to one of 15 intelligent categories using embeddings + similarity search
- **Semantic Similarity Search**: Find similar past tickets using Qdrant vector database
- **FastAPI Backend**: Clean REST API with Swagger documentation
- **PostgreSQL**: Persistent ticket storage
- **Qdrant**: High-performance vector search
- **Streamlit UI**: Simple and clean interface for creating and viewing tickets
- **Docker Support**: Full containerized deployment

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Database**: PostgreSQL + SQLAlchemy
- **Vector DB**: Qdrant
- **Embeddings**: `BAAI/bge-large-en-v1.5`
- **Frontend**: Streamlit
- **Deployment**: Docker + Docker Compose

## Project Structure

```text
Support_Ticket_Intelligence_Assistant/
├── app/
│   ├── main.py
│   ├── core/
│   ├── db/
│   ├── schemas/
│   ├── services/
│   └── api/
├── streamlit_app/
│   └── app.py
├── ml/
│   └── clustering/
│       ├── embeddings.npy
│       ├── merged_category.csv
│       └── cluster_names.json
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
