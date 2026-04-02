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
│   └── artifacts/
│       ├── embeddings.npy
│       ├── merged_category.csv
│       └── cluster_names.json
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md

```
Quick Start (Local)

Start the services:
docker-compose up --build

Open:
FastAPI: http://localhost:8000/docs
Streamlit: http://localhost:8501


Manual Local Setup (without Docker)

Start Qdrant:Bashdocker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
Start FastAPI:Bashuvicorn app.main:app --reload --port 8000
Start Streamlit:Bashstreamlit run streamlit_app/app.py

API Endpoints

POST /tickets/ → Create a new ticket
POST /tickets/classify → Classify a ticket into one of 15 categories
GET /tickets/{id}/similar → Find similar tickets
GET /tickets/ → List all tickets

Categories
The system automatically classifies tickets into these 10 generic categories:

Security & Data Breach Concerns
Platform Operations & System Performance    
Data Security & Compliance                   
Financial Analytics & Investment Insights     
Platform Integration & Workflow Optimization  
Marketing, Strategy & Client Experience        
Billing, Payments & Subscription Management     
Hardware & End-User Device Support              
Network & Connectivity Operations              
General Operational Support                     
User Onboarding & Training    

How It Works

New ticket → Text is cleaned using the same preprocessing as training
Embedding is generated using BAAI/bge-large-en-v1.5
System finds top 5 most similar past tickets in Qdrant
Majority category from similar tickets is assigned
Ticket and embedding are stored for future similarity searches

Docker Commands
Bash# Start all services
docker-compose up --build

# Start in background
docker-compose up -d --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f
