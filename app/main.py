from fastapi import FastAPI
from app.api.v1 import tickets
from app.core.config import settings
from app.db.models import Base
from app.db.session import engine
from app.services.ticket_services import initialize_qdrant

# Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)


app = FastAPI(title=settings.PROJECT_NAME, redirect_slashes=True)

app.include_router(tickets.router, prefix="/api/v1", tags=["tickets"])

@app.on_event("startup")
def startup_event():
    initialize_qdrant()
    from app.services.ticket_services import embedder
    embedder.encode("warmup", normalize_embeddings=True)
    print("Embedder warmed up.")

@app.get("/")
def root():
    return {"message": "Backend is running (Unsupervised mode)"}