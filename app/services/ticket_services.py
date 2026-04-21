import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
from joblib import load

from app.db.models import Ticket
from app.schemas.ticket import TicketCreate, TicketResponse
from app.core.qdrant import qdrant
from app.services.preprocessing import clean_text

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

umap_model = load("ml/artifacts/umap_surrogate.joblib")
reduced_embeddings = np.load("ml/artifacts/reduced_embeddings.npy")
meta_cluster_ids = np.load("ml/artifacts/meta_cluster_ids.npy")

with open("ml/artifacts/final_cluster_names.json", "r") as f:
    final_cluster_names = json.load(f)

prio_model = load("ml/artifacts/multi_lin_prio_model.joblib")
tfidf_vectorizer = load("ml/artifacts/multi_lin_tfidf_vec.joblib")

desc_cat_df = pd.read_csv("ml/artifacts/desc_cats.csv")
descriptions = desc_cat_df['description'].tolist()


def _embed_and_reduce(text: str) -> list:
    """Generate a 5D reduced embedding for the given text."""
    raw = embedder.encode(text, normalize_embeddings=True).reshape(1, -1)
    reduced = umap_model.predict(raw)
    return reduced[0].tolist()


def _get_category_from_points(points) -> str:
    """Majority vote category from nearest neighbor points."""
    if not points:
        return "Other / Rare Issues"

    categories = []
    for p in points:
        cluster_id = p.payload.get("cluster_id", -1)
        key = f"Meta-Group {cluster_id}"
        categories.append(final_cluster_names.get(key, "Other / Rare Issues"))

    return Counter(categories).most_common(1)[0][0]


def initialize_qdrant():
    """
    Initialize Qdrant collections.
    - tickets: training data for classification
    - user_tickets: production tickets for similar/search
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    print(f"Existing collections: {existing}")

    if "tickets" in existing:
        info = qdrant.get_collection("tickets")
        print(f"tickets OK — dim={info.config.params.vectors.size}, points={info.points_count}")
    else:
        print("Creating tickets collection and seeding training data...")
        dimension = reduced_embeddings.shape[1]
        qdrant.create_collection(
            collection_name="tickets",
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )

        BATCH_SIZE = 500
        batch = []
        for i in range(len(reduced_embeddings)):
            cluster_id = int(meta_cluster_ids[i])
            key = f"Meta-Group {cluster_id}"
            category = final_cluster_names.get(key, "Other / Rare Issues")

            batch.append(PointStruct(
                id=i,
                vector=reduced_embeddings[i].tolist(),
                payload={
                    "ticket_id": i,
                    "cluster_id": cluster_id,
                    "category": category,
                    "description": descriptions[i]
                }
            ))

            if len(batch) == BATCH_SIZE:
                qdrant.upsert(collection_name="tickets", points=batch)
                print(f"Seeded {i+1}/{len(reduced_embeddings)}")
                batch = []

        if batch:
            qdrant.upsert(collection_name="tickets", points=batch)

        print(f"tickets seeded with {len(reduced_embeddings)} points.")

    if "user_tickets" in existing:
        info = qdrant.get_collection("user_tickets")
        print(f"user_tickets OK — dim={info.config.params.vectors.size}, points={info.points_count}")
    else:
        print("Creating user_tickets collection...")
        dimension = reduced_embeddings.shape[1]
        qdrant.create_collection(
            collection_name="user_tickets",
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
        print("user_tickets collection created.")


def predict_priority(text: str) -> str:
    """Predict ticket priority using TF-IDF + Logistic Regression."""
    cleaned = clean_text(text)
    X = tfidf_vectorizer.transform([cleaned])
    return prio_model.predict(X)[0]


def create_ticket(db: Session, ticket: TicketCreate) -> TicketResponse:
    """
    Create, classify and store a new support ticket.
    - Classification uses tickets collection (training data)
    - Storage uses user_tickets collection (production data)
    """
    full_text = f"{ticket.title} {ticket.description}"
    cleaned_text = clean_text(full_text)

    reduced_vector = _embed_and_reduce(cleaned_text)

    result = qdrant.query_points(
        collection_name="tickets",
        query=reduced_vector,
        limit=3,
        with_payload=True
    )
    points = result.points if hasattr(result, 'points') else result

    category = _get_category_from_points(points)
    priority_pred = predict_priority(full_text)

    db_ticket = Ticket(
        customer_id=ticket.customer_id,
        title=ticket.title,
        description=ticket.description,
        created_at=datetime.utcnow(),
        category=category,
        priority=priority_pred
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)

    try:
        qdrant.upsert(
            collection_name="user_tickets",
            points=[
                PointStruct(
                    id=db_ticket.id,
                    vector=reduced_vector,
                    payload={
                        "ticket_id": db_ticket.id,
                        "title": ticket.title,
                        "description": ticket.description,
                        "category": category,
                        "priority": priority_pred
                    }
                )
            ]
        )
        print(f"Ticket {db_ticket.id} pushed to user_tickets")
    except Exception as e:
        print(f"user_tickets upsert failed for ticket {db_ticket.id}: {e}")

    return TicketResponse(
        id=db_ticket.id,
        customer_id=db_ticket.customer_id,
        title=db_ticket.title,
        description=db_ticket.description,
        created_at=db_ticket.created_at.isoformat(),
        category=category,
        priority=priority_pred
    )


def classify_ticket(full_text: str):
    """Classify a ticket without saving it."""
    cleaned_text = clean_text(full_text)
    reduced_vector = _embed_and_reduce(cleaned_text)

    result = qdrant.query_points(
        collection_name="tickets",
        query=reduced_vector,
        limit=3,
        with_payload=True
    )
    points = result.points if hasattr(result, 'points') else result
    category = _get_category_from_points(points)
    priority_pred = predict_priority(full_text)

    return {"category": category, "priority": priority_pred}


def search_by_keywords(keywords_text: str, limit: int = 10):
    """Search production tickets by keywords or natural language."""
    cleaned = clean_text(keywords_text)
    query_embedding = _embed_and_reduce(cleaned)

    result = qdrant.query_points(
        collection_name="user_tickets",
        query=query_embedding,
        limit=limit,
        with_payload=True
    )

    points = result.points if hasattr(result, 'points') else result
    tickets = []

    for point in points:
        if not point.payload:
            continue
        tickets.append({
            "ticket_id": point.payload.get("ticket_id"),
            "title": point.payload.get("title"),
            "description": point.payload.get("description"),
            "category": point.payload.get("category", "Other / Rare Issues"),
            "similarity_score": round(point.score, 4)
        })

    return tickets