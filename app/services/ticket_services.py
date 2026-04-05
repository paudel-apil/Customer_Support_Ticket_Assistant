import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

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
    raw = embedder.encode(text, normalize_embeddings=True).reshape(1, -1)
    reduced = umap_model.predict(raw) 
    return reduced[0].tolist()


def _get_category_from_points(points) -> str:
    if not points:
        return "Other / Rare Issues"

    categories = []
    for p in points:
        cluster_id = p.payload.get("cluster_id", -1)
        key = f"Meta-Group {cluster_id}"
        categories.append(final_cluster_names.get(key, "Other / Rare Issues"))

    return Counter(categories).most_common(1)[0][0]


def initialize_qdrant():
    try:
        qdrant.delete_collection("tickets")
    except:
        pass

    dimension = reduced_embeddings.shape[1] 
    qdrant.create_collection(
        collection_name="tickets",
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

    points = []
    for i in range(len(reduced_embeddings)):
        cluster_id = int(meta_cluster_ids[i])
        key = f"Meta-Group {cluster_id}"
        category = final_cluster_names.get(key, "Other / Rare Issues")

        points.append(PointStruct(
            id=i,
            vector=reduced_embeddings[i].tolist(), 
            payload={
                "ticket_id": i,
                "cluster_id": cluster_id,
                "category": category,
                "description": descriptions[i]
            }
        ))

    qdrant.upsert(collection_name="tickets", points=points)
    print(f"Qdrant seeded with {len(points)} points at dimension={dimension}.")


def predict_priority(text: str) -> str:
    cleaned = clean_text(text)
    X = tfidf_vectorizer.transform([cleaned])
    return prio_model.predict(X)[0]


def create_ticket(db: Session, ticket: TicketCreate) -> TicketResponse:
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

    qdrant.upsert(
        collection_name="tickets",
        points=[
            PointStruct(
                id=db_ticket.id,
                vector=reduced_vector,
                payload={
                    "ticket_id": db_ticket.id,
                    "title": ticket.title,
                    "description": ticket.description,
                    "category": category,
                    "priority": priority_pred,
                    "is_user_ticket": True
                }
            )
        ]
    )

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
