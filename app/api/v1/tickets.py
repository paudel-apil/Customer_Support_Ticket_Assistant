from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Ticket
from app.schemas.ticket import TicketCreate, TicketResponse, TicketListResponse
from app.services.ticket_services import create_ticket, classify_ticket, qdrant, search_by_keywords
from app.services.ticket_services import _embed_and_reduce

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("/", response_model=TicketResponse, status_code=201)
def create_ticket_endpoint(ticket: TicketCreate, db: Session = Depends(get_db)):
    """Create and classify a new support ticket."""
    try:
        return create_ticket(db, ticket)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
def classify_ticket_endpoint(ticket: TicketCreate):
    """Classify a ticket without saving it."""
    try:
        full_text = f"{ticket.title} {ticket.description}"
        return classify_ticket(full_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_tickets_endpoint(data: dict):
    """Search production tickets by keywords or natural language."""
    try:
        query = data.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query text is required")
        results = search_by_keywords(query, limit=10)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=TicketListResponse)
def list_tickets_endpoint(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None)
):
    """List all tickets with optional filters."""
    try:
        query = db.query(Ticket)
        if category:
            query = query.filter(Ticket.category == category)
        if priority:
            query = query.filter(Ticket.priority == priority)

        tickets = query.all()
        ticket_responses = [
            TicketResponse(
                id=t.id,
                customer_id=t.customer_id,
                title=t.title,
                description=t.description,
                created_at=t.created_at.isoformat() if t.created_at else None,
                category=t.category,
                priority=t.priority
            )
            for t in tickets
        ]
        return TicketListResponse(tickets=ticket_responses, total=len(ticket_responses))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticket_id}/similar")
def get_similar_tickets_endpoint(
    ticket_id: int,
    db: Session = Depends(get_db),
    limit: int = 3
):
    """Find similar production tickets using vector search."""

    retrieved = qdrant.retrieve(
        collection_name="user_tickets",
        ids=[ticket_id],
        with_vectors=True,
        with_payload=True
    )

    if not retrieved or not retrieved[0].vector:
        db_ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not db_ticket:
            return []
        full_text = f"{db_ticket.title} {db_ticket.description}"
        query_vector = _embed_and_reduce(full_text)
    else:
        query_vector = retrieved[0].vector

    result = qdrant.query_points(
        collection_name="user_tickets",
        query=query_vector,
        limit=limit + 1,
        with_payload=True
    )

    points = result.points if hasattr(result, "points") else result
    similar = []

    for p in points:
        t_id = p.payload.get("ticket_id")
        if t_id == ticket_id:
            continue

        db_ticket_similar = db.query(Ticket).filter(Ticket.id == t_id).first()
        similar.append({
            "ticket_id": t_id,
            "title": p.payload.get("title"),
            "description": db_ticket_similar.description if db_ticket_similar else p.payload.get("description"),
            "category": db_ticket_similar.category if db_ticket_similar else p.payload.get("category")
        })

        if len(similar) == limit:
            break

    return similar