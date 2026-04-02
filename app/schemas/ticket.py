from pydantic import BaseModel
from typing import Optional

class TicketCreate(BaseModel):
    customer_id: str
    title: str
    description: str

class TicketResponse(BaseModel):
    id: int
    customer_id: str
    title: str
    description: str
    created_at: str
    category: Optional[str] = None
    priority: str 
    
    class Config:
        from_attributes = True

class TicketListResponse(BaseModel):
    tickets: list[TicketResponse]
    total: int