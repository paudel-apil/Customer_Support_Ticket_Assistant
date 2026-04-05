from pydantic import BaseModel
from typing import Optional

class TicketCreate(BaseModel):
    """
    Schema for creating a new support ticket. 
    
    This model defines the required input fields when a user submits
    a ticket via the API. 
    """
    customer_id: str
    title: str
    description: str

class TicketResponse(BaseModel):
    """
    Schema representing a ticket returned by the API.
    
    This model is used for responses after ticket creation or retrieval,
    including classification results.
    """
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
    """
    Schema for returning a list of tickets.
    """
    tickets: list[TicketResponse]
    total: int