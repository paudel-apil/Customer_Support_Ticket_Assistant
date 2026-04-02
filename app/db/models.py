from sqlalchemy import Column, Integer, String, DateTime, Text
from app.core.config import Base  

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    title = Column(String)
    description = Column(Text)
    created_at = Column(DateTime)
    category = Column(String, default="Other / Rare Issues")
    priority = Column(String, default="medium")