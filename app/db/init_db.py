from app.db.session import engine
from app.core.config import Base
from app.db.models import Ticket  

def init_database():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")

if __name__ == "__main__":
    init_database()