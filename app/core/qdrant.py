from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.core.config import settings

qdrant = QdrantClient(":memory:")

def init_qdrant():
    """
    Initialization of qdrant on memory
    """
    try:
        collections = qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "tickets" not in collection_names:
            qdrant.create_collection(
                collection_name="tickets",
                vectors_config=VectorParams(
                    size=1024,  
                    distance=Distance.COSINE
                )
            )
            print("Qdrant collection 'tickets' created in-memory")
        else:
            print("Qdrant collection 'tickets' already exists")
    except Exception as e:
        print(f"Qdrant initialization: {e}")

init_qdrant()