from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from core.config import settings
from sentence_transformers import SentenceTransformer

qdrant = QdrantClient(settings.QDRANT_URL)

embedder = SentenceTransformer('ml/artifacts/metadata_embedder')
try:
    qdrant.delete_collection("tickets")
    print("Old collection deleted.")
except:
    print("No old collection to delete.")

print("Creating new collection with dim:", embedder.get_sentence_embedding_dimension())
qdrant.create_collection(
    collection_name="tickets",
    vectors_config=VectorParams(
        size=embedder.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    )
)
print("New collection ready.")