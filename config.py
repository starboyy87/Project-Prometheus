# Database configurations
NEO4J_URI = "bolt://localhost:7687" # Default URI for local Neo4j Desktop DB
NEO4J_USERNAME = "neo4j" # Default username
NEO4J_PASSWORD = "94409440"  # !!! REPLACE WITH YOUR NEO4J PASSWORD !!!

QDRANT_HOST = "localhost" # Not strictly needed for local file-based Qdrant
QDRANT_PORT = 6333      # Not strictly needed for local file-based Qdrant
QDRANT_PATH = "C:\\prometheus\\data\\qdrant"  # Local storage path for Qdrant

# Processing configurations
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100 # For potential future batch processing

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Paths
import os

# Define the base directory
BASE_DIR = "C:\\prometheus"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
LOG_DIR = os.path.join(BASE_DIR, "logs")