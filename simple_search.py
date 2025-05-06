# File: C:\prometheus\scripts\simple_search.py

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    import config # Try importing config from the parent directory
except ModuleNotFoundError:
    print("Error: Could not find config.py. Make sure it's in C:\\prometheus\\")
    sys.exit(1)

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Setup logging
os.makedirs(config.LOG_DIR, exist_ok=True)
log_file_path = os.path.join(config.LOG_DIR, f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

# --- Global Variables (Lazy Initialization) ---
embedding_model = None
neo4j_driver = None
qdrant_client = None

def initialize_components():
    """Initialize models and clients if not already done."""
    global embedding_model, neo4j_driver, qdrant_client

    if embedding_model is None:
        logger.info(f"Loading sentence transformer model: {config.EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        logger.info("Sentence transformer model loaded.")

    if neo4j_driver is None:
        logger.info("Initializing Neo4j driver...")
        try:
            neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            neo4j_driver.verify_connectivity()
            logger.info("Neo4j driver initialized and connection verified.")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    if qdrant_client is None:
        logger.info("Initializing Qdrant client...")
        # Ensure the path exists, but Qdrant client handles creation on first use too
        os.makedirs(config.QDRANT_PATH, exist_ok=True)
        qdrant_client = QdrantClient(path=config.QDRANT_PATH)
        logger.info(f"Qdrant client initialized using path: {config.QDRANT_PATH}")


def search(query, top_k=5):
    """Search for relevant chunks using vector search (Qdrant) and enrich with Neo4j data."""
    global embedding_model, neo4j_driver, qdrant_client
    collection_name = "documents"

    try:
        # Initialize components if needed
        initialize_components()

        # --- Generate Query Embedding ---
        logger.info(f"Generating embedding for query: '{query}'")
        query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
        logger.info("Query embedding generated.")

        # --- Search in Qdrant ---
        logger.info(f"Searching Qdrant collection '{collection_name}' for top {top_k} results...")
        try:
            vector_search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True # Ensure payload is returned
            )
            logger.info(f"Qdrant search returned {len(vector_search_results)} results.")
        except Exception as e:
             # Handle case where collection might not exist yet
             if "not found" in str(e).lower():
                  logger.error(f"Qdrant collection '{collection_name}' not found. Have any documents been processed yet?")
                  return []
             else:
                  logger.error(f"Qdrant search failed: {e}")
                  raise

        # --- Process and Enrich Results ---
        results = []
        if not vector_search_results:
             logger.info("No relevant chunks found in Qdrant.")
             return []

        # Extract doc_ids needed for Neo4j query
        doc_ids = list(set([result.payload.get("doc_id") for result in vector_search_results if result.payload]))
        doc_info_map = {}

        # Fetch document info from Neo4j in one batch query
        if doc_ids:
            logger.info(f"Fetching document details from Neo4j for {len(doc_ids)} document IDs...")
            try:
                with neo4j_driver.session(database="neo4j") as session:
                    neo_results = session.run(
                        """
                        MATCH (d:Document)
                        WHERE d.id IN $doc_ids
                        RETURN d.id as id, d.name as name, d.path as path
                        """,
                        doc_ids=doc_ids
                    )
                    for record in neo_results:
                        doc_info_map[record["id"]] = {"name": record["name"], "path": record["path"]}
                logger.info(f"Fetched details for {len(doc_info_map)} documents from Neo4j.")
            except Exception as e:
                logger.error(f"Neo4j error fetching document details: {e}")
                # Continue without enrichment if Neo4j fails, or raise error
                # For now, we'll log the error and proceed

        # Combine Qdrant results with Neo4j data
        for result in vector_search_results:
            payload = result.payload
            if not payload:
                logger.warning(f"Qdrant result missing payload: {result.id}")
                continue

            chunk_id = payload.get("chunk_id", result.id) # Use Qdrant ID if chunk_id missing
            chunk_text = payload.get("text", "")
            doc_id = payload.get("doc_id")
            score = result.score

            doc_info = doc_info_map.get(doc_id, {}) # Get enrichment data
            doc_name = doc_info.get("name", payload.get("doc_name", "Unknown")) # Fallback to payload doc_name
            doc_path = doc_info.get("path", "Unknown")

            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_path": doc_path,
                "text": chunk_text,
                "score": score
            })

        # Sort results by score (descending) as Qdrant already does this
        # results.sort(key=lambda x: x['score'], reverse=True) # Usually redundant

        return results

    except Exception as e:
        logger.error(f"Error during search for query '{query}': {e}", exc_info=True)
        raise # Re-raise after logging

def close_resources():
    """Close Neo4j driver if it was initialized."""
    global neo4j_driver
    if neo4j_driver and hasattr(neo4j_driver, 'close'):
        logger.info("Closing Neo4j driver.")
        neo4j_driver.close()
        neo4j_driver = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search the graph RAG system")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    try:
        results = search(args.query, args.top_k)
        print(f"\n--- Search Results for: '{args.query}' ---")

        if results:
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Score: {result['score']:.4f}")
                print(f"Document: {result['doc_name']} (ID: {result['doc_id']})")
                # print(f"Path: {result['doc_path']}") # Optional: Show path
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Text: {result['text'][:300]}...") # Show snippet
                print("-" * 20)
        else:
            print("\nNo relevant results found.")

        print(f"\nSearch log file: {log_file_path}")

    except Exception as e:
        print(f"\nSearch failed. Check logs for details: {log_file_path}")
        # Error is already logged in search function
    finally:
        close_resources() # Ensure resources are closed when run as script