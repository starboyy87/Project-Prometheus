# File: C:\prometheus\scripts\simple_processor.py

import os
import sys
import logging
from datetime import datetime
import uuid

# Add parent directory to path to import config
# Adjust path separators for Windows if necessary, though os.path.join handles it
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    import config # Try importing config from the parent directory
except ModuleNotFoundError:
    print("Error: Could not find config.py. Make sure it's in C:\\prometheus\\")
    sys.exit(1)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import spacy
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance # Ensure PointStruct is imported


# Setup logging
os.makedirs(config.LOG_DIR, exist_ok=True)
log_file_path = os.path.join(config.LOG_DIR, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

# --- Global Variables (Lazy Initialization) ---
nlp_model = None
embedding_model = None
neo4j_driver = None
qdrant_client = None

def initialize_components():
    """Initialize models and clients if not already done."""
    global nlp_model, embedding_model, neo4j_driver, qdrant_client

    if nlp_model is None:
        logger.info("Loading spaCy model...")
        try:
            nlp_model = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded.")
        except OSError:
             logger.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
             raise

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
            # Verify connection
            neo4j_driver.verify_connectivity()
            logger.info("Neo4j driver initialized and connection verified.")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    if qdrant_client is None:
        logger.info("Initializing Qdrant client...")
        os.makedirs(config.QDRANT_PATH, exist_ok=True)
        qdrant_client = QdrantClient(path=config.QDRANT_PATH)
        logger.info(f"Qdrant client initialized using path: {config.QDRANT_PATH}")

def setup_databases():
    """Ensure necessary collections and constraints exist."""
    global neo4j_driver, qdrant_client
    if qdrant_client is None or neo4j_driver is None:
        initialize_components() # Ensure clients are ready

    # Setup Qdrant Collection
    collection_name = "documents"
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            logger.info(f"Creating '{collection_name}' collection in Qdrant...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
            )
            logger.info(f"'{collection_name}' collection created in Qdrant.")
        else:
             logger.info(f"Qdrant collection '{collection_name}' already exists.")

    except Exception as e:
        logger.error(f"Error setting up Qdrant collection: {e}")
        raise

    # Setup Neo4j Schema (Constraints and Indexes)
    logger.info("Setting up Neo4j schema (constraints)...")
    try:
        with neo4j_driver.session(database="neo4j") as session: # Use the default database for schema setup
            # Create constraints for unique nodes
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            # Unique entity identified by name AND type
            session.run("CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE")
            # Consider adding indexes for faster lookups if needed later
            # session.run("CREATE INDEX chunk_index IF NOT EXISTS FOR (c:Chunk) ON (c.index)")
            # session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        logger.info("Neo4j schema setup complete.")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {e}")
        raise

def process_text_file(file_path):
    """Process a text file and store it in Neo4j and Qdrant"""
    global nlp_model, embedding_model, neo4j_driver, qdrant_client

    try:
        logger.info(f"Processing file: {file_path}")

        # Initialize components if not already done
        initialize_components()
        # Ensure databases are set up
        setup_databases()

        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
             logger.error(f"Error reading file {file_path}: {e}")
             raise

        # Generate document ID and name
        doc_name = os.path.basename(file_path)
        # Use a consistent UUID for the document ID
        doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path)) # Use file path for consistency
        doc_id = f"doc_{doc_uuid}"
        logger.info(f"Generated Document ID: {doc_id} for Name: {doc_name}")

        # --- Neo4j: Create Document Node ---
        try:
            with neo4j_driver.session(database="neo4j") as session: # Specify database if needed
                # Use MERGE to avoid creating duplicate document nodes if re-processed
                result = session.run(
                    """
                    MERGE (d:Document {id: $id})
                    ON CREATE SET
                        d.name = $name,
                        d.path = $path,
                        d.created_at = datetime(),
                        d.last_processed_at = datetime()
                    ON MATCH SET
                        d.last_processed_at = datetime()
                    RETURN d.id as docId
                    """,
                    id=doc_id,
                    name=doc_name,
                    path=file_path # Store the original path
                )
                logger.info(f"Neo4j: Ensured Document node exists with ID: {result.single()['docId']}")
                # Optional: Clean up old chunks/relationships if re-processing
                # session.run("MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk) DETACH DELETE c", doc_id=doc_id)
                # logger.info(f"Neo4j: Cleaned up old chunks for {doc_id} (if any)")

        except Exception as e:
            logger.error(f"Neo4j Error creating document node {doc_id}: {e}")
            raise

        # --- Text Splitting ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(text_content)
        logger.info(f"Document split into {len(chunks)} chunks")

        if not chunks:
            logger.warning(f"No text chunks generated for document {doc_name}. Skipping further processing.")
            return None

        # --- Process Chunks in Batches (Example: Batch size 100) ---
        total_entities = 0
        qdrant_points = []
        chunk_batch_size = config.BATCH_SIZE # Use config for batch size

        for i in range(0, len(chunks), chunk_batch_size):
            batch_chunks = chunks[i:i + chunk_batch_size]
            batch_chunk_ids = [f"{doc_id}_chunk_{j}" for j in range(i, i + len(batch_chunks))]
            logger.info(f"Processing batch {i // chunk_batch_size + 1}: Chunks {i} to {i + len(batch_chunks) - 1}")

            # --- Embeddings ---
            logger.info(f"Generating embeddings for {len(batch_chunks)} chunks...")
            embeddings = embedding_model.encode(batch_chunks, normalize_embeddings=True).tolist()
            logger.info("Embeddings generated.")

            # --- Prepare Qdrant Points ---
            batch_qdrant_points = []
            for k, chunk_text in enumerate(batch_chunks):
                chunk_index = i + k
                chunk_id = batch_chunk_ids[k]
                batch_qdrant_points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=embeddings[k],
                        payload={
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "doc_name": doc_name, # Add doc name to payload
                            "text": chunk_text,
                            "index": chunk_index
                        }
                    )
                )
            qdrant_points.extend(batch_qdrant_points) # Collect points for batch upsert later

            # --- Neo4j: Create Chunk Nodes and Relationships ---
            logger.info("Neo4j: Creating chunk nodes and Document-CONTAINS->Chunk relationships...")
            try:
                with neo4j_driver.session(database="neo4j") as session:
                     # Use UNWIND for batching Cypher operations
                    session.run(
                        """
                        MATCH (d:Document {id: $doc_id})
                        UNWIND $chunk_data as chunk
                        MERGE (c:Chunk {id: chunk.id})
                        ON CREATE SET
                            c.text = chunk.text,
                            c.index = chunk.index
                        MERGE (d)-[:CONTAINS]->(c)
                        """,
                        doc_id=doc_id,
                        chunk_data=[
                            {"id": batch_chunk_ids[k], "text": chunk_text, "index": i + k}
                            for k, chunk_text in enumerate(batch_chunks)
                        ]
                    )
                logger.info(f"Neo4j: Processed {len(batch_chunks)} chunks in this batch.")
            except Exception as e:
                logger.error(f"Neo4j error creating chunk nodes batch starting at index {i}: {e}")
                # Decide if you want to continue with the next batch or raise exception
                continue # Or raise e

            # --- NER and Neo4j Entity/Relationship Creation ---
            logger.info("Extracting entities and creating Neo4j relationships...")
            batch_entities = 0
            for k, chunk_text in enumerate(batch_chunks):
                chunk_id = batch_chunk_ids[k]
                try:
                    doc = nlp_model(chunk_text)
                    entities_to_create = []
                    for ent in doc.ents:
                        # Basic filtering (optional)
                        if len(ent.text.strip()) > 1: # Avoid single characters/spaces
                            entities_to_create.append({"name": ent.text, "type": ent.label_})
                            batch_entities += 1

                    if entities_to_create:
                         # Neo4j: Merge entities and Chunk-MENTIONS->Entity relationships
                        with neo4j_driver.session(database="neo4j") as session:
                            session.run(
                                """
                                MATCH (c:Chunk {id: $chunk_id})
                                UNWIND $entities as entity_data
                                MERGE (e:Entity {name: entity_data.name, type: entity_data.type})
                                MERGE (c)-[r:MENTIONS]->(e)
                                ON CREATE SET r.count = 1
                                ON MATCH SET r.count = r.count + 1
                                """,
                                chunk_id=chunk_id,
                                entities=entities_to_create
                            )
                except Exception as e:
                    logger.error(f"Error processing entities for chunk {chunk_id}: {e}")
                    continue # Continue with next chunk in batch

            logger.info(f"Extracted {batch_entities} entities in this batch.")
            total_entities += batch_entities

        # --- Qdrant: Batch Upsert ---
        if qdrant_points:
            logger.info(f"Qdrant: Upserting {len(qdrant_points)} points (all batches)...")
            try:
                 # Use upsert for potentially overwriting existing chunks if doc is re-processed
                 qdrant_client.upsert(
                    collection_name="documents",
                    points=qdrant_points,
                    wait=True # Wait for operation to complete for consistency
                 )
                 logger.info("Qdrant: Upsert complete.")
            except Exception as e:
                 logger.error(f"Qdrant error upserting points: {e}")
                 raise
        else:
             logger.warning("No points to upsert to Qdrant.")


        logger.info(f"Document {doc_id} ('{doc_name}') processed successfully.")
        logger.info(f"Total Chunks: {len(chunks)}, Total Entities Found: {total_entities}")

        return {
            "status": "success",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunks": len(chunks),
            "entities": total_entities,
            "log_file": log_file_path
        }

    except Exception as e:
        logger.error(f"FATAL Error processing document {file_path}: {e}", exc_info=True) # Log traceback
        # No need to close drivers here if they are meant to be reused, manage lifecycle separately
        raise # Re-raise the exception after logging

def close_resources():
    """Close Neo4j driver if it was initialized."""
    global neo4j_driver
    if neo4j_driver and hasattr(neo4j_driver, 'close'):
        logger.info("Closing Neo4j driver.")
        neo4j_driver.close()
        neo4j_driver = None
    # Qdrant client using file path doesn't typically need explicit closing unless using specific modes.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a text file for the graph RAG system")
    parser.add_argument("file_path", type=str, help="Path to the text file to process")

    args = parser.parse_args()

    if os.path.exists(args.file_path):
        try:
            result = process_text_file(args.file_path)
            if result:
                 print(f"\n--- Processing Summary ---")
                 print(f"Status: {result['status']}")
                 print(f"Log File: {result['log_file']}")
                 print(f"Document ID: {result['doc_id']}")
                 print(f"Document Name: {result['doc_name']}")
                 print(f"Chunks Created: {result['chunks']}")
                 print(f"Entities Found: {result['entities']}")
                 print(f"--------------------------")
            else:
                 print(f"\nProcessing resulted in no output (e.g., empty file). Check logs: {log_file_path}")

        except Exception as e:
            print(f"\nProcessing failed. Check logs for details: {log_file_path}")
            # Error is already logged in process_text_file
        finally:
             close_resources() # Ensure resources are closed when run as script
    else:
        print(f"Error: File not found at {args.file_path}")
        logger.error(f"File not found: {args.file_path}")