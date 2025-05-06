import os
import fitz  # PyMuPDF
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import hashlib
import base64
import uuid
import time
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from langchain_neo4j import Neo4jGraph  # Updated import from langchain-neo4j package

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

# --- Environment & Logging Setup ---
load_dotenv(dotenv_path=Path(__file__).parent / 'env' / '.env') # Load from env/.env

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "process_document.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("process_document")

# --- Get API Keys & DB Credentials ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") # Should be set
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")

# --- Configuration ---
RAW_DIR = Path(r"c:\prometheus\data\raw")
PROCESSED_DIR = Path(r"c:\prometheus\data\processed")
EXTRACTED_IMAGES_DIR = Path("extracted_images") # Relative path, created in C:\prometheus
TEXT_CHUNK_SIZE = int(os.getenv('TEXT_CHUNK_SIZE', 1000))
TEXT_CHUNK_OVERLAP = int(os.getenv('TEXT_CHUNK_OVERLAP', 200))

# Batch processing configuration
QDRANT_BATCH_SIZE = int(os.getenv('QDRANT_BATCH_SIZE', 100))  # Batch size for text chunks
QDRANT_IMAGE_BATCH_SIZE = int(os.getenv('QDRANT_IMAGE_BATCH_SIZE', 20))  # Batch size for images
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', "models/embedding-001")
TEXT_VECTOR_SIZE = 768 # Fixed for models/embedding-001
QDRANT_TEXT_COLLECTION = os.getenv('QDRANT_TEXT_COLLECTION', "doc_chunks_poc")
# --- Image Config ---
QDRANT_IMAGE_COLLECTION = os.getenv('QDRANT_IMAGE_COLLECTION', "image_embeddings_poc")
IMAGE_EMBEDDING_MODEL_NAME = os.getenv('IMAGE_EMBEDDING_MODEL', "openai/clip-vit-base-patch32")
IMAGE_VECTOR_SIZE = 512 # Fixed for clip-vit-base-patch32
IMAGE_VECTOR_DISTANCE = Distance.COSINE
# --- LLM & Graph Config ---
LLM_MODEL_FOR_GRAPH = os.getenv('LLM_MODEL_FOR_GRAPH', "gemini-2.5-pro-latest")
GRAPH_ALLOWED_NODES = os.getenv('GRAPH_ALLOWED_NODES', "Chunk,Entity,PartNumber,Specification,Location,Organization,Person").split(',')
GRAPH_ALLOWED_RELATIONSHIPS = os.getenv('GRAPH_ALLOWED_RELATIONSHIPS', "MENTIONS,HAS_SPEC,PART_OF,CONNECTS_TO,LOCATED_IN").split(',')
# ---------------------------

# --- Validate Essential Config ---
def validate_config():
    valid = True # Start assuming valid
    if not GOOGLE_API_KEY:
        logger.critical("CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables or .env file.")
        valid = False
    # Stricter Neo4j check: if any are set, all must be set
    neo4j_vars = [NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]
    if any(neo4j_vars) and not all(neo4j_vars):
        logger.critical("CRITICAL ERROR: If using Neo4j, NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must all be set.")
        valid = False
    elif not any(neo4j_vars):
         logger.warning("Neo4j credentials not set. Neo4j graph operations will be skipped.")

    if not RAW_DIR.exists():
        logger.critical(f"CRITICAL ERROR: Raw directory '{RAW_DIR}' does not exist.")
        valid = False
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Test writability (optional, might fail in restricted environments)
        # test_file = PROCESSED_DIR / ".cascade_write_test"
        # test_file.touch()
        # test_file.unlink()
    except OSError as e:
        logger.critical(f"CRITICAL ERROR: Could not create or write to processed directory '{PROCESSED_DIR}': {e}")
        valid = False

    try:
        EXTRACTED_IMAGES_DIR.mkdir(exist_ok=True)
    except OSError as e:
         logger.critical(f"CRITICAL ERROR: Could not create extracted images directory '{EXTRACTED_IMAGES_DIR}': {e}")
         valid = False

    return valid

# --- Test LLM Configuration ---
def test_llm_configuration():
    logger.info(f"Testing LLM configuration with model: {LLM_MODEL_FOR_GRAPH}...")
    try:
        # Verify API key is set
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set. LLM configuration test failed.")
            return False
            
        # Check if model name is valid
        valid_models = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-2.5-pro-latest"]
        if LLM_MODEL_FOR_GRAPH not in valid_models:
            logger.warning(f"Model {LLM_MODEL_FOR_GRAPH} may not be valid. Valid models include: {', '.join(valid_models)}")
            # Continue anyway as Google may add new models
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            temperature=0,
            model=LLM_MODEL_FOR_GRAPH,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Test with a simple prompt
        response = llm.invoke("Hello, are you working correctly? Please respond with a single short sentence.")
        logger.info(f"LLM test response: {response.content}")
        
        # Verify the response is not empty
        if not response.content or len(response.content.strip()) == 0:
            logger.warning("LLM returned empty response. This may indicate an issue.")
            return False
            
        return True
    except Exception as e:
        logger.error(f"LLM configuration test failed: {e}")
        return False

# --- Initialize Clients ---
def initialize_clients():
    logger.info("Initializing clients...")
    clients = {
        'qdrant': None, 'neo4j': None, 'text_embedder': None,
        'clip_model': None, 'clip_processor': None, 'graph_llm': None,
        'llm_transformer': None, 'device': None, 'neo4j_connected': False
    }
    start_time_init = time.time()

    # Qdrant Client
    qdrant_port_resolved = 6333
    try:
        qdrant_port_resolved = int(QDRANT_PORT) if QDRANT_PORT else 6333
    except ValueError:
        logger.warning(f"Invalid QDRANT_PORT value '{QDRANT_PORT}'. Using default 6333.")
    qdrant_host_resolved = QDRANT_HOST if QDRANT_HOST else "localhost"

    logger.info(f"Connecting to Qdrant at {qdrant_host_resolved}:{qdrant_port_resolved}...")
    try:
        # Significantly increase timeout to handle large document processing
        clients['qdrant'] = QdrantClient(host=qdrant_host_resolved, port=qdrant_port_resolved, timeout=600) # 10 minutes timeout
        clients['qdrant'].get_collections() # Test connection
        logger.info("Qdrant client connected with extended timeout (600s).")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to connect to Qdrant: {e}")
        return None

    # Neo4j Driver
    if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
        try:
            # Test connection first with detailed error reporting
            if test_neo4j_connection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD):
                logger.info(f"Connecting to Neo4j at {NEO4J_URI}...")
                clients['neo4j'] = GraphDatabase.driver(
                    NEO4J_URI, 
                    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                    max_connection_lifetime=3600  # 1 hour max connection lifetime
                )
                clients['neo4j_connected'] = True
                logger.info("Neo4j connection successful and ready for use.")
            else:
                logger.warning("Neo4j connection test failed. Neo4j operations will be skipped.")
                clients['neo4j'] = None
        except Exception as e:
            logger.warning(f"Warning: Error connecting to Neo4j: {e}. Neo4j operations will be skipped.")
            clients['neo4j'] = None
    else:
        logger.warning("Neo4j credentials not fully set. Neo4j graph operations will be skipped.")

    # Text Embeddings Model
    logger.info(f"Initializing Google Text Embeddings model: {TEXT_EMBEDDING_MODEL}...")
    try:
        clients['text_embedder'] = GoogleGenerativeAIEmbeddings(
            model=TEXT_EMBEDDING_MODEL,
            task_type="retrieval_document",
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Google Text Embeddings model initialized.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to initialize Google Text Embeddings: {e}")
        return None

    # CLIP Model
    logger.info(f"Loading Multimodal Embedding Model: {IMAGE_EMBEDDING_MODEL_NAME}...")
    clients['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for CLIP: {clients['device']}")
    try:
        clients['clip_processor'] = AutoProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL_NAME)
        clients['clip_model'] = AutoModel.from_pretrained(IMAGE_EMBEDDING_MODEL_NAME)
        # Add try-except around moving model to device
        try:
            clients['clip_model'] = clients['clip_model'].to(clients['device'])
            logger.info("Multimodal Embedding Model loaded successfully.")
        except Exception as device_e:
            logger.critical(f"CRITICAL ERROR: Failed to move CLIP model to device '{clients['device']}': {device_e}")
            # Attempt to fallback to CPU if CUDA failed
            if clients['device'] == 'cuda':
                logger.warning("Falling back to CPU for CLIP model.")
                clients['device'] = 'cpu'
                try:
                    clients['clip_model'] = clients['clip_model'].to(clients['device'])
                    logger.info("Multimodal Embedding Model loaded successfully on CPU.")
                except Exception as cpu_e:
                    logger.critical(f"CRITICAL ERROR: Failed to load CLIP model even on CPU: {cpu_e}")
                    return None
            else:
                 return None # Failed on CPU initially
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to load CLIP processor or model structure: {e}")
        return None

    # LLM for Graph Extraction
    if clients['neo4j']:
        logger.info(f"Initializing LLM for Graph Extraction: {LLM_MODEL_FOR_GRAPH}...")
        try:
            # Test LLM configuration first
            if not test_llm_configuration():
                logger.warning("LLM configuration test failed. Graph extraction will be skipped.")
                return clients
                
            clients['graph_llm'] = ChatGoogleGenerativeAI(
                temperature=0,
                model=LLM_MODEL_FOR_GRAPH,
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("LLM for Graph Extraction initialized.")

            logger.info("Initializing LLMGraphTransformer...")
            clients['llm_transformer'] = LLMGraphTransformer(
                llm=clients['graph_llm'],
                allowed_nodes=GRAPH_ALLOWED_NODES,
                allowed_relationships=GRAPH_ALLOWED_RELATIONSHIPS,
            )
            logger.info("LLMGraphTransformer initialized.")
        except Exception as e:
            logger.warning(f"Warning: Failed to initialize LLM or Transformer for Graph Extraction: {e}")
            clients['graph_llm'] = None
            clients['llm_transformer'] = None
    else:
        logger.info("Skipping LLM/Transformer initialization for Graph (Neo4j driver not available).")

    end_time_init = time.time()
    logger.info(f"Client initialization finished in {end_time_init - start_time_init:.2f} seconds.")
    return clients

# --- Ensure Qdrant Collections Exist ---
def setup_qdrant_collections(qdrant_client):
    logger.info("Ensuring Qdrant collections exist...")
    start_time_qdrant_setup = time.time()
    success = True

    # Check and create text collection
    text_collection_name = QDRANT_TEXT_COLLECTION
    try:
        if not qdrant_client.collection_exists(collection_name=text_collection_name):
            logger.info(f"Creating Qdrant text collection: '{text_collection_name}'...")
            qdrant_client.create_collection(
                collection_name=text_collection_name,
                vectors_config=models.VectorParams(size=TEXT_VECTOR_SIZE, distance=models.Distance.COSINE)
            )
            logger.info(f"Successfully created text collection '{text_collection_name}'.")
        else:
            logger.info(f"Qdrant text collection '{text_collection_name}' already exists.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed operation on Qdrant text collection '{text_collection_name}': {e}")
        success = False

    # Check and create image collection
    image_collection_name = QDRANT_IMAGE_COLLECTION
    try:
        if not qdrant_client.collection_exists(collection_name=image_collection_name):
            logger.info(f"Creating Qdrant image collection: '{image_collection_name}'...")
            qdrant_client.create_collection(
                collection_name=image_collection_name,
                vectors_config=VectorParams(size=IMAGE_VECTOR_SIZE, distance=IMAGE_VECTOR_DISTANCE)
            )
            logger.info(f"Successfully created image collection '{image_collection_name}'.")
        else:
            logger.info(f"Qdrant image collection '{image_collection_name}' already exists.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed operation on Qdrant image collection '{image_collection_name}': {e}")
        success = False

    end_time_qdrant_setup = time.time()
    if success:
        logger.info(f"Qdrant collection setup finished successfully in {end_time_qdrant_setup - start_time_qdrant_setup:.2f} seconds.")
    else:
        logger.error(f"Qdrant collection setup failed after {end_time_qdrant_setup - start_time_qdrant_setup:.2f} seconds.")
    return success

# --- Helper: Create deterministic UUID ---
def create_deterministic_uuid(identifier: str):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, identifier))

# --- Helper: Test Neo4j Connection ---
def test_neo4j_connection(uri, username, password):
    """Test Neo4j connection and report detailed error information if it fails."""
    logger.info(f"Testing Neo4j connection to {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        
        # Test basic query execution
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            if test_value != 1:
                raise Exception(f"Unexpected test result: {test_value}")
                
        logger.info("Neo4j connection test successful!")
        driver.close()
        return True
    except ServiceUnavailable as e:
        logger.error(f"Neo4j server unavailable: {e}. Check if Neo4j is running and accessible at {uri}.")
        return False
    except AuthError as e:
        logger.error(f"Neo4j authentication failed: {e}. Check username and password.")
        return False
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {e}")
        return False

# --- Main Processing Function for a Single PDF ---
def process_single_pdf(pdf_path: Path, clients: dict):
    logger.info(f"Starting processing for: {pdf_path.name}")
    start_time_pdf_proc = time.time()
    doc_identifier = pdf_path.stem.replace(" ", "_")
    base_metadata = {'source_doc': pdf_path.name}

    # 1. Extract Text
    logger.info("Extracting text content...")
    text_content = ""
    num_pages = 0
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        all_pages_text = [page.extract_text() or "" for page in reader.pages]
        text_content = "\n".join(all_pages_text)
        logger.info(f"Successfully processed text from {num_pages} pages.")
        if not text_content.strip(): logger.warning("No meaningful text content extracted from PDF.")
    except Exception as e: logger.error(f"Error extracting text with pypdf: {e}")

    # 2. Extract Images
    logger.info(f"Extracting images...")
    extracted_image_paths = []
    doc = None # Initialize doc to None
    try:
        doc = fitz.open(pdf_path)
        img_count = 0
        for page_index in range(len(doc)):
            image_list = doc.get_page_images(page_index, full=True)
            if not image_list: continue
            for image_index, img_info in enumerate(image_list):
                xref = img_info[0]
                if xref == 0: continue # Skip invalid xref
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image: continue
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    # Use a more robust unique identifier
                    img_unique_id = f"{doc_identifier}_page{page_index+1}_img{img_count+1}_xref{xref}"
                    safe_filename = f"{img_unique_id}.{image_ext}".replace(" ", "_")
                    image_path = EXTRACTED_IMAGES_DIR / safe_filename
                    with open(image_path, "wb") as image_file: image_file.write(image_bytes)
                    extracted_image_paths.append(image_path)
                    img_count += 1
                except Exception as img_e:
                     logger.warning(f"     - Could not save image xref={xref} on page {page_index+1}. Error: {img_e}")
        logger.info(f"Finished image extraction. Found and saved {len(extracted_image_paths)} images to '{EXTRACTED_IMAGES_DIR}'.")
    except Exception as e:
        logger.warning(f"Warning: Error during image extraction process for {pdf_path.name}: {e}")
        # Optionally decide if this is critical; here we log and continue
    finally:
        if doc: # Ensure document is closed even if errors occurred
             try:
                 doc.close()
             except Exception as close_e:
                 logger.warning(f"Error closing fitz document {pdf_path.name}: {close_e}")

    # 3. Chunk Text
    logger.info("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP)
    text_chunks = text_splitter.split_text(text_content)
    logger.info(f"Split text into {len(text_chunks)} chunks.")

    langchain_docs = []
    for i, chunk in enumerate(text_chunks):
        metadata = base_metadata.copy()
        metadata['chunk_index'] = i
        metadata['chunk_id'] = create_deterministic_uuid(f"{doc_identifier}_chunk_{i}")
        langchain_docs.append(Document(page_content=chunk, metadata=metadata))

    # 4. Generate Text Embeddings and Store in Qdrant (with batching)
    logger.info("Generating text embeddings and storing in Qdrant using batched processing...")
    
    # Define batch size for processing chunks
    BATCH_SIZE = int(os.getenv('QDRANT_BATCH_SIZE', 100))
    total_chunks = len(langchain_docs)
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    logger.info(f"Processing {total_chunks} chunks in {total_batches} batches (batch size: {BATCH_SIZE})")
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Process in batches
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_chunks)
        current_batch = langchain_docs[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(current_batch)} chunks")
        
        try:
            # Generate embeddings for this batch
            batch_contents = [doc.page_content for doc in current_batch]
            batch_vectors = clients['text_embedder'].embed_documents(batch_contents)
            
            # Create points for this batch
            batch_points = [
                PointStruct(
                    id=doc.metadata['chunk_id'],
                    vector=vector,
                    payload=doc.metadata
                )
                for doc, vector in zip(current_batch, batch_vectors)
            ]
            
            # Upsert this batch to Qdrant
            if batch_points:
                clients['qdrant'].upsert(
                    collection_name=QDRANT_TEXT_COLLECTION, 
                    points=batch_points, 
                    wait=True
                )
                successful_chunks += len(batch_points)
                logger.info(f"Successfully upserted batch {batch_idx+1}/{total_batches} with {len(batch_points)} vectors")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx+1}/{total_batches}: {e}")
            failed_chunks += len(current_batch)
            # Continue with next batch instead of failing completely
    
    # Report overall results
    if failed_chunks > 0:
        logger.warning(f"Completed with some errors: {successful_chunks} chunks succeeded, {failed_chunks} chunks failed")
    else:
        logger.info(f"Successfully processed all {successful_chunks} chunks")
        
    # Only fail if all chunks failed
    if successful_chunks == 0 and failed_chunks > 0:
        logger.error("All chunks failed to process")
        return False
    
    return True  # Continue processing even if some chunks failed

    # 5. Generate Image Embeddings and Store in Qdrant (with batching)
    logger.info("Generating image embeddings and storing in Qdrant using batched processing...")
    
    # Define batch size for processing images
    IMAGE_BATCH_SIZE = int(os.getenv('QDRANT_IMAGE_BATCH_SIZE', 20))  # Smaller batch size for images
    total_images = len(extracted_image_paths)
    
    if total_images == 0:
        logger.info("No images to process.")
    else:
        total_image_batches = (total_images + IMAGE_BATCH_SIZE - 1) // IMAGE_BATCH_SIZE  # Ceiling division
        logger.info(f"Processing {total_images} images in {total_image_batches} batches (batch size: {IMAGE_BATCH_SIZE})")
        
        successful_images = 0
        failed_images = 0
        
        # Process images in batches
        for batch_idx in range(total_image_batches):
            start_idx = batch_idx * IMAGE_BATCH_SIZE
            end_idx = min(start_idx + IMAGE_BATCH_SIZE, total_images)
            current_batch = extracted_image_paths[start_idx:end_idx]
            
            logger.info(f"Processing image batch {batch_idx+1}/{total_image_batches} with {len(current_batch)} images")
            batch_points = []
            
            # Process each image in the current batch
            for image_path in current_batch:
                try:
                    # Ensure the image file exists and is accessible
                    if not os.path.exists(image_path):
                        logger.warning(f"Image file not found: {image_path}")
                        failed_images += 1
                        continue
                        
                    # Load and convert to RGB to ensure consistent 3-channel format
                    try:
                        img = Image.open(image_path).convert("RGB")
                    except Exception as img_e:
                        logger.warning(f"Failed to open image {image_path.name}: {img_e}")
                        failed_images += 1
                        continue
                        
                    # Process with CLIP
                    with torch.no_grad():
                        try:
                            # Create inputs on CPU first
                            inputs = clients['clip_processor'](images=img, return_tensors="pt")
                            # Move to appropriate device
                            inputs = {k: v.to(clients['device']) for k, v in inputs.items()}
                            # Get features
                            img_features = clients['clip_model'].get_image_features(**inputs)
                            # Move back to CPU and convert to list
                            img_vector = img_features.cpu().numpy().flatten().tolist()
                        except Exception as clip_e:
                            logger.warning(f"CLIP processing failed for {image_path.name}: {clip_e}")
                            failed_images += 1
                            continue
        
                    # Create metadata and add to batch points
                    img_id = create_deterministic_uuid(image_path.name) # Use filename for deterministic ID
                    payload = base_metadata.copy()
                    payload['image_path'] = str(image_path)
                    payload['image_filename'] = image_path.name
                    payload['processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    payload['batch'] = f"{batch_idx+1}/{total_image_batches}"
        
                    batch_points.append(PointStruct(id=img_id, vector=img_vector, payload=payload))
                    successful_images += 1
                    
                except Exception as e:
                    logger.warning(f"Could not process or embed image {image_path.name}: {e}")
                    failed_images += 1
            
            # Upsert this batch to Qdrant
            if batch_points:
                try:
                    clients['qdrant'].upsert(
                        collection_name=QDRANT_IMAGE_COLLECTION, 
                        points=batch_points, 
                        wait=True
                    )
                    logger.info(f"Successfully upserted image batch {batch_idx+1}/{total_image_batches} with {len(batch_points)} vectors")
                except Exception as e:
                    logger.error(f"Error upserting image batch {batch_idx+1}: {e}")
                    failed_images += len(batch_points)
                    successful_images -= len(batch_points)
                    # Continue with next batch
        
        # Report overall image processing results
        logger.info(f"Image processing summary: {successful_images} successful, {failed_images} failed")
        
        if successful_images == 0 and total_images > 0:
            logger.warning("No images were successfully processed and embedded.")
        elif successful_images > 0:
            logger.info(f"Successfully embedded {successful_images} images in Qdrant collection '{QDRANT_IMAGE_COLLECTION}'.")

    # 6. Extract Graph and Store in Neo4j (if enabled and successful so far)
    if clients['neo4j'] and clients['llm_transformer']:
        logger.info("Extracting graph data and storing in Neo4j...")
        try:
            # Convert Langchain docs to graph documents
            graph_documents = clients['llm_transformer'].convert_to_graph_documents(langchain_docs)
            logger.info(f"Converted {len(langchain_docs)} documents to {len(graph_documents)} graph documents.")

            # Process each graph document and store directly in Neo4j
            logger.info("Processing graph documents and storing in Neo4j...")
            total_nodes = 0
            total_relationships = 0
            
            # Note: We're not using the Neo4jGraph instance directly for storage
            # but it could be useful for querying the graph later
            # neo4j_graph = Neo4jGraph(
            #     url=NEO4J_URI,
            #     username=NEO4J_USERNAME,
            #     password=NEO4J_PASSWORD
            # )
            
            # Process documents in batches for better transaction efficiency
            for doc_idx, graph_doc in enumerate(graph_documents):
                logger.info(f"Processing graph document {doc_idx+1}/{len(graph_documents)}")
                
                # Use a single session for each document to improve performance
                with clients['neo4j'].session() as session:
                    # Process all nodes in a single transaction
                    if graph_doc.nodes:
                        try:
                            # Define the transaction function for nodes
                            def process_nodes_tx(tx):
                                node_count = 0
                                for node in graph_doc.nodes:
                                    # Prepare properties dict, filtering out None values
                                    props = {k: v for k, v in node.properties.items() if v is not None}
                                    
                                    # Create Cypher for this node
                                    cypher_query = f"""
                                    MERGE (n:{node.type} {{id: $id}})
                                    SET n += $props
                                    """
                                    
                                    tx.run(cypher_query, id=node.id, props=props)
                                    node_count += 1
                                return node_count
                            
                            # Execute the transaction
                            nodes_added = session.execute_write(process_nodes_tx)
                            total_nodes += nodes_added
                            logger.info(f"Added {nodes_added} nodes in document {doc_idx+1}")
                        except Exception as node_tx_e:
                            logger.warning(f"Error in node transaction for document {doc_idx+1}: {node_tx_e}")
                    
                    # Process all relationships in a single transaction
                    if graph_doc.relationships:
                        try:
                            # Define the transaction function for relationships
                            def process_rels_tx(tx):
                                rel_count = 0
                                for rel in graph_doc.relationships:
                                    # Prepare properties dict, filtering out None values
                                    props = {k: v for k, v in rel.properties.items() if v is not None}
                                    
                                    # Create Cypher for this relationship
                                    cypher_query = f"""
                                    MATCH (source:{rel.source.type} {{id: $source_id}})
                                    MATCH (target:{rel.target.type} {{id: $target_id}})
                                    MERGE (source)-[r:{rel.type}]->(target)
                                    """
                                    
                                    # Add properties if they exist
                                    if props:
                                        cypher_query += "SET r += $props"
                                    
                                    tx.run(
                                        cypher_query,
                                        source_id=rel.source.id,
                                        target_id=rel.target.id,
                                        props=props
                                    )
                                    rel_count += 1
                                return rel_count
                            
                            # Execute the transaction
                            rels_added = session.execute_write(process_rels_tx)
                            total_relationships += rels_added
                            logger.info(f"Added {rels_added} relationships in document {doc_idx+1}")
                        except Exception as rel_tx_e:
                            logger.warning(f"Error in relationship transaction for document {doc_idx+1}: {rel_tx_e}")
            
            logger.info(f"Successfully stored {total_nodes} nodes and {total_relationships} relationships in Neo4j.")

        except Exception as e:
            logger.error(f"Error during graph extraction or storage for {pdf_path.name}: {e}")
            return False # CRITICAL: Mark processing as failed if graph storage fails
    else:
        logger.info("Skipping graph extraction (Neo4j or LLM Transformer not available).")

    end_time_pdf_proc = time.time()
    logger.info(f"Processing for {pdf_path.name} finished in {end_time_pdf_proc - start_time_pdf_proc:.2f} seconds.")
    return True # Indicate success

# --- Main Execution Block ---
def main():
    if not validate_config():
        logger.critical("Configuration validation failed. Exiting.")
        exit(1)

    clients = initialize_clients()
    if not clients:
        logger.critical("Client initialization failed. Exiting.")
        exit(1)

    if not setup_qdrant_collections(clients['qdrant']):
        logger.critical("Qdrant collection setup failed. Exiting.")
        exit(1)
        
    # Verify Neo4j is ready if needed
    if clients.get('neo4j') and not clients.get('neo4j_connected', False):
        logger.warning("Neo4j driver was initialized but connection was not verified. Graph operations may fail.")
        # Continue anyway as we'll skip graph operations if connection fails

    # Find PDF files to process
    pdf_files_to_process = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files_to_process:
        logger.info(f"No PDF files found in {RAW_DIR}. Exiting.")
        return

    logger.info(f"Found {len(pdf_files_to_process)} PDF files to process in {RAW_DIR}")

    overall_start_time = time.time()
    processed_count = 0
    failed_count = 0

    # --- Start Processing Loop ---
    for pdf_path in pdf_files_to_process:
        logger.info(f"\n{'='*20} Processing File: {pdf_path.name} {'='*20}")
        file_processed_successfully = False
        try:
            file_processed_successfully = process_single_pdf(pdf_path, clients)
        except Exception as e:
            logger.error(f"Unexpected critical error processing {pdf_path.name}: {e}", exc_info=True)
            file_processed_successfully = False # Ensure it's marked as failed
        finally:
            # Move file only if processing was successful
            if file_processed_successfully:
                try:
                    destination_path = PROCESSED_DIR / pdf_path.name
                    shutil.move(str(pdf_path), str(destination_path))
                    logger.info(f"Successfully processed and moved '{pdf_path.name}' to '{PROCESSED_DIR}'")
                    processed_count += 1
                except Exception as move_e:
                    logger.error(f"Error moving processed file {pdf_path.name} to {PROCESSED_DIR}: {move_e}")
                    failed_count += 1 # Count as failed if move fails
            else:
                logger.error(f"Processing failed for '{pdf_path.name}'. File left in '{RAW_DIR}'.")
                failed_count += 1

    overall_end_time = time.time()
    logger.info("\n----- Processing Summary -----")
    logger.info(f"Total files processed: {processed_count}")
    logger.info(f"Total files failed: {failed_count}")
    logger.info(f"Total time: {overall_end_time - overall_start_time:.2f} seconds")

    # Close Neo4j driver if it was opened
    if clients.get('neo4j'):
        try:
            clients['neo4j'].close()
            logger.info("Neo4j driver closed.")
        except Exception as e:
            logger.warning(f"Error closing Neo4j driver: {e}")

if __name__ == "__main__":
    main()
