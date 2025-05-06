# query_system.py
import os
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from transformers import AutoProcessor, AutoModel # For CLIP
import torch # For CLIP
from neo4j import GraphDatabase 
from prometheus_config import config, get_neo4j_config, get_qdrant_config

# --- Configuration ---
# Use the prometheus_config system for secure configuration management

# --- Essential Configuration ---
GOOGLE_API_KEY = config.get("GOOGLE_API_KEY")

# Get Qdrant configuration
qdrant_config = get_qdrant_config()
QDRANT_HOST = qdrant_config.get('host')
QDRANT_PORT = qdrant_config.get('port')

# Get Neo4j configuration
neo4j_config = get_neo4j_config()
NEO4J_URI = neo4j_config.get('uri')
NEO4J_USERNAME = neo4j_config.get('username')
NEO4J_PASSWORD = neo4j_config.get('password')

# --- Qdrant/Model Settings ---
QDRANT_TEXT_COLLECTION = "doc_chunks_poc"
QDRANT_IMAGE_COLLECTION = "image_embeddings_poc"
TEXT_EMBEDDING_MODEL = "models/embedding-001" # For query embedding -> text search
IMAGE_EMBEDDING_MODEL_NAME = "openai/clip-vit-base-patch32" # For query embedding -> image search
LLM_MODEL = "gemini-2.5-pro-preview-03-25" # Using the latest preview Pro model (check availability/limits)

# --- Retrieval Settings ---
TEXT_SEARCH_LIMIT = 3 # Number of text chunks to retrieve
IMAGE_SEARCH_LIMIT = 3 # Number of images to reference

# --- Early Check for API Key ---
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in configuration. Please set it before running.")
    print("Run 'python secrets_manager.py status' to check your configuration status.")
    exit()

# Check for Neo4j credentials
neo4j_config_present = True # Assume present initially
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    print("Warning: Neo4j configuration is incomplete. Missing URI, username, or password.")
    print("Neo4j graph search capabilities will be disabled.")
    print("Run 'python secrets_manager.py status' to check your configuration status.")
    neo4j_config_present = False # Flag that config is missing


# --- Initialize Clients ---
print("Initializing clients...")
neo4j_driver_instance = None # Initialize driver instance variable

# Qdrant Client
print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Verify collections exist
    qdrant_client.get_collection(collection_name=QDRANT_TEXT_COLLECTION)
    qdrant_client.get_collection(collection_name=QDRANT_IMAGE_COLLECTION)
    print("Qdrant client connected and collections verified.")
except Exception as e:
    print(f"CRITICAL Error: Failed to connect to Qdrant or find collections: {e}")
    print("Ensure Qdrant is running and collections were created by process_document.py.")
    exit()

# <<< ADDED NEO4J DRIVER INITIALIZATION >>>
if neo4j_config_present: # Only try if config was found
    try:
        print(f"Connecting to Neo4j at {NEO4J_URI}...")
        # Create the driver instance
        neo4j_driver_instance = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        neo4j_driver_instance.verify_connectivity()
        print("Neo4j connection successful.")
    except Exception as e:
        print(f"Warning: Error connecting to Neo4j: {e}. Graph search will be disabled.")
        neo4j_driver_instance = None # Ensure driver is None on connection failure
# <<< END NEO4J DRIVER INITIALIZATION >>>


# Google Embeddings (for text query -> text search)
print(f"Initializing Google Embeddings model: {TEXT_EMBEDDING_MODEL}...")
try:
    google_text_embeddings = GoogleGenerativeAIEmbeddings(
        model=TEXT_EMBEDDING_MODEL,
        task_type="retrieval_query",
        google_api_key=GOOGLE_API_KEY # Explicitly pass key
    )
    print("Google Embeddings model initialized.")
except Exception as e:
    print(f"CRITICAL Error: Failed to initialize Google Text Embeddings: {e}")
    print("Check your GOOGLE_API_KEY and internet connection.")
    exit()

# CLIP Model (for text query -> image search)
print(f"Loading CLIP Model for query embedding: {IMAGE_EMBEDDING_MODEL_NAME}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for CLIP: {device}")
    clip_processor = AutoProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(IMAGE_EMBEDDING_MODEL_NAME).to(device)
    print("CLIP Model loaded.")
except Exception as e:
    print(f"CRITICAL Error: Failed to load CLIP model: {e}")
    print("Ensure 'transformers' and 'torch' are installed and you have internet.")
    exit()

# Google LLM (Gemini 2.5 Pro Preview)
print(f"Initializing LLM: {LLM_MODEL}...")
try:
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3, # Adjust creativity (0.0 = deterministic, 1.0 = creative)
        convert_system_message_to_human=True # Good practice for some chat models
        )
    print("LLM initialized.")
except Exception as e:
    print(f"CRITICAL Error: Failed to initialize LLM: {e}")
    print(f"Check your GOOGLE_API_KEY, ensure model name '{LLM_MODEL}' is correct and accessible, and check internet connection.")
    exit()

# --- Core Query Function ---
def perform_multimodal_rag(query_text):
    """
    Performs RAG by searching text and image collections and querying the LLM.
    """
    print(f"\nProcessing query: '{query_text}'")

    # 1. Generate Google Text Embedding for text search
    print("Generating query embedding for text search...")
    try:
        query_vector_text = google_text_embeddings.embed_query(query_text)
        print(f"  - Text embedding generated (Size: {len(query_vector_text)})")
    except Exception as e:
        print(f"  - Error generating text embedding: {e}")
        return "Sorry, I encountered an error generating the text search embedding."

    # 2. Generate CLIP Text Embedding for image search
    print("Generating query embedding for image search...")
    query_vector_clip = None # Initialize
    try:
        inputs = clip_processor(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        query_vector_clip = text_features.cpu().numpy().flatten().tolist()
        print(f"  - Image search embedding generated (Size: {len(query_vector_clip)})")
    except Exception as e:
        print(f"  - Warning: Error generating image search embedding: {e}")
        # Continue without image search capability for this query

    # 3. Search Text Collection using VectorRetrieverAgent
    print(f"Searching text collection '{QDRANT_TEXT_COLLECTION}'...")
    retrieved_texts = []
    try:
        # Use the VectorRetrieverAgent for text retrieval
        text_results = vector_agent.retrieve(
            query_vector=query_vector_text,
            collection=QDRANT_TEXT_COLLECTION,
            top_k=TEXT_SEARCH_LIMIT
        )
        retrieved_texts = [hit.payload.get("text", "[missing text]") for hit in text_results if hit.payload]
        print(f"  - Found {len(retrieved_texts)} relevant text chunks.")
    except Exception as e:
        print(f"  - Error searching text collection: {e}")
        # Continue with potentially empty text results

    # 4. Search Image Collection using VectorRetrieverAgent
    print(f"Searching image collection '{QDRANT_IMAGE_COLLECTION}'...")
    retrieved_image_paths = []
    if query_vector_clip:
        try:
            # Use the VectorRetrieverAgent for image retrieval
            image_results = vector_agent.retrieve(
                query_vector=query_vector_clip,
                collection=QDRANT_IMAGE_COLLECTION,
                top_k=IMAGE_SEARCH_LIMIT
            )
            retrieved_image_paths = [hit.payload.get("file_path", "[missing path]") for hit in image_results if hit.payload]
            print(f"  - Found {len(retrieved_image_paths)} relevant image paths.")
        except Exception as e:
            print(f"  - Error searching image collection: {e}")
            # Continue with potentially empty image results
    else:
        print("  - Skipping image search due to embedding error.")

    # 5. Search Graph (Neo4j) using GraphRetrieverAgent
    print("Searching graph database (Neo4j)...")
    retrieved_graph_info = []
    if graph_agent and neo4j_driver_instance:
        try:
            # Use the GraphRetrieverAgent for graph retrieval
            graph_results = graph_agent.retrieve(query_text, top_n=5)
            
            # Also query for specific entities
            entity_results = graph_agent.query_entities(query_text, top_n=3)
            
            # Process the graph results into a readable format
            if graph_results:
                for result in graph_results:
                    if 'n' in result and 'r' in result and 'm' in result:
                        n = result['n']
                        r = result['r']
                        m = result['m']
                        retrieved_graph_info.append(f"{n.get('text', 'Unknown')} {r.type} {m.get('text', 'Unknown')}")
            
            # Add entity information
            if entity_results:
                for result in entity_results:
                    if 'e' in result:
                        e = result['e']
                        retrieved_graph_info.append(f"Entity: {e.get('name', 'Unknown')} - {e.get('type', 'Unknown')}")
                        
            print(f"  - Found {len(retrieved_graph_info)} graph relationships/entities.")
        except Exception as e:
            print(f"  - Error during graph search: {e}")
    else:
        print("  - Skipping Neo4j search (driver or agent not available).")

    # 6. Prepare Context for LLM
    print("Preparing context for LLM...")
    context = ""
    
    # Add text chunks to context
    if retrieved_texts:
        context += "\n\nRELEVANT TEXT CHUNKS:\n"
        for i, text in enumerate(retrieved_texts, 1):
            context += f"\nChunk {i}:\n{text}\n"
    else:
        context += "\nNo relevant text chunks found.\n"
    
    # Add graph information to context
    if retrieved_graph_info:
        context += "\n\nRELEVANT GRAPH INFORMATION:\n"
        for i, info in enumerate(retrieved_graph_info, 1):
            context += f"\n{i}. {info}"
    
    # Add image references to context
    if retrieved_image_paths:
        context += "\n\nRELEVANT IMAGES:\n"
        for i, path in enumerate(retrieved_image_paths, 1):
            context += f"\nImage {i}: {path}\n"
    
    # 7. Prepare the system message
    system_message = f"""You are an intelligent assistant for a manufacturing company.
Answer the user's query based on the provided context information.
If the context doesn't contain relevant information, say you don't know.
If there are relevant images, mention them and explain their relevance.
Be concise, accurate, and helpful.

Context information:
{context}
"""

    # 8. Query the LLM
    print("Querying LLM...")
    try:
        response = llm.invoke(system_message + f"\nUser Query: {query_text}")
        answer = response.content
        print("  - LLM response received.")
    except Exception as e:
        print(f"  - Error querying LLM: {e}")
        answer = "Sorry, I encountered an error while processing your query."
    
    # 9. Return the answer
    return answer

# --- Main Query Loop ---
if __name__ == "__main__":
    # Import and initialize agents
    try:
        from agents import GraphRetrieverAgent, VectorRetrieverAgent
        
        # Initialize VectorRetrieverAgent with existing qdrant_client
        vector_agent = VectorRetrieverAgent(qdrant_client=qdrant_client)
        print("VectorRetrieverAgent initialized successfully.")
        
        # Initialize GraphRetrieverAgent with existing neo4j_driver_instance if available
        graph_agent = None
        if neo4j_driver_instance:
            graph_agent = GraphRetrieverAgent(neo4j_driver=neo4j_driver_instance)
            print("GraphRetrieverAgent initialized successfully.")
        else:
            print("GraphRetrieverAgent initialization skipped (Neo4j driver not available).")
            
        # Perform health checks
        vector_health = vector_agent.health_check()
        print(f"VectorRetrieverAgent health: {vector_health['status']}")
        
        if graph_agent:
            graph_health = graph_agent.health_check()
            print(f"GraphRetrieverAgent health: {graph_health['status']}")
    except Exception as e:
        print(f"Error initializing agents: {e}")
        print("Continuing without agent functionality.")
        vector_agent = None
        graph_agent = None
    
    print("\nQuery system ready. Type 'quit' or 'exit' to stop.")
    while True:
        user_query = None # Initialize to prevent potential UnboundLocalError in finally
        try:
            user_query = input("\nEnter your query: ")
            if user_query.lower() in ["quit", "exit"]:
                break
            if not user_query.strip():
                print("Please enter a query.")
                continue

            # <<< Pass driver to function (will be needed later) >>>
            answer = perform_multimodal_rag(user_query)
            print("\nLLM Response:")
            print(answer)

        except EOFError: # Handle Ctrl+D or unexpected end of input
             break
        except KeyboardInterrupt: # Handle Ctrl+C
             break
        # Error handling can be added here for specific query processing errors

    # Moved Neo4j closing and exit message outside the loop's try/except
    # to ensure they run after the loop terminates via break.
    if neo4j_driver_instance:
        print("\nClosing Neo4j driver connection...")
        neo4j_driver_instance.close()

    print("\nExiting query system.")