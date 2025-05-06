import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

# --- Configuration ---
# Qdrant connection details
QDRANT_URL = "http://localhost:6333"

# Neo4j connection details
# !!! IMPORTANT: Replace "YOUR_NEO4J_PASSWORD" with your actual Neo4j password !!!
NEO4J_URI = "bolt://localhost:7687"  # Use bolt://localhost:7687 if neo4j:// doesn't work
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "94409440" # <-- PUT YOUR NEO4J PASSWORD HERE

# --- Test Qdrant Connection ---
print(f"Attempting to connect to Qdrant at {QDRANT_URL}...")
try:
    qdrant_client = QdrantClient(url=QDRANT_URL)
    # Attempt a simple operation to confirm connectivity
    qdrant_client.get_collections()
    print("SUCCESS: Connected to Qdrant.")
except Exception as e:
    print(f"FAILED: Could not connect to Qdrant.")
    print(f"Error: {e}")

print("-" * 20) # Separator

# --- Test Neo4j Connection ---
print(f"Attempting to connect to Neo4j at {NEO4J_URI} as user '{NEO4J_USERNAME}'...")
if NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD":
     print("WARNING: Neo4j password is still set to the placeholder value in the script!")
     print("Please edit connection_test.py and set the NEO4J_PASSWORD variable.")
else:
    neo4j_driver = None
    try:
        # Establish Neo4j connection
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        # Verify connection works
        neo4j_driver.verify_connectivity()
        print("SUCCESS: Connected to Neo4j.")
    except AuthError:
        print("FAILED: Could not connect to Neo4j due to authentication error.")
        print("Please check your NEO4J_USERNAME and NEO4J_PASSWORD in the script.")
    except ServiceUnavailable as e:
        print(f"FAILED: Could not connect to Neo4j at {NEO4J_URI}.")
        print(f"Is the database running and accessible? Check the URI.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"FAILED: An unexpected error occurred while connecting to Neo4j.")
        print(f"Error: {e}")
    finally:
        # Always close the driver connection if it was opened
        if neo4j_driver:
            neo4j_driver.close()
            # print("(Neo4j driver closed)") # Optional confirmation

print("-" * 20)
print("Connection tests finished.")