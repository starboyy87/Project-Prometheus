import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import os
import config
import shutil # For cleanup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qdrant():
    """Test Qdrant local setup"""
    test_collection_name = "test_connection_collection"
    vector_size = 4 # Simple small vector for testing

    try:
        # Ensure Qdrant data directory exists
        os.makedirs(config.QDRANT_PATH, exist_ok=True)
        logger.info(f"Using Qdrant local storage path: {config.QDRANT_PATH}")

        # Initialize local Qdrant client pointing to the path
        client = QdrantClient(path=config.QDRANT_PATH)
        logger.info("Qdrant client initialized.")

        # Clean up previous test collection if it exists
        try:
            if client.collection_exists(collection_name=test_collection_name):
                logger.warning(f"Deleting existing test collection: {test_collection_name}")
                client.delete_collection(collection_name=test_collection_name)
        except Exception as cleanup_e:
             logger.error(f"Could not clean up previous test collection (maybe it didn't exist): {cleanup_e}")


        # Create a test collection
        logger.info(f"Creating test collection: {test_collection_name}")
        client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Test collection created.")

        # Insert a test vector
        test_id = 1
        test_vector = np.random.rand(vector_size).tolist()
        logger.info("Upserting test point...")
        client.upsert(
            collection_name=test_collection_name,
            points=[
                PointStruct(
                    id=test_id,
                    vector=test_vector,
                    payload={"test": "data"}
                )
            ],
            wait=True # Ensure the operation completes
        )
        logger.info("Test point upserted.")

        # Retrieve the point to verify insertion
        retrieved_point = client.retrieve(
             collection_name=test_collection_name,
             ids=[test_id]
        )
        if not retrieved_point:
             raise ValueError("Failed to retrieve the inserted test point.")
        logger.info(f"Successfully retrieved test point: {retrieved_point[0].id}")


        # Search for the vector
        logger.info("Searching for test point...")
        search_result = client.search(
            collection_name=test_collection_name,
            query_vector=test_vector,
            limit=1
        )
        if not search_result or search_result[0].id != test_id:
             raise ValueError(f"Failed to find the correct test point via search. Found: {search_result}")

        logger.info(f"Qdrant test: Successfully created collection, upserted, retrieved, and searched.")

        # Clean up the test collection
        logger.info(f"Deleting test collection: {test_collection_name}")
        client.delete_collection(collection_name=test_collection_name)
        logger.info("Test collection deleted.")

        return True
    except Exception as e:
        logger.error(f"Qdrant test failed: {e}")
        # Attempt to clean up directory if it was just created for the test
        # Be cautious with automated deletion - perhaps comment out if issues occur
        # if os.path.exists(config.QDRANT_PATH) and not os.listdir(config.QDRANT_PATH):
        #     try:
        #         shutil.rmtree(config.QDRANT_PATH)
        #         logger.info(f"Cleaned up potentially empty Qdrant directory: {config.QDRANT_PATH}")
        #     except Exception as rmtree_e:
        #         logger.error(f"Failed to clean up Qdrant directory: {rmtree_e}")
        return False

if __name__ == "__main__":
    if test_qdrant():
        print("\n✅ Successfully set up and tested local Qdrant!")
        print(f"   Qdrant data will be stored in: {config.QDRANT_PATH}")
    else:
        print("\n❌ Failed to set up Qdrant. Please check the console output above for errors.")
        print(f"   Check if you have write permissions for the directory: {config.QDRANT_PATH}")