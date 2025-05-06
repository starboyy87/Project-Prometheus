import logging
from neo4j import GraphDatabase
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """Test connection to Neo4j"""
    try:
        # Ensure the Neo4j database is running in Neo4j Desktop
        logger.info(f"Attempting to connect to Neo4j at {config.NEO4J_URI}")
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' AS message")
            message = result.single()["message"]
            logger.info(f"Neo4j test query result: {message}")
        driver.close()
        return True
    except Exception as e:
        # Catch potential auth errors or connection issues
        logger.error(f"Neo4j connection failed: {e}")
        if "authentication" in str(e).lower():
            logger.error("Hint: Check if NEO4J_PASSWORD in config.py matches the password set in Neo4j Desktop.")
        elif "connection refused" in str(e).lower() or "timed out" in str(e).lower():
             logger.error("Hint: Ensure the 'prometheus-db' database is started in Neo4j Desktop.")
        return False

if __name__ == "__main__":
    if test_neo4j_connection():
        print("\n✅ Successfully connected to Neo4j!")
    else:
        print("\n❌ Failed to connect to Neo4j. Please check the console output above for errors, verify your password in config.py, and ensure the database is running in Neo4j Desktop.")