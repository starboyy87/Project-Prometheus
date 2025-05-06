"""
Example demonstrating how to use the PrometheusConfig module
in your Project Prometheus application.
"""

import logging
from prometheus_config import config, get_neo4j_config, get_qdrant_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate config usage."""
    logger.info("Starting configuration example...")
    
    # Access individual configuration values
    google_api_key = config.get('GOOGLE_API_KEY')
    logger.info(f"Using Google API Key: {'*' * 5}{google_api_key[-4:] if google_api_key else 'Not configured'}")
    
    # Access typed configuration values
    max_latency = config.get_int('MAX_QUERY_LATENCY_MS')
    logger.info(f"Maximum query latency: {max_latency}ms")
    
    # Get database configuration
    neo4j_config = get_neo4j_config()
    logger.info(f"Neo4j URI: {neo4j_config['uri']}")
    logger.info(f"Neo4j User: {neo4j_config['username']}")
    logger.info(f"Neo4j Password: {'*' * 8}")
    
    # Get Qdrant configuration
    qdrant_config = get_qdrant_config()
    logger.info(f"Qdrant Host: {qdrant_config['host']}")
    logger.info(f"Qdrant Port: {qdrant_config['port']}")
    if 'api_key' in qdrant_config:
        logger.info(f"Qdrant API Key: {'*' * 8}")
    
    # Example of handling missing configuration values
    try:
        required_config = config.get('REQUIRED_VALUE')
        logger.info(f"Required value: {required_config}")
    except Exception as e:
        logger.warning(f"Caught exception: {e}")
    
    # Example of providing a default for missing configuration
    optional_config = config.get('OPTIONAL_VALUE', 'default-value')
    logger.info(f"Optional value with default: {optional_config}")

if __name__ == "__main__":
    main()
