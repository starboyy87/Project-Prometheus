"""
Test script for Project Prometheus agent system
Tests security, performance, and functionality
"""
import os
import time
import logging
import argparse
from typing import Dict, List, Any
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from prometheus_config import config, get_neo4j_config, get_qdrant_config

# Import our modules
from agents import GraphRetrieverAgent, VectorRetrieverAgent
from security import verify_api_key, log_security_event, rate_limiter

# Configure logging
log_level = config.get('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "test_agents.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_agents")

class AgentTester:
    def __init__(self):
        """Initialize the agent tester"""
        # Get configuration from the config system
        neo4j_config = get_neo4j_config()
        self.neo4j_uri = neo4j_config.get('uri')
        self.neo4j_username = neo4j_config.get('username')
        self.neo4j_password = neo4j_config.get('password')
        
        qdrant_config = get_qdrant_config()
        self.qdrant_host = qdrant_config.get('host')
        self.qdrant_port = qdrant_config.get('port')
        
        # Initialize clients
        self.neo4j_driver = None
        self.qdrant_client = None
        
        # Initialize agents
        self.graph_agent = None
        self.vector_agent = None
        
        # Test results
        self.results = {
            "security_tests": {},
            "performance_tests": {},
            "functionality_tests": {}
        }
    
    def setup(self) -> bool:
        """Set up the test environment"""
        try:
            # Connect to Neo4j
            logger.info(f"Connecting to Neo4j at {self.neo4j_uri}")
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection successful")
            
            # Connect to Qdrant
            logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            # Verify Qdrant connection by getting collections
            self.qdrant_client.get_collections()
            logger.info("Qdrant connection successful")
            
            # Initialize agents
            self.graph_agent = GraphRetrieverAgent(neo4j_driver=self.neo4j_driver)
            self.vector_agent = VectorRetrieverAgent(qdrant_client=self.qdrant_client)
            
            logger.info("Test environment setup complete")
            return True
        except Exception as e:
            logger.error(f"Error setting up test environment: {e}")
            return False
    
    def teardown(self) -> None:
        """Clean up after tests"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")
    
    def test_security(self) -> Dict[str, Any]:
        """Test security features"""
        results = {}
        
        # Test 1: API key verification
        logger.info("Testing API key verification")
        results["api_key_verification"] = {
            "valid_key": verify_api_key("neo4j", self.neo4j_password),
            "invalid_key": not verify_api_key("neo4j", "invalid_key_123")
        }
        
        # Test 2: Rate limiting
        logger.info("Testing rate limiting")
        rate_limit_results = {"allowed": 0, "blocked": 0}
        test_client = "test_client_123"
        
        # Try to make requests beyond the rate limit
        for i in range(150):  # Assuming limit is 100 requests per minute
            if rate_limiter.is_allowed(test_client):
                rate_limit_results["allowed"] += 1
            else:
                rate_limit_results["blocked"] += 1
        
        results["rate_limiting"] = rate_limit_results
        
        # Test 3: Security logging
        logger.info("Testing security logging")
        log_security_event("test_event", {"test": "data"})
        results["security_logging"] = True  # Just testing that it doesn't raise an exception
        
        self.results["security_tests"] = results
        return results
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance features"""
        results = {}
        
        if not self.graph_agent or not self.vector_agent:
            logger.error("Agents not initialized")
            return {"error": "Agents not initialized"}
        
        # Test 1: Graph retrieval performance
        logger.info("Testing graph retrieval performance")
        graph_times = []
        for i in range(5):
            start_time = time.time()
            self.graph_agent.retrieve("test query", top_n=3)
            end_time = time.time()
            graph_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        results["graph_retrieval"] = {
            "avg_time_ms": sum(graph_times) / len(graph_times) if graph_times else 0,
            "max_time_ms": max(graph_times) if graph_times else 0,
            "min_time_ms": min(graph_times) if graph_times else 0
        }
        
        # Test 2: Vector retrieval performance with caching
        logger.info("Testing vector retrieval performance with caching")
        # Create a test vector
        test_vector = [0.1] * 768  # Assuming 768-dimensional vectors
        
        # First request (cache miss)
        start_time = time.time()
        self.vector_agent.retrieve(test_vector, "doc_chunks_poc", top_k=3)
        first_request_time = (time.time() - start_time) * 1000
        
        # Second request (should be cache hit)
        start_time = time.time()
        self.vector_agent.retrieve(test_vector, "doc_chunks_poc", top_k=3)
        second_request_time = (time.time() - start_time) * 1000
        
        results["vector_retrieval"] = {
            "first_request_ms": first_request_time,
            "second_request_ms": second_request_time,
            "speedup_factor": first_request_time / second_request_time if second_request_time > 0 else 0
        }
        
        # Test 3: Cache statistics
        logger.info("Testing cache statistics")
        cache_stats = self.vector_agent.get_cache_stats()
        results["cache_stats"] = cache_stats
        
        self.results["performance_tests"] = results
        return results
    
    def test_functionality(self) -> Dict[str, Any]:
        """Test agent functionality"""
        results = {}
        
        if not self.graph_agent or not self.vector_agent:
            logger.error("Agents not initialized")
            return {"error": "Agents not initialized"}
        
        # Test 1: Graph agent health check
        logger.info("Testing graph agent health check")
        graph_health = self.graph_agent.health_check()
        results["graph_agent_health"] = {
            "status": graph_health["status"],
            "metrics_available": len(graph_health["metrics"]) > 0
        }
        
        # Test 2: Vector agent health check
        logger.info("Testing vector agent health check")
        vector_health = self.vector_agent.health_check()
        results["vector_agent_health"] = {
            "status": vector_health["status"],
            "metrics_available": len(vector_health["metrics"]) > 0
        }
        
        # Test 3: Graph agent query functionality
        logger.info("Testing graph agent query functionality")
        try:
            # Test basic retrieval
            retrieval_results = self.graph_agent.retrieve("test", top_n=3)
            
            # Test entity queries
            entity_results = self.graph_agent.query_entities("test", top_n=3)
            
            # Test relationship queries (may fail if no entities of this type exist)
            try:
                relationship_results = self.graph_agent.query_relationships("Entity", limit=3)
                relationship_success = True
            except Exception:
                relationship_success = False
            
            results["graph_agent_queries"] = {
                "retrieval_success": True,
                "entity_query_success": True,
                "relationship_query_success": relationship_success
            }
        except Exception as e:
            logger.error(f"Error testing graph agent queries: {e}")
            results["graph_agent_queries"] = {
                "retrieval_success": False,
                "entity_query_success": False,
                "relationship_query_success": False,
                "error": str(e)
            }
        
        # Test 4: Vector agent query functionality
        logger.info("Testing vector agent query functionality")
        try:
            # Create a test vector
            test_vector = [0.1] * 768  # Assuming 768-dimensional vectors
            
            # Test basic retrieval
            retrieval_results = self.vector_agent.retrieve(test_vector, "doc_chunks_poc", top_k=3)
            
            # Test filter-based retrieval
            try:
                filter_results = self.vector_agent.retrieve_by_filter("doc_chunks_poc", {"metadata.type": "text"}, top_k=3)
                filter_success = True
            except Exception:
                filter_success = False
            
            results["vector_agent_queries"] = {
                "retrieval_success": True,
                "filter_query_success": filter_success
            }
        except Exception as e:
            logger.error(f"Error testing vector agent queries: {e}")
            results["vector_agent_queries"] = {
                "retrieval_success": False,
                "filter_query_success": False,
                "error": str(e)
            }
        
        self.results["functionality_tests"] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        if not self.setup():
            return {"error": "Failed to set up test environment"}
        
        try:
            logger.info("Starting security tests")
            self.test_security()
            
            logger.info("Starting performance tests")
            self.test_performance()
            
            logger.info("Starting functionality tests")
            self.test_functionality()
            
            return self.results
        finally:
            self.teardown()
    
    def print_results(self) -> None:
        """Print test results to the console"""
        print("\n" + "="*50)
        print("PROMETHEUS AGENT SYSTEM TEST RESULTS")
        print("="*50)
        
        # Security tests
        print("\nSECURITY TESTS:")
        print("-"*50)
        security = self.results.get("security_tests", {})
        
        if "api_key_verification" in security:
            api_key = security["api_key_verification"]
            print(f"API Key Verification:")
            print(f"  - Valid key accepted: {api_key.get('valid_key', False)}")
            print(f"  - Invalid key rejected: {api_key.get('invalid_key', False)}")
        
        if "rate_limiting" in security:
            rate_limit = security["rate_limiting"]
            print(f"Rate Limiting:")
            print(f"  - Allowed requests: {rate_limit.get('allowed', 0)}")
            print(f"  - Blocked requests: {rate_limit.get('blocked', 0)}")
        
        # Performance tests
        print("\nPERFORMANCE TESTS:")
        print("-"*50)
        performance = self.results.get("performance_tests", {})
        
        if "graph_retrieval" in performance:
            graph = performance["graph_retrieval"]
            print(f"Graph Retrieval Performance:")
            print(f"  - Average time: {graph.get('avg_time_ms', 0):.2f} ms")
            print(f"  - Maximum time: {graph.get('max_time_ms', 0):.2f} ms")
            print(f"  - Minimum time: {graph.get('min_time_ms', 0):.2f} ms")
        
        if "vector_retrieval" in performance:
            vector = performance["vector_retrieval"]
            print(f"Vector Retrieval Performance:")
            print(f"  - First request (cache miss): {vector.get('first_request_ms', 0):.2f} ms")
            print(f"  - Second request (cache hit): {vector.get('second_request_ms', 0):.2f} ms")
            print(f"  - Speedup factor: {vector.get('speedup_factor', 0):.2f}x")
        
        if "cache_stats" in performance:
            cache = performance["cache_stats"]
            print(f"Cache Statistics:")
            print(f"  - Cache size: {cache.get('cache_size', 0)}")
            print(f"  - Cache hits: {cache.get('cache_hits', 0)}")
            print(f"  - Cache misses: {cache.get('cache_misses', 0)}")
            print(f"  - Hit rate: {cache.get('hit_rate_percent', 0):.2f}%")
        
        # Functionality tests
        print("\nFUNCTIONALITY TESTS:")
        print("-"*50)
        functionality = self.results.get("functionality_tests", {})
        
        if "graph_agent_health" in functionality:
            graph_health = functionality["graph_agent_health"]
            print(f"Graph Agent Health:")
            print(f"  - Status: {graph_health.get('status', 'unknown')}")
            print(f"  - Metrics available: {graph_health.get('metrics_available', False)}")
        
        if "vector_agent_health" in functionality:
            vector_health = functionality["vector_agent_health"]
            print(f"Vector Agent Health:")
            print(f"  - Status: {vector_health.get('status', 'unknown')}")
            print(f"  - Metrics available: {vector_health.get('metrics_available', False)}")
        
        if "graph_agent_queries" in functionality:
            graph_queries = functionality["graph_agent_queries"]
            print(f"Graph Agent Queries:")
            print(f"  - Retrieval: {'Success' if graph_queries.get('retrieval_success', False) else 'Failed'}")
            print(f"  - Entity Query: {'Success' if graph_queries.get('entity_query_success', False) else 'Failed'}")
            print(f"  - Relationship Query: {'Success' if graph_queries.get('relationship_query_success', False) else 'Failed'}")
            if "error" in graph_queries:
                print(f"  - Error: {graph_queries['error']}")
        
        if "vector_agent_queries" in functionality:
            vector_queries = functionality["vector_agent_queries"]
            print(f"Vector Agent Queries:")
            print(f"  - Retrieval: {'Success' if vector_queries.get('retrieval_success', False) else 'Failed'}")
            print(f"  - Filter Query: {'Success' if vector_queries.get('filter_query_success', False) else 'Failed'}")
            if "error" in vector_queries:
                print(f"  - Error: {vector_queries['error']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prometheus Agent System Tester")
    parser.add_argument("--security", action="store_true", help="Run only security tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--functionality", action="store_true", help="Run only functionality tests")
    args = parser.parse_args()
    
    tester = AgentTester()
    
    # If no specific tests are requested, run all tests
    run_all = not (args.security or args.performance or args.functionality)
    
    if not tester.setup():
        print("Failed to set up test environment")
        return
    
    try:
        if run_all or args.security:
            tester.test_security()
        
        if run_all or args.performance:
            tester.test_performance()
        
        if run_all or args.functionality:
            tester.test_functionality()
        
        tester.print_results()
    finally:
        tester.teardown()

if __name__ == "__main__":
    main()
