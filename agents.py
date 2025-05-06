# agents.py
import logging
import os
import time
import psutil
import functools
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional
from prometheus_config import config, get_neo4j_config, get_qdrant_config

# Import security module
from security import verify_api_key, require_permission, log_security_event, rate_limiter

# Configuration
SECURITY = {
    "neo4j": {
        "roles": ["reader", "writer", "admin"],
        "default_role": "reader",
        "auth_required": True
    },
    "qdrant": {
        "api_key_required": config.get("QDRANT_API_KEY", "") != "",  # True if API key is configured
        "rate_limit": {
            "requests_per_minute": config.get_int("RATE_LIMIT_REQUESTS_PER_MINUTE", 1000),
            "window_size_seconds": 60
        }
    }
}

PERFORMANCE = {
    "monitoring": {
        "enabled": True,
        "log_level": config.get("LOG_LEVEL", "INFO"),
        "metrics": [
            "query_latency",
            "retrieval_time",
            "embedding_time",
            "cpu_utilization"
        ]
    },
    "thresholds": {
        "max_query_latency_ms": config.get_int("MAX_QUERY_LATENCY_MS", 300),
        "warning_threshold_ms": config.get_int("WARNING_QUERY_LATENCY_MS", 200)
    }
}

AGENTS = {
    "graph_retriever": {
        "max_retries": 3,
        "retry_delay_seconds": 2,
        "batch_size": 50
    },
    "vector_retriever": {
        "top_k": 5,
        "distance_threshold": 0.8,
        "batch_size": 100
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "agents.log")),
        logging.StreamHandler()
    ]
)

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Get CPU usage before function execution
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().percent
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Calculate metrics after execution
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        cpu_after = psutil.cpu_percent(interval=None)
        memory_after = psutil.virtual_memory().percent
        
        # Calculate resource usage deltas
        cpu_delta = cpu_after - cpu_before
        memory_delta = memory_after - memory_before
        
        func_name = func.__name__
        
        # Create performance metrics
        metrics = {
            "function": func_name,
            "duration_ms": duration,
            "cpu_usage_percent": cpu_delta,
            "memory_usage_percent": memory_delta,
            "timestamp": time.time()
        }
        
        # Log performance metrics
        logging.info(f"PERFORMANCE: {func_name} took {duration:.2f}ms, CPU: {cpu_delta:.1f}%, Memory: {memory_delta:.1f}%")
        
        # Check against performance thresholds
        if duration > PERFORMANCE['thresholds']['max_query_latency_ms']:
            logging.warning(f"ALERT: {func_name} exceeded max latency threshold: {duration:.2f}ms")
        elif duration > PERFORMANCE['thresholds']['warning_threshold_ms']:
            logging.warning(f"WARNING: {func_name} approaching warning threshold: {duration:.2f}ms")
        
        # Store metrics for later analysis
        store_metrics(metrics)
        
        return result
    return wrapper

def store_metrics(metrics: Dict[str, Any]):
    """
    Store performance metrics for later analysis.
    In a production system, this would write to a database or monitoring service.
    """
    # For now, just append to a log file in a specific format for easy parsing
    with open(os.path.join("logs", "performance_metrics.log"), "a") as f:
        f.write(f"{metrics['timestamp']},{metrics['function']},{metrics['duration_ms']:.2f},{metrics['cpu_usage_percent']:.2f},{metrics['memory_usage_percent']:.2f}\n")

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.logger = self._setup_logging()
        self.health_status = {
            "status": "healthy", 
            "last_check": time.time(),
            "metrics": {}
        }
        self.client_id = f"agent_{name}_{id(self)}"
    
    def _setup_logging(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        return logger
    
    def _check_rate_limit(self, operation: str) -> bool:
        """
        Check if the operation is allowed based on rate limits
        """
        operation_key = f"{self.client_id}_{operation}"
        return rate_limiter.is_allowed(operation_key)
    
    @timing_decorator
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the agent and return status information
        """
        # Update last check time
        self.health_status["last_check"] = time.time()
        
        # Add system metrics
        self.health_status["metrics"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return self.health_status
        
    def log_operation(self, operation: str, details: Dict[str, Any], success: bool):
        """
        Log an operation performed by the agent
        """
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(f"OPERATION: {operation} - {status} - {details}")
        
        # Log security events for sensitive operations
        sensitive_operations = ["retrieve", "query_entities", "update", "delete"]
        if operation in sensitive_operations:
            log_security_event(f"{self.name}_{operation}", {
                "status": status,
                "agent": self.name,
                "details": details
            })

class GraphRetrieverAgent(BaseAgent):
    def __init__(self, neo4j_driver=None, uri=None, username=None, password=None, api_key=None):
        super().__init__("GraphRetriever")
        # Use Neo4j password as API key if not specified
        self.api_key = api_key or config.get("NEO4J_PASSWORD", "")
        
        if neo4j_driver:
            self.neo4j_driver = neo4j_driver
        else:
            self.neo4j_driver = self._initialize_neo4j(uri, username, password)
    
    def _initialize_neo4j(self, uri, username, password):
        try:
            # Get Neo4j config from the configuration system
            neo4j_config = get_neo4j_config()
            
            if not uri:
                uri = neo4j_config.get('uri')
            if not username:
                username = neo4j_config.get('username')
            if not password:
                password = neo4j_config.get('password')
            
            # Verify API key for Neo4j access
            if not verify_api_key("neo4j", self.api_key):
                self.logger.error("Invalid Neo4j API key")
                raise PermissionError("Invalid Neo4j API key")
                
            driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_lifetime=3600
            )
            driver.verify_connectivity()
            self.logger.info(f"Successfully connected to Neo4j at {uri}")
            return driver
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j: {e}")
            raise
    
    @timing_decorator
    def retrieve(self, query: str, top_n: int = 5, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve graph relationships matching the query
        """
        operation_details = {"query": query, "top_n": top_n}
        
        # Check rate limits
        if not self._check_rate_limit("retrieve"):
            self.log_operation("retrieve", operation_details, False)
            return []
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)-[r]->(m)
                    WHERE n.text CONTAINS $query OR m.text CONTAINS $query
                    RETURN n, r, m
                    LIMIT $top_n
                    """,
                    {"query": query, "top_n": top_n}
                )
                data = [record.data() for record in result]
                self.log_operation("retrieve", {**operation_details, "results": len(data)}, True)
                return data
        except Exception as e:
            self.logger.error(f"Error in graph retrieval: {e}")
            self.log_operation("retrieve", operation_details, False)
            return []  # Return empty list instead of raising to avoid breaking the application

    @timing_decorator
    def query_entities(self, query: str, top_n: int = 5, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Query for entities related to the search query
        """
        operation_details = {"query": query, "top_n": top_n}
        
        # Check rate limits
        if not self._check_rate_limit("query_entities"):
            self.log_operation("query_entities", operation_details, False)
            return []
            
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query
                    RETURN e
                    LIMIT $top_n
                    """,
                    {"query": query, "top_n": top_n}
                )
                data = [record.data() for record in result]
                self.log_operation("query_entities", {**operation_details, "results": len(data)}, True)
                return data
        except Exception as e:
            self.logger.error(f"Error in entity retrieval: {e}")
            self.log_operation("query_entities", operation_details, False)
            return []
            
    @timing_decorator
    def query_relationships(self, entity_type: str, relationship_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query for specific entity relationships
        """
        operation_details = {"entity_type": entity_type, "relationship_type": relationship_type, "limit": limit}
        
        # Check rate limits
        if not self._check_rate_limit("query_relationships"):
            self.log_operation("query_relationships", operation_details, False)
            return []
            
        try:
            with self.neo4j_driver.session() as session:
                # Build the Cypher query based on provided parameters
                if relationship_type:
                    cypher = f"""
                    MATCH (n:{entity_type})-[r:{relationship_type}]->(m)
                    RETURN n, r, m
                    LIMIT $limit
                    """
                else:
                    cypher = f"""
                    MATCH (n:{entity_type})-[r]->(m)
                    RETURN n, r, m
                    LIMIT $limit
                    """
                    
                result = session.run(cypher, {"limit": limit})
                data = [record.data() for record in result]
                self.log_operation("query_relationships", {**operation_details, "results": len(data)}, True)
                return data
        except Exception as e:
            self.logger.error(f"Error in relationship retrieval: {e}")
            self.log_operation("query_relationships", operation_details, False)
            return []

class VectorRetrieverAgent(BaseAgent):
    def __init__(self, qdrant_client=None, host=None, port=None, api_key=None):
        super().__init__("VectorRetriever")
        self.api_key = api_key or config.get("QDRANT_API_KEY", "")
        
        if qdrant_client:
            self.qdrant_client = qdrant_client
        else:
            self.qdrant_client = self._initialize_qdrant(host, port)
        
        # Cache for frequently accessed vectors
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = config.get_int("CACHE_SIZE", 1000)  # Get from config
    
    def _initialize_qdrant(self, host, port):
        try:
            # Get Qdrant config from the configuration system
            qdrant_config = get_qdrant_config()
            
            if not host:
                host = qdrant_config.get('host')
            if not port:
                port = qdrant_config.get('port')
            
            # Verify API key for Qdrant access
            if not verify_api_key("qdrant", self.api_key):
                self.logger.error("Invalid Qdrant API key")
                raise PermissionError("Invalid Qdrant API key")
                
            client = QdrantClient(
                host=host,
                port=port
            )
            
            self.logger.info(f"Successfully connected to Qdrant at {host}:{port}")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def _get_cache_key(self, collection: str, query_vector: List[float], top_k: int) -> str:
        """
        Generate a cache key for vector retrieval
        """
        # Use first 5 and last 5 elements of vector to create a simpler key
        vector_sample = query_vector[:5] + query_vector[-5:] if len(query_vector) > 10 else query_vector
        vector_str = ",".join([f"{v:.6f}" for v in vector_sample])
        return f"{collection}_{vector_str}_{top_k}"
    
    def _manage_cache(self):
        """
        Manage the cache size by removing least recently used items
        """
        if len(self.cache) > self.max_cache_size:
            # Remove 20% of the oldest entries
            items_to_remove = int(self.max_cache_size * 0.2)
            sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k]["last_accessed"])
            for key in sorted_keys[:items_to_remove]:
                del self.cache[key]
    
    @timing_decorator
    def retrieve(self, query_vector: List[float], collection: str, top_k: int = 5, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve vectors from the collection that match the query vector
        """
        operation_details = {"collection": collection, "top_k": top_k, "vector_size": len(query_vector)}
        
        # Check rate limits
        if not self._check_rate_limit("retrieve"):
            self.log_operation("retrieve", operation_details, False)
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(collection, query_vector, top_k)
        if cache_key in self.cache:
            self.cache_hits += 1
            self.cache[cache_key]["last_accessed"] = time.time()
            self.cache[cache_key]["hits"] += 1
            self.log_operation("retrieve", {**operation_details, "cache_hit": True}, True)
            return self.cache[cache_key]["results"]
        
        # Cache miss, perform the search
        self.cache_misses += 1
        try:
            results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k
            )
            
            # Store in cache
            self.cache[cache_key] = {
                "results": results,
                "last_accessed": time.time(),
                "hits": 1
            }
            
            # Manage cache size
            self._manage_cache()
            
            self.log_operation("retrieve", {**operation_details, "results": len(results)}, True)
            return results
        except Exception as e:
            self.logger.error(f"Error in vector retrieval: {e}")
            self.log_operation("retrieve", operation_details, False)
            return []  # Return empty list instead of raising to avoid breaking the application
    
    @timing_decorator
    def retrieve_by_filter(self, collection: str, filter_dict: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve vectors from the collection using metadata filters
        """
        operation_details = {"collection": collection, "filter": filter_dict, "top_k": top_k}
        
        # Check rate limits
        if not self._check_rate_limit("retrieve_by_filter"):
            self.log_operation("retrieve_by_filter", operation_details, False)
            return []
            
        try:
            results = self.qdrant_client.scroll(
                collection_name=collection,
                filter=filter_dict,
                limit=top_k
            )[0]  # scroll returns (points, next_page_offset)
            
            self.log_operation("retrieve_by_filter", {**operation_details, "results": len(results)}, True)
            return results
        except Exception as e:
            self.logger.error(f"Error in filter-based retrieval: {e}")
            self.log_operation("retrieve_by_filter", operation_details, False)
            return []
    
    @timing_decorator
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache performance
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0
        
        stats = {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "most_accessed": []
        }
        
        # Get the most frequently accessed items
        if self.cache:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["hits"], reverse=True)[:5]
            stats["most_accessed"] = [
                {"key": key.split("_")[0], "hits": value["hits"]} 
                for key, value in sorted_items
            ]
        
        return stats
