"""
Project Prometheus Configuration Module

This module provides a centralized way to access configuration and secrets
throughout the Project Prometheus application. It handles environment variable
loading, validation, and provides a simple interface for accessing configuration.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PrometheusConfig")

# Define the environment file path
ENV_FILE_PATH = Path("env/.env")

# Define required environment variables with their types and default values
CONFIG_SCHEMA = {
    "GOOGLE_API_KEY": {"type": str, "required": True, "sensitive": True},
    "NEO4J_URI": {"type": str, "required": True, "default": "bolt://localhost:7687"},
    "NEO4J_USERNAME": {"type": str, "required": True, "default": "neo4j"},
    "NEO4J_PASSWORD": {"type": str, "required": True, "sensitive": True},
    "QDRANT_HOST": {"type": str, "required": True, "default": "localhost"},
    "QDRANT_PORT": {"type": int, "required": True, "default": 6333},
    "QDRANT_API_KEY": {"type": str, "required": False, "sensitive": True},
    "PASSWORD_SALT": {"type": str, "required": False, "sensitive": True},
    "RATE_LIMIT_REQUESTS_PER_MINUTE": {"type": int, "required": False, "default": 100},
    "LOG_LEVEL": {"type": str, "required": False, "default": "INFO"},
    "CACHE_SIZE": {"type": int, "required": False, "default": 1000},
    "CACHE_TTL_SECONDS": {"type": int, "required": False, "default": 3600},
    "MAX_QUERY_LATENCY_MS": {"type": int, "required": False, "default": 300},
    "WARNING_QUERY_LATENCY_MS": {"type": int, "required": False, "default": 200},
}

class MissingConfigError(Exception):
    """Exception raised when a required configuration value is missing."""
    pass

class PrometheusConfig:
    """Configuration manager for the Project Prometheus system."""
    
    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to the environment file (defaults to env/.env)
        """
        self.env_file = env_file or ENV_FILE_PATH
        self._load_environment()
        self._config_cache: Dict[str, Any] = {}
        self._validate_required_config()
    
    def _load_environment(self) -> None:
        """Load environment variables from the specified file."""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        else:
            logger.warning(f"Environment file not found: {self.env_file}")
    
    def _validate_required_config(self) -> None:
        """Validate that all required configuration values are present."""
        missing = []
        for key, schema in CONFIG_SCHEMA.items():
            if schema.get("required", False):
                value = os.getenv(key)
                if value is None and "default" not in schema:
                    missing.append(key)
        
        if missing:
            logger.warning(f"Missing required configuration: {', '.join(missing)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if not found
            
        Returns:
            The configuration value with appropriate type conversion
        
        Raises:
            MissingConfigError: If the configuration is required and not found
        """
        # Check cache first
        if key in self._config_cache:
            return self._config_cache[key]
        
        # Get schema for this key
        schema = CONFIG_SCHEMA.get(key)
        if schema is None:
            logger.warning(f"Accessing undefined configuration key: {key}")
            value = os.getenv(key, default)
            self._config_cache[key] = value
            return value
        
        # Get value with schema validation
        value = os.getenv(key)
        
        # If value is not found, use default from schema or provided default
        if value is None:
            if "default" in schema:
                value = schema["default"]
            else:
                value = default
        
        # If still no value and it's required, raise an error
        if value is None and schema.get("required", False):
            raise MissingConfigError(f"Missing required configuration: {key}")
        
        # Type conversion
        if value is not None:
            try:
                if schema["type"] == int:
                    value = int(value)
                elif schema["type"] == float:
                    value = float(value)
                elif schema["type"] == bool:
                    value = value.lower() in ("yes", "true", "t", "1")
            except (ValueError, TypeError):
                logger.warning(f"Invalid type for {key}: expected {schema['type'].__name__}")
        
        # Cache the result
        self._config_cache[key] = value
        return value
    
    def get_int(self, key: str, default: Optional[int] = None) -> int:
        """Get an integer configuration value."""
        value = self.get(key, default)
        if value is None:
            return 0
        if not isinstance(value, int):
            try:
                return int(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid integer value for {key}: {value}")
                return 0
        return value
    
    def get_float(self, key: str, default: Optional[float] = None) -> float:
        """Get a float configuration value."""
        value = self.get(key, default)
        if value is None:
            return 0.0
        if not isinstance(value, float):
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid float value for {key}: {value}")
                return 0.0
        return value
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean configuration value."""
        value = self.get(key, default)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("yes", "true", "t", "1")
        return bool(value)
    
    def get_dict(self, prefix: str) -> Dict[str, Any]:
        """
        Get a dictionary of all configuration values with the given prefix.
        
        Args:
            prefix: The prefix to filter by (e.g., 'NEO4J_')
            
        Returns:
            Dictionary of configuration values
        """
        result = {}
        for key in os.environ:
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                short_key = key[len(prefix):].lower()
                result[short_key] = self.get(key)
        return result
    
    def get_neo4j_config(self) -> Dict[str, str]:
        """Get Neo4j connection configuration."""
        return {
            'uri': self.get('NEO4J_URI'),
            'username': self.get('NEO4J_USERNAME'),
            'password': self.get('NEO4J_PASSWORD')
        }
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant connection configuration."""
        config = {
            'host': self.get('QDRANT_HOST'),
            'port': self.get_int('QDRANT_PORT')
        }
        
        # Add API key if available
        api_key = self.get('QDRANT_API_KEY', None)
        if api_key:
            config['api_key'] = api_key
            
        return config
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.get('ENVIRONMENT', 'development').lower() == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.get('ENVIRONMENT', 'development').lower() == 'production'

# Create a global instance for easy imports
config = PrometheusConfig()

# Helper functions for easy access
def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return config.get(key, default)

def get_neo4j_config() -> Dict[str, str]:
    """Get Neo4j connection configuration."""
    return config.get_neo4j_config()

def get_qdrant_config() -> Dict[str, Any]:
    """Get Qdrant connection configuration."""
    return config.get_qdrant_config()
