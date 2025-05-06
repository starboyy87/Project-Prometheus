import os
from typing import Optional, Dict
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import logging
from pathlib import Path

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

class ConfigManager:
    def __init__(self, env_file: str = '.env'):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to the environment file
        """
        self.logger = logging.getLogger(__name__)
        self.env_file = env_file
        self._load_environment()
        self._validate_configuration()
        
    def _load_environment(self) -> None:
        """Load environment variables from the specified file."""
        try:
            load_dotenv(self.env_file)
            self.logger.info(f"Loaded environment variables from {self.env_file}")
        except Exception as e:
            self.logger.error(f"Error loading environment variables: {e}")
            raise ConfigError(f"Failed to load environment variables: {e}")

    def _validate_configuration(self) -> None:
        """Validate that all required environment variables are present and valid."""
        required_vars = {
            'GOOGLE_API_KEY': str,
            'NEO4J_URI': str,
            'NEO4J_USERNAME': str,
            'NEO4J_PASSWORD': str,
            'QDRANT_HOST': str,
            'QDRANT_PORT': int,
            'QDRANT_API_KEY': str,
            'PASSWORD_SALT': str,
            'RATE_LIMIT_REQUESTS_PER_MINUTE': int,
            'LOG_LEVEL': str,
            'CACHE_SIZE': int,
            'CACHE_TTL_SECONDS': int,
            'MAX_QUERY_LATENCY_MS': int,
            'WARNING_QUERY_LATENCY_MS': int
        }

        for var, expected_type in required_vars.items():
            value = os.getenv(var)
            if not value:
                raise ConfigError(f"Missing required environment variable: {var}")
            
            try:
                if expected_type == int:
                    int(value)
                elif expected_type == float:
                    float(value)
            except ValueError:
                raise ConfigError(f"Invalid value for {var}: must be {expected_type.__name__}")

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        """
        Get a configuration value with optional encryption.
        
        Args:
            key: The environment variable name
            default: Default value if not found
            
        Returns:
            The configuration value
        """
        value = os.getenv(key, default)
        if not value:
            raise ConfigError(f"Configuration value not found: {key}")
        return value

    def get_int_config(self, key: str, default: Optional[int] = None) -> int:
        """Get an integer configuration value."""
        return int(self.get_config(key, str(default)))

    def get_float_config(self, key: str, default: Optional[float] = None) -> float:
        """Get a float configuration value."""
        return float(self.get_config(key, str(default)))

    def get_bool_config(self, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean configuration value."""
        value = self.get_config(key, str(default)).lower()
        return value in ['true', '1', 'yes', 'on']

    def generate_encryption_key(self) -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()

    def encrypt_value(self, value: str, encryption_key: str) -> str:
        """Encrypt a value using the provided encryption key."""
        f = Fernet(encryption_key.encode())
        return f.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str, encryption_key: str) -> str:
        """Decrypt a value using the provided encryption key."""
        f = Fernet(encryption_key.encode())
        return f.decrypt(encrypted_value.encode()).decode()

    def save_encrypted_config(self, config: Dict[str, str], encryption_key: str) -> None:
        """
        Save configuration with encrypted sensitive values.
        
        Args:
            config: Dictionary of configuration values
            encryption_key: The encryption key to use
        """
        encrypted_config = {}
        sensitive_keys = ['GOOGLE_API_KEY', 'NEO4J_PASSWORD', 'QDRANT_API_KEY', 'PASSWORD_SALT']
        
        for key, value in config.items():
            if key in sensitive_keys:
                encrypted_config[key] = self.encrypt_value(value, encryption_key)
            else:
                encrypted_config[key] = value

        with open('.env.encrypted', 'w') as f:
            for key, value in encrypted_config.items():
                f.write(f"{key}={value}\n")

    def load_encrypted_config(self, encryption_key: str) -> Dict[str, str]:
        """
        Load configuration with decrypted sensitive values.
        
        Args:
            encryption_key: The encryption key to use
            
        Returns:
            Dictionary of decrypted configuration values
        """
        encrypted_config = {}
        with open('.env.encrypted', 'r') as f:
            for line in f:
                if line.strip():
                    key, value = line.strip().split('=', 1)
                    encrypted_config[key] = value

        decrypted_config = {}
        for key, value in encrypted_config.items():
            if key in ['GOOGLE_API_KEY', 'NEO4J_PASSWORD', 'QDRANT_API_KEY', 'PASSWORD_SALT']:
                decrypted_config[key] = self.decrypt_value(value, encryption_key)
            else:
                decrypted_config[key] = value

        return decrypted_config

# Initialize global config manager
config_manager = ConfigManager('env/.env')
