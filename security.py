"""
Security module for Project Prometheus
Handles authentication, authorization, and security logging
"""
import os
import logging
import time
import hashlib
import secrets
import base64
from functools import wraps
from typing import Dict, List, Optional, Callable, Any
from prometheus_config import config

# Setup logging
log_level = config.get('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "security.log")),
        logging.StreamHandler()
    ]
)

security_logger = logging.getLogger("security")

# Security configuration
API_KEYS = {
    "qdrant": config.get("QDRANT_API_KEY", ""),
    "neo4j": config.get("NEO4J_PASSWORD", "")
}

# Role definitions
ROLES = {
    "reader": {
        "permissions": ["read"],
        "description": "Can only read data"
    },
    "writer": {
        "permissions": ["read", "write"],
        "description": "Can read and write data"
    },
    "admin": {
        "permissions": ["read", "write", "delete", "admin"],
        "description": "Full access to all operations"
    }
}

# In-memory session store (would be replaced by a proper database in production)
active_sessions = {}

def generate_api_key() -> str:
    """Generate a secure API key"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')

def hash_password(password: str) -> str:
    """Hash a password for storage"""
    salt = config.get("PASSWORD_SALT", "prometheus_salt")
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == hashed

def create_session(user_id: str, role: str) -> str:
    """Create a new session for a user"""
    session_id = secrets.token_urlsafe(32)
    active_sessions[session_id] = {
        "user_id": user_id,
        "role": role,
        "created_at": time.time(),
        "expires_at": time.time() + (24 * 60 * 60)  # 24 hours
    }
    security_logger.info(f"Session created for user {user_id} with role {role}")
    return session_id

def validate_session(session_id: str) -> Optional[Dict]:
    """Validate a session and return user info if valid"""
    session = active_sessions.get(session_id)
    if not session:
        security_logger.warning(f"Invalid session ID: {session_id}")
        return None
    
    if session["expires_at"] < time.time():
        security_logger.warning(f"Expired session for user {session['user_id']}")
        del active_sessions[session_id]
        return None
    
    return session

def require_permission(permission: str) -> Callable:
    """Decorator to require a specific permission for a function"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_id = kwargs.get("session_id")
            if not session_id:
                security_logger.error("No session ID provided")
                raise PermissionError("Authentication required")
            
            session = validate_session(session_id)
            if not session:
                security_logger.error("Invalid session")
                raise PermissionError("Invalid or expired session")
            
            role = session["role"]
            if permission not in ROLES[role]["permissions"]:
                security_logger.error(f"Permission denied: {permission} for role {role}")
                raise PermissionError(f"Permission denied: {permission}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def verify_api_key(service: str, key: str) -> bool:
    """Verify an API key for a specific service"""
    if service not in API_KEYS:
        security_logger.error(f"Unknown service: {service}")
        return False
    
    expected_key = API_KEYS[service]
    if not expected_key:
        security_logger.warning(f"No API key configured for {service}")
        return True  # Allow if no key is configured
    
    if key != expected_key:
        security_logger.warning(f"Invalid API key for {service}")
        return False
    
    return True

def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log a security event"""
    security_logger.info(f"SECURITY EVENT: {event_type} - {details}")

# Rate limiting implementation
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self.request_timestamps = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if a request is allowed based on rate limits"""
        current_time = time.time()
        
        # Initialize if this is the first request from this client
        if client_id not in self.request_timestamps:
            self.request_timestamps[client_id] = []
        
        # Remove timestamps older than the window
        self.request_timestamps[client_id] = [
            ts for ts in self.request_timestamps[client_id]
            if current_time - ts < self.window_size
        ]
        
        # Check if the client has exceeded the rate limit
        if len(self.request_timestamps[client_id]) >= self.requests_per_minute:
            security_logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Add the current timestamp
        self.request_timestamps[client_id].append(current_time)
        return True

# Create a global rate limiter instance
rate_limit_rpm = config.get_int('RATE_LIMIT_REQUESTS_PER_MINUTE', 100)
rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
