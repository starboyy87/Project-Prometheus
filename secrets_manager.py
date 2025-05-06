"""
Project Prometheus Secrets Manager

This script provides tools to securely manage environment variables and secrets
for the Project Prometheus system. It reads from the existing env/.env file
and provides functionality to validate, backup, and update secrets.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv, set_key, find_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SecretManager")

# Define the environment file path
ENV_FILE_PATH = Path("env/.env")

# Define required environment variables
REQUIRED_VARIABLES = [
    "GOOGLE_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
]

# Define sensitive environment variables
SENSITIVE_VARIABLES = [
    "GOOGLE_API_KEY",
    "NEO4J_PASSWORD",
    "QDRANT_API_KEY",
]

def load_secrets(env_file: Path = ENV_FILE_PATH) -> Dict[str, str]:
    """Load secrets from the environment file."""
    if not env_file.exists():
        logger.error(f"Environment file not found: {env_file}")
        sys.exit(1)
    
    load_dotenv(env_file)
    
    secrets = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    key, value = line.split('=', 1)
                    secrets[key] = value
                except ValueError:
                    logger.warning(f"Skipping invalid line in .env file: {line}")
    
    return secrets

def validate_secrets(secrets: Dict[str, str]) -> List[str]:
    """Validate that all required secrets are present and have values."""
    missing = []
    for var in REQUIRED_VARIABLES:
        if var not in secrets or not secrets[var]:
            missing.append(var)
    return missing

def backup_env_file(env_file: Path = ENV_FILE_PATH) -> Optional[Path]:
    """Create a backup of the environment file."""
    if not env_file.exists():
        logger.error(f"Environment file not found: {env_file}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = env_file.parent / f".env.backup.{timestamp}"
    
    try:
        shutil.copy2(env_file, backup_file)
        logger.info(f"Created backup of environment file: {backup_file}")
        return backup_file
    except Exception as e:
        logger.error(f"Failed to backup environment file: {e}")
        return None

def create_secret(key: str, value: str, env_file: Path = ENV_FILE_PATH) -> bool:
    """Create or update a secret in the environment file."""
    try:
        if not env_file.exists():
            env_file.parent.mkdir(parents=True, exist_ok=True)
            env_file.touch()
        
        set_key(str(env_file), key, value)
        logger.info(f"Updated secret: {key}")
        return True
    except Exception as e:
        logger.error(f"Failed to update secret {key}: {e}")
        return False

def print_status():
    """Print the status of the secrets management."""
    try:
        secrets = load_secrets()
        missing = validate_secrets(secrets)
        
        print("\n=== Project Prometheus Secrets Status ===")
        print(f"Environment file: {ENV_FILE_PATH}")
        print(f"Total secrets defined: {len(secrets)}")
        
        if missing:
            print(f"\n⚠️  Missing required secrets: {', '.join(missing)}")
        else:
            print("\n✅ All required secrets are defined")
        
        # Print a masked version of the secrets
        print("\nDefined secrets:")
        for key, value in secrets.items():
            if key in SENSITIVE_VARIABLES:
                # Show only first and last 2 characters
                masked_value = value[:2] + "****" + value[-2:] if len(value) > 6 else "****"
                print(f"  {key}: {masked_value}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error checking status: {e}")

def backup_secrets():
    """Create a backup of the secrets file."""
    backup_path = backup_env_file()
    if backup_path:
        print(f"\n✅ Backup created at: {backup_path}")
    else:
        print("\n❌ Failed to create backup")

def update_secret(key: str):
    """Update a specific secret."""
    value = input(f"Enter new value for {key}: ")
    
    if not value:
        print("Empty value provided. Secret not updated.")
        return
    
    # Backup before updating
    backup_secrets()
    
    # Update the secret
    if create_secret(key, value):
        print(f"\n✅ Updated secret: {key}")
    else:
        print(f"\n❌ Failed to update secret: {key}")

def import_secrets(import_file: str):
    """Import secrets from another file."""
    import_path = Path(import_file)
    if not import_path.exists():
        print(f"❌ Import file not found: {import_file}")
        return
    
    try:
        # Load secrets from the import file
        import_secrets = {}
        with open(import_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        import_secrets[key] = value
                    except ValueError:
                        continue
        
        if not import_secrets:
            print("❌ No valid secrets found in import file")
            return
        
        # Backup before importing
        backup_secrets()
        
        # Import each secret
        success_count = 0
        for key, value in import_secrets.items():
            if create_secret(key, value):
                success_count += 1
        
        print(f"\n✅ Imported {success_count} secrets from {import_file}")
    except Exception as e:
        print(f"❌ Error importing secrets: {e}")

def main():
    """Main function to run the secrets manager."""
    if len(sys.argv) < 2:
        print("\nProject Prometheus Secrets Manager")
        print("\nUsage:")
        print("  python secrets_manager.py status           - Show current secrets status")
        print("  python secrets_manager.py backup           - Create a backup of the secrets file")
        print("  python secrets_manager.py update <KEY>     - Update a specific secret")
        print("  python secrets_manager.py import <FILE>    - Import secrets from a file")
        sys.exit(1)

    command = sys.argv[1]

    if command == "status":
        print_status()
    
    elif command == "backup":
        backup_secrets()
    
    elif command == "update":
        if len(sys.argv) < 3:
            print("❌ Missing key to update")
            print("Usage: python secrets_manager.py update <KEY>")
            sys.exit(1)
        
        key = sys.argv[2]
        update_secret(key)
    
    elif command == "import":
        if len(sys.argv) < 3:
            print("❌ Missing file to import from")
            print("Usage: python secrets_manager.py import <FILE>")
            sys.exit(1)
        
        import_file = sys.argv[2]
        import_secrets(import_file)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'status', 'backup', 'update', or 'import'")
        sys.exit(1)

if __name__ == "__main__":
    main()
