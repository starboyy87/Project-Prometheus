from config_manager import ConfigManager
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def main():
    logging.basicConfig(level=logging.INFO)
    env_path = 'env/.env'
    config_manager = ConfigManager(env_path)
    
    if len(sys.argv) < 2:
        print("Usage: python manage_secrets.py [encrypt|decrypt|generate_key]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "encrypt":
        if len(sys.argv) < 3:
            print("Usage: python manage_secrets.py encrypt <encryption_key>")
            sys.exit(1)
            
        encryption_key = sys.argv[2]
        env_file = Path(env_path)
        if not env_file.exists():
            print(f"Error: Environment file {env_path} not found")
            sys.exit(1)
        
        # Load existing environment variables
        load_dotenv(env_path)
        
        # Read all environment variables
        config = {}
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
                    
        # Create encrypted backup
        config_manager.save_encrypted_config(config, encryption_key)
        print(f"Configuration successfully encrypted and saved to .env.encrypted")
        print(f"Original configuration at {env_path} remains unchanged")
    
    elif command == "decrypt":
        if len(sys.argv) < 4:
            print("Usage: python manage_secrets.py decrypt <encryption_key> [--apply]")
            print("  --apply: Apply decrypted values to env/.env file (BE CAREFUL!)")
            sys.exit(1)
            
        encryption_key = sys.argv[2]
        apply_changes = len(sys.argv) > 3 and sys.argv[3] == "--apply"
        
        try:
            decrypted_config = config_manager.load_encrypted_config(encryption_key)
            print("\nDecrypted Configuration:")
            for key, value in decrypted_config.items():
                print(f"{key}={value}")
                
            if apply_changes:
                apply = input("\nWARNING: This will update your env/.env file. Continue? (yes/no): ")
                if apply.lower() == "yes":
                    # Update the .env file with decrypted values
                    for key, value in decrypted_config.items():
                        set_key(env_path, key, value)
                    print(f"Updated {env_path} with decrypted values")
                else:
                    print("Operation cancelled")
        except Exception as e:
            print(f"Error decrypting configuration: {e}")
    
    elif command == "generate_key":
        encryption_key = config_manager.generate_encryption_key()
        print(f"Generated Encryption Key: {encryption_key}")
        print("\nImportant: Save this key in a secure place. You'll need it to decrypt your configuration.")
        
        save_key = input("Would you like to save this key to a file? (yes/no): ")
        if save_key.lower() == "yes":
            key_file = input("Enter path to save the key (default: key.txt): ") or "key.txt"
            with open(key_file, 'w') as f:
                f.write(encryption_key)
            print(f"Key saved to {key_file}")
            print("IMPORTANT: Keep this file secure and back it up safely!")
    
    elif command == "status":
        # Check if .env file exists
        env_file = Path(env_path)
        encrypted_file = Path(".env.encrypted")
        
        print("Secrets Management Status:")
        print(f"  Environment file: {'✓ Found' if env_file.exists() else '✗ Not found'} ({env_path})")
        print(f"  Encrypted backup: {'✓ Found' if encrypted_file.exists() else '✗ Not found'} (.env.encrypted)")
        
        if env_file.exists():
            # Count variables in .env file
            var_count = 0
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        var_count += 1
            print(f"  Environment variables: {var_count} defined")
    else:
        print("Invalid command. Use 'encrypt', 'decrypt', 'generate_key', or 'status'")
        print("\nUsage:")
        print("  python manage_secrets.py status                    - Check secrets status")
        print("  python manage_secrets.py generate_key             - Generate a new encryption key")
        print("  python manage_secrets.py encrypt <key>            - Encrypt configuration")
        print("  python manage_secrets.py decrypt <key> [--apply]  - Decrypt configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
