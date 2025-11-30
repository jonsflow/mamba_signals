"""Dynamic config loader based on environment."""
import os
from pathlib import Path

# Load .env file if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Determine which config to load based on environment
SYSTEM_TYPE = os.getenv("SYSTEM_TYPE", "dev").lower()

if SYSTEM_TYPE == "gpu":
    # Load GPU config
    from gpu_config import Config, default_config
    print(f"[Config] Loaded GPU config (SYSTEM_TYPE={SYSTEM_TYPE})")
elif SYSTEM_TYPE == "dev":
    # Load dev config (default)
    from config import Config, default_config
    print(f"[Config] Loaded DEV config (SYSTEM_TYPE={SYSTEM_TYPE})")
else:
    raise ValueError(f"Unknown SYSTEM_TYPE: {SYSTEM_TYPE}. Use 'dev' or 'gpu'")

__all__ = ["Config", "default_config", "SYSTEM_TYPE"]
