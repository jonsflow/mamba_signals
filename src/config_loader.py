"""Dynamic config loader based on environment."""
import os
from pathlib import Path

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
