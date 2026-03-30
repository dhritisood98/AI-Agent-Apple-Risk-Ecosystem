# src/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

@dataclass
class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY")
    nim_base_url: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

# single settings instance for imports
settings = Settings()