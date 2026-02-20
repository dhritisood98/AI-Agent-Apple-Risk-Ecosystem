import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

USER_AGENT = os.environ.get("USER_AGENT", "Mozilla/5.0")