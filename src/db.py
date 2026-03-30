import os
from supabase import create_client, Client
from dotenv import load_dotenv, find_dotenv  # 1. Import find_dotenv

# 2. This will look for the .env file in your root folder automatically
load_dotenv(find_dotenv()) 

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        # This is where your script is currently stopping
        raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in environment")
        
    return create_client(url, key)