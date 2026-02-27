from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]  # or SUPABASE_KEY if that’s what you use
sb = create_client(url, key)

resp = sb.table("rag_results").insert({
    "query_id": "smoke_test_1",
    "query_text": "Does insert work?",
    "context": "test context",
    "llm_response": "test response"
}).execute()

print(resp)