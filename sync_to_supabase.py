import os
import json
from supabase import create_client, Client
from openai import OpenAI
# Assuming your custom embedders.py is in the src folder
from src.embedders import Embedder, get_embedder_spec

# --- 1. CONFIGURATION ---
SUPABASE_URL = "https://qxiqpqqbmoemnelsgbuy.supabase.co"
SUPABASE_KEY = "sb_secret_yQJhFCrr1mPFzlXdCedKrA_DnjomnbB"
NVIDIA_API_KEY = "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"

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key=NVIDIA_API_KEY
)

# Initialize Local Nomic Embedder (768 dimensions)
spec = get_embedder_spec("nomic_768")
embedder = Embedder(spec)

def sync_ios_logic_to_supabase():
    # UPDATED: Path to your local iOS repository structure
    base_path = os.getcwd()
    source_dir = os.path.join(base_path, "repos", "fingerprintjs-ios", "Sources", "FingerprintJS")
    
    if not os.path.exists(source_dir):
        print(f"❌ ERROR: Source directory not found: {source_dir}")
        return

    print(f"🚀 Starting ingestion from iOS Source: {source_dir}")

    # Use os.walk to get files in all subfolders (Harvesters, Providers, etc.)
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # FILTER: Target Swift files, exclude tests and internal mocks
            is_swift = filename.endswith(".swift")
            is_test = any(x in filename for x in ["Tests", "Spy", "Mock"])

            if is_swift and not is_test:
                file_path = os.path.join(root, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()

                print(f"📝 Analyzing iOS logic: {filename}...")

                # --- 2. SUMMARIZE: Using NVIDIA Llama-3.1 ---
                prompt = (
                    f"Analyze this iOS fingerprinting Swift code. Provide a summary for a fraud analyst: "
                    f"1. What signal it harvests. 2. How it identifies the device. 3. Potential for Apple to block this. "
                    f"Return ONLY a JSON object with keys: 'feature', 'ios_signal', 'risk_impact'.\n\n"
                    f"Code:\n{code_content}"
                )

                try:
                    summary_res = ai_client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                    
                    data = json.loads(summary_res.choices[0].message.content)
                    # Combine fields for the embedding content
                    summary_text = f"Feature: {data['feature']}. Signal: {data['ios_signal']}. Risk: {data['risk_impact']}"

                    # --- 3. EMBED: Using Local Nomic ---
                    print(f"🔢 Generating vector embedding for: {filename}...")
                    vector = embedder.embed_document(summary_text)

                    # --- 4. PUSH: Send to Supabase ---
                    # Note: file_path is saved as a relative path for cleaner data
                    relative_path = os.path.relpath(file_path, base_path)
                    
                    supabase.table("code_knowledge").insert({
                        "file_path": relative_path,
                        "feature": data['feature'],
                        "content": summary_text,
                        "embedding": vector
                    }).execute()

                    print(f"✅ Successfully synced {filename} to Supabase!\n")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    sync_ios_logic_to_supabase()
    print("🏁 Sync Complete. Your iOS Risk Knowledge Base is live in Supabase.")