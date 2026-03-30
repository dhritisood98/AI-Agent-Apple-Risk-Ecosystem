import os
import json
import hashlib
from supabase import create_client, Client
from openai import OpenAI
from src.embedders import Embedder, get_embedder_spec

# --- 1. CONFIGURATION ---
# Pulling from Environment Variables for GitHub Actions compatibility
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key=NVIDIA_API_KEY
)

# Initialize Local Nomic Embedder (768 dimensions)
spec = get_embedder_spec("nomic_768")
embedder = Embedder(spec)

def _sha256(text: str) -> str:
    """Generates a unique fingerprint for the file content."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def sync_ios_logic_to_supabase():
    base_path = os.getcwd()
    # Path relative to your repository root
    source_dir = os.path.join(base_path, "repos", "fingerprintjs-ios", "Sources", "FingerprintJS")
    
    if not os.path.exists(source_dir):
        print(f"❌ ERROR: Source directory not found: {source_dir}")
        return

    print(f"🚀 Starting ingestion from iOS Source: {source_dir}")

    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            is_swift = filename.endswith(".swift")
            is_test = any(x in filename for x in ["Tests", "Spy", "Mock"])

            if is_swift and not is_test:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, base_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()

                # --- NEW: THE GATEKEEPER ---
                new_hash = _sha256(code_content)

                # Check if this specific file path and hash already exist
                existing = supabase.table("code_knowledge") \
                    .select("content_hash") \
                    .eq("file_path", relative_path) \
                    .limit(1) \
                    .execute()

                if existing.data and existing.data[0].get("content_hash") == new_hash:
                    print(f"⏭️ Skipping {filename}: Content hasn't changed.")
                    continue

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
                    summary_text = f"Feature: {data['feature']}. Signal: {data['ios_signal']}. Risk: {data['risk_impact']}"

                    # --- 3. EMBED: Using Local Nomic ---
                    print(f"🔢 Generating vector embedding for: {filename}...")
                    vector = embedder.embed_document(summary_text)

                    # --- 4. PUSH: Using UPSERT to prevent duplicates ---
                    supabase.table("code_knowledge").upsert({
                        "file_path": relative_path,
                        "feature": data['feature'],
                        "content": summary_text,
                        "embedding": vector,
                        "content_hash": new_hash  # Save the hash for future checks
                    }, on_conflict="file_path").execute()

                    print(f"✅ Successfully synced {filename} to Supabase!\n")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    sync_ios_logic_to_supabase()
    print("🏁 Sync Complete. Your iOS Risk Knowledge Base is live in Supabase.")