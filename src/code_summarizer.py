import os
import json
from openai import OpenAI

# 1. Setup your NVIDIA client
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key="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"
)

def run_summarization():
    base_path = os.getcwd()
    # Updated to target the entire FingerprintJS directory
    source_dir = os.path.join(base_path, "repos", "fingerprintjs-ios", "Sources", "FingerprintJS")
    output_dir = os.path.join(base_path, "code_summaries")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"📂 Scanning all files in: {source_dir}")

    files_processed = 0
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # FILTER: We want ALL Swift files now, excluding tests/mocks
            is_swift = filename.endswith(".swift")
            is_test = "Test" in filename or "Spy" in filename or "Mock" in filename

            if is_swift and not is_test:
                file_path = os.path.join(root, filename)
                files_processed += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    print(f"🚀 Analyzing: {filename}...")

                    # UPDATED PROMPT: Ask for the "Role" and "Content" of the file
                    prompt = (
                        f"Analyze this iOS Swift file from the FingerprintJS library. \n"
                        f"1. What is the functional role of this file (e.g., Data Provider, Harvester, Hashing Logic, or Coordinator)? \n"
                        f"2. Summarize the specific technical content/logic inside. \n"
                        f"3. How does this contribute to the final device fingerprint? \n"
                        f"Return JSON with keys: 'file_role', 'content_summary', 'fingerprint_contribution'. \n\n"
                        f"Code:\n{content}"
                    )

                    # Note: Using 'meta/llama-3.1-70b-instruct' if 'nvidia/' gave a 404 earlier
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct", 
                        messages=[{"role": "user", "content": prompt}],
                        response_format={ "type": "json_object" }
                    )

                    with open(f"{output_dir}/{filename}.json", 'w') as out:
                        out.write(completion.choices[0].message.content)
                        
                except Exception as e:
                    print(f"⚠️ Error processing {filename}: {e}")

    print(f"\n✅ Success! {files_processed} files analyzed in {output_dir}")

if __name__ == "__main__":
    run_summarization()