def build_rationale_prompt(file_name: str, summary: str, effective_risk: str,
                           zs_level: str, top_bulletin: str, triggering_cve: str) -> str:
    """Prompt for LLM to generate a 2-sentence traceable rationale after scoring is complete."""
    cve_line = f"Triggering CVE: {triggering_cve}\n" if triggering_cve else ""
    return f"""You are a mobile security analyst at TransUnion. Risk scoring has already been completed deterministically. Your only job is to write a 2-sentence explanation.

File: {file_name}
What this file does: {summary[:400]}

Risk verdict (already decided): {effective_risk}
Intrinsic sensitivity (zero-shot): {zs_level}
{cve_line}Most relevant Apple bulletin excerpt:
{top_bulletin[:400]}

Write exactly 2 sentences:
1. Why this file is at risk based on the bulletin above.
2. Which specific iOS signal or capability could be affected.

Be specific. Do not repeat the risk level. Do not add headings."""


def build_prompt(query, results, system_instructions=""):
    prompt = ""

    if system_instructions:
        prompt += system_instructions.strip() + "\n\n"

    prompt += f"User Query:\n{query}\n\n"
    prompt += "Retrieved Context:\n"

    if not results:
        prompt += "No relevant context retrieved.\n"
        return prompt

    for i, res in enumerate(results, start=1):
        prompt += f"\nContext {i}:\n"
        prompt += f"- Chunk ID: {res.get('snapshot_chunk_id', 'unknown')}\n"
        prompt += f"- Similarity: {res.get('similarity', 0):.3f}\n"
        prompt += f"- Text: {res.get('chunk_text', '')}\n"

    prompt += "\nPlease answer using the retrieved context when possible."
    return prompt