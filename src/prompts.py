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