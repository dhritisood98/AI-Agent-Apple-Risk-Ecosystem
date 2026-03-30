from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import requests

@dataclass
class LLMResult:
    text: str

class NoLLM:
    def generate(self, prompt: str, model: str = "") -> LLMResult:
        return LLMResult(text="(LLM disabled)")

class NIMClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, model: str) -> LLMResult:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only the final answer. No JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return LLMResult(text=data["choices"][0]["message"]["content"])