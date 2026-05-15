from __future__ import annotations

import os
from langchain_openai import ChatOpenAI


def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=3000,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "HITL PR Agent",
        },
    )