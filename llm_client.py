from __future__ import annotations
import os
import streamlit as st
from groq import Groq

# Priority list of top models on Groq
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "llama-3.2-3b-preview",
    "llama3-8b-8192"
    "openai/gpt-oss-120b"
]

def get_groq_client():
    key = None
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    return Groq(api_key=key.strip())

def get_active_model(client: Groq) -> str:
    """Auto-detects which models are currently online and enabled on your Groq account."""
    try:
        available = [m.id for m in client.models.list().data]
        for candidate in CANDIDATE_MODELS:
            if candidate in available:
                return candidate
        # Fallback: pick any text model that isn't whisper
        for m in available:
            if "whisper" not in m.lower():
                return m
    except Exception:
        pass
    return "llama-3.3-70b-versatile"

def chat_stream(messages: list[dict], temperature: float = 0.4, max_tokens: int = 800):
    client = get_groq_client()
    if not client:
        yield "\n\n⚠️ **GROQ_API_KEY missing:** Please add your `GROQ_API_KEY` in Streamlit App Settings (Secrets)."
        return

    model_to_use = get_active_model(client)

    try:
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        yield f"\n\n⚠️ **Groq API Error ({model_to_use}):** {e}"
