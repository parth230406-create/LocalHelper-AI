from __future__ import annotations
import os
import streamlit as st
from groq import Groq

MODEL_ID = "llama-3.1-8b-instant"

def get_groq_client():
    key = None
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    return Groq(api_key=key)

def chat_stream(messages: list[dict], temperature: float = 0.4, max_tokens: int = 800):
    client = get_groq_client()
    if not client:
        yield "\n\n⚠️ **GROQ_API_KEY missing:** Please add your `GROQ_API_KEY` in Streamlit App Settings (Secrets)."
        return

    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
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
        yield f"\n\n⚠️ **Groq API Error:** {e}"
