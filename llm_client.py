from __future__ import annotations
import os
import streamlit as st
from huggingface_hub import InferenceClient

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def get_token() -> str:
    # Check Streamlit Cloud Secrets first, then local environment
    if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    return os.environ.get("HF_TOKEN", "")

def chat_stream(messages: list[dict], temperature: float = 0.4, max_tokens: int = 800):
    token = get_token()
    if not token:
        yield "\n\n⚠️ **HF_TOKEN missing:** Please add your `HF_TOKEN` in the Streamlit Cloud App Settings (Secrets)."
        return

    try:
        client = InferenceClient(model=MODEL_ID, token=token)
        stream = client.chat_completion(
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n\n⚠️ **Inference API Error:** {e}"