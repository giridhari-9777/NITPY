# src/agents/groq_client.py
# ============================================================
# Groq API Client — replaces Ollama for cloud deployment
# Free API — no local server needed
# Supports: llama3-8b-8192, mixtral-8x7b, gemma-7b-it
# ============================================================

import os
import re
import requests

# ── Model mapping ────────────────────────────────────────────
# Maps your existing Ollama model names → Groq model names
MODEL_MAP = {
    "llama3"       : "llama3-8b-8192",
    "llama3:8b"    : "llama3-8b-8192",
    "mistral"      : "mixtral-8x7b-32768",
    "mistral:7b"   : "mixtral-8x7b-32768",
    "gemma3"       : "gemma-7b-it",
    "gemma"        : "gemma-7b-it",
    "qwen2.5"      : "llama3-8b-8192",   # fallback
    "phi4"         : "llama3-8b-8192",   # fallback
    "deepseek-r1"  : "llama3-8b-8192",   # fallback
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama3-8b-8192"


def get_groq_key() -> str:
    """Get Groq API key from environment."""
    return os.environ.get("GROQ_API_KEY", "").strip()


def call_groq(
    prompt      : str,
    model       : str = "llama3",
    max_tokens  : int = 500,
    temperature : float = 0.1,
    system      : str = "You are an expert oncologist."
) -> str:
    """
    Call Groq API — drop-in replacement for Ollama.
    """

    key = get_groq_key()
    if not key:
        print("Warning: GROQ_API_KEY is not set.")
        return ""

    groq_model = MODEL_MAP.get(model.lower(), DEFAULT_MODEL)

    headers = {
        "Authorization" : f"Bearer {key}",
        "Content-Type"  : "application/json",
    }

    payload = {
        "model"       : groq_model,
        "messages"    : [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens"  : max_tokens,
        "temperature" : temperature,
        "top_p"       : 1.0,
        "stream"      : False,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers = headers,
            json    = payload,
            timeout = 60
        )

        if resp.status_code == 200:
            data = resp.json()
            text = (
                data
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            # Clean any special tokens
            for tok in [
                "<|im_end|>", "<|im_start|>", "<|end|>",
                "<|assistant|>", "[/INST]", "</s>"
            ]:
                text = text.replace(tok, "")

            # Strip DeepSeek thinking tokens
            text = re.sub(
                r"<think>.*?</think>", "",
                text, flags=re.DOTALL
            ).strip()

            return text

        elif resp.status_code == 429:
            return (
                "I am currently handling many requests. "
                "Please try again in a moment."
            )

        elif resp.status_code == 401:
            return (
                "API key error. Please check GROQ_API_KEY "
                "in environment variables."
            )

        else:
            print(
                f"Groq API error: {resp.status_code} "
                f"— {resp.text[:200]}"
            )
            return ""

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."

    except Exception as e:
        print(f"Groq call failed: {e}")
        return ""


def call_groq_chat(
    messages    : list,
    model       : str   = "llama3",
    max_tokens  : int   = 500,
    temperature : float = 0.1
) -> str:
    """
    Call Groq API with full message history.
    """
    key = get_groq_key()
    if not key:
        return ""

    groq_model = MODEL_MAP.get(model.lower(), DEFAULT_MODEL)

    headers = {
        "Authorization" : f"Bearer {key}",
        "Content-Type"  : "application/json",
    }

    payload = {
        "model"       : groq_model,
        "messages"    : messages,
        "max_tokens"  : max_tokens,
        "temperature" : temperature,
        "stream"      : False,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers = headers,
            json    = payload,
            timeout = 60
        )

        if resp.status_code == 200:
            data = resp.json()
            return (
                data
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        else:
            print(f"Groq chat error: {resp.status_code}")
            return ""

    except Exception as e:
        print(f"Groq chat failed: {e}")
        return ""
